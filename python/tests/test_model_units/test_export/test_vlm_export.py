# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the VLM export recipe: vision geometry, model dispatch, attention swap."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from coreai_models.models.macos.qwen3_vl import Qwen3VLForCausalLMEmbeddings
from coreai_models.vlm import export as vlm_export
from coreai_models.vlm.export import SUPPORTED_MODELS, StaticVisionEncoder


class _FakeVisualModel(nn.Module):
    """Minimal stand-in for the HF visual backbone."""

    def __init__(self, hidden: int = 32, patch_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, hidden, bias=False)
        self.blocks = nn.ModuleList()
        self.merger = nn.Linear(hidden, hidden, bias=False)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 32)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 8)


_COMMON = dict(image_size=32, patch_size=16, spatial_merge_size=1, temporal_patch_size=2)


class TestStaticVisionEncoderGridT:
    def test_single_image_default(self):
        enc = StaticVisionEncoder(_FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON)
        assert enc.grid_t == 1
        assert enc.num_frames == 1

    def test_single_image_explicit(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        assert enc.grid_t == 1

    def test_multi_frame_divisible(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        assert enc.grid_t == 2
        assert enc.num_patches == 2 * 2 * 2  # grid_t * grid_h * grid_w

    def test_multi_frame_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            StaticVisionEncoder(
                _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=3
            )

    def test_patchify_single_image_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        pixels = torch.randn(1, 3, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)

    def test_patchify_multi_frame_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        pixels = torch.randn(1, 3 * 4, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)


# ---------------------------------------------------------------------------
# Text decoder dispatch
# ---------------------------------------------------------------------------


class _StopAfterDispatch(Exception):
    """Ends the stubbed export once the chosen decoder class is observable."""


def _refuse_download(*_args, **_kwargs):
    raise AssertionError("weights must not be fetched")


def _stub_auto_config(hidden_size: int = 8, vocab_size: int = 16):
    text_config = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    return SimpleNamespace(
        from_pretrained=lambda *_a, **_k: SimpleNamespace(text_config=text_config)
    )


class _OtherDecoder:
    """A decoder class no registered spec uses, to tell dispatch from a fixed choice."""


class TestTextDecoderDispatch:
    def test_qwen3_vl_declares_its_reauthored_decoder(self):
        assert SUPPORTED_MODELS["qwen3-vl"].text_decoder_class is Qwen3VLForCausalLMEmbeddings

    def test_text_export_loads_the_decoder_the_spec_declares(self, monkeypatch, tmp_path):
        spec = replace(SUPPORTED_MODELS["qwen3-vl"], text_decoder_class=_OtherDecoder)
        seen = {}

        def _capture(*, model_class, **_kwargs):
            seen["model_class"] = model_class
            raise _StopAfterDispatch

        monkeypatch.setattr(vlm_export, "snapshot_download", lambda *_a, **_k: str(tmp_path))
        monkeypatch.setattr(vlm_export, "AutoConfig", _stub_auto_config())
        monkeypatch.setattr(vlm_export, "load_model_from_safetensors", _capture)

        with pytest.raises(_StopAfterDispatch):
            asyncio.run(
                vlm_export.export_text_bundle(
                    spec,
                    max_ctx=64,
                    num_layers=1,
                    output_dir=tmp_path,
                    overwrite=True,
                )
            )
        assert seen["model_class"] is _OtherDecoder

    def test_vision_only_model_refuses_text_export_before_download(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vlm_export, "snapshot_download", _refuse_download)
        monkeypatch.setattr(vlm_export, "load_model_from_safetensors", _refuse_download)

        with pytest.raises(ValueError, match=r"qwen3\.5-0\.8b.*vision-encoder"):
            asyncio.run(
                vlm_export.export_text_bundle(
                    SUPPORTED_MODELS["qwen3.5-0.8b"],
                    max_ctx=64,
                    num_layers=1,
                    output_dir=tmp_path,
                    overwrite=True,
                )
            )
        assert list(tmp_path.iterdir()) == []


class TestVisionOnlyBundle:
    def test_seeds_a_manifest_the_vision_export_can_patch(self, tmp_path):
        spec = SUPPORTED_MODELS["qwen3.5-0.8b"]
        bundle_path = vlm_export._prepare_vision_only_bundle(spec, tmp_path)

        assert bundle_path == tmp_path / spec.output_name
        metadata = json.loads((bundle_path / "metadata.json").read_text())
        assert metadata["assets"] == {}
        assert metadata["vision"]["image_token_id"] == spec.image_token_id
        assert metadata["vision"]["image_token_count"] == spec.num_visual_tokens

    def test_keeps_an_existing_manifest(self, tmp_path):
        spec = SUPPORTED_MODELS["qwen3-vl"]
        bundle_path = tmp_path / spec.output_name
        bundle_path.mkdir()
        (bundle_path / "metadata.json").write_text('{"assets": {"main": "kept.aimodel"}}')

        vlm_export._prepare_vision_only_bundle(spec, tmp_path)
        metadata = json.loads((bundle_path / "metadata.json").read_text())
        assert metadata["assets"] == {"main": "kept.aimodel"}

    def test_cli_refuses_vision_only_together_with_skip_vision(self):
        with pytest.raises(SystemExit):
            vlm_export.build_parser().parse_args(["qwen3-vl", "--skip-vision", "--vision-only"])

    def test_cli_names_the_switch_for_a_model_without_a_text_decoder(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["coreai.vlm.export", "qwen3.5-0.8b"])
        monkeypatch.setattr(vlm_export, "_run", _refuse_download)

        with pytest.raises(SystemExit, match=r"Error: 'qwen3\.5-0\.8b'.*--vision-only"):
            vlm_export.main()

    def test_run_exports_only_the_vision_encoder(self, monkeypatch, tmp_path):
        spec = SUPPORTED_MODELS["qwen3.5-0.8b"]
        args = vlm_export.build_parser().parse_args(
            ["qwen3.5-0.8b", "--vision-only", "--output-dir", str(tmp_path)]
        )
        seen = {}

        async def _refuse_text_bundle(*_a, **_k):
            raise AssertionError("the text bundle must not be exported")

        async def _record_vision(_spec, bundle_path, *_a, **_k):
            seen["bundle_path"] = bundle_path
            return "vision.aimodel"

        monkeypatch.setattr(vlm_export, "export_text_bundle", _refuse_text_bundle)
        monkeypatch.setattr(vlm_export, "export_vision_encoder", _record_vision)

        bundle_path = asyncio.run(vlm_export._run(spec, args))
        assert seen["bundle_path"] == bundle_path == tmp_path / spec.output_name
        assert (bundle_path / "metadata.json").exists()


# ---------------------------------------------------------------------------
# ANE-friendly attention
# ---------------------------------------------------------------------------

_TOWER_GEOMETRY = dict(image_size=32, patch_size=16, spatial_merge_size=1, temporal_patch_size=2)

_VISION_TOWERS = [
    pytest.param(Qwen3VLVisionConfig, Qwen3VLVisionModel, id="qwen3-vl"),
    pytest.param(Qwen3_5VisionConfig, Qwen3_5VisionModel, id="qwen3.5"),
]
