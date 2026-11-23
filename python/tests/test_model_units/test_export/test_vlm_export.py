# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the VLM export recipe: vision geometry, model dispatch, attention swap."""

import asyncio
import copy
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
    """A decoder class no registered spec uses."""


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
# Neural Engine form
# ---------------------------------------------------------------------------

_TOWER_GEOMETRY = dict(image_size=32, patch_size=16, spatial_merge_size=1, temporal_patch_size=2)

_VISION_TOWERS = [
    pytest.param(Qwen3VLVisionConfig, Qwen3VLVisionModel, id="qwen3-vl"),
    pytest.param(Qwen3_5VisionConfig, Qwen3_5VisionModel, id="qwen3.5"),
]


def _tiny_vision_tower(config_cls, model_cls, seed: int = 0):
    """A randomly initialized vision tower small enough to run in a unit test."""
    torch.manual_seed(seed)
    config = config_cls(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=1,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=4,
        deepstack_visual_indexes=[0],
    )
    return model_cls(config).eval()


def _neural_engine_encoder(tower) -> StaticVisionEncoder:
    return StaticVisionEncoder(
        tower,
        **_TOWER_GEOMETRY,
        patchified_input=True,
        linear_patch_embed=True,
        f16_attention=True,
    ).eval()


def _count_f16_attention_calls(monkeypatch) -> list:
    """Record the attention modules that run the dtype-preserving implementation."""
    calls: list = []
    implementation = vlm_export._f16_attention_forward

    def counting(self, *args, **kwargs):
        calls.append(self)
        return implementation(self, *args, **kwargs)

    monkeypatch.setattr(vlm_export, "_f16_attention_forward", counting)
    return calls


@pytest.mark.parametrize(("config_cls", "model_cls"), _VISION_TOWERS)
class TestF16AttentionScope:
    """The f16 attention is bound to the Neural Engine encoder's own attention modules."""

    def test_plain_encoder_is_unaffected_by_a_neural_engine_encoder(self, config_cls, model_cls):
        plain = StaticVisionEncoder(
            _tiny_vision_tower(config_cls, model_cls), **_TOWER_GEOMETRY
        ).eval()
        pixels = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            before = plain(pixels)
        assert torch.isfinite(before).all()

        _neural_engine_encoder(_tiny_vision_tower(config_cls, model_cls, seed=1))

        with torch.no_grad():
            after = plain(pixels)
        assert after.shape == before.shape
        assert torch.equal(before, after)

    def test_attention_class_is_left_untouched(self, config_cls, model_cls):
        tower = _tiny_vision_tower(config_cls, model_cls)
        attention_cls = type(tower.blocks[0].attn)

        _neural_engine_encoder(tower)

        assert attention_cls.forward is not vlm_export._f16_attention_forward
        for block in tower.blocks:
            assert block.attn.forward.__func__ is vlm_export._f16_attention_forward

    def test_neural_engine_encoder_runs_the_replacement(self, monkeypatch, config_cls, model_cls):
        calls = _count_f16_attention_calls(monkeypatch)
        tower = _tiny_vision_tower(config_cls, model_cls)
        encoder = _neural_engine_encoder(tower)

        patches = torch.randn(1, encoder.num_patches, encoder.patch_dim)
        with torch.no_grad():
            features = encoder(patches)

        assert calls == [block.attn for block in tower.blocks]
        assert torch.isfinite(features).all()

    def test_export_traces_the_replacement(self, monkeypatch, config_cls, model_cls):
        calls = _count_f16_attention_calls(monkeypatch)
        tower = _tiny_vision_tower(config_cls, model_cls)
        encoder = _neural_engine_encoder(tower)

        patches = torch.randn(1, encoder.num_patches, encoder.patch_dim)
        torch.export.export(encoder, (patches,))

        assert {id(attention) for attention in calls} == {id(block.attn) for block in tower.blocks}


@pytest.mark.parametrize(("config_cls", "model_cls"), _VISION_TOWERS)
class TestNeuralEngineFormParity:
    """The Neural Engine form computes the plain encoder's features from host-patchified input.

    Both encoders wrap their own copy of one randomly initialized tower: the attention
    swap binds to the tower's modules, so they cannot share one.
    """

    def _pair(self, config_cls, model_cls) -> tuple[StaticVisionEncoder, StaticVisionEncoder]:
        tower = _tiny_vision_tower(config_cls, model_cls)
        with torch.no_grad():
            # The default init leaves attention scores near zero, where a missing scale or a
            # misrouted rotary half barely moves the output.
            for param in tower.parameters():
                param.normal_(0.0, 0.5)
        plain = StaticVisionEncoder(copy.deepcopy(tower), **_TOWER_GEOMETRY)
        neural_engine = _neural_engine_encoder(tower)
        return plain.eval(), neural_engine

    def test_f32_matches_the_plain_encoder(self, config_cls, model_cls):
        plain, neural_engine = self._pair(config_cls, model_cls)
        pixels = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            expected = plain(pixels)
            got = neural_engine(plain._patchify(pixels).unsqueeze(0))
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    def test_f16_stays_close_to_the_f32_plain_encoder(self, config_cls, model_cls):
        plain, neural_engine = self._pair(config_cls, model_cls)
        pixels = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            expected = plain(pixels)
            got = neural_engine.half()(plain._patchify(pixels).unsqueeze(0).half())
        assert got.dtype == torch.float16
        torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=2e-2)


class TestNeuralEngineFormSingleImage:
    def test_cli_refuses_multi_frame(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["coreai.vlm.export", "qwen3-vl", "--vision-neural-engine", "--num-frames", "4"],
        )
        monkeypatch.setattr(vlm_export, "_run", _refuse_download)
        with pytest.raises(SystemExit):
            vlm_export.main()

    def test_export_refuses_multi_frame_before_download(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            vlm_export, "AutoConfig", SimpleNamespace(from_pretrained=_refuse_download)
        )
        with pytest.raises(ValueError, match="single-image"):
            asyncio.run(
                vlm_export.export_vision_encoder(
                    SUPPORTED_MODELS["qwen3-vl"],
                    tmp_path,
                    overwrite=True,
                    num_frames=4,
                    neural_engine=True,
                    vision_dtype="f16",
                )
            )

    def test_export_requires_f16(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            vlm_export, "AutoConfig", SimpleNamespace(from_pretrained=_refuse_download)
        )
        with pytest.raises(ValueError, match="f16"):
            asyncio.run(
                vlm_export.export_vision_encoder(
                    SUPPORTED_MODELS["qwen3-vl"],
                    tmp_path,
                    overwrite=True,
                    neural_engine=True,
                    vision_dtype="f32",
                )
            )

    def test_cli_refuses_f32(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["coreai.vlm.export", "qwen3-vl", "--vision-neural-engine", "--vision-dtype", "f32"],
        )
        monkeypatch.setattr(vlm_export, "_run", _refuse_download)
        with pytest.raises(SystemExit):
            vlm_export.main()

    def test_cli_defaults_to_f16(self, monkeypatch, tmp_path):
        args = vlm_export.build_parser().parse_args(
            ["qwen3-vl", "--vision-only", "--vision-neural-engine", "--output-dir", str(tmp_path)]
        )
        seen = {}

        async def _record_vision(*_a, **kwargs):
            seen.update(kwargs)
            return "vision.aimodel"

        monkeypatch.setattr(vlm_export, "export_vision_encoder", _record_vision)
        asyncio.run(vlm_export._run(SUPPORTED_MODELS["qwen3-vl"], args))
        assert seen["vision_dtype"] == "f16"
        assert seen["neural_engine"] is True


# ---------------------------------------------------------------------------
# Graph entry, compression and metadata
# ---------------------------------------------------------------------------


def _export_module(tower, neural_engine: bool) -> tuple[nn.Module, tuple[int, ...]]:
    """The module ``export_vision_encoder`` hands the converter, and its pixel shape."""
    if neural_engine:
        encoder = _neural_engine_encoder(tower).half()
        module = vlm_export.BatchedF16VisionEncoder(encoder, input_cast=torch.float16)
        return module.eval(), (1, encoder.num_patches, encoder.patch_dim)
    encoder = StaticVisionEncoder(tower, **_TOWER_GEOMETRY).eval()
    return vlm_export.BatchedF16VisionEncoder(encoder).eval(), (1, 3, 32, 32)


def _assert_exports_f32_in_f16_out(module: nn.Module, pixel_shape: tuple[int, ...]) -> None:
    pixels = torch.randn(*pixel_shape, dtype=torch.float32)
    program = torch.export.export(module, (pixels,))
    (pixel_node,) = [n for n in program.graph.nodes if n.op == "placeholder" and "pixel" in n.name]
    assert pixel_node.meta["val"].dtype == torch.float32
    with torch.no_grad():
        out = program.module()(pixels)
    assert out.dtype == torch.float16
    assert out.shape == (1, 4, 32)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("neural_engine", [False, True], ids=["f32", "neural-engine-f16"])
class TestVisionGraphEntry:
    def test_exports_with_f32_pixels_and_f16_features(self, neural_engine):
        module, pixel_shape = _export_module(
            _tiny_vision_tower(Qwen3VLVisionConfig, Qwen3VLVisionModel), neural_engine
        )
        _assert_exports_f32_in_f16_out(module, pixel_shape)

    def test_4bit_quantized_encoder_exports(self, neural_engine):
        module, pixel_shape = _export_module(
            _tiny_vision_tower(Qwen3VLVisionConfig, Qwen3VLVisionModel), neural_engine
        )
        compressed = vlm_export._compress_vision_encoder(module, pixel_shape, "4bit")
        assert any(hasattr(m, "parametrizations") for m in compressed.modules()), (
            "no weight was quantized"
        )
        _assert_exports_f32_in_f16_out(compressed, pixel_shape)


class TestVisionCompression:
    def test_none_returns_the_module(self):
        module = nn.Linear(2, 2)
        assert vlm_export._compress_vision_encoder(module, (1, 2), "none") is module

    def test_rejects_an_unknown_preset(self):
        with pytest.raises(ValueError, match="4bit_weight_palettized_group32"):
            vlm_export._compress_vision_encoder(nn.Linear(2, 2), (1, 2), "8bit")

    def test_cli_offers_the_presets(self):
        parser = vlm_export.build_parser()
        for preset in vlm_export.VISION_COMPRESSION_PRESETS:
            assert parser.parse_args(["qwen3-vl", "--vision-compression", preset])
        with pytest.raises(SystemExit):
            parser.parse_args(["qwen3-vl", "--vision-compression", "8bit"])

    def test_palettization_uses_the_preset_on_f32_pixels(self, monkeypatch):
        from coreai_models.export import compression
        from coreai_models.export.presets import get_preset

        seen = {}

        def _capture(module, example_inputs, config, *_a, **_k):
            seen.update(module=module, example_inputs=example_inputs, config=config)
            return module

        monkeypatch.setattr(compression, "palettize_pytorch_model", _capture)
        module = nn.Linear(2, 2)
        out = vlm_export._compress_vision_encoder(
            module, (1, 4, 2), "4bit_weight_palettized_group32"
        )
        assert out is module
        assert (
            seen["config"]
            == get_preset("4bit_weight_palettized_group32")["torch_palettization_config"]
        )
        (pixels,) = seen["example_inputs"]
        assert pixels.shape == (1, 4, 2)
        assert pixels.dtype == torch.float32


class TestVisionMetadata:
    def _bundle(self, tmp_path, vision: dict):
        (tmp_path / "metadata.json").write_text(
            json.dumps({"assets": {"main": "m.aimodel"}, "vision": vision})
        )
        return tmp_path

    def test_single_frame_export_drops_video_fields(self, tmp_path):
        spec = SUPPORTED_MODELS["qwen3-vl"]
        stale = {
            **vlm_export._vision_metadata(spec),
            "image_token_count": 4 * spec.num_visual_tokens,
            "max_video_frames": 4,
            "tokens_per_frame": spec.num_visual_tokens,
            "temporal_patch_size": 2,
        }
        bundle = self._bundle(tmp_path, stale)

        vlm_export._write_vision_metadata(bundle, spec, 1, spec.num_visual_tokens)

        metadata = json.loads((bundle / "metadata.json").read_text())
        assert metadata["vision"] == vlm_export._vision_metadata(spec)
        assert metadata["assets"] == {"main": "m.aimodel", "vision": "vision.aimodel"}

    def test_multi_frame_export_records_frames(self, tmp_path):
        spec = SUPPORTED_MODELS["qwen3-vl"]
        bundle = self._bundle(tmp_path, vlm_export._vision_metadata(spec))

        vlm_export._write_vision_metadata(bundle, spec, 4, 2 * spec.num_visual_tokens)

        vision = json.loads((bundle / "metadata.json").read_text())["vision"]
        assert vision["image_token_count"] == 2 * spec.num_visual_tokens
        assert vision["max_video_frames"] == 4
        assert vision["tokens_per_frame"] == spec.num_visual_tokens

    def test_vision_only_manifest_uses_the_shared_version(self, tmp_path):
        from coreai_models._constants import METADATA_VERSION

        bundle = vlm_export._prepare_vision_only_bundle(SUPPORTED_MODELS["qwen3.5-0.8b"], tmp_path)
        metadata = json.loads((bundle / "metadata.json").read_text())
        assert metadata["metadata_version"] == METADATA_VERSION


# ---------------------------------------------------------------------------
# Patch order shared with the Swift runtime
# ---------------------------------------------------------------------------


class TestPatchOrderSharedWithSwift:
    """``VisionPatchifyTests.swift`` asserts these same rows for the host-side patchify.

    An 8x8 image whose CHW values are their own flat index, at patch 2, merge 2 and two
    temporal copies: row ``r`` holds, for each channel, both temporal copies of the 2x2
    pixels below and right of ``topLeft``.
    """

    @staticmethod
    def _patch_row(top_left: int) -> list[int]:
        row = []
        for channel in (0, 64, 128):
            pixels = [top_left + channel + offset for offset in (0, 1, 8, 9)]
            row += pixels + pixels
        return row

    def test_rows_match_the_swift_expectations(self):
        visual = _FakeVisualModel(patch_dim=2 * 3 * 2 * 2)
        encoder = StaticVisionEncoder(
            visual, image_size=8, patch_size=2, spatial_merge_size=2, temporal_patch_size=2
        )
        pixels = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        patches = encoder._patchify(pixels).to(torch.int64)

        assert patches.shape == (16, 24)
        for row, top_left in ((0, 0), (2, 16), (5, 6), (15, 54)):
            assert patches[row].tolist() == self._patch_row(top_left), row


class TestCliCombinations:
    @pytest.mark.parametrize(
        "extra",
        [
            ["--skip-vision", "--vision-neural-engine"],
            ["--skip-vision", "--vision-dtype", "f16"],
            ["--skip-vision", "--vision-compression", "4bit"],
            ["--vision-only", "--compression", "4bit"],
            ["--vision-only", "--num-layers", "2"],
        ],
    )
    def test_refuses_options_the_run_would_ignore(self, monkeypatch, extra):
        monkeypatch.setattr("sys.argv", ["coreai.vlm.export", "qwen3-vl", *extra])
        monkeypatch.setattr(vlm_export, "_run", _refuse_download)
        with pytest.raises(SystemExit):
            vlm_export.main()


def test_vision_compression_choices_are_existing_presets():
    from coreai_models.export.presets import ALL_PRESET_NAMES

    assert set(vlm_export.VISION_COMPRESSION_PRESETS) <= set(ALL_PRESET_NAMES)
