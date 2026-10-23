# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the size-ladder validation in export/gemma4_vision.py.

Only exercises `_validated_image_sizes` (pure, no checkpoint/network access) --
`export_gemma4_vision_encoder` itself needs a gated HF checkpoint download, like
`test_gemma4.py::test_e2b_full_parity_hf_vs_torch`, which is skipped for the same
reason.
"""

import pytest

from coreai_models.export.gemma4_vision import SHIPPING_IMAGE_SIZE_LADDER, _validated_image_sizes


def test_shipping_ladder_is_valid() -> None:
    """The shipping default must satisfy its own alignment rule (patch 16, pool 3
    for E2B/E4B) -- a regression guard against editing the constant without
    checking divisibility."""
    result = _validated_image_sizes(
        SHIPPING_IMAGE_SIZE_LADDER, patch_size=16, pooling_kernel_size=3
    )
    assert result == tuple(sorted(SHIPPING_IMAGE_SIZE_LADDER))


def test_sorts_unsorted_input() -> None:
    result = _validated_image_sizes((768, 384, 1152), patch_size=16, pooling_kernel_size=3)
    assert result == (384, 768, 1152)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _validated_image_sizes((), patch_size=16, pooling_kernel_size=3)


def test_rejects_a_size_not_a_multiple_of_the_block() -> None:
    # 16*3=48; 400 is not a multiple of 48.
    with pytest.raises(ValueError, match="not a multiple"):
        _validated_image_sizes((384, 400, 768), patch_size=16, pooling_kernel_size=3)


def test_accepts_a_single_bucket() -> None:
    result = _validated_image_sizes((768,), patch_size=16, pooling_kernel_size=3)
    assert result == (768,)


@pytest.mark.parametrize("sizes", [(0,), (-48,), (384, 384), (384, 0, 768), (384, 768, 384)])
def test_image_sizes_reject_non_positive_and_duplicate_sizes(sizes):
    with pytest.raises(ValueError):
        _validated_image_sizes(sizes, patch_size=16, pooling_kernel_size=3)


@pytest.mark.parametrize("patch_size,pool", [(0, 3), (16, 0), (-16, 3), (16, -3)])
def test_image_sizes_reject_a_non_positive_block(patch_size, pool):
    with pytest.raises(ValueError):
        _validated_image_sizes((384,), patch_size=patch_size, pooling_kernel_size=pool)


def test_validate_image_sizes_for_model_uses_the_checkpoint_geometry(monkeypatch):
    from types import SimpleNamespace

    from coreai_models.export import gemma4_vision

    config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16, pooling_kernel_size=3))
    monkeypatch.setattr(gemma4_vision.AutoConfig, "from_pretrained", lambda _: config)
    assert gemma4_vision.validate_image_sizes_for_model("m", (768, 384)) == (384, 768)
    with pytest.raises(ValueError, match="multiple"):
        gemma4_vision.validate_image_sizes_for_model("m", (100,))


def test_patch_metadata_keeps_original_when_write_fails(tmp_path) -> None:
    import json

    from coreai_models.export.gemma4_vision import _patch_metadata

    meta = tmp_path / "metadata.json"
    original = json.dumps({"kind": "llm"})
    meta.write_text(original)
    with pytest.raises(TypeError):
        _patch_metadata(
            tmp_path,
            patch_size=16,
            pooling_kernel_size=3,
            image_sizes=(48,),
            image_token_id=object(),  # not JSON serializable
            boi_token_id=None,
            eoi_token_id=None,
            pad_token_id=None,
        )
    assert meta.read_text() == original


def test_patch_metadata_records_sizes_and_derived_fields(tmp_path) -> None:
    import json

    from coreai_models.export.gemma4_vision import _patch_metadata, encoder_function_name

    (tmp_path / "metadata.json").write_text(json.dumps({"kind": "llm", "assets": {"main": "m"}}))
    _patch_metadata(
        tmp_path,
        patch_size=16,
        pooling_kernel_size=3,
        image_sizes=(384, 768, 1152),
        image_token_id=7,
        boi_token_id=8,
        eoi_token_id=None,
        pad_token_id=0,
    )
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["kind"] == "vlm" and meta["assets"]["vision"] == "vision.aimodel"
    assert meta["vision"]["image_sizes"] == [384, 768, 1152]
    assert meta["vision"]["image_size"] == 1152 and meta["vision"]["image_token_count"] == 576
    assert "eoi_token_id" not in meta["vision"] and meta["vision"]["boi_token_id"] == 8
    assert meta["language"]["function_map"]["vision"] == [
        encoder_function_name(s) for s in (384, 768, 1152)
    ]
    assert encoder_function_name(384) == "encode_image_384"


def _tiny_encoder():
    pytest.importorskip("transformers")
    try:
        from transformers import Gemma4VisionConfig
    except ImportError:  # pragma: no cover - requires transformers>=5.5
        pytest.skip("gemma4 requires transformers>=5.5")
    from coreai_models.models.ios.gemma4_vision import Gemma4VisionEncoderForiOS

    config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=4,
        pooling_kernel_size=3,
        position_embedding_size=64,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        use_clipped_linears=True,
        standardize=False,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    return Gemma4VisionEncoderForiOS(config, text_hidden_size=48, image_size=24).eval()


def test_write_vision_asset_converts_and_saves_every_size(tmp_path) -> None:
    """Runs the conversion and save that `export_gemma4_vision_encoder` performs, on a tiny
    encoder, so a call that the installed Core AI version does not support fails here."""
    import torch

    from coreai_models.export.gemma4_vision import _write_vision_asset, encoder_function_name

    encoder = _tiny_encoder()
    vision_path = tmp_path / "vision.aimodel"
    _write_vision_asset(
        encoder,
        encoder,
        (24, 48),
        target_dtype=torch.float32,
        vision_path=vision_path,
        hf_model_id="google/gemma-4-E2B-it",
        overwrite=False,
    )
    assert (vision_path / "metadata.json").is_file()
    program = (vision_path / "main.mlirb").read_bytes()
    for size in (24, 48):
        assert encoder_function_name(size).encode() in program

    with pytest.raises(FileExistsError, match="--overwrite"):
        _write_vision_asset(
            encoder,
            encoder,
            (24,),
            target_dtype=torch.float32,
            vision_path=vision_path,
            hf_model_id="google/gemma-4-E2B-it",
            overwrite=False,
        )
