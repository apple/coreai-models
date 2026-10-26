# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity test for the iOS Gemma4 vision encoder vs HuggingFace.

Re-authors a tiny, randomly-initialized Gemma4 vision tower + multimodal projector
(``models/ios/gemma4_vision.py``) and checks its image-soft-token output against the
HF reference path (``Gemma4VisionModel`` -> ``Gemma4MultimodalEmbedder``, i.e. the
math behind ``Gemma4Model.get_image_features``) for a fixed square grid. No weights
download -- the random config exercises every op (QK-norm, scale-less V-norm, 2D RoPE,
clipped linears, 3x3 pooling, projector) that must match for real-weight parity.

Requires transformers>=5.5 (``gemma4``).
"""

import pytest
import torch

pytest.importorskip("transformers")

try:
    from transformers import Gemma4TextConfig, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )
except Exception:  # pragma: no cover - requires transformers>=5.5
    Gemma4VisionConfig = None

from coreai_models.models.ios.gemma4_vision import Gemma4VisionEncoderForiOS  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gemma4VisionConfig is None, reason="gemma4 requires transformers>=5.5"
)

# Tiny fixed grid: patch 4, image 24 -> 6x6 = 36 patches -> pool 3x3 -> 2x2 = 4 soft tokens.
PATCH = 4
IMAGE = 24
GRID = IMAGE // PATCH  # 6
NUM_PATCHES = GRID * GRID  # 36


def _vision_config() -> "Gemma4VisionConfig":
    return Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=PATCH,
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


def _grid_position_ids() -> torch.Tensor:
    """Row-major (x=col, y=row) patch positions, matching the encoder's baked grid."""
    rows = torch.arange(GRID).repeat_interleave(GRID)
    cols = torch.arange(GRID).repeat(GRID)
    return torch.stack([cols, rows], dim=-1).unsqueeze(0)  # (1, N, 2)


def _convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Inlined HF ``Gemma4ImageProcessor.convert_image_to_patches`` (avoids the
    torchvision import the real module pulls in). ``image``: ``(C, H, W)``."""
    c, h, w = image.shape
    nph, npw = h // patch_size, w // patch_size
    x = image.reshape(c, nph, patch_size, npw, patch_size)
    x = x.permute(1, 3, 2, 4, 0)
    return x.reshape(nph * npw, -1)


def _psnr(ref: torch.Tensor, out: torch.Tensor) -> float:
    mse = torch.mean((ref - out) ** 2).item()
    if mse == 0:
        return float("inf")
    peak = (ref.max() - ref.min()).item()
    return (
        20.0 * torch.log10(torch.tensor(peak)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()
    )


def _build_model(
    *, image_size: int = IMAGE, target_dtype: torch.dtype = torch.float32
) -> "Gemma4VisionEncoderForiOS":
    """A tiny, randomly-initialized iOS encoder loaded from an HF-checkpoint-layout
    state dict, shared by every test below that needs a real (loaded, not
    zero-initialized) model rather than testing HF parity directly."""
    torch.manual_seed(0)
    vcfg = _vision_config()
    tcfg = Gemma4TextConfig(hidden_size=48, vocab_size=64)

    hf_vision = Gemma4VisionModel(vcfg).eval().float()
    hf_embed = Gemma4MultimodalEmbedder(vcfg, tcfg).eval().float()
    with torch.no_grad():
        for m in hf_vision.modules():
            if hasattr(m, "use_clipped_linears") and m.use_clipped_linears:
                m.input_min.fill_(-3.0)
                m.input_max.fill_(3.0)
                m.output_min.fill_(-5.0)
                m.output_max.fill_(5.0)

    full_sd: dict[str, torch.Tensor] = {}
    for k, v in hf_vision.state_dict().items():
        full_sd["model.vision_tower." + k] = v
    for k, v in hf_embed.state_dict().items():
        full_sd["model.embed_vision." + k] = v

    return Gemma4VisionEncoderForiOS.from_state_dict(
        full_sd,
        vcfg,
        text_hidden_size=tcfg.hidden_size,
        image_size=image_size,
        target_dtype=target_dtype,
    )


def test_vision_encoder_parity() -> None:
    torch.manual_seed(0)
    vcfg = _vision_config()
    tcfg = Gemma4TextConfig(hidden_size=48, vocab_size=64)  # only hidden_size matters here

    # HF reference modules with random weights.
    hf_vision = Gemma4VisionModel(vcfg).eval().float()
    hf_embed = Gemma4MultimodalEmbedder(vcfg, tcfg).eval().float()

    # Give the clipped linears real (finite) clip ranges so the clamp path is exercised.
    with torch.no_grad():
        for m in hf_vision.modules():
            if hasattr(m, "use_clipped_linears") and m.use_clipped_linears:
                m.input_min.fill_(-3.0)
                m.input_max.fill_(3.0)
                m.output_min.fill_(-5.0)
                m.output_max.fill_(5.0)

    # Assemble a checkpoint-layout state dict for the iOS loader.
    full_sd: dict[str, torch.Tensor] = {}
    for k, v in hf_vision.state_dict().items():
        full_sd["model.vision_tower." + k] = v
    for k, v in hf_embed.state_dict().items():
        full_sd["model.embed_vision." + k] = v

    ours = Gemma4VisionEncoderForiOS.from_state_dict(
        full_sd,
        vcfg,
        text_hidden_size=tcfg.hidden_size,
        image_size=IMAGE,
        target_dtype=torch.float32,
    )

    # Shared inputs. A random NCHW image in [0,1] (the layout Swift feeds); the HF side
    # patchifies it with the real processor helper, ours patchifies in-graph.
    image = torch.rand(1, 3, IMAGE, IMAGE, dtype=torch.float32)
    hf_patches = _convert_image_to_patches(image[0], PATCH).unsqueeze(0)  # (1, N, 3*p^2)
    pos_ids = _grid_position_ids()

    with torch.no_grad():
        vout = hf_vision(pixel_values=hf_patches, pixel_position_ids=pos_ids)
        ref = hf_embed(vout.last_hidden_state)  # (num_soft_tokens, text_hidden)
        # All-zero (fully-visible) mask: this test fills the whole bucket with real
        # content, matching HF's reference call above (no padding involved).
        attn_mask = torch.zeros(1, ours.num_patches, 1, ours.num_patches, dtype=torch.float32)
        got = ours(image, attn_mask).squeeze(0)  # (num_soft_tokens, text_hidden)

    assert got.shape == ref.shape == (ours.num_soft_tokens, tcfg.hidden_size)

    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    psnr = _psnr(ref, got)
    assert cos > 0.9999, f"cosine {cos} too low"
    assert psnr > 60.0, f"PSNR {psnr} dB too low"


def test_masked_padding_matches_smaller_unpadded_bucket() -> None:
    """The core correctness property of the ``resize_to`` size ladder: masking a
    larger bucket's padding patches out of attention must reproduce -- for the real
    region -- exactly what a smaller bucket sized to just that real content produces
    with no padding at all. This is what lets the runner support arbitrary input
    image shapes: pick the smallest bucket that fits, letterbox into it, and mask
    the rest (see the module docstring and ``resize_to``).

    Pads both axes (real content is a 24x24 square inside a 36x36 canvas), so the
    real pooled cells are *not* a contiguous prefix of the flattened output --
    exercises the harder case; ``test_masked_padding_one_axis_full`` below covers
    the simpler, single-axis-padded case.
    """
    vcfg = _vision_config()
    model = _build_model()

    # Baseline: the small bucket (IMAGE=24, grid 6x6, pooled 2x2 = 4 tokens), fully
    # real, no padding -- same shape as the base parity test above.
    torch.manual_seed(1)
    real_image = torch.rand(1, 3, IMAGE, IMAGE, dtype=torch.float32)
    small_mask = torch.zeros(1, model.num_patches, 1, model.num_patches, dtype=torch.float32)
    with torch.no_grad():
        small_out = model(real_image, small_mask).squeeze(0)  # (4, hidden)

    # Resize to a larger bucket (grid 9x9, pooled 3x3 = 9 tokens) and letterbox the
    # same real content into the top-left, zero-padding the rest -- padding is
    # exactly `pooling_kernel_size` pool-blocks wide/tall, so it's pool-block-aligned.
    large_image_size = IMAGE + PATCH * vcfg.pooling_kernel_size  # 24 + 12 = 36
    model.resize_to(large_image_size)
    padded_image = torch.zeros(1, 3, large_image_size, large_image_size, dtype=torch.float32)
    padded_image[:, :, :IMAGE, :IMAGE] = real_image

    # Mask out every padding patch as a key (row-major patch index = row*grid + col).
    grid = model.grid
    real_patches_per_side = IMAGE // PATCH  # 6
    is_real = torch.zeros(grid, grid, dtype=torch.bool)
    is_real[:real_patches_per_side, :real_patches_per_side] = True
    key_is_padding = ~is_real.reshape(model.num_patches)
    # Mask shape is (1, key_seq, 1, query_seq) -- matches primitives/ios/sdpa.py's
    # SDPA.forward causal_mask convention. Key axis is dim 1, not the last dim.
    large_mask = torch.zeros(1, model.num_patches, 1, model.num_patches, dtype=torch.float32)
    large_mask[:, key_is_padding, :, :] = -1e9

    with torch.no_grad():
        large_out = model(padded_image, large_mask).squeeze(0)  # (9, hidden)

    # Real pooled cells: the top-left 2x2 block of the 3x3 pooled grid (row-major) --
    # exactly what the runner would keep and drop the other 5 padding-only cells.
    pooled_grid = model.pooled_grid  # 3
    real_pooled_per_side = IMAGE // (PATCH * vcfg.pooling_kernel_size)  # 2
    real_indices = [
        r * pooled_grid + c
        for r in range(real_pooled_per_side)
        for c in range(real_pooled_per_side)
    ]

    for i, idx in enumerate(real_indices):
        torch.testing.assert_close(large_out[idx], small_out[i], atol=1e-4, rtol=1e-3)


def test_masked_padding_one_axis_full() -> None:
    """Complementary case to ``test_masked_padding_matches_smaller_unpadded_bucket``:
    real content fills the bucket's full width, only the height is padded, so the
    real pooled cells are a contiguous *prefix* of the flattened output -- the
    simpler, and probably more common, real-world aspect-ratio case (e.g. a wide
    photo letterboxed into a square canvas).

    Validated differently than the harder non-contiguous case above: since the
    model is square-only (one ``image_size`` -> one grid), there's no smaller square
    bucket that exactly matches a full-width/partial-height rectangle to compare
    against. Instead this checks the more fundamental property masking is supposed
    to provide directly -- the real region's output must be *invariant* to whatever
    content sits in the masked-out padding, since a masked key contributes exactly
    zero to attention regardless of its value. Two runs with the same real content
    and mask but different (all-zero vs. random) padding pixels must agree exactly
    on the real region.
    """
    pool_k = _vision_config().pooling_kernel_size
    model = _build_model()  # image_size=IMAGE=24; resize_to below moves it to 36

    large_image_size = IMAGE + PATCH * pool_k  # 24 + 12 = 36, grid=9
    model.resize_to(large_image_size)
    grid = model.grid  # 9

    # Real region: full width, top `real_height_patches` rows only -- one fewer
    # pooled row than the full pooled grid, so it's a genuine (pool-aligned) partial
    # height rather than the trivial all-real case.
    real_pooled_rows_target = model.pooled_grid - 1
    real_height_patches = real_pooled_rows_target * pool_k
    real_height_px = real_height_patches * PATCH

    is_real = torch.zeros(grid, grid, dtype=torch.bool)
    is_real[:real_height_patches, :] = True  # full width, partial height
    mask = torch.zeros(1, model.num_patches, 1, model.num_patches, dtype=torch.float32)
    mask[:, ~is_real.reshape(model.num_patches), :, :] = -1e9

    torch.manual_seed(3)
    real_content = torch.rand(1, 3, real_height_px, large_image_size, dtype=torch.float32)

    zero_padded = torch.zeros(1, 3, large_image_size, large_image_size, dtype=torch.float32)
    zero_padded[:, :, :real_height_px, :] = real_content
    garbage_padded = torch.rand(1, 3, large_image_size, large_image_size, dtype=torch.float32)
    garbage_padded[:, :, :real_height_px, :] = real_content

    with torch.no_grad():
        out_zero_pad = model(zero_padded, mask).squeeze(0)
        out_garbage_pad = model(garbage_padded, mask).squeeze(0)

    # Real pooled cells: the top `real_height_patches / pool_k` rows, every column --
    # a contiguous prefix of the flattened (row-major) pooled output.
    real_pooled_rows = real_height_patches // pool_k
    real_count = real_pooled_rows * model.pooled_grid
    assert real_count > 0 and real_count < model.num_soft_tokens, (
        "test setup error: expected a genuine partial (real < full) height"
    )
    torch.testing.assert_close(
        out_zero_pad[:real_count], out_garbage_pad[:real_count], atol=1e-5, rtol=1e-4
    )


def test_resize_to_preserves_dtype() -> None:
    """`resize_to`'s position/RoPE buffers must stay in the model's working dtype
    across a resize -- a plain `torch.zeros(...)` (no explicit dtype) would silently
    revert to fp32, breaking `forward()` at the first `x + self.pos_embeds` add once
    the model is loaded at fp16 (the runner's actual dtype). Regression test for a
    bug caught by code review, not by any test, while adding the size ladder."""
    model = _build_model(target_dtype=torch.float16)
    assert model.pos_embeds.dtype == torch.float16
    assert model.rope_cos.dtype == torch.float16
    assert model.rope_sin.dtype == torch.float16

    larger = IMAGE + PATCH * model.pool_k
    model.resize_to(larger)
    assert model.pos_embeds.dtype == torch.float16
    assert model.rope_cos.dtype == torch.float16
    assert model.rope_sin.dtype == torch.float16

    pixel_values = torch.zeros(1, 3, larger, larger, dtype=torch.float16)
    attn_mask = torch.zeros(1, model.num_patches, 1, model.num_patches, dtype=torch.float16)
    with torch.no_grad():
        out = model(pixel_values, attn_mask)  # raises on a dtype mismatch if broken
    assert out.dtype == torch.float16


def test_resize_to_multiple_buckets_in_sequence() -> None:
    """Mirrors the actual export loop (`export_gemma4_vision_encoder`): resize
    through an ascending ladder, then back down, checking geometry at every step and
    that revisiting an earlier size reproduces the exact same buffers (resize_to is
    a pure function of (image_size, the stored raw position table), not stateful
    across calls)."""
    model = _build_model()
    ladder = [IMAGE, IMAGE + PATCH * 3, IMAGE + PATCH * 6]  # e.g. 24, 36, 48

    first_pass_pos_embeds = {}
    for size in ladder:
        model.resize_to(size)
        grid = size // PATCH
        assert model.grid == grid
        assert model.num_patches == grid * grid
        assert model.pooled_grid == grid // model.pool_k
        assert model.pos_embeds.shape == (1, model.num_patches, 1, model.hidden_size)
        first_pass_pos_embeds[size] = model.pos_embeds.clone()

    # Walk back down through the same sizes; each revisit must reproduce the first
    # pass's buffers exactly (same raw table, same coordinates -> same result).
    for size in reversed(ladder):
        model.resize_to(size)
        torch.testing.assert_close(model.pos_embeds, first_pass_pos_embeds[size])
