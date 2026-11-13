# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine Sana Sprint components (``models/ios/sana_*.py``) vs diffusers / transformers.

Tiny random configs, float32: every rewrite in the re-authored modules is exact, so each must
match its source to well above the 70 dB re-authoring gate.
"""

import pytest
import torch
import torch.nn.functional as F

from coreai_models.diffusion.sana import TEXT_SEQUENCE_LENGTH, SanaTextEncoderWrapper
from coreai_models.models.ios.sana_dc_ae_decoder import (
    SanaDCAEDecoder,
    _merged_upsample_conv,
    _shuffle_shortcut,
)
from coreai_models.models.ios.sana_gemma2_encoder import SanaGemma2Encoder
from coreai_models.models.ios.sana_transformer import SanaTransformer
from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm


def _psnr(got: torch.Tensor, want: torch.Tensor) -> float:
    mse = (got.double() - want.double()).pow(2).mean()
    return float(10 * torch.log10(want.double().abs().max() ** 2 / mse))


def _randomize(module: torch.nn.Module, seed: int) -> None:
    """Non-trivial norm weights, biases and tables (diffusers initializes many to 0 or 1)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.add_(0.1 * torch.randn(p.shape, generator=gen))


def test_max_abs_rms_norm_add_floor_matches_clamp() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 1, 16) * 300
    x[..., 3] = 0  # an all-zero column stays zero under both floors
    a, b = MaxAbsRMSNorm(64, 1e-5, axis=-3), MaxAbsRMSNorm(64, 1e-5, axis=-3, floor="add")
    with torch.no_grad():
        a.weight.normal_()
        b.weight.copy_(a.weight)
        assert _psnr(b(x), a(x)) > 100
        assert torch.equal(b(x)[..., 3], torch.zeros_like(x[:, :, :, 3]))


def test_merged_upsample_conv_matches_conv_of_nearest() -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(6, 5, 3, padding=1)
    x = torch.randn(1, 6, 7, 9)
    want = conv(F.interpolate(x, scale_factor=2, mode="nearest"))
    assert _psnr(_merged_upsample_conv(conv)(x), want) > 120


@pytest.mark.parametrize("cin,cout", [(8, 8), (8, 4), (16, 8)])
def test_shuffle_shortcut_matches_pixel_shuffle(cin: int, cout: int) -> None:
    x = torch.randn(1, cin, 5, 6)
    r = cout * 4 // cin
    want = F.pixel_shuffle(x.repeat_interleave(r, dim=1), 2)
    assert torch.equal(_shuffle_shortcut(cin, cout)(x), want)


def _tiny_sana_transformer():
    from diffusers import SanaTransformer2DModel

    torch.manual_seed(0)
    src = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=4,
        attention_head_dim=32,
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=64,
        cross_attention_dim=128,
        caption_channels=48,
        mlp_ratio=2.5,
        sample_size=8,
        patch_size=1,
        norm_elementwise_affine=False,
        qk_norm="rms_norm_across_heads",
        guidance_embeds=True,
        guidance_embeds_scale=0.1,
        interpolation_scale=None,
    ).eval()
    _randomize(src, seed=1)
    return src


def _tiny_sana_transformer_inputs() -> tuple[torch.Tensor, ...]:
    """Latent, text, mask, σ and the raw guidance scale."""
    mask = torch.ones(1, TEXT_SEQUENCE_LENGTH)
    mask[:, 20:] = 0
    text = torch.randn(1, TEXT_SEQUENCE_LENGTH, 48)
    return torch.randn(1, 4, 8, 8), text, mask, torch.tensor([0.8]), torch.tensor([4.5])


def test_transformer_matches_diffusers() -> None:
    src = _tiny_sana_transformer()
    m = SanaTransformer.from_source_model(src)
    latent, text, mask, sigma, guidance = _tiny_sana_transformer_inputs()
    with torch.no_grad():
        want = src(
            latent,
            encoder_hidden_states=text,
            encoder_attention_mask=mask,
            timestep=sigma,
            guidance=guidance * 0.1,
            return_dict=False,
        )[0]
        got = m(latent, text, mask, sigma, guidance)
    assert got.shape == want.shape
    assert _psnr(got, want) > 70


def test_dc_ae_decoder_matches_diffusers() -> None:
    from diffusers import AutoencoderDC

    torch.manual_seed(0)
    vae = AutoencoderDC(
        in_channels=3,
        latent_channels=4,
        attention_head_dim=8,
        encoder_block_types="ResBlock",
        decoder_block_types=(
            "ResBlock",
            "ResBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
        ),
        encoder_block_out_channels=(8, 16, 32, 32),
        decoder_block_out_channels=(8, 16, 32, 32),
        encoder_layers_per_block=(1, 1, 1, 1),
        decoder_layers_per_block=(1, 2, 1, 2),
        encoder_qkv_multiscales=((), (), (5,), (5,)),
        decoder_qkv_multiscales=((), (), (5,), (5,)),
        upsample_block_type="interpolate",
        decoder_norm_types="rms_norm",
        decoder_act_fns="silu",
    ).eval()
    _randomize(vae.decoder, seed=2)
    m = SanaDCAEDecoder.from_source_model(vae.decoder, latent_size=4)
    z = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        want = vae.decoder(z)
        got = m(z)
    assert got.shape == want.shape == (1, 3, 32, 32)
    assert _psnr(got, want) > 70


def _tiny_gemma2() -> torch.nn.Module:
    from transformers import Gemma2Config, Gemma2Model

    torch.manual_seed(0)
    cfg = Gemma2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=4096,
        attn_implementation="sdpa",
    )
    src = Gemma2Model(cfg).eval()
    _randomize(src, seed=3)
    return src


def _tiny_gemma2_inputs(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = TEXT_SEQUENCE_LENGTH + 10
    ids = torch.randint(1, vocab_size, (1, length), generator=torch.Generator().manual_seed(4))
    mask = torch.ones(1, length, dtype=torch.long)
    mask[:, 200:] = 0  # right padding, as the runtime pads
    return ids, mask


def test_gemma2_encoder_matches_transformers_sdpa() -> None:
    src = _tiny_gemma2()
    cfg = src.config
    length = TEXT_SEQUENCE_LENGTH + 10
    ids, mask = _tiny_gemma2_inputs(cfg.vocab_size)
    m = SanaGemma2Encoder.from_source_model(src, length)
    with torch.no_grad():
        want = SanaTextEncoderWrapper(src)(ids, mask)
        got = m(ids, mask)
    keep = torch.cat([mask[:, :1], mask[:, -(TEXT_SEQUENCE_LENGTH - 1) :]], 1)[0].bool()
    assert got.shape == want.shape
    assert _psnr(got[:, keep], want[:, keep]) > 70


def test_gemma2_encoder_palettizes_with_the_neural_engine_preset() -> None:
    """``te6bit-pal`` palettizes every conv, leaves the token table dense, and still encodes."""
    from coreai_models.diffusion.pipeline import _palettize_weights
    from coreai_models.diffusion.presets import PRESETS

    src = _tiny_gemma2()
    ids, mask = _tiny_gemma2_inputs(src.config.vocab_size)
    m = SanaGemma2Encoder.from_source_model(src, ids.shape[1])
    with torch.no_grad():
        dense = m(ids, mask)
    config = PRESETS["te6bit-pal"]["component_palettization"]["text_encoder"]
    m = _palettize_weights(m, config, (ids, mask))
    state = m.state_dict()
    convs = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d)]
    assert convs and all(f"{n}.parametrizations.weight.0.lut" in state for n in convs)
    assert "embed_tokens.weight" in state
    with torch.no_grad():
        got = m.half()(ids, mask).float()
    keep = torch.cat([mask[:, :1], mask[:, -(TEXT_SEQUENCE_LENGTH - 1) :]], 1)[0].bool()
    assert torch.isfinite(got).all()
    assert _psnr(got[:, keep], dense[:, keep]) > 25


def test_gemma2_encoder_takes_an_int8_token_table() -> None:
    """``te6bit-tok8-tx8bit-pal`` stores the token table as int8 rows and still encodes."""
    from coreai_models.diffusion.pipeline import _palettize_weights, _quantize_component_weights
    from coreai_models.diffusion.presets import PRESETS

    src = _tiny_gemma2()
    ids, mask = _tiny_gemma2_inputs(src.config.vocab_size)
    m = SanaGemma2Encoder.from_source_model(src, ids.shape[1])
    with torch.no_grad():
        dense = m(ids, mask)
    preset = PRESETS["te6bit-tok8-tx8bit-pal"]
    _quantize_component_weights(
        m, m, (ids, mask), preset["component_quantization"]["text_encoder"], set()
    )
    m = _palettize_weights(m, preset["component_palettization"]["text_encoder"], (ids, mask))
    state = m.state_dict()
    table = state["embed_tokens.parametrizations.weight.0.quantized_data"]
    assert table.dtype == torch.int8
    assert state["embed_tokens.parametrizations.weight.0.scale"].shape == (table.shape[0], 1)
    with torch.no_grad():
        got = m.half()(ids, mask).float()
    keep = torch.cat([mask[:, :1], mask[:, -(TEXT_SEQUENCE_LENGTH - 1) :]], 1)[0].bool()
    assert torch.isfinite(got).all()
    assert _psnr(got[:, keep], dense[:, keep]) > 25


def test_transformer_palettizes_with_the_neural_engine_preset() -> None:
    """The preset palettizes the block convs and keeps the conditioning path dense."""
    from coreai_models.diffusion.pipeline import _palettize_weights
    from coreai_models.diffusion.presets import PRESETS

    m = SanaTransformer.from_source_model(_tiny_sana_transformer())
    inputs = _tiny_sana_transformer_inputs()
    with torch.no_grad():
        dense = m(*inputs)
    config = PRESETS["te6bit-tok8-tx8bit-pal"]["component_palettization"]["transformer"]
    m = _palettize_weights(m, config, inputs)
    state = m.state_dict()
    convs = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d)]
    palettized = {n for n in convs if f"{n}.parametrizations.weight.0.lut" in state}
    kept = {"time_linear", "caption_linear_1", "caption_linear_2", "patch_embed", "proj_out"}
    kept |= {n for n in convs if "embedder." in n or n.endswith("ff.conv_depth")}
    assert palettized == set(convs) - kept
    assert any(n.startswith("blocks.1.") for n in palettized)
    latent, text, mask, sigma, guidance = inputs
    with torch.no_grad():
        got = m.to_neural_engine_dtypes()(latent.half(), text.half(), mask.half(), sigma, guidance)
    assert torch.isfinite(got).all()
    assert _psnr(got.float(), dense) > 25
