# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for the Gemma 4 vision-language export path.

These build *tiny, randomly-initialized* HF Gemma 4 configs (no gated weights,
no network) and check that:

  1. :class:`Gemma4StaticVisionEncoder` reproduces HF
     ``Gemma4ForConditionalGeneration.get_image_features`` for a single full
     tile (bit-close up to the export f16 cast).
  2. :class:`Gemma4ForCausalLMEmbeddings` (the VLM decoder graph that takes
     ``inputs_embeds``) produces identical logits to the text-only
     :class:`Gemma4ForCausalLM` when fed the matching (embed_scale-multiplied)
     token embeddings.

Mirrors the Qwen3-VL vision/text parity approach.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore")


def _psnr(ref: torch.Tensor, out: torch.Tensor) -> float:
    r = ref.float().reshape(-1)
    o = out.float().reshape(-1)
    assert r.shape == o.shape, f"shape mismatch {r.shape} vs {o.shape}"
    mse = torch.mean((r - o) ** 2).item()
    if mse == 0.0:
        return float("inf")
    denom = torch.mean(r**2).item()
    return 10 * math.log10(denom / mse)


def _tiny_text_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=200,
        hidden_size_per_layer_input=8,
        sliding_window=16,
        max_position_embeddings=64,
    )


def _tiny_full_config(image_size: int = 96, patch_size: int = 16, pooling_kernel_size: int = 3):
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4VisionConfig,
    )

    vc = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
        position_embedding_size=64,
        standardize=False,
    )
    return Gemma4Config(text_config=_tiny_text_config(), vision_config=vc, audio_config=None)


def test_vision_encoder_parity_with_hf() -> None:
    """Gemma4StaticVisionEncoder == HF get_image_features for a single tile."""
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )

    from coreai_models.models.macos.gemma4_vision import build_vision_encoder

    image_size, patch_size = 96, 16
    cfg = _tiny_full_config(image_size=image_size, patch_size=patch_size)
    torch.manual_seed(0)
    hf = Gemma4ForConditionalGeneration(cfg).eval()

    grid = image_size // patch_size
    num_patches = grid * grid
    pixels = torch.rand(1, 3, image_size, image_size)

    # HF reference: patchify + grid position ids, then get_image_features.
    x = pixels[0].reshape(3, grid, patch_size, grid, patch_size)
    x = x.permute(1, 3, 2, 4, 0).reshape(num_patches, 3 * patch_size * patch_size).unsqueeze(0)
    gx, gy = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    pos = torch.stack([gx, gy], dim=-1).reshape(num_patches, 2).unsqueeze(0).long()
    with torch.no_grad():
        ref_out = hf.get_image_features(x, pos)
    ref = ref_out.pooler_output if hasattr(ref_out, "pooler_output") else ref_out

    enc = build_vision_encoder(hf, image_size=image_size)
    with torch.no_grad():
        out = enc(pixels)

    assert tuple(out.shape) == (1, ref.shape[-2], ref.shape[-1])
    psnr = _psnr(ref, out)
    # f16-cast output vs f32 reference; 60 dB is a comfortable floor.
    assert psnr > 60.0, f"vision parity PSNR too low: {psnr:.2f} dB"


def test_text_embeddings_variant_matches_id_variant() -> None:
    """Gemma4ForCausalLMEmbeddings == Gemma4ForCausalLM given matching embeddings."""
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from coreai_models.models.macos.gemma4_text import (
        Gemma4ForCausalLM,
        Gemma4ForCausalLMEmbeddings,
    )

    tc = _tiny_text_config()
    torch.manual_seed(0)
    mid = Gemma4ForCausalLM(tc, model_device="cpu").eval().to(torch.float32)
    memb = Gemma4ForCausalLMEmbeddings(tc, model_device="cpu").eval().to(torch.float32)

    shared = {k: v for k, v in mid.state_dict().items() if not k.startswith("model.embed_tokens")}
    memb.load_state_dict(shared, strict=False)

    max_ctx, query_len = 64, 8
    input_ids = torch.randint(1, tc.vocab_size, (1, query_len))
    position_ids = torch.arange(query_len, dtype=torch.int32).unsqueeze(0)
    k_c, v_c = mid._build_sliding_kv_cache_tensors(tc, tc.sliding_window, torch.float32)
    k_f, v_f = mid._build_full_kv_cache_tensors(tc, max_ctx, torch.float32)
    ple = torch.zeros(
        1, query_len, tc.num_hidden_layers * tc.hidden_size_per_layer_input, dtype=torch.float32
    )

    with torch.no_grad():
        logits_id = mid(
            input_ids, position_ids, k_c.clone(), v_c.clone(), k_f.clone(), v_f.clone(), ple
        )
        # inputs_embeds as the exported embed.aimodel produces them (embed_scale baked in).
        embeds = mid.model.embed_tokens(input_ids) * mid.model.embed_scale
        logits_emb = memb(
            embeds, position_ids, k_c.clone(), v_c.clone(), k_f.clone(), v_f.clone(), ple
        )

    assert logits_id.shape == logits_emb.shape
    assert torch.allclose(logits_id, logits_emb, atol=1e-5), (
        f"max abs diff {(logits_id - logits_emb).abs().max().item():.3e}"
    )


def test_gemma4_vlm_spec_and_metadata_shape() -> None:
    """The registered Gemma4 VLM spec exposes the expected geometry + metadata."""
    from coreai_models.vlm.gemma4 import GEMMA4_E2B

    assert GEMMA4_E2B.image_size == 768
    assert GEMMA4_E2B.num_visual_tokens == 256  # (768/16)**2 / 3**2
    assert GEMMA4_E2B.image_token_id == 258880
    # Gemma4 preprocessing is rescale-only (patch embedder centers via 2*(x-0.5)).
    assert GEMMA4_E2B.image_mean == (0.0, 0.0, 0.0)
    assert GEMMA4_E2B.image_std == (1.0, 1.0, 1.0)


def test_gemma4_vlm_registered_in_registry() -> None:
    """`coreai.vlm.export gemma4-e2b-vlm` resolves to a gemma4-family preset."""
    from coreai_models.model_registry import presets_for_type, try_lookup_preset

    preset = try_lookup_preset("gemma4-e2b-vlm", model_type="vlm")
    assert preset is not None
    assert preset.family == "gemma4"
    assert preset.hf_id == "google/gemma-4-E2B-it"
    assert any(p.short_name == "gemma4-e2b-vlm" for p in presets_for_type("vlm"))
