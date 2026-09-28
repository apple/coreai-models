# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the DiffusionGemma encoder/decoder modules and config.

These tests exercise the module structure, forward-pass shapes, weight-key
remapping, and config parsing on tiny random-initialized configs — no model
weights or HuggingFace reference are required. A ``__main__`` runner is provided
so the suite can also run under ``coverage`` directly (the sandbox blocks the
pytest rerun-failures socket plugin).
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import torch

import coreai_models.models.macos.diffusion_gemma as dgm
from coreai_models.models.macos.diffusion_gemma import (
    DiffusionGemmaAttention,
    DiffusionGemmaDecoder,
    DiffusionGemmaEncoder,
    DiffusionGemmaRMSNorm,
    ScaledEmbedding,
    _encoder_cache_update,
    _mutate_diffusion_gemma_state_dict,
    apply_rotary_pos_emb,
)
from coreai_models.models.macos.diffusion_gemma_config import (
    DiffusionGemmaConfig,
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaTextConfig,
)
from coreai_models.primitives.macos.cache import KVCache

SOFTCAP = 30.0


def _tiny_config() -> DiffusionGemmaTextConfig:
    # 2 layers: one sliding + one full (exercises tied-KV + proportional RoPE).
    return DiffusionGemmaTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        intermediate_size=48,
        moe_intermediate_size=24,
        num_experts=4,
        top_k_experts=2,
        vocab_size=100,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )


def _init_linears(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_text_config_from_dict_filters_unknown() -> None:
    cfg = DiffusionGemmaTextConfig.from_dict({"hidden_size": 128, "not_a_field": 7})
    assert cfg.hidden_size == 128
    assert not hasattr(cfg, "not_a_field")


def test_text_config_properties() -> None:
    cfg = _tiny_config()
    assert cfg.is_full_attention_layer(1) and not cfg.is_full_attention_layer(0)
    assert cfg.cache_head_dim == 32  # max(16, 32)
    assert cfg.cache_num_key_value_heads == 2  # max(2, 1)
    assert cfg.rope_sliding["rope_type"] == "default"
    assert cfg.rope_full["rope_type"] == "proportional"


def test_config_from_dict_and_local() -> None:
    d = {
        "text_config": {"hidden_size": 64, "num_hidden_layers": 2},
        "canvas_length": 48,
        "image_token_id": 999,
    }
    cfg = DiffusionGemmaConfig.from_dict(d)
    assert cfg.text_config.hidden_size == 64
    assert cfg.canvas_length == 48
    assert cfg.image_token_id == 999
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(d))
        cfg2 = DiffusionGemmaConfig.from_local(str(path))
        assert cfg2.canvas_length == 48


def test_generation_config_defaults() -> None:
    gc = DiffusionGemmaGenerationConfig()
    assert gc.max_denoising_steps == 48
    assert gc.t_max == 0.8 and gc.t_min == 0.4
    assert gc.entropy_bound == 0.1


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_rmsnorm_with_and_without_scale() -> None:
    x = torch.randn(2, 3, 8)
    scaled = DiffusionGemmaRMSNorm(8, with_scale=True)
    with torch.no_grad():
        scaled.weight.fill_(2.0)
    weightless = DiffusionGemmaRMSNorm(8, with_scale=False)
    out_scaled = scaled(x)
    out_weightless = weightless(x)
    assert out_scaled.shape == x.shape
    # with_scale=True multiplies the pure-norm output by the weight (2.0 here).
    assert torch.allclose(out_scaled, out_weightless * 2.0, atol=1e-5)
    # Pure RMSNorm: mean square ~ 1 per row.
    ms = out_weightless.float().pow(2).mean(-1)
    assert torch.allclose(ms, torch.ones_like(ms), atol=1e-2)


def test_scaled_embedding_scales_output() -> None:
    emb = ScaledEmbedding(10, 8, embed_scale=4.0)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    out = emb(ids)
    assert out.shape == (1, 3, 8)
    # Output equals the raw embedding times the scale.
    raw = torch.nn.functional.embedding(ids, emb.weight)
    assert torch.allclose(out, raw * 4.0, atol=1e-5)


def test_apply_rotary_pos_emb_shape_and_identity() -> None:
    x = torch.randn(1, 4, 2, 8)  # [B, T, heads, head_dim]
    cos = torch.ones(1, 4, 4)
    sin = torch.zeros(1, 4, 4)
    out = apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)
    # cos=1, sin=0 is the identity rotation.
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


def test_attention_inv_freq_shapes() -> None:
    cfg = _tiny_config()
    sliding = DiffusionGemmaAttention(cfg, layer_idx=0, is_encoder=True)
    full = DiffusionGemmaAttention(cfg, layer_idx=1, is_encoder=True)
    assert sliding.inv_freq.shape[0] == cfg.head_dim // 2
    assert full.inv_freq.shape[0] == cfg.global_head_dim // 2
    # Proportional RoPE zeroes the tail (non-rotated) frequencies.
    assert full.inv_freq[-1].item() == 0.0
    assert full.v_proj is None  # full layers tie V to K


def test_encoder_cache_update_pads_and_narrows() -> None:
    # Unified cache [layers, 1, cache_kv=2, ctx=6, cache_hd=4]; write a layer with
    # native (n_kv=1, hd=3) -> padded to (2, 4) on write, narrowed back on read.
    k_cache = torch.zeros(1, 1, 2, 6, 4)
    v_cache = torch.zeros_like(k_cache)
    cache = KVCache(k_cache, v_cache)
    k = torch.randn(1, 1, 5, 3)  # [B, n_kv=1, T=5, hd=3]
    v = torch.randn(1, 1, 5, 3)
    k_out, v_out = _encoder_cache_update(
        cache, layer_idx=0, offset=0, k=k, v=v, seq_len=5, n_kv=1, hd=3
    )
    assert k_out.shape == (1, 1, 5, 3)
    assert v_out.shape == (1, 1, 5, 3)
    assert torch.allclose(k_out, k, atol=1e-5)


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


def test_encoder_forward_shapes_and_softcap() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    enc = DiffusionGemmaEncoder(cfg).eval()
    _init_linears(enc)
    # Tied embeddings: lm_head shares the embedding weight.
    assert enc.lm_head.weight is enc.model.embed_tokens.weight

    seq_len, ctx = 5, 16
    n_kv, head_dim = enc.num_key_value_heads, enc.head_dim
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.int32)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    k_cache = torch.zeros(cfg.num_hidden_layers, 1, n_kv, ctx, head_dim)
    v_cache = torch.zeros_like(k_cache)

    with torch.no_grad():
        logits = enc(input_ids, position_ids, k_cache, v_cache)

    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert logits.abs().max().item() <= SOFTCAP + 1e-3  # final logit softcapping


def test_decoder_forward_shapes_and_softcap() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    dec = DiffusionGemmaDecoder(cfg).eval()
    _init_linears(dec)

    canvas, enc_len = 8, 6
    n_kv, head_dim = dec.num_key_value_heads, dec.head_dim
    decoder_input_ids = torch.randint(0, cfg.vocab_size, (1, canvas), dtype=torch.int32)
    prev_soft = torch.zeros(1, canvas, cfg.hidden_size)
    position_ids = torch.arange(enc_len, enc_len + canvas, dtype=torch.int32).unsqueeze(0)
    encoder_k = torch.randn(cfg.num_hidden_layers, 1, n_kv, enc_len, head_dim)
    encoder_v = torch.randn_like(encoder_k)
    temperature = torch.tensor([0.8])

    with torch.no_grad():
        logits, soft_out = dec(
            decoder_input_ids, prev_soft, position_ids, encoder_k, encoder_v, temperature
        )

    assert logits.shape == (1, canvas, cfg.vocab_size)
    assert soft_out.shape == (1, canvas, cfg.hidden_size)
    assert torch.isfinite(logits).all() and torch.isfinite(soft_out).all()
    # Temperature-scaled softcapped logits stay bounded by softcap / temperature.
    assert logits.abs().max().item() <= SOFTCAP / 0.8 + 1e-2


def test_decoder_self_conditioning_changes_output() -> None:
    # A non-zero self-conditioning signal must change the decoder output.
    cfg = _tiny_config()
    torch.manual_seed(0)
    dec = DiffusionGemmaDecoder(cfg).eval()
    _init_linears(dec)
    canvas, enc_len = 8, 6
    n_kv, head_dim = dec.num_key_value_heads, dec.head_dim
    ids = torch.randint(0, cfg.vocab_size, (1, canvas), dtype=torch.int32)
    pos = torch.arange(enc_len, enc_len + canvas, dtype=torch.int32).unsqueeze(0)
    ek = torch.randn(cfg.num_hidden_layers, 1, n_kv, enc_len, head_dim)
    ev = torch.randn_like(ek)
    temp = torch.tensor([1.0])
    with torch.no_grad():
        zero_soft = torch.zeros(1, canvas, cfg.hidden_size)
        rand_soft = torch.randn(1, canvas, cfg.hidden_size)
        out_zero, _ = dec(ids, zero_soft, pos, ek, ev, temp)
        out_rand, _ = dec(ids, rand_soft, pos, ek, ev, temp)
    assert not torch.allclose(out_zero, out_rand, atol=1e-3)


# ---------------------------------------------------------------------------
# State-dict remapping
# ---------------------------------------------------------------------------


def test_mutate_state_dict_remaps_hf_keys() -> None:
    cfg = _tiny_config()
    dec = DiffusionGemmaDecoder(cfg)
    h, moe_int, e = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts
    sd = {
        "model.decoder.embed_tokens.weight": torch.zeros(cfg.vocab_size, h),
        "model.decoder.norm.weight": torch.zeros(h),
        "model.decoder.self_conditioning.pre_norm.weight": torch.zeros(h),
        "model.decoder.layers.0.input_layernorm.weight": torch.zeros(h),
        "model.decoder.layers.0.self_attn.q_proj.weight": torch.zeros(64, h),
        "model.decoder.layers.0.layer_scalar": torch.ones(1),
        "model.encoder.language_model.layers.0.layer_scalar": torch.ones(1),
        "model.decoder.layers.0.experts.gate_up_proj": torch.zeros(e, 2 * moe_int, h),
        "model.decoder.layers.0.experts.down_proj": torch.zeros(e, h, moe_int),
        "model.decoder.layers.0.router.proj.weight": torch.zeros(e, h),
        "model.decoder.layers.0.router.scale": torch.zeros(h),
        "model.decoder.layers.0.router.per_expert_scale": torch.zeros(e),
        "model.encoder.vision_tower.foo": torch.zeros(1),  # dropped
    }
    _mutate_diffusion_gemma_state_dict(sd, dec)

    assert "model.embed_tokens.weight" in sd
    assert "model.norm.weight" in sd
    assert "model.self_conditioning.pre_norm.weight" in sd
    assert "model.layers.0.input_layernorm.weight" in sd
    assert "model.layers.0.self_attn.q_proj.weight" in sd
    assert "model.layers.0.decoder_scalar" in sd
    assert "model.layers.0.encoder_scalar" in sd
    assert sd["model.layers.0.moe.switch_glu.gate_proj.weight"].shape == (1, e, moe_int, h)
    assert sd["model.layers.0.moe.switch_glu.up_proj.weight"].shape == (1, e, moe_int, h)
    assert sd["model.layers.0.moe.switch_glu.down_proj.weight"].shape == (1, e, h, moe_int)
    assert "model.layers.0.moe.router_proj.weight" in sd
    assert "model.layers.0.moe.router_scale" in sd
    assert "model.layers.0.moe.per_expert_scale" in sd
    # HF-layout keys and vision-tower junk are gone.
    assert "model.decoder.layers.0.input_layernorm.weight" not in sd
    assert "model.encoder.vision_tower.foo" not in sd


def test_registry_resolves_diffusion_gemma() -> None:
    from coreai_models.models.registry import get_model_entry

    entry = get_model_entry("diffusion_gemma")
    assert entry.macos_class is DiffusionGemmaEncoder
    assert entry.hf_config_attr == "text_config"


def test_load_state_dict_round_trip_reties_embeddings() -> None:
    cfg = _tiny_config()
    for cls in (DiffusionGemmaEncoder, DiffusionGemmaDecoder):
        model = cls(cfg)
        # Exercise the mutate wrapper (no-op on an already-mapped dict) and the
        # load_state_dict override that re-ties lm_head to the embedding.
        model._mutate_state_dict({})
        model.load_state_dict(model.state_dict())
        assert model.lm_head.weight is model.model.embed_tokens.weight


def test_decoder_casts_bf16_logits_to_f16() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    dec = DiffusionGemmaDecoder(cfg).to(torch.bfloat16).eval()
    _init_linears(dec)
    canvas, enc_len = 8, 6
    n_kv, head_dim = dec.num_key_value_heads, dec.head_dim
    ids = torch.randint(0, cfg.vocab_size, (1, canvas), dtype=torch.int32)
    soft = torch.zeros(1, canvas, cfg.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(enc_len, enc_len + canvas, dtype=torch.int32).unsqueeze(0)
    ek = torch.zeros(cfg.num_hidden_layers, 1, n_kv, enc_len, head_dim, dtype=torch.bfloat16)
    ev = torch.zeros_like(ek)
    with torch.no_grad():
        logits, _ = dec(ids, soft, pos, ek, ev, torch.tensor([0.8], dtype=torch.bfloat16))
    # bf16 compute path down-casts the emitted logits to float16.
    assert logits.dtype == torch.float16


# ---------------------------------------------------------------------------
# Weight loading (HF download paths, exercised with mocks)
# ---------------------------------------------------------------------------


def test_config_from_pretrained_mocked() -> None:
    class _FakeHFConfig:
        def to_dict(self) -> dict:
            return {"text_config": {"hidden_size": 64}}

    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=_FakeHFConfig()):
        cfg = DiffusionGemmaConfig.from_pretrained("some/model")
    assert cfg.text_config.hidden_size == 64


def test_generation_config_from_pretrained_success_and_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gcp = Path(tmp, "generation_config.json")
        gcp.write_text(
            json.dumps(
                {"max_denoising_steps": 12, "t_max": 0.9, "sampler_config": {"entropy_bound": 0.2}}
            )
        )
        with mock.patch("huggingface_hub.hf_hub_download", return_value=str(gcp)):
            gc = DiffusionGemmaGenerationConfig.from_pretrained("some/model")
        assert gc.max_denoising_steps == 12 and gc.t_max == 0.9 and gc.entropy_bound == 0.2
    # Missing file -> defaults (exception fallback path).
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=OSError("nope")):
        gc2 = DiffusionGemmaGenerationConfig.from_pretrained("some/model")
    assert gc2.max_denoising_steps == 48


def test_load_state_dict_from_hub_mocked() -> None:
    from coreai_models.models import base

    with (
        mock.patch("huggingface_hub.snapshot_download", return_value="/tmp/x"),
        mock.patch.object(base, "_resolve_safetensors_files", return_value=[]),
        mock.patch.object(base, "_build_safetensors_key_index", return_value=({}, {})),
        mock.patch.object(base, "_load_tensors_for_keys", return_value={}),
    ):
        sd = dgm._load_state_dict_from_hub("some/model", torch.float32, num_layers=2)
    assert sd == {}


def test_load_encoder_and_decoder_mocked() -> None:
    full_cfg = DiffusionGemmaConfig(text_config=_tiny_config())
    with (
        mock.patch.object(DiffusionGemmaConfig, "from_pretrained", return_value=full_cfg),
        mock.patch.object(dgm, "_load_state_dict_from_hub", return_value={}),
    ):
        enc = dgm.load_diffusion_gemma_encoder(
            "m", target_dtype=torch.float32, max_context_length=64, num_layers=2
        )
        dec = dgm.load_diffusion_gemma_decoder("m", target_dtype=torch.float32, num_layers=2)
    assert enc.lm_head.weight is enc.model.embed_tokens.weight
    assert dec.lm_head.weight is dec.model.embed_tokens.weight


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    for fn in _TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(_TESTS)} tests passed")
