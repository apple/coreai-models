# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Muse Glimmer model.

Note: Muse Glimmer is not in our transformers version, so we cannot
do HF parity tests. These tests verify structural correctness, weight
loading, and numerical stability.
"""

from types import SimpleNamespace

import pytest
import torch

from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _make_glimmer_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=8,
        intermediate_size=128,
        vocab_size=200,
        max_position_embeddings=32,
        head_dim=16,
        attention_bias=False,
        hidden_activation="silu",
        rms_norm_eps=1e-5,
        sliding_window=8,
        final_logit_softcapping=20.0,
        tie_word_embeddings=False,
        output_multiplier=0.196,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        layer_rope_theta=[500000, 500000, 500000, 0, 500000, 500000, 500000, 0],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMuseGlimmerForCausalLM:
    """Test Muse Glimmer structural correctness and numerical stability."""

    def test_forward_produces_finite_output(self):
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        model.to(torch.float32).eval()

        input_ids = torch.randint(0, 200, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (1, 6, config.vocab_size)
        assert torch.isfinite(out).all()

    def test_logit_softcapping(self):
        """Logits should be bounded by softcap value."""
        config = _make_glimmer_config(final_logit_softcapping=20.0)
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        model.to(torch.float32).eval()

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = model(input_ids, position_ids, k_cache, v_cache)

        assert out.abs().max() <= 20.0

    def test_output_multiplier_affects_output(self):
        """Different output_multiplier should produce different logits."""
        config1 = _make_glimmer_config(output_multiplier=1.0, final_logit_softcapping=None)
        config2 = _make_glimmer_config(output_multiplier=0.196, final_logit_softcapping=None)

        torch.manual_seed(42)
        model1 = MuseGlimmerForCausalLM(config1, model_device="cpu").to(torch.float32).eval()
        torch.manual_seed(42)
        model2 = MuseGlimmerForCausalLM(config2, model_device="cpu").to(torch.float32).eval()

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k1, v1 = KVCache.create_cache_tensors(config1, dtype=torch.float32)
        k2, v2 = KVCache.create_cache_tensors(config2, dtype=torch.float32)

        with torch.no_grad():
            out1 = model1(input_ids, position_ids, k1, v1)
            out2 = model2(input_ids, position_ids, k2, v2)

        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_gated_attention_structure(self):
        """Attention should have gate_proj with correct dimensions."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        attn = model.model.layers[0].self_attn

        assert hasattr(attn, "gate_proj")
        n_heads = config.num_attention_heads
        head_dim = config.head_dim
        assert attn.gate_proj.weight.shape == (n_heads * head_dim, config.hidden_size)

    def test_sandwich_norms_structure(self):
        """Each layer should have 4 norms (pre+post for both attn and MLP)."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        layer = model.model.layers[0]

        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "pre_feedforward_layernorm")
        assert hasattr(layer, "post_feedforward_layernorm")

    def test_per_layer_rope_control(self):
        """Local layers should have RoPE, global layers should not."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLM(config, model_device="cpu")

        # Layer 0: sliding (has RoPE)
        assert model.model.layers[0].self_attn.has_rope is True
        assert model.model.layers[0].self_attn.is_sliding is True

        # Layer 3: full/global (no RoPE)
        assert model.model.layers[3].self_attn.has_rope is False
        assert model.model.layers[3].self_attn.is_sliding is False

    def test_sliding_window_pattern(self):
        """Should be [S,S,S,G] repeating."""
        config = _make_glimmer_config()
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        pattern = [layer.self_attn.is_sliding for layer in model.model.layers]
        expected = [True, True, True, False, True, True, True, False]
        assert pattern == expected

    def test_deterministic_output(self):
        """Same input should produce same output."""
        config = _make_glimmer_config()
        torch.manual_seed(42)
        model = MuseGlimmerForCausalLM(config, model_device="cpu").to(torch.float32).eval()

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k1, v1 = KVCache.create_cache_tensors(config, dtype=torch.float32)
        k2, v2 = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out1 = model(input_ids, position_ids, k1, v1)
            out2 = model(input_ids, position_ids, k2, v2)

        torch.testing.assert_close(out1, out2)

    def test_mutate_state_dict_normalizes_keys(self):
        """_mutate_state_dict should handle both raw and stripped key forms."""
        config = _make_glimmer_config(num_hidden_layers=1)
        model = MuseGlimmerForCausalLM(config, model_device="cpu")

        # Simulate raw checkpoint keys
        sd = {}
        sd["model.language_model.embed_tokens.weight"] = torch.randn(200, 64)
        sd["model.language_model.layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
        sd["model.language_model.layers.0.self_attn.k_proj.weight"] = torch.randn(32, 64)
        sd["model.language_model.layers.0.self_attn.v_proj.weight"] = torch.randn(32, 64)
        sd["model.language_model.layers.0.self_attn.o_proj.weight"] = torch.randn(64, 64)
        sd["model.language_model.layers.0.self_attn.gate_proj.weight"] = torch.randn(64, 64)
        sd["model.vision_tower.layers.0.attn.q_proj.weight"] = torch.randn(64, 64)
        sd["lm_head.weight"] = torch.randn(200, 64)

        model._mutate_state_dict(sd)

        assert "model.embed_tokens.weight" in sd
        assert "model.layers.0.self_attn.q_proj.weight" in sd
        assert "lm_head.weight" in sd
        assert "model.vision_tower.layers.0.attn.q_proj.weight" not in sd
        assert "model.language_model.embed_tokens.weight" not in sd

    def test_mutate_state_dict_stripped_keys(self):
        """_mutate_state_dict should handle already-stripped keys."""
        config = _make_glimmer_config(num_hidden_layers=1)
        model = MuseGlimmerForCausalLM(config, model_device="cpu")

        sd = {}
        sd["layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
        sd["embed_tokens.weight"] = torch.randn(200, 64)
        sd["norm.weight"] = torch.randn(64)
        sd["lm_head.weight"] = torch.randn(200, 64)

        model._mutate_state_dict(sd)

        assert "model.layers.0.self_attn.q_proj.weight" in sd
        assert "model.embed_tokens.weight" in sd
        assert "model.norm.weight" in sd
        assert "lm_head.weight" in sd

    def test_qk_scale_factor(self):
        """qk_scale_factor should be stored on attention and applied to Q."""
        config = _make_glimmer_config(qk_scale_factor=3.87)
        model = MuseGlimmerForCausalLM(config, model_device="cpu")
        attn = model.model.layers[0].self_attn
        assert attn.qk_scale_factor == pytest.approx(3.87, rel=1e-5)
        assert hasattr(attn, "qk_norm")
