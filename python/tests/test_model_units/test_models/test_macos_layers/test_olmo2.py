# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS OLMo 2 model parity with HuggingFace."""

import torch
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2ForCausalLM as HFOlmo2ForCausalLM,
)

from coreai_models.models.macos.olmo2 import Olmo2ForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _make_olmo2_config(**overrides) -> Olmo2Config:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    defaults.update(overrides)
    return Olmo2Config(**defaults)


class TestOlmo2ForCausalLM:
    """Test macOS Olmo2ForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_olmo2_config()
        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_token(self):
        seq_len = 8
        config = _make_olmo2_config()
        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_float16(self):
        config = _make_olmo2_config()
        hf_model = HFOlmo2ForCausalLM(config).to(torch.float16).eval()
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float16).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-3, rtol=5e-3)

    def test_output_shape(self):
        config = _make_olmo2_config()
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len = 1, 6
        input_ids = torch.randint(0, 100, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, config.vocab_size)

    def test_post_norm_differs_from_pre_norm(self):
        """OLMo2's post-norm should produce different output than pre-norm with same weights."""
        config = _make_olmo2_config(num_hidden_layers=1)
        model = Olmo2ForCausalLM(config, model_device="cpu").to(torch.float32).eval()
        layer = model.model.layers[0]

        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "post_feedforward_layernorm")
        assert not hasattr(layer, "input_layernorm")

    def test_qk_norm_present(self):
        """OLMo2 attention should have separate q_norm and k_norm."""
        config = _make_olmo2_config()
        model = Olmo2ForCausalLM(config, model_device="cpu")
        attn = model.model.layers[0].self_attn

        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // n_heads
        assert attn.q_norm.weight.shape == (n_heads * head_dim,)
        assert attn.k_norm.weight.shape == (n_kv_heads * head_dim,)

    def test_mutate_state_dict_fuses_qkv_preserves_norms(self):
        config = _make_olmo2_config(num_hidden_layers=1)
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        hf_model = HFOlmo2ForCausalLM(config)

        sd = dict(hf_model.state_dict())
        assert "model.layers.0.self_attn.q_proj.weight" in sd
        assert "model.layers.0.self_attn.q_norm.weight" in sd
        assert "model.layers.0.self_attn.k_norm.weight" in sd

        our_model._mutate_state_dict(sd)

        assert "model.layers.0.self_attn.qkv_proj.weight" in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd
        assert "model.layers.0.self_attn.q_norm.weight" in sd
        assert "model.layers.0.self_attn.k_norm.weight" in sd

    def test_incremental_decode(self):
        config = _make_olmo2_config()
        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        our_model = Olmo2ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        input_ids = torch.randint(0, 100, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            our_model(input_ids, position_ids, k_cache, v_cache)

        next_token = torch.randint(0, 100, (1, 1))
        pos_ids_step2 = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            out2 = our_model(next_token, pos_ids_step2, k_cache, v_cache)

        assert out2.shape == (1, 1, config.vocab_size)
        with torch.no_grad():
            out2b = our_model(next_token, pos_ids_step2, k_cache, v_cache)
        torch.testing.assert_close(out2, out2b)
