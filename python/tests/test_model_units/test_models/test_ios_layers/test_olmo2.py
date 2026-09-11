# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS OLMo 2 model parity with HuggingFace."""

import torch
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2ForCausalLM as HFOlmo2ForCausalLM,
)

from coreai_models.models.ios.olmo2 import Olmo2ForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler


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


def _make_ne_causal_mask(
    seq_len: int, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a causal mask for the iOS SDPA.

    The iOS SDPA expects shape (1, max_seq_len, 1, seq_len) where
    mask[0, j, 0, i] = -inf if position j should not attend to position i.
    """
    mask = torch.zeros(1, max_seq_len, 1, seq_len, dtype=dtype)
    for i in range(seq_len):
        mask[0, i + 1 :, 0, i] = float("-inf")
    return mask


def _load_ne_model(hf_model: HFOlmo2ForCausalLM, config: Olmo2Config) -> Olmo2ForCausalLMForiOS:
    ne_model = Olmo2ForCausalLMForiOS(
        config, model_device="cpu", disable_embedding_quantization=True
    )
    ne_model.to(torch.float32).eval()
    sd = dict(hf_model.state_dict())
    ne_model._mutate_state_dict(sd)
    ne_model.load_state_dict(sd, assign=True, strict=True)
    return ne_model


class TestNEOlmo2ForCausalLM:
    """Test iOS Olmo2ForCausalLMForiOS against HuggingFace reference."""

    def test_forward_parity_multi_token(self):
        """Multi-token prefill: iOS model matches HF logits."""
        seq_len = 4
        max_seq = 32
        config = _make_olmo2_config(max_position_embeddings=max_seq)

        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        ne_model = _load_ne_model(hf_model, config)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_single_token(self):
        """Single-token decode: iOS model matches HF logits."""
        max_seq = 32
        config = _make_olmo2_config(max_position_embeddings=max_seq)

        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        ne_model = _load_ne_model(hf_model, config)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(1, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_two_layers(self):
        """Two-layer model: verify parity scales with depth (exercises post-norm residuals)."""
        seq_len = 4
        max_seq = 32
        config = _make_olmo2_config(num_hidden_layers=2, max_position_embeddings=max_seq)

        hf_model = HFOlmo2ForCausalLM(config).to(torch.float32).eval()
        ne_model = _load_ne_model(hf_model, config)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float32)

        with torch.no_grad():
            ne_out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        ne_logits = ne_out.squeeze(1)
        torch.testing.assert_close(ne_logits, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape is (batch, 1, seq_len, vocab_size) for iOS layout."""
        max_seq = 32
        config = _make_olmo2_config(max_position_embeddings=max_seq)
        ne_model = Olmo2ForCausalLMForiOS(
            config, model_device="cpu", disable_embedding_quantization=True
        )
        ne_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 4, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = _make_ne_causal_mask(seq_len, max_seq)
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float32)

        with torch.no_grad():
            out = ne_model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)

        assert out.shape == (batch, 1, seq_len, vocab)

    def test_mutate_state_dict_adds_conv2d_dims(self):
        """_mutate_state_dict reshapes linear weights to Conv2d (4D) and keeps flat QK-norms."""
        config = _make_olmo2_config(num_hidden_layers=1)
        ne_model = Olmo2ForCausalLMForiOS(
            config, model_device="cpu", disable_embedding_quantization=True
        )
        hf_model = HFOlmo2ForCausalLM(config)

        sd = dict(hf_model.state_dict())
        assert "model.layers.0.self_attn.q_norm.weight" in sd
        assert "model.layers.0.self_attn.k_norm.weight" in sd

        ne_model._mutate_state_dict(sd)

        # Attention and MLP weights are unsqueezed to 4D (Conv2d format).
        q_key = "extend.model.layers.0.self_attn.q_proj.weight"
        assert q_key in sd
        assert sd[q_key].dim() == 4
        gate_key = "extend.model.layers.0.mlp.gate_proj.weight"
        assert gate_key in sd
        assert sd[gate_key].dim() == 4

        # Flat QK-norm weights are preserved untouched (still 1D).
        q_norm_key = "extend.model.layers.0.self_attn.q_norm.weight"
        k_norm_key = "extend.model.layers.0.self_attn.k_norm.weight"
        assert q_norm_key in sd
        assert k_norm_key in sd
        assert sd[q_norm_key].dim() == 1
        assert sd[k_norm_key].dim() == 1

        # Embedding table is relocated under load_embeddings.
        assert "load_embeddings.embedding_table" in sd
