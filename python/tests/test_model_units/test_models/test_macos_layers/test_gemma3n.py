# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Gemma 3n model parity with HuggingFace."""

import torch
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM as HFGemma3nForCausalLM,
)

from coreai_models.models.macos.gemma3n import Gemma3nForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _make_gemma3n_config(**overrides) -> Gemma3nTextConfig:
    n_layers = overrides.pop("num_hidden_layers", 10)
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=n_layers,
        intermediate_size=[128] * n_layers,
        vocab_size=200,
        vocab_size_per_layer_input=200,
        max_position_embeddings=32,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
        hidden_size_per_layer_input=16,
        altup_num_inputs=4,
        altup_active_idx=0,
        altup_correct_scale=True,
        laurel_rank=8,
        num_kv_shared_layers=0,
        activation_sparsity_pattern=[0.0] * n_layers,
        sliding_window=8,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return Gemma3nTextConfig(**defaults)


def _build_models(config):
    torch.manual_seed(42)
    hf_model = HFGemma3nForCausalLM(config).to(torch.float32).eval()
    our_model = Gemma3nForCausalLM(config, model_device="cpu")
    our_model.to(torch.float32).eval()
    sd = dict(hf_model.state_dict())
    our_model._mutate_state_dict(sd)
    our_model.load_state_dict(sd, assign=True, strict=True)
    return hf_model, our_model


class TestGemma3nForCausalLM:
    """Test macOS Gemma3nForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        hf_model, our_model = _build_models(config)

        input_ids = torch.randint(0, 200, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_float16(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        torch.manual_seed(42)
        hf_model = HFGemma3nForCausalLM(config).to(torch.float16).eval()
        our_model = Gemma3nForCausalLM(config, model_device="cpu")
        our_model.to(torch.float16).eval()
        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-2, rtol=5e-2)

    def test_forward_parity_multi_token(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        hf_model, our_model = _build_models(config)

        input_ids = torch.randint(0, 200, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_with_kv_sharing(self):
        """KV sharing: last 2 layers reuse K/V from source layers."""
        config = _make_gemma3n_config(num_hidden_layers=10, num_kv_shared_layers=2)
        hf_model, our_model = _build_models(config)

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = Gemma3nForCausalLM.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long(), use_cache=True)

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_with_sparsity(self):
        """Gaussian TopK activation sparsity in early layers."""
        sparsity = [0.95, 0.95, 0.0, 0.0, 0.0]
        config = _make_gemma3n_config(num_hidden_layers=5, activation_sparsity_pattern=sparsity)
        hf_model, our_model = _build_models(config)

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_full_config(self):
        """Full config: KV sharing + sparsity + 10 layers."""
        sparsity = [0.95, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        config = _make_gemma3n_config(
            num_hidden_layers=10,
            num_kv_shared_layers=2,
            activation_sparsity_pattern=sparsity,
        )
        hf_model, our_model = _build_models(config)

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = Gemma3nForCausalLM.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long(), use_cache=True)

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        our_model = Gemma3nForCausalLM(config, model_device="cpu").to(torch.float32).eval()

        batch, seq_len = 1, 6
        input_ids = torch.randint(0, 200, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, config.vocab_size)

    def test_kv_sharing_cache_compaction(self):
        """Cache should have fewer slots than layers when KV sharing is enabled."""
        config = _make_gemma3n_config(num_hidden_layers=10, num_kv_shared_layers=2)
        model = Gemma3nForCausalLM(config, model_device="cpu")

        assert model.model.num_cache_slots == 8
        k_cache, v_cache = Gemma3nForCausalLM.create_cache_tensors(config, dtype=torch.float32)
        assert k_cache.shape[0] == 8

    def test_kv_sharing_slot_mapping(self):
        """Shared layers should map to source layer's cache slot."""
        config = _make_gemma3n_config(num_hidden_layers=10, num_kv_shared_layers=2)
        model = Gemma3nForCausalLM(config, model_device="cpu")

        layers = model.model.layers
        # Layer 8 (sliding) should share with last non-shared sliding layer
        assert layers[8].self_attn.is_kv_shared
        assert layers[8].self_attn.cache_slot == layers[7].self_attn.cache_slot
        # Layer 9 (full) should share with last non-shared full layer (layer 4)
        assert layers[9].self_attn.is_kv_shared
        assert layers[9].self_attn.cache_slot == layers[4].self_attn.cache_slot

    def test_altup_predict_correct_shape(self):
        """AltUp should maintain list of [B, S, D] tensors through predict/correct."""
        config = _make_gemma3n_config(num_hidden_layers=5)
        model = Gemma3nForCausalLM(config, model_device="cpu").to(torch.float32).eval()
        altup = model.model.layers[0].altup

        hidden = [torch.randn(1, 3, 64) for _ in range(4)]
        predictions = altup.predict(hidden)
        assert isinstance(predictions, list)
        assert len(predictions) == 4
        for p in predictions:
            assert p.shape == (1, 3, 64)

        activated = torch.randn(1, 3, 64)
        corrected = altup.correct(predictions, activated)
        assert isinstance(corrected, list)
        assert len(corrected) == 4
        for c in corrected:
            assert c.shape == (1, 3, 64)

    def test_sliding_window_pattern(self):
        """Layers should alternate: 4 sliding + 1 full."""
        config = _make_gemma3n_config(num_hidden_layers=10)
        model = Gemma3nForCausalLM(config, model_device="cpu")
        pattern = [layer.self_attn.is_sliding for layer in model.model.layers]
        expected = [True, True, True, True, False, True, True, True, True, False]
        assert pattern == expected

    def test_tie_word_embeddings(self):
        config = _make_gemma3n_config(num_hidden_layers=5, tie_word_embeddings=True)
        hf_model = HFGemma3nForCausalLM(config).eval()
        our_model = Gemma3nForCausalLM(config, model_device="cpu").eval()
        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        assert our_model.lm_head.weight is our_model.model.embed_tokens.weight
