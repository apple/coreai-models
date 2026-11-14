# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the model registry."""

import pytest
import torch
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3ForCausalLM as HFSmolLM3ForCausalLM,
)

from coreai_models.models.registry import get_model_entry, list_models
from coreai_models.primitives.macos.cache import KVCache


class TestModelRegistry:
    """Test model registry lookups."""

    def test_list_models_contains_qwen3(self):
        """The registry should list 'qwen3' as a supported model."""
        models = list_models()
        assert "qwen3" in models

    def test_list_models_returns_sorted(self):
        """list_models returns a sorted list."""
        models = list_models()
        assert models == sorted(models)

    def test_get_qwen3_entry(self):
        """get_model_entry('qwen3') returns an entry with both macOS and iOS classes."""
        entry = get_model_entry("qwen3")
        assert entry.macos_class is not None
        assert entry.ios_class is not None

    def test_get_qwen3_macos_class_is_correct(self):
        """The macOS class for qwen3 is Qwen3ForCausalLM."""
        from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM

        entry = get_model_entry("qwen3")
        assert entry.macos_class is Qwen3ForCausalLM

    def test_get_qwen3_ne_class_is_correct(self):
        """The iOS class for qwen3 is Qwen3ForCausalLMForiOS."""
        from coreai_models.models.ios.qwen3 import (
            Qwen3ForCausalLMForiOS,
        )

        entry = get_model_entry("qwen3")
        assert entry.ios_class is Qwen3ForCausalLMForiOS

    def test_get_smollm3_entry_maps_to_qwen2(self):
        """SmolLM3 uses Qwen2 architecture and resolves correctly."""
        entry = get_model_entry("smollm3")
        qwen2_entry = get_model_entry("qwen2")
        assert entry.macos_class is qwen2_entry.macos_class
        assert entry.ios_class is qwen2_entry.ios_class

    @pytest.mark.parametrize("no_rope_layer_interval", [2, 1000])  # NoPE at layers 1, 3 / none
    def test_smollm3_matches_hugging_face(self, no_rope_layer_interval):
        """SmolLM3 macOS logits must match Hugging Face (NoPE layers skip RoPE)."""
        config = SmolLM3Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=64,
            no_rope_layer_interval=no_rope_layer_interval,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        torch.manual_seed(0)
        hf_model = HFSmolLM3ForCausalLM(config).eval()
        model = get_model_entry("smollm3").macos_class(config, model_device="cpu").eval()
        state_dict = dict(hf_model.state_dict())
        model._mutate_state_dict(state_dict)
        model.load_state_dict(state_dict, assign=True, strict=True)

        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        position_ids = torch.arange(32, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_ids, position_ids, k_cache, v_cache)
            want = hf_model(input_ids=input_ids, position_ids=position_ids.long()).logits

        torch.testing.assert_close(logits, want, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("no_rope_layer_interval", [2, 1000])  # NoPE at layers 1, 3 / none
    def test_smollm3_matches_hugging_face_ios(self, no_rope_layer_interval):
        """SmolLM3 iOS logits must match Hugging Face (NoPE layers skip RoPE)."""
        from coreai_models.primitives.ios.cache import KVCacheHandler

        config = SmolLM3Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            intermediate_size=128,
            vocab_size=100,
            max_position_embeddings=64,
            no_rope_layer_interval=no_rope_layer_interval,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        torch.manual_seed(0)
        hf_model = HFSmolLM3ForCausalLM(config).to(torch.float32).eval()
        model = (
            get_model_entry("smollm3")
            .ios_class(config, model_device="cpu", disable_embedding_quantization=True)
            .to(torch.float32)
            .eval()
        )
        state_dict = dict(hf_model.state_dict())
        model._mutate_state_dict(state_dict)
        model.load_state_dict(state_dict, assign=True, strict=True)

        seq_len = 4
        # The iOS KV cache spans max_position_embeddings; the mask must cover it.
        max_seq = config.max_position_embeddings
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        in_step = torch.tensor([0], dtype=torch.int32)
        causal_mask = torch.zeros(1, max_seq, 1, seq_len, dtype=torch.float32)
        for i in range(seq_len):
            causal_mask[0, i + 1 :, 0, i] = float("-inf")
        k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_ids, position_ids, in_step, causal_mask, k_cache, v_cache)
            want = hf_model(input_ids=input_ids, position_ids=position_ids.long()).logits

        torch.testing.assert_close(logits.squeeze(1), want, atol=1e-5, rtol=1e-5)

    def test_unknown_model_raises_key_error(self):
        """Requesting a nonexistent model should raise KeyError."""
        with pytest.raises(KeyError, match="nonexistent_model"):
            get_model_entry("nonexistent_model")

    def test_unknown_model_error_lists_available(self):
        """The KeyError message should list available model types."""
        with pytest.raises(KeyError, match="qwen3"):
            get_model_entry("nonexistent_model")
