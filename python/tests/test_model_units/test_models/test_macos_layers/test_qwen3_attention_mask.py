# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Opt-in additive-attention-mask override for macOS Qwen3 attention.

Proves, at batch=1, that an explicit additive ``attn_mask`` reproduces the default
``is_causal`` path. Default-off, so the existing export contract is untouched.
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from coreai_models.models.macos.qwen3 import Attention, Qwen3ForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        head_dim=16,
    )
    config.rope_scaling = None
    config.rope_theta = 10000.0
    return config


def _causal_additive_mask(seq_len: int) -> torch.Tensor:
    """Additive causal mask broadcastable to [B, heads, q, k]."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(
        1, 1, seq_len, seq_len
    )


def test_attention_mask_mode_matches_causal_single_batch() -> None:
    """Mask-mode attention fed a causal additive mask == default is_causal path (B=1)."""
    torch.manual_seed(0)
    config = _tiny_config()
    seq_len = 6

    causal = Attention(config, layer_idx=0).eval()
    masked = Attention(config, layer_idx=0, use_attention_mask=True).eval()
    masked.load_state_dict(causal.state_dict())

    x = torch.rand(1, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        out_causal = causal(x, position_ids)
        out_masked = masked(x, position_ids, attn_mask=_causal_additive_mask(seq_len))

    torch.testing.assert_close(out_masked, out_causal, atol=1e-5, rtol=1e-5)


def test_qwen3_for_causal_lm_mask_mode_matches_causal_single_batch() -> None:
    """End-to-end: mask-mode Qwen3ForCausalLM threads attn_mask == default is_causal (B=1)."""
    torch.manual_seed(0)
    config = _tiny_config()
    seq_len = 6

    causal = Qwen3ForCausalLM(config, model_device="cpu")
    causal.to(torch.float32).eval()
    masked = Qwen3ForCausalLM(config, model_device="cpu", use_attention_mask=True)
    masked.to(torch.float32).eval()
    masked.load_state_dict(causal.state_dict())

    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    kc, vc = KVCache.create_cache_tensors(config, dtype=torch.float32, seq_len=seq_len)
    kc2, vc2 = KVCache.create_cache_tensors(config, dtype=torch.float32, seq_len=seq_len)

    with torch.no_grad():
        out_causal = causal(input_ids, position_ids, kc, vc)
        out_masked = masked(
            input_ids, position_ids, kc2, vc2, attn_mask=_causal_additive_mask(seq_len)
        )

    torch.testing.assert_close(out_masked, out_causal, atol=1e-5, rtol=1e-5)
