# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse Glimmer DFlash drafter for CoreAI model export.

Meta's speculative decoding companion for Muse Glimmer 30B (Apache 2.0).
Architecture:
- 5 transformer layers, all sliding window (2048)
- 32Q / 8KV heads, head_dim 128
- Separate q_norm / k_norm (not shared like the target)
- No gated attention, no sandwich norms
- Shares embed_tokens and lm_head with the target model
- Ring buffer KV cache with explicit attention mask
"""

import torch
import torch.nn as nn
from typing_extensions import override

from coreai_models._constants import (
    MAIN_GRAPH_NAME,
    SLIDING_KEY_CACHE_NAME,
    SLIDING_VALUE_CACHE_NAME,
)
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import RingKVCache, ring_window_causal_mask
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=False)

        rope_theta = (
            config.rope_parameters.get("rope_theta", 500000.0)
            if isinstance(config.rope_parameters, dict)
            else 500000.0
        )
        self.rope = RoPE()
        with torch.device("cpu"):
            self._rope_freqs = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        query = self.q_norm(
            self.q_proj(x)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        key = self.k_norm(
            self.k_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.v_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        freqs = self._rope_freqs.to(device=query.device)
        query = self.rope(query, position_ids=position_ids, freqs=freqs)
        key = self.rope(key, position_ids=position_ids, freqs=freqs)

        if cache is not None:
            offset = position_ids[0, 0]
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, query_len=query_len
            )

        attn_output = (
            self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache, attn_mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class DrafterRingModel(nn.Module):
    """Muse Glimmer DFlash drafter backbone (no lm_head)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)

        if cache is not None:
            query_len = input_ids.shape[-1]
            offset = position_ids[0, 0]
            attn_mask = ring_window_causal_mask(
                query_len=query_len,
                capacity=cache.capacity(),
                offset=offset,
                window_size=self.config.sliding_window,
                device=input_ids.device,
            )
        else:
            attn_mask = None

        for layer in self.layers:
            h = layer(h, position_ids, cache, attn_mask)
        return self.norm(h)


class MuseGlimmerDrafterForCausalLM(BaseForCausalLM):
    """Muse Glimmer DFlash drafter with sliding-window ring buffer states."""

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config) -> None:
        self.model = DrafterRingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        sliding_k_cache: torch.Tensor,
        sliding_v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = RingKVCache(sliding_k_cache, sliding_v_cache)
        out = self.model(input_ids, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: (SLIDING_KEY_CACHE_NAME, SLIDING_VALUE_CACHE_NAME)}

    @classmethod
    def export_state_classification(cls) -> dict[str, str]:
        return {
            SLIDING_KEY_CACHE_NAME: "sliding_cache",
            SLIDING_VALUE_CACHE_NAME: "sliding_cache",
        }
