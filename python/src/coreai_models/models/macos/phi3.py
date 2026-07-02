# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math

import torch
import torch.nn as nn
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3ForCausalLM as HFPhi3ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

USE_FUSED_KV = True


def _compute_phi3_attention_factor(config: Phi3Config) -> float:
    """Compute the longrope attention_factor for Phi-3/4 models."""
    original_max_pos = getattr(config, "original_max_position_embeddings", None)
    if original_max_pos is None:
        original_max_pos = config.max_position_embeddings

    factor = config.max_position_embeddings / original_max_pos
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(original_max_pos))


class Phi3RoPE(nn.Module):
    """RoPE with longrope attention_factor scaling for Phi-3/4."""

    def __init__(self, dims: int, base: float, attention_factor: float) -> None:
        super().__init__()
        self._rope = RoPE(base=base, dims=dims)
        self._attention_factor = attention_factor
        self._dims = dims

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self._attention_factor == 1.0:
            return self._rope(x, position_ids=position_ids)
        out = self._rope(x, position_ids=position_ids)
        if self._dims is not None and self._dims < x.shape[-1]:
            rotated = out[..., :self._dims] * self._attention_factor
            passthrough = out[..., dims:]
            return torch.cat([rotated, passthrough], dim=-1)
        return out * self._attention_factor


class Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads

        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)

        # Phi-3/4 uses partial rotary with longrope attention_factor
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rope_dims = int(head_dim * partial_rotary_factor)
        attention_factor = _compute_phi3_attention_factor(config)
        self.rope = Phi3RoPE(
            dims=rope_dims,
            base=resolve_rope_theta(config),
            attention_factor=attention_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        if USE_FUSED_KV:
            query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
            query_key = self.rope(query_key, position_ids=rope_positions)
            query = query_key.narrow(1, 0, n_heads)
            key = query_key.narrow(1, n_heads, n_kv_heads)
        else:
            query = qkv.narrow(1, 0, n_heads)
            key = qkv.narrow(1, n_heads, n_kv_heads)
            query = self.rope(query, position_ids=rope_positions)
            key = self.rope(key, position_ids=rope_positions)

        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
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
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Phi3Model(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
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
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Phi3ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFPhi3ForCausalLM

    @override
    def _init_model(self, config: Phi3Config) -> None:
        self.model = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Phi-3/4 HF checkpoint has fused gate_up_proj [2*intermediate, hidden].
        # Our MLP uses separate gate_proj and up_proj. Split them.
        keys_to_process = [k for k in state_dict if "mlp.gate_up_proj.weight" in k]
        for key in keys_to_process:
            fused = state_dict.pop(key)
            # gate_up_proj is [2*intermediate_size, hidden_size]
            # First half is gate, second half is up
            half = fused.shape[0] // 2
            prefix = key.replace("mlp.gate_up_proj.weight", "mlp.")
            state_dict[prefix + "gate_proj.weight"] = fused[:half]
            state_dict[prefix + "up_proj.weight"] = fused[half:]

        # qkv_proj is already in [Q|K|V] format matching our Attention module,
        # so no transformation needed for attention weights.

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
