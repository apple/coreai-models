# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

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
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads

        # Use separate projections for GQA (n_heads != n_kv_heads) to work around
        # a CoreAI MLIR narrow lowering bug (rdar://184090277). MHA uses fused QKV.
        self.use_separate_qkv = n_heads != n_kv_heads
        if self.use_separate_qkv:
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        else:
            self.qkv_proj = nn.Linear(
                dim,
                n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
                bias=False,
            )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        sliding_window = getattr(config, "sliding_window", None)
        max_pos = getattr(config, "max_position_embeddings", None)
        if sliding_window and max_pos and sliding_window < max_pos:
            self.sdpa = SDPA(is_causal=True, scale=head_dim**-0.5, window_size=sliding_window)
        else:
            self.sdpa = SDPA(is_causal=True, scale=head_dim**-0.5)

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rope_dims = int(head_dim * partial_rotary_factor)
        rope_theta = resolve_rope_theta(config)
        assert rope_theta is not None, "Phi models require rope_theta in config"
        rope_scaling = getattr(config, "rope_scaling", None)
        original_max_pos = getattr(config, "original_max_position_embeddings", None)
        native_max_pos = getattr(config, "_native_max_position_embeddings", max_pos)
        self.rope = initialize_rope(
            dims=rope_dims,
            base=rope_theta,
            scaling_config=rope_scaling,
            max_position_embeddings=max_pos or original_max_pos,
            original_max_position_embeddings=original_max_pos,
            config_max_position_embeddings=native_max_pos,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        if self.use_separate_qkv:
            query = (
                self.q_proj(x)
                .reshape(batch_size, query_len, n_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            key = (
                self.k_proj(x)
                .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            value = (
                self.v_proj(x)
                .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            query = self.rope(query, position_ids=rope_positions)
            key = self.rope(key, position_ids=rope_positions)
        else:
            qkv = (
                self.qkv_proj(x)
                .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
            query_key = self.rope(query_key, position_ids=rope_positions)
            query = query_key.narrow(1, 0, n_heads)
            key = query_key.narrow(1, n_heads, n_kv_heads)
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


class FusedGateUpMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(nn.functional.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = FusedGateUpMLP(hidden_size, config.intermediate_size)

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

    @classmethod
    @override
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        # Preserve the native max_position_embeddings before clamping so that
        # LongRoPE can compute attention_factor from the model's full context ratio.
        if max_context_length is not None and hasattr(hf_config, "max_position_embeddings"):
            hf_config._native_max_position_embeddings = hf_config.max_position_embeddings
        return super()._get_reauthored_config(hf_config, max_context_length, num_layers)

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
        is_gqa = self.config.num_attention_heads != self.config.num_key_value_heads
        if is_gqa:
            n_heads = self.config.num_attention_heads
            n_kv_heads = self.config.num_key_value_heads
            head_dim = getattr(self.config, "head_dim", None) or (
                self.config.hidden_size // n_heads
            )
            q_size = n_heads * head_dim
            k_size = n_kv_heads * head_dim
            v_size = n_kv_heads * head_dim
            for key in [k for k in list(state_dict.keys()) if "qkv_proj.weight" in k]:
                qkv_weight = state_dict.pop(key)
                prefix = key.replace("qkv_proj.weight", "")
                q, k, v = qkv_weight.split([q_size, k_size, v_size], dim=0)
                state_dict[prefix + "q_proj.weight"] = q
                state_dict[prefix + "k_proj.weight"] = k
                state_dict[prefix + "v_proj.weight"] = v

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
