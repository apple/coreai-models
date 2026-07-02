# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM as HFMixtralForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

USE_FUSED_KV = True


class SparseMoeBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x).to(torch.float32)

        top_logits, active_experts_indices = torch.topk(
            router_logits, self.top_k, dim=-1, largest=True
        )
        active_experts_scores = torch.softmax(top_logits, dim=-1).to(x.dtype)

        y_active_experts = self.switch_mlp(x, active_experts_indices)
        active_experts_scores = active_experts_scores.unsqueeze(-1)
        y_active_experts_weighted_by_scores = y_active_experts * active_experts_scores
        y_active_experts_summary = torch.sum(y_active_experts_weighted_by_scores, dim=-2)
        return y_active_experts_summary.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = dim // n_heads
        self.head_dim = head_dim

        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True, scale=head_dim**-0.5)
        self.rope = initialize_rope(base=resolve_rope_theta(config))

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

        if USE_FUSED_KV:
            query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
        else:
            query = qkv.narrow(1, 0, n_heads)
            key = qkv.narrow(1, n_heads, n_kv_heads)

        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        if USE_FUSED_KV:
            query_key = self.rope(query_key, position_ids=rope_positions)
            query = query_key.narrow(1, 0, n_heads)
            key = query_key.narrow(1, n_heads, n_kv_heads)
        else:
            query = self.rope(query, position_ids=rope_positions)
            key = self.rope(key, position_ids=rope_positions)

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
    def __init__(self, config: MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.block_sparse_moe = SparseMoeBlock(
            dim=hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        return h + r


class MixtralModel(nn.Module):
    def __init__(self, config: MixtralConfig) -> None:
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


class MixtralForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFMixtralForCausalLM

    @override
    def _init_model(self, config: MixtralConfig) -> None:
        self.model = MixtralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict"
            raise ValueError(err)

        for i in range(max_layer + 1):
            combined_weight = []
            need_to_fuse = True
            for proj in ["q_proj", "k_proj", "v_proj"]:
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                if weight_key not in state_dict:
                    need_to_fuse = False
                    continue
                combined_weight.append(state_dict[weight_key])
                del state_dict[weight_key]
            if need_to_fuse:
                state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.concat(
                    combined_weight, axis=0
                )

        # Handle MoE weights: w1->gate_proj, w2->down_proj, w3->up_proj
        for i in range(max_layer + 1):
            prefix = f"model.layers.{i}.block_sparse_moe"

            if f"{prefix}.experts.0.w1.weight" not in state_dict:
                continue

            num_experts = 0
            while f"{prefix}.experts.{num_experts}.w1.weight" in state_dict:
                num_experts += 1

            weight_mappings = [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]

            for v1, v2 in weight_mappings:
                first_key = f"{prefix}.experts.0.{v1}.weight"
                first_weight = state_dict[first_key]

                output = torch.empty(
                    (1, num_experts) + first_weight.shape,
                    dtype=first_weight.dtype,
                    device=first_weight.device,
                )

                for e in range(num_experts):
                    expert_weight = state_dict.pop(f"{prefix}.experts.{e}.{v1}.weight")
                    output[0, e] = expert_weight

                state_dict[f"{prefix}.switch_mlp.{v2}.weight"] = output
