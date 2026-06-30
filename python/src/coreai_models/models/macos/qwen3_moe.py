# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

USE_FUSED_KV = True


class Attention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        if USE_FUSED_KV:
            self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads)
        else:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=True, scale=head_dim**-0.5)
        assert is_default_rope_scaling(config), f"unsupported rope_scaling: {config.rope_scaling}"
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

        if USE_FUSED_KV:
            query_key = self.qk_norm(query_key)
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)

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
            self.sdpa(query=query, key=key, value=value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class SparseMoeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, hidden_dim, num_experts)
        self.norm_topk_prob = norm_topk_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x).to(torch.float32)

        if self.norm_topk_prob:
            top_logits, active_experts_indices = torch.topk(
                router_logits, self.top_k, dim=-1, largest=True
            )
            active_experts_scores = torch.softmax(top_logits, dim=-1)
        else:
            gates = torch.softmax(router_logits, dim=-1)
            active_experts_scores, active_experts_indices = torch.topk(
                gates, self.top_k, dim=-1, largest=True
            )
        active_experts_indices = active_experts_indices.to(torch.uint16)

        y_active_experts = self.switch_mlp(x, active_experts_indices)
        active_experts_scores = active_experts_scores.unsqueeze(-1)
        y_active_experts_weighted_by_scores = y_active_experts * active_experts_scores
        y_active_experts_summary = torch.sum(y_active_experts_weighted_by_scores, axis=-2)
        return y_active_experts_summary.to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        if config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = SparseMoeBlock(
                dim=hidden_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
            )
        else:
            self.mlp = MLP(hidden_size, config.intermediate_size)

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


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
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


class Qwen3MoeForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFQwen3MoeForCausalLM

    @override
    def _init_model(self, config: Qwen3MoeConfig) -> None:
        self.model = Qwen3MoeModel(config)
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
            # Fuse q/k/v projections into combined qkv_proj
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

            # Fuse q_norm/k_norm into qk_norm
            if USE_FUSED_KV:
                q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
                k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"

                if q_norm_key in state_dict and k_norm_key in state_dict:
                    layer = self.model.layers[i]
                    n_heads = layer.self_attn.n_heads
                    n_kv_heads = layer.self_attn.n_kv_heads
                    head_dim = layer.self_attn.head_dim

                    q_norm_weight = state_dict[q_norm_key].unsqueeze(0).unsqueeze(0)
                    k_norm_weight = state_dict[k_norm_key].unsqueeze(0).unsqueeze(0)

                    q_repeated = q_norm_weight.expand(n_heads, 1, head_dim)
                    k_repeated = k_norm_weight.expand(n_kv_heads, 1, head_dim)
                    fused_weight = torch.cat([q_repeated, k_repeated], dim=0)

                    state_dict[f"model.layers.{i}.self_attn.qk_norm.weight"] = fused_weight

                    del state_dict[q_norm_key]
                    del state_dict[k_norm_key]

        # Handle MoE weights: stack per-expert weights into SwitchGLU layout
        for i in range(max_layer + 1):
            prefix = f"model.layers.{i}.mlp"

            if f"{prefix}.experts.0.gate_proj.weight" not in state_dict:
                continue

            num_experts = 0
            while f"{prefix}.experts.{num_experts}.gate_proj.weight" in state_dict:
                num_experts += 1

            for proj in ["gate_proj", "down_proj", "up_proj"]:
                first_key = f"{prefix}.experts.0.{proj}.weight"
                first_weight = state_dict[first_key]

                output = torch.empty(
                    (1, num_experts) + first_weight.shape,
                    dtype=first_weight.dtype,
                    device=first_weight.device,
                )

                for e in range(num_experts):
                    expert_weight = state_dict.pop(f"{prefix}.experts.{e}.{proj}.weight")
                    output[0, e] = expert_weight

                state_dict[f"{prefix}.switch_mlp.{proj}.weight"] = output
