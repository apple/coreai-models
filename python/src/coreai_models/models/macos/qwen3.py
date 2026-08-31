# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA

USE_FUSED_KV = True


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
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

        self.sdpa = SDPA(is_causal=True)
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
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
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


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
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
        position_ids: torch.IntTensor = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Qwen3ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFQwen3ForCausalLM

    # Emit a second, prefill-only ``prefill`` entrypoint beside ``main``.
    exports_prefill_graph = True

    @override
    def _init_model(self, config: Qwen3Config) -> None:
        self.model = Qwen3Model(config)
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
    ) -> torch.Tensor | tuple:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        if self.prefill_mode:
            # A bare `return` causes torch export to trace a leaf node with value
            # `None` rather than having no leaf nodes whatsoever. Remedied with
            # empty tuple.
            return ()
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
