# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne as Gemma3RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

USE_FUSED_KV = True


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(up_tensor * gate_tensor)


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(
            self.embed_scale, dtype=self.weight.dtype, device=self.weight.device
        )


class Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        is_local = (layer_idx + 1) % config._sliding_window_pattern != 0
        self.sdpa = SDPA(
            scale=config.query_pre_attn_scalar**-0.5,
            window_size=config.sliding_window if is_local else 0,
            is_causal=True,
        )

        # Gemma-3 rope. Transformers ≥ 4.x moved both `rope_theta` /
        # `rope_local_base_freq` and `rope_scaling` into nested per-attention
        # dicts under `config.rope_parameters` (keyed by `sliding_attention`
        # and `full_attention`). Read from the nested layout first, fall back
        # to the legacy flat attributes.
        attn_key = "sliding_attention" if is_local else "full_attention"
        nested = getattr(config, "rope_parameters", None) or {}
        nested_attn = nested.get(attn_key) if isinstance(nested, dict) else None
        if isinstance(nested_attn, dict):
            base = nested_attn.get("rope_theta")
            rope_type = nested_attn.get("rope_type", "default")
            rope_factor = nested_attn.get("factor", 1.0) if rope_type == "linear" else 1.0
        else:
            legacy_attr = "rope_local_base_freq" if is_local else "rope_theta"
            base = getattr(config, legacy_attr, None) or resolve_rope_theta(config)
            scaling = getattr(config, "rope_scaling", None)
            if isinstance(scaling, dict) and scaling.get("rope_type") == "linear":
                rope_factor = scaling.get("factor", 1.0)
            else:
                rope_factor = 1.0

        self.rope = RoPE(
            base=base,
            scale=1.0 if is_local else float(1 / rope_factor),
        )

        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        if USE_FUSED_KV:
            self.qk_norm = Gemma3RMSNorm(
                head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads
            )
        else:
            self.q_norm = Gemma3RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = Gemma3RMSNorm(head_dim, eps=config.rms_norm_eps)

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


class TransformerBlock(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        eps = config.rms_norm_eps
        self.input_layernorm = Gemma3RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = Embedding(
            config.vocab_size,
            hidden_size,
            config.pad_token_id,
            embed_scale=hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(hidden_size, eps=config.rms_norm_eps)

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


class Gemma3ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFGemma3ForCausalLM

    # Emit a second, prefill-only ``prefill`` entrypoint beside ``main``.
    exports_prefill_graph = True

    @override
    def _init_model(self, config: Gemma3TextConfig) -> None:
        self.model = Gemma3Model(config)
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
