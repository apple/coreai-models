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

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)

        self.o_proj = nn.Conv2d(n_heads * head_dim, dim, kernel_size=1, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(head_dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = (
            query.transpose(-3, -1)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .transpose(-2, -3)
        )
        key = (
            key.transpose(-3, -1)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .transpose(-2, -3)
        )

        query = self.q_norm(query)
        key = self.k_norm(key)

        seq_len = rope_cos.shape[1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        query = (
            query.transpose(-2, -3)
            .reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )
        key = (
            key.transpose(-3, -2)
            .reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx,
                in_step,
                key,
                value,
                query_len,
            )

        output = self.sdpa(query, key, value, causal_mask)
        output = self.o_proj(output)
        return output.transpose(-3, -1)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(dim=hidden_size, hidden_dim=config.intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            cache,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            token_embeddings = layer(
                token_embeddings,
                rope_cos,
                rope_sin,
                in_step,
                causal_mask,
                cache,
            )
        return self.norm(token_embeddings)


class Qwen3Extend(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = resolve_rope_theta(config)
        self.rope = RoPECache(head_dim, config.max_position_embeddings, rope_theta)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            self.kv_cache,
        )
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            return self.lm_head(out.transpose(-2, -3))

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table,
                self.emb_scale,
                self.emb_zero_point,
                out.dtype,
            )

        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )

        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
        return (embedding_table @ out).transpose(-2, -1)


class Qwen3ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFQwen3ForCausalLM

    def _init_model(self, config: Qwen3Config) -> None:
        self.extend = Qwen3Extend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Rewrite HF weight keys into the iOS module layout, in place.

        Called repeatedly on partial state_dicts: the full dict, one
        single-layer slice at a time, and a shared-params-only slice
        (embeddings / lm_head). Handles any subset of keys and does not assume
        all layers are present.
        """
        present_layers = set()
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            present_layers.add(int(name_split[2]))

        # Access present layers' keys unconditionally so a malformed layer fails
        # loudly instead of being silently skipped; a layer-less slice is valid
        # only if it carries shared params.
        has_shared_keys = (
            "model.embed_tokens.weight" in state_dict or "lm_head.weight" in state_dict
        )
        if not present_layers and not has_shared_keys:
            err = (
                "state_dict has no recognizable keys: expected per-layer weights "
                "('model.layers.N.*') and/or shared params "
                "('model.embed_tokens.weight', 'lm_head.weight')."
            )
            raise ValueError(err)

        for layer_idx in sorted(present_layers):
            # Reshape attention weights for Conv2d
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
                state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            # Reshape MLP weights for Conv2d
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                weight_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
                state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

        # Handle embeddings (shared param. will be absent from per-layer slices)
        if "model.embed_tokens.weight" in state_dict:
            embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
            if not self.disable_embedding_quantization:
                embedding_table, scale, zero_point = quantize_per_tensor(
                    embedding_table, nbits=8, symmetric=True
                )
            else:
                scale = torch.tensor(1.0, dtype=embedding_table.dtype)
                zero_point = torch.tensor(0, dtype=torch.int8)

            state_dict["load_embeddings.embedding_table"] = embedding_table
            state_dict["gather_embeddings.scale"] = scale
            state_dict["gather_embeddings.zero_point"] = zero_point
            state_dict["extend.emb_scale"] = scale
            state_dict["extend.emb_zero_point"] = zero_point

            state_dict.pop("model.embed_tokens.weight")

        # Qwen3Model is held inside Qwen3Extend — add "extend." prefix
        new_state_dict = {}
        keys_to_pop = set()

        for k, _v in state_dict.items():
            if k.startswith("model.") and "gather_embeddings" not in k:
                new_key = f"extend.{k}"
                new_state_dict[new_key] = state_dict[k]
                keys_to_pop.add(k)

        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings and "lm_head.weight" in state_dict:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]

        state_dict.pop("lm_head.weight", None)
