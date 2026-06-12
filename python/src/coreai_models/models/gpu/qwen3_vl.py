"""Qwen3-VL text decoder for CoreAI model export.

Like qwen3.py, but for the VL checkpoint layout: text decoder weights live
under model.language_model.*, vision encoder weights (model.visual.*) are dropped.
"""

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
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
        self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads)
        self.sdpa = SDPA(is_causal=True)
        self.rope = initialize_rope(base=config.rope_theta)

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
        query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        query_key = self.qk_norm(query_key)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query_key = self.rope(query_key, position_ids=rope_positions)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

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
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
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


class Qwen3VLModel(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
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


class Qwen3VLForCausalLM(BaseForCausalLM):
    """Engine-compatible Qwen3-VL text decoder (input_ids variant)."""

    _HF_MODEL_CLASS = HFQwen3VLForConditionalGeneration

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

        rope_theta = getattr(text_config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(text_config, "rope_parameters", None) or getattr(
                text_config, "rope_scaling", None
            )
            if rope_params and "rope_theta" in rope_params:
                rope_theta = rope_params["rope_theta"]
            else:
                rope_theta = 5_000_000
        text_config.rope_theta = float(rope_theta)
        text_config.rope_scaling = None
        return text_config

    @override
    def _init_model(self, config: Qwen3VLTextConfig) -> None:
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
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
        # Keys arrive in one of two forms depending on the loading path:
        # (a) Raw checkpoint keys:    "model.language_model.layers.0.self_attn.q_proj.weight"
        # (b) Already-stripped keys:  "layers.0.self_attn.q_proj.weight"
        #     (when from_hf_memory_efficient strips "model.language_model." prefix)
        # Step 1: Normalize all keys to "model.layers.N.*" / "model.embed_tokens.*" / etc.
        keys_to_process = list(state_dict.keys())
        for key in keys_to_process:
            if key.startswith("model.visual."):
                del state_dict[key]
            elif key.startswith("model.language_model."):
                # Raw checkpoint form: strip "model.language_model." → add "model."
                new_key = "model." + key[len("model.language_model."):]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith("layers.") or key.startswith("norm.") or key == "embed_tokens.weight":
                # Already-stripped form: add "model." prefix
                state_dict["model." + key] = state_dict.pop(key)
            # Keys already in "model.*" form pass through unchanged

        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict: no transformer layers found"
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
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Embeddings variant (takes inputs_embeds instead of input_ids)
# ---------------------------------------------------------------------------

class Qwen3VLModelEmbeddings(nn.Module):
    """Variant of Qwen3VLModel that accepts pre-computed embeddings."""

    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Qwen3VLForCausalLMEmbeddings(BaseForCausalLM):
    """Engine-compatible Qwen3-VL text decoder (inputs_embeds variant).

    forward(inputs_embeds: [1, seq, hidden], position_ids: [1, total_pos], k_cache, v_cache) -> logits
    """

    _HF_MODEL_CLASS = HFQwen3VLForConditionalGeneration

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

        rope_theta = getattr(text_config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(text_config, "rope_parameters", None) or getattr(
                text_config, "rope_scaling", None
            )
            if rope_params and "rope_theta" in rope_params:
                rope_theta = rope_params["rope_theta"]
            else:
                rope_theta = 5_000_000
        text_config.rope_theta = float(rope_theta)
        text_config.rope_scaling = None
        return text_config

    @override
    def _init_model(self, config: Qwen3VLTextConfig) -> None:
        self.model = Qwen3VLModelEmbeddings(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(inputs_embeds, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Normalize keys to "model.layers.N.*" form (same two-form handling as Qwen3VLForCausalLM)
        keys_to_process = list(state_dict.keys())
        for key in keys_to_process:
            if key.startswith("model.visual."):
                del state_dict[key]
            elif key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith("layers.") or key.startswith("norm.") or key == "embed_tokens.weight":
                state_dict["model." + key] = state_dict.pop(key)

        # Drop embed_tokens or convert to lm_head.weight for tied embeddings
        for k in list(state_dict.keys()):
            if "embed_tokens" in k:
                if getattr(self.config, "tie_word_embeddings", False):
                    state_dict["lm_head.weight"] = state_dict.pop(k)
                else:
                    del state_dict[k]

        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict: no transformer layers found"
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
