# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse Glimmer text decoder for CoreAI model export.

Meta's 30B on-device agentic model (Apache 2.0). Architecture features:
- Local/Global attention: [S,S,S,G] repeating (39 sliding + 13 full)
- RoPE on local layers only (global layers skip RoPE)
- Extreme GQA: 32Q / 2KV heads
- Gated attention: learned gate_proj on attention output
- Sandwich norm: pre+post norm on both attention and MLP
- qk_scale_factor: custom attention scaling (not 1/sqrt(d))
- output_multiplier: scales final hidden state before lm_head
- Logit softcapping: tanh(logits/cap) * cap
"""

import gc
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from typing_extensions import Self, override

from coreai_models.models.base import (
    BaseForCausalLM,
    _load_tensors_for_keys,
    _resolve_safetensors_files,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm, RMSNormPlusOne
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, n_heads * head_dim, bias=False)

        self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.qk_scale_factor = getattr(config, "qk_scale_factor", 1.0)

        layer_types = config.layer_types
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"

        layer_rope_theta = config.layer_rope_theta
        rope_theta = layer_rope_theta[layer_idx] if layer_rope_theta else 500000.0
        self.has_rope = rope_theta > 0

        if self.is_sliding:
            self.sdpa = SDPA(is_causal=True, window_size=config.sliding_window)
        else:
            self.sdpa = SDPA(is_causal=True)

        if self.has_rope:
            self.rope = RoPE()
            with torch.device("cpu"):
                self._rope_freqs = 1.0 / (
                    rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
                )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        query = (
            self.qk_norm(
                self.q_proj(x)
                .reshape(batch_size, query_len, n_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            * self.qk_scale_factor
        )
        key = self.qk_norm(
            self.k_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.v_proj(x)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        gate = torch.sigmoid(self.gate_proj(x))

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        if self.has_rope:
            freqs = self._rope_freqs.to(device=query.device)
            query = self.rope(query, position_ids=rope_positions, freqs=freqs)
            key = self.rope(key, position_ids=rope_positions, freqs=freqs)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        attn_output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )

        return self.o_proj(attn_output * gate)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        post_eps = getattr(config, "post_norm_eps", config.rms_norm_eps)
        self.input_layernorm = RMSNormPlusOne(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(hidden_size, eps=post_eps)
        self.pre_feedforward_layernorm = RMSNormPlusOne(hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNormPlusOne(hidden_size, eps=post_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        r = self.post_attention_layernorm(r)
        h = x + r
        r = self.mlp(self.pre_feedforward_layernorm(h))
        r = self.post_feedforward_layernorm(r)
        return h + r


class MuseGlimmerModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.output_multiplier = getattr(config, "output_multiplier", 1.0)
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
        # MuseGlimmerTextNormedEmbedding: weight-less RMSNorm on embeddings
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        h = self.norm(h)
        if self.output_multiplier != 1.0:
            h = h * self.output_multiplier
        return h


class MuseGlimmerForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = None  # Not in our transformers version

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        return text_config

    @override
    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        disable_embedding_quantization: bool = False,
    ) -> Self:
        return cls.from_hf_memory_efficient(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            mmap_path=mmap_path,
            num_layers=num_layers,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        )

    @override
    @classmethod
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = "text_config",
        hf_state_dict_prefix: str = "model.language_model.",
        disable_embedding_quantization: bool = False,
    ) -> Self:
        import re

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )

        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        cfg_dict = raw.get(hf_config_attr, raw) if hf_config_attr else raw
        hf_config = SimpleNamespace(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict

        config = cls._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)
        model = cls(config=config, model_device="meta")
        model.to(dtype=target_dtype)

        safetensors_files = _resolve_safetensors_files(model_dir)

        # Build key index with Muse Glimmer's actual key layout:
        #   model.language_model.layers.N.*  → per-layer
        #   model.language_model.embed_tokens.weight, .norm.weight → shared
        #   lm_head.weight → shared (no prefix)
        #   model.vision_* → skip
        layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.")
        from safetensors import safe_open

        per_layer: dict[int, dict[str, str]] = {}
        shared: dict[str, str] = {}
        for path in safetensors_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith("model.vision_tower.") or key.startswith("model.vision_"):
                        continue
                    match = layer_pattern.match(key)
                    if match:
                        layer_idx = int(match.group(1))
                        if num_layers is not None and layer_idx >= num_layers:
                            continue
                        per_layer.setdefault(layer_idx, {})[key] = path
                    else:
                        shared[key] = path

        # Load shared params (embed_tokens, norm, lm_head)
        shared_dict = _load_tensors_for_keys(shared, target_dtype)
        # Normalize keys: strip "model.language_model." prefix where present
        normalized: dict[str, torch.Tensor] = {}
        prefix = "model.language_model."
        for k, v in shared_dict.items():
            if k.startswith(prefix):
                normalized["model." + k[len(prefix) :]] = v
            else:
                normalized[k] = v
        del shared_dict
        model.load_state_dict(normalized, assign=True, strict=False)
        del normalized
        gc.collect()

        # Load one layer at a time
        for layer_idx in sorted(per_layer.keys()):
            layer_key_to_file = per_layer.pop(layer_idx)
            layer_sd = _load_tensors_for_keys(layer_key_to_file, target_dtype)
            del layer_key_to_file
            # Strip prefix → "layers.N.*", then add "model." → "model.layers.N.*"
            remapped: dict[str, torch.Tensor] = {}
            for k, v in layer_sd.items():
                remapped["model." + k[len(prefix) :]] = v
            del layer_sd
            model.load_state_dict(remapped, assign=True, strict=False)
            del remapped
            gc.collect()

        # qk_norm has no checkpoint weights — initialize to ones (identity RMSNorm)
        for layer in model.model.layers:
            layer.self_attn.qk_norm.weight = nn.Parameter(
                torch.ones(model.config.head_dim, dtype=target_dtype)
            )

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")

        return model

    @override
    def _init_model(self, config) -> None:
        self.model = MuseGlimmerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._softcap = getattr(config, "final_logit_softcapping", None)
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
        logits = self.lm_head(out)
        if self._softcap:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Keys arrive in one of two forms:
        # (a) Raw: "model.language_model.layers.0.self_attn.q_proj.weight"
        # (b) Already-stripped by from_hf_memory_efficient: "layers.0.self_attn.q_proj.weight"
        # Normalize all to "model.layers.N.*" / "model.embed_tokens.*" / "lm_head.*"
        prefix = "model.language_model."
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith("model.vision_tower.") or key.startswith("model.vision_"):
                del state_dict[key]
            elif key.startswith(prefix):
                state_dict["model." + key[len(prefix) :]] = state_dict.pop(key)
            elif (
                key.startswith("layers.") or key.startswith("norm.") or key == "embed_tokens.weight"
            ):
                state_dict["model." + key] = state_dict.pop(key)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
