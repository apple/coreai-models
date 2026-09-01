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

import gc
import json
import os
import re
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from typing_extensions import Self, override

from coreai_models._constants import (
    MAIN_GRAPH_NAME,
    SLIDING_KEY_CACHE_NAME,
    SLIDING_VALUE_CACHE_NAME,
)
from coreai_models.models.base import (
    BaseForCausalLM,
    TraceSpec,
    _load_tensors_for_keys,
    _resolve_safetensors_files,
)
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
        seq_len = position_ids.shape[-1]
        offset = seq_len - query_len
        rope_positions = position_ids.narrow(-1, offset, query_len)
        query = self.rope(query, position_ids=rope_positions, freqs=freqs)
        key = self.rope(key, position_ids=rope_positions, freqs=freqs)

        if cache is not None:
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
        query_len = input_ids.shape[-1]
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        h = self.embed_tokens(input_ids)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)

        if cache is not None:
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
            SLIDING_KEY_CACHE_NAME: "sliding_kv_cache",
            SLIDING_VALUE_CACHE_NAME: "sliding_kv_cache",
        }

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    @override
    def from_hf(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        disable_embedding_quantization: bool = False,
        *,
        target_model_id: str = "meta-models/Muse-Glimmer-30B",
    ) -> Self:
        """Load drafter from HuggingFace, borrowing embed_tokens/lm_head from the target.

        The drafter checkpoint (e.g. ``meta-models/Muse-Glimmer-30B-assistant``)
        carries only the 5 transformer layers and final norm.  ``embed_tokens``
        and ``lm_head`` are shared with the full 30B target and must be loaded
        from *target_model_id*.

        Args:
            huggingface_model_id: Drafter checkpoint id
                (e.g. ``meta-models/Muse-Glimmer-30B-assistant``).
            target_model_id: Full target model id to borrow embeddings from.
            Other args: see :meth:`BaseForCausalLM.from_hf`.
        """
        from safetensors import safe_open

        # ---- 1. Download both checkpoints (safetensors + config only) --------
        allow = ["*.safetensors", "*.safetensors.index.json", "config.json"]
        drafter_dir = snapshot_download(huggingface_model_id, allow_patterns=allow)
        target_dir = snapshot_download(target_model_id, allow_patterns=allow)

        # ---- 2. Build drafter config, inject vocab_size from target ----------
        with open(os.path.join(drafter_dir, "config.json")) as f:
            drafter_raw = json.load(f)
        with open(os.path.join(target_dir, "config.json")) as f:
            target_raw = json.load(f)

        # Target is multimodal — vocab_size lives under text_config
        text_cfg = target_raw.get("text_config", target_raw)
        drafter_raw["vocab_size"] = text_cfg["vocab_size"]

        config = SimpleNamespace(**drafter_raw)
        if max_context_length is not None:
            config.max_position_embeddings = max_context_length
        if num_layers is not None:
            config.num_hidden_layers = num_layers

        # ---- 3. Create model on meta device ----------------------------------
        model = cls(config=config, model_device="meta")
        model.to(dtype=target_dtype)

        # ---- 4. Load drafter weights (skip encoder.*, add model. prefix) -----
        drafter_files = _resolve_safetensors_files(drafter_dir)
        drafter_keys: dict[str, str] = {}
        for path in drafter_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith("encoder."):
                        continue
                    if num_layers is not None:
                        m = re.match(r"layers\.(\d+)\.", key)
                        if m and int(m.group(1)) >= num_layers:
                            continue
                    drafter_keys[key] = path

        drafter_sd = _load_tensors_for_keys(drafter_keys, target_dtype)
        # "layers.0.*" → "model.layers.0.*", "norm.weight" → "model.norm.weight"
        remapped: dict[str, torch.Tensor] = {}
        for k, v in drafter_sd.items():
            remapped["model." + k] = v
        del drafter_sd
        model.load_state_dict(remapped, assign=True, strict=False)
        del remapped
        gc.collect()

        # ---- 5. Load embed_tokens and lm_head from target (2 tensors) -------
        target_files = _resolve_safetensors_files(target_dir)
        embed_hf_key = "model.language_model.embed_tokens.weight"
        lm_head_hf_key = "lm_head.weight"
        target_keys: dict[str, str] = {}
        for path in target_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key in (embed_hf_key, lm_head_hf_key):
                        target_keys[key] = path

        if embed_hf_key not in target_keys or lm_head_hf_key not in target_keys:
            found = list(target_keys)
            raise RuntimeError(
                f"Expected '{embed_hf_key}' and '{lm_head_hf_key}' in target checkpoint, "
                f"found: {found}"
            )

        target_sd = _load_tensors_for_keys(target_keys, target_dtype)
        shared: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": target_sd[embed_hf_key],
            "lm_head.weight": target_sd[lm_head_hf_key],
        }
        del target_sd
        model.load_state_dict(shared, assign=True, strict=False)
        del shared
        gc.collect()

        # ---- 6. Validate no meta params remain -------------------------------
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")

        return model

    @classmethod
    @override
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls.from_hf(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            mmap_path=mmap_path,
            num_layers=num_layers,
        )

    # ------------------------------------------------------------------
    # Export contract: 2-state sliding-only (caches are static)
    # ------------------------------------------------------------------

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        """Reference tensors for the drafter's 2-state sliding-only export.

        Sliding caches are STATIC at ``config.sliding_window`` -- they never
        grow, so no dynamic cache dim is needed.
        """
        window_size = config.sliding_window
        n_layers = config.num_hidden_layers
        n_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        position_ids = torch.arange(spec.offset + spec.query_len, dtype=torch.int32).unsqueeze(0)

        sliding_k_cache = torch.zeros(
            n_layers, 1, n_kv_heads, window_size, head_dim, dtype=target_dtype
        )
        sliding_v_cache = torch.zeros(
            n_layers, 1, n_kv_heads, window_size, head_dim, dtype=target_dtype
        )

        return {
            MAIN_GRAPH_NAME: {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "sliding_k_cache": sliding_k_cache,
                "sliding_v_cache": sliding_v_cache,
            }
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        """Dynamic shapes for the drafter's 2-state sliding-only export.

        Sliding caches are pinned to ``config.sliding_window`` (no dynamic dim).
        ``position_ids`` is full-history (length = offset + query_len);
        ``input_ids`` is the query only (length = query_len).
        """
        max_ctx = spec.max_context_length
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
                "position_ids": {
                    1: torch.export.Dim("seq_pos", min=spec.query_len, max=max_ctx - 1)
                },
                "sliding_k_cache": None,
                "sliding_v_cache": None,
            }
        }
