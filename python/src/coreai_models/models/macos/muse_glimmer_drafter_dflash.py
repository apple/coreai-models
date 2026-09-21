# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse Glimmer DFlash drafter -- two-phase speculative decoding prototype.

Architecture (same backbone as the standard ring drafter):
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
    DRAFT_GRAPH_NAME,
    INJECT_KV_GRAPH_NAME,
    SLIDING_KEY_CACHE_NAME,
    SLIDING_VALUE_CACHE_NAME,
)
from coreai_models.models.base import (
    BaseForCausalLM,
    TraceSpec,
    _load_tensors_for_keys,
    _resolve_safetensors_files,
    move_model_to_disk,
)
from coreai_models.primitives.macos.cache import RingKVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def dflash_draft_mask(
    query_len: int,
    capacity: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Bidirectional attention mask for the draft phase."""
    slot = torch.arange(capacity, device=device)
    last_pos = offset + query_len - 1

    # Reconstruct absolute position stored in each ring slot.
    # Same derivation as ring_window_causal_mask but without the causal check.
    r = last_pos % capacity
    diff = r - slot  # range (-capacity, capacity)
    ring_back = torch.where(diff >= 0, diff, diff + capacity)
    k_pos = last_pos - ring_back  # (capacity,)

    total_valid = offset + query_len
    valid = (k_pos >= 0) & (k_pos < total_valid)
    return valid.unsqueeze(0).expand(query_len, capacity)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Attention with separate inject_kv and draft paths."""

    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
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

        # Non-causal; explicit mask required
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

    # ---- Phase 1: KV injection (no attention) ----------------------------

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Project features through Wk/Wv and write to ring cache.

        ``position_ids`` is FULL-HISTORY (length = offset + n_features): the
        injection offset is derived from shapes (``offset = seq_len -
        n_features``) rather than read out of the tensor with ``.item()``, so
        the graph exports cleanly. This matches ``forward`` (the draft path).
        For base-offset-0 injection (length-``n_features`` positions) offset is
        0, identical to the old ``int(position_ids[0, 0])`` behaviour.
        """
        batch_size, n_features, _ = features.shape
        n_kv_heads = self.n_kv_heads

        # Shape-derived injection offset (== first absolute position injected).
        seq_len = position_ids.shape[-1]
        offset = seq_len - n_features
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, n_features)

        # K projection + norm + RoPE
        key = self.k_norm(
            self.k_proj(features)
            .reshape(batch_size, n_features, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        freqs = self._rope_freqs.to(device=features.device)
        key = self.rope(key, position_ids=rope_positions, freqs=freqs)

        # V projection (no norm, no RoPE)
        value = (
            self.v_proj(features)
            .reshape(batch_size, n_features, n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        cache.update_and_fetch(self.layer_idx, offset=offset, k=key, v=value, query_len=n_features)

    # ---- Phase 2: Draft forward (full attention) -------------------------

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward for draft tokens."""
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

        # Write draft KV to cache, fetch full cache (injected + draft)
        key, value = cache.update_and_fetch(self.layer_idx, offset, key, value, query_len=query_len)

        attn_output = (
            self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """KV injection: project features and write to cache."""
        self.self_attn.inject_kv(features, position_ids, cache)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Draft forward: attention + MLP."""
        r = self.self_attn(self.input_layernorm(x), position_ids, cache, attn_mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DFlashDrafterModel(nn.Module):
    """DFlash drafter backbone with two-phase forward.

    Phase 1 (``inject_kv``): target encoder features -> per-layer Wk/Wv -> KV cache.
    Phase 2 (``draft``): ``[last_token, MASK * K]`` -> full transformer -> hidden states.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Write target encoder features to the KV cache."""
        for layer in self.layers:
            layer.inject_kv(features, position_ids, cache)  # type: ignore[operator]

    def draft(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> torch.Tensor:
        """Run draft tokens through the transformer, attending to injected KV."""
        query_len = input_ids.shape[-1]
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len  # = n_injected
        torch._check_is_size(offset)

        h = self.embed_tokens(input_ids)
        # NO embed norm for DFlash — HF explicitly bypasses it:
        # "The assistant needs embedding without norm" (candidate_generator.py:1676)

        # Bidirectional mask over valid ring positions
        attn_mask = dflash_draft_mask(
            query_len=query_len,
            capacity=cache.capacity(),
            offset=offset,
            device=input_ids.device,
        )

        for layer in self.layers:
            h = layer(h, position_ids, cache, attn_mask)
        return self.norm(h)


class MuseGlimmerDFlashDrafterForCausalLM(BaseForCausalLM):
    """DFlash drafter with lm_head, exported as two shared-KV entrypoints.

    The macOS export emits two CoreML entrypoints from this one eager model,
    both binding the SAME ring-KV state (``slidingKeyCache`` /
    ``slidingValueCache``):

    - ``inject_kv`` (write-only): target encoder features -> per-layer Wk/Wv ->
      ring cache. Declares no outputs, mirroring the prefill graph's discipline.
    - ``draft`` (read+write): ``[anchor, MASK * K]`` -> transformer attending to
      the injected KV -> ``logits`` ``[1, K, vocab]``.

    The eager ``inject_kv`` / ``draft`` methods remain for Python-side acceptance
    testing; ``inject_kv_graph`` / ``draft_graph`` are the flat-tensor trace
    wrappers the exporter drives.

    Usage (eager)::

        model = MuseGlimmerDFlashDrafterForCausalLM(config)
        model.inject_kv(features, inject_positions, cache)
        logits = model.draft(draft_input_ids, full_position_ids, cache)
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config) -> None:
        self.model = DFlashDrafterModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @override
    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    # ------------------------------------------------------------------
    # Eager API (Python-side acceptance / parity testing)
    # ------------------------------------------------------------------

    def inject_kv(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> None:
        """Write encoder features to KV cache."""
        self.model.inject_kv(features, position_ids, cache)

    def draft(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: RingKVCache,
    ) -> torch.Tensor:
        """Draft K tokens, return logits ``(batch, K, vocab_size)``."""
        hidden = self.model.draft(input_ids, position_ids, cache)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------
    # Flat-tensor trace wrappers (the two exported entrypoints)
    # ------------------------------------------------------------------

    def inject_kv_graph(
        self,
        features: torch.Tensor,
        position_ids: torch.IntTensor,
        sliding_k_cache: torch.Tensor,
        sliding_v_cache: torch.Tensor,
    ) -> tuple[()]:
        """``inject_kv`` entrypoint: rebuild the ring cache, inject, return ()."""
        cache = RingKVCache(sliding_k_cache, sliding_v_cache)
        self.model.inject_kv(features, position_ids, cache)
        return ()

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def draft_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        sliding_k_cache: torch.Tensor,
        sliding_v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """``draft`` entrypoint: rebuild the (shared) ring cache, draft, return logits."""
        cache = RingKVCache(sliding_k_cache, sliding_v_cache)
        hidden = self.model.draft(input_ids, position_ids, cache)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------
    # Export contract: two entrypoints sharing one sliding ring-KV buffer
    # ------------------------------------------------------------------

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            INJECT_KV_GRAPH_NAME: ("features", "position_ids"),
            DRAFT_GRAPH_NAME: ("input_ids", "position_ids"),
        }

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        # Identical tuple for both graphs => the converter binds ONE physical
        # ring buffer across both entrypoints.
        shared = (SLIDING_KEY_CACHE_NAME, SLIDING_VALUE_CACHE_NAME)
        return {INJECT_KV_GRAPH_NAME: shared, DRAFT_GRAPH_NAME: shared}

    @classmethod
    @override
    def export_state_classification(cls) -> dict[str, str]:
        # Keyed by state name (one dict covers both graphs); matches the ring drafter.
        return {
            SLIDING_KEY_CACHE_NAME: "sliding_kv_cache",
            SLIDING_VALUE_CACHE_NAME: "sliding_kv_cache",
        }

    @classmethod
    @override
    def export_output_names(cls) -> dict[str, tuple[str, ...]]:
        return {INJECT_KV_GRAPH_NAME: (), DRAFT_GRAPH_NAME: ("logits",)}

    # ------------------------------------------------------------------
    # Reference inputs + dynamic shapes (per graph; sliding caches STATIC)
    # ------------------------------------------------------------------

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        """Reference tensors for both entrypoints.

        Sliding caches are STATIC at ``config.sliding_window``. ``features`` /
        ``input_ids`` are query-length while ``position_ids`` is full-history
        (length = ``spec.offset + spec.query_len``), so the shape-derived offset
        is ``spec.offset`` -- exercising the fixed (non-``.item()``) inject path.
        """
        window_size = config.sliding_window
        n_layers = config.num_hidden_layers
        n_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        def _caches() -> tuple[torch.Tensor, torch.Tensor]:
            k = torch.zeros(n_layers, 1, n_kv_heads, window_size, head_dim, dtype=target_dtype)
            v = torch.zeros(n_layers, 1, n_kv_heads, window_size, head_dim, dtype=target_dtype)
            return k, v

        position_ids = torch.arange(spec.offset + spec.query_len, dtype=torch.int32).unsqueeze(0)

        inject_k, inject_v = _caches()
        draft_k, draft_v = _caches()
        return {
            INJECT_KV_GRAPH_NAME: {
                "features": torch.zeros(1, spec.query_len, config.hidden_size, dtype=target_dtype),
                "position_ids": position_ids,
                "sliding_k_cache": inject_k,
                "sliding_v_cache": inject_v,
            },
            DRAFT_GRAPH_NAME: {
                "input_ids": torch.randint(
                    1, config.vocab_size, (1, spec.query_len), dtype=torch.int32
                ),
                "position_ids": position_ids,
                "sliding_k_cache": draft_k,
                "sliding_v_cache": draft_v,
            },
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        """Dynamic shapes for both entrypoints (sliding caches pinned static)."""
        max_ctx = spec.max_context_length
        return {
            INJECT_KV_GRAPH_NAME: {
                "features": {1: torch.export.Dim("inject_len", max=max_ctx - 2)},
                "position_ids": {
                    1: torch.export.Dim("inject_pos", min=spec.query_len, max=max_ctx - 1)
                },
                "sliding_k_cache": None,
                "sliding_v_cache": None,
            },
            DRAFT_GRAPH_NAME: {
                "input_ids": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
                "position_ids": {
                    1: torch.export.Dim("seq_pos", min=spec.query_len, max=max_ctx - 1)
                },
                "sliding_k_cache": None,
                "sliding_v_cache": None,
            },
        }

    # ------------------------------------------------------------------
    # Weight loading (borrow embed_tokens/lm_head from the target)
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
        """Load the DFlash drafter from HuggingFace, borrowing embed/lm_head from target.

        The drafter checkpoint carries only the transformer layers and final
        norm; ``embed_tokens`` and ``lm_head`` are shared with the full 30B
        target and loaded from *target_model_id*. Structurally identical to the
        ring drafter's loader.

        Args:
            huggingface_model_id: Drafter checkpoint id.
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

        # Move weights to disk-backed mmap if a path is provided, matching
        # BaseForCausalLM.from_hf (which threads mmap_path through the same way).
        if mmap_path is not None:
            move_model_to_disk(model, path=mmap_path)

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
