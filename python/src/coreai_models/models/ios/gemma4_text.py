# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma4 text decoder iOS implementation.

Supports E2B and E4B variants. Text decoder only (no vision/audio).

Key differences from other iOS models (Qwen3, Llama):
  - Dual head dimensions (sliding=256, global=512)
  - Shared KV cache (later layers reuse KV from earlier same-type layers)
  - V-norm (scale-free RMSNorm on values)
  - Per-Layer Embeddings (PLE) as external input
  - 4 RMSNorm layers per block (pre+post attention, pre+post MLP)
  - Layer scalar (learned per-layer buffer)
  - Proportional RoPE for global layers (partial_rotary_factor=0.25)
  - Double-wide MLP for E2B shared layers
  - GELU activation (not SiLU)
  - RMSNorm (not RMSNormPlusOne)
  - Final logit soft cap (``c * tanh(logits / c)``) deliberately *not* applied here:
    ``tanh`` is best run on the CPU rather than in the graph, so the graph emits raw
    logits and the runner applies the cap on the CPU. The cap is published to the runner as
    ``language.overrides.final_logit_softcapping`` in bundle metadata (see
    ``models/gemma4/export.py``) and applied by Swift's ``LogitSoftcapProcessor``.
  - Two compacted KV caches: a full-context global cache and a small sliding-window
    ring cache, each holding only its type's storing layers (correct sliding-window
    attention enables large contexts)
"""

from typing import Any

import torch
import torch.nn as nn
from coreai.authoring.types import AllocationType, HardwareConstraints
from safetensors.torch import save_file
from transformers import Gemma4ForConditionalGeneration
from typing_extensions import override

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    KEY_CACHE_OUTPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    VALUE_CACHE_OUTPUT_NAME,
)
from coreai_models.models.base import BaseForCausalLMForiOS, TraceSpec
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import apply_rope
from coreai_models.primitives.ios.sdpa import SDPA, BlockedSDPA

# Flash chunk width for the global (full-attention) ``BlockedSDPA`` over the flat cache:
# the largest key-axis chunk whose per-block score/softmax stays within the accelerator's
# supported tensor sizes (8192 empirically). Overridable via the config attribute
# ``kv_block_size``.
DEFAULT_KV_BLOCK_SIZE = 8192

# Gemma4-specific graph I/O names, extending the shared iOS ones in `_constants`.
ROPE_COS_INPUT_NAME = "rope_cos"
ROPE_SIN_INPUT_NAME = "rope_sin"
SLIDING_CAUSAL_MASK_INPUT_NAME = "sliding_causal_mask"
SLIDING_IN_STEP_INPUT_NAME = "sliding_in_step"
SLIDING_KEY_CACHE_INPUT_NAME = "sliding_key_cache"
SLIDING_VALUE_CACHE_INPUT_NAME = "sliding_value_cache"
SLIDING_KEY_CACHE_OUTPUT_NAME = "new_sliding_k_cache"
SLIDING_VALUE_CACHE_OUTPUT_NAME = "new_sliding_v_cache"
PLE_EMBEDDINGS_INPUT_NAME = "ple_embeddings"

# Widest iOS static-shape query length (a prefill chunk); sizes the sliding-cache ring.
MAX_QUERY_LEN = max(BaseForCausalLMForiOS.IOS_STATIC_QUERY_LENS)


def sliding_ring_size(sliding_window: int, max_query_len: int) -> int:
    """Minimal sliding-cache ring depth S that holds a full prefill chunk's window
    union and never wraps a contiguous q_len write.

    S >= sliding_window + max_query_len - 1, rounded up to a multiple of
    max_query_len.
    """
    raw = sliding_window + max_query_len - 1
    return ((raw + max_query_len - 1) // max_query_len) * max_query_len


def _ple_quant_scale(ple_weight: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """``quantize_per_tensor``'s symmetric INT8 scale for ``ple_weight * embed_scale``,
    without an fp32 copy of the table: scaling by a positive constant commutes with
    ``max|x|``."""
    return torch.clamp(ple_weight.abs().max().float() * embed_scale, min=1e-6) / 127


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (for v_norm)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self._eps, dtype=x.dtype)
        square = x * x
        mean_square = square.mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)
        return x * inv_rms


class GeGLUMLP(nn.Module):
    """MLP with GELU-gated activation (GeGLU) using Conv2d for iOS."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _, dim = x.shape
        x = x.reshape(batch_size * query_len, dim, 1, 1)

        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        down_tensor = self.down_proj(up_tensor * gate_tensor)

        return down_tensor.reshape(batch_size, query_len, 1, dim)


class Attention(nn.Module):
    """Gemma4 iOS attention supporting both sliding and global modes."""

    def __init__(
        self,
        config,
        is_sliding: bool,
        is_kv_shared: bool,
        slot: int,
    ) -> None:
        super().__init__()
        self.is_sliding = is_sliding
        self.is_kv_shared = is_kv_shared
        # Slot in this layer's type-specific cache: the read+write slot for
        # storing layers, the source layer's slot for shared (read-only) layers.
        self.slot = slot

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if is_sliding else config.global_head_dim

        # Global layers run the chunked flash ``BlockedSDPA`` over the flat cache;
        # sliding layers use plain ``SDPA`` over the small ring.
        self.sdpa: nn.Module
        if not is_sliding:
            block_size = getattr(config, "kv_block_size", None) or DEFAULT_KV_BLOCK_SIZE
            self.sdpa = BlockedSDPA(head_dim=self.head_dim, scale=1.0, block_size=block_size)
        else:
            self.sdpa = SDPA(head_dim=self.head_dim, scale=1.0)

        self.q_proj = nn.Conv2d(dim, self.n_heads * self.head_dim, kernel_size=1, bias=False)
        if not is_kv_shared:
            self.k_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, dim, kernel_size=1, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not is_kv_shared:
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        write_offset: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        """Attention for one layer.

        ``write_offset`` is the single dynamic flat write offset into this layer's cache:
        the ring offset (``in_step % S``) for sliding layers, the absolute position
        (``in_step``) for global layers — so each region has ≤1 dynamic-offset slice.
        ``causal_mask`` is the flat ``(1, S, 1, q_len)`` sliding mask or the flat
        ``(1, ctx, 1, q_len)`` global mask; global layers walk it in block_size chunks
        inside ``BlockedSDPA``.
        """
        batch_size, query_len, _, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads
        head_dim = self.head_dim

        x_conv = x.transpose(-3, -1)
        query = self.q_proj(x_conv)

        # Reshape Q for norm + RoPE, keeping (B, S, n_heads, head_dim) so norm (over
        # head_dim) and RoPE broadcast over heads at dim 2 — no n_heads<->S transpose needed.
        query = query.transpose(-3, -1).reshape(batch_size, query_len, n_heads, head_dim)
        query = self.q_norm(query)
        query = apply_rope(query, rope_cos, rope_sin, head_axis=2)
        # Q back to BC1S for SDPA
        query = query.reshape(batch_size, query_len, 1, n_heads * head_dim).transpose(-3, -1)

        if self.is_kv_shared:
            assert cache is not None
            # Read the whole flat cache slot from the source layer (read-only):
            # (1, C, 1, seq) for both sliding (seq=S) and global (seq=ctx).
            key = cache.k_cache[self.slot]
            value = cache.v_cache[self.slot]
            output = self.sdpa(query, key, value, causal_mask)
        else:
            key = self.k_proj(x_conv)
            value = self.v_proj(x_conv)

            # Reshape K for norm + RoPE (B, S, n_kv_heads, head_dim)
            key = key.transpose(-3, -1).reshape(batch_size, query_len, n_kv_heads, head_dim)
            key = self.k_norm(key)
            key = apply_rope(key, rope_cos, rope_sin, head_axis=2)

            # K back to BC1S
            key = key.reshape(batch_size, query_len, 1, n_kv_heads * head_dim).transpose(-3, -1)

            # V: apply v-norm in (B, S, n_kv_heads, head_dim) then convert back to BC1S
            value = value.transpose(-3, -1).reshape(batch_size, query_len, n_kv_heads, head_dim)
            value = self.v_norm(value)
            value = value.reshape(batch_size, query_len, 1, n_kv_heads * head_dim).transpose(-3, -1)

            if cache is not None:
                # Flat cache write at a single dynamic offset (in_step % S for sliding,
                # in_step for global). S and ctx are multiples of every q_len, so the
                # contiguous [offset, offset+q_len) write never wraps or straddles — a
                # plain mutable_slice_update.
                key, value = cache.update_and_fetch(self.slot, write_offset, key, value, query_len)

            output = self.sdpa(query, key, value, causal_mask)

        output = self.o_proj(output)
        return output.transpose(-3, -1)


class TransformerBlock(nn.Module):
    """Gemma4 iOS decoder layer with PLE injection and layer scalar."""

    def __init__(
        self,
        config,
        is_sliding: bool,
        is_kv_shared: bool,
        slot: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.self_attn = Attention(
            config=config,
            is_sliding=is_sliding,
            is_kv_shared=is_kv_shared,
            slot=slot,
        )
        self.mlp = GeGLUMLP(hidden_size, intermediate_size)

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Conv2d(
                hidden_size, self.hidden_size_per_layer_input, kernel_size=1, bias=False
            )
            self.per_layer_projection = nn.Conv2d(
                self.hidden_size_per_layer_input, hidden_size, kernel_size=1, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        write_offset: torch.IntTensor,
        causal_mask: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        # Self-attention (pre+post norm)
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos,
            rope_sin,
            write_offset,
            causal_mask,
            cache,
        )
        h = x + self.post_attention_layernorm(r)

        # MLP (pre+post norm)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        h = h + self.post_feedforward_layernorm(r)

        # Per-Layer Embedding injection
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            batch_size, seq_len, _, ple_dim = per_layer_input.shape
            hidden_size = h.shape[-1]

            h_conv = h.reshape(batch_size * seq_len, hidden_size, 1, 1)
            gate = nn.functional.gelu(self.per_layer_input_gate(h_conv), approximate="tanh")

            pli_conv = per_layer_input.reshape(batch_size * seq_len, ple_dim, 1, 1)
            ple_out = self.per_layer_projection(gate * pli_conv)
            ple_out = ple_out.reshape(batch_size, seq_len, 1, hidden_size)
            h = h + self.post_per_layer_input_norm(ple_out)

        h = h * self.layer_scalar
        return h


def _compute_kv_layout(config):
    """Per-layer routing for the two-cache compacted KV design.

    Sliding and global layers use separate caches. Only the first
    ``num_hidden_layers - num_kv_shared_layers`` layers of each type ever write
    KV (the "storing" layers); shared layers read the last storing layer of
    their own type.

    Returns ``(sliding_storing, global_storing, layout)`` where the first two are
    the lists of storing layer indices per type, and ``layout`` is one
    ``(is_sliding, is_kv_shared, slot)`` tuple per layer. ``slot`` indexes into
    that type's storing list — the read+write slot for storing layers, the
    source layer's slot for shared (read-only) layers.
    """
    layer_types = config.layer_types
    num_layers = config.num_hidden_layers
    num_shared = config.num_kv_shared_layers
    first_shared = num_layers - num_shared if num_shared > 0 else num_layers

    sliding_storing = [i for i in range(first_shared) if layer_types[i] == "sliding_attention"]
    global_storing = [i for i in range(first_shared) if layer_types[i] == "full_attention"]

    layout = []
    for i in range(num_layers):
        is_sliding = layer_types[i] == "sliding_attention"
        is_kv_shared = i >= first_shared and num_shared > 0
        storing = sliding_storing if is_sliding else global_storing
        if is_kv_shared:
            source_idx = next(
                j for j in range(first_shared - 1, -1, -1) if layer_types[j] == layer_types[i]
            )
            slot = storing.index(source_idx)
        else:
            slot = storing.index(i)
        layout.append((is_sliding, is_kv_shared, slot))

    return sliding_storing, global_storing, layout


class Gemma4Model(nn.Module):
    """Gemma4 iOS text decoder model (without LM head)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        # Block size for the blocked global KV cache.
        self.kv_block_size = getattr(config, "kv_block_size", None) or DEFAULT_KV_BLOCK_SIZE

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            ple_dim = num_layers * config.hidden_size_per_layer_input
            self.per_layer_model_projection = nn.Conv2d(
                hidden_size, ple_dim, kernel_size=1, bias=False
            )
            self.per_layer_model_projection_scale = hidden_size**-0.5
            self.per_layer_projection_norm = RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
            self.per_layer_input_scale = 2.0**-0.5

        sliding_storing, global_storing, layout = _compute_kv_layout(config)
        self.n_sliding_storing = len(sliding_storing)
        self.n_global_storing = len(global_storing)

        layers = []
        for i in range(num_layers):
            is_sliding, is_kv_shared, slot = layout[i]

            if config.use_double_wide_mlp and is_kv_shared:
                intermediate_size = config.intermediate_size * 2
            else:
                intermediate_size = config.intermediate_size

            layers.append(
                TransformerBlock(
                    config,
                    is_sliding=is_sliding,
                    is_kv_shared=is_kv_shared,
                    slot=slot,
                    intermediate_size=intermediate_size,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # RoPE cos/sin are precomputed in the runner and passed in as graph inputs: a 131k
        # position index overflows a 16-bit position input, and a 32-bit one feeding the
        # RoPE gather fails to compile.

    def _compute_per_layer_inputs(
        self,
        ple_embeddings: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PLE: pre-gathered PLE embedding + projected main embedding."""
        num_layers = self.config.num_hidden_layers
        ple_dim = self.config.hidden_size_per_layer_input
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        hidden_size = inputs_embeds.shape[-1]

        per_layer_emb = ple_embeddings.reshape(batch_size, seq_len, num_layers, ple_dim)

        embeds_conv = inputs_embeds.reshape(batch_size * seq_len, hidden_size, 1, 1)
        per_layer_proj = self.per_layer_model_projection(embeds_conv)
        per_layer_proj = per_layer_proj.reshape(batch_size, seq_len, num_layers, ple_dim)
        per_layer_proj = per_layer_proj * torch.tensor(
            self.per_layer_model_projection_scale, dtype=per_layer_proj.dtype
        )
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)

        return (per_layer_proj + per_layer_emb) * torch.tensor(
            self.per_layer_input_scale, dtype=per_layer_proj.dtype
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        sliding_in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_causal_mask: torch.Tensor,
        global_cache: KVCacheHandler | None = None,
        sliding_cache: KVCacheHandler | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = token_embeddings

        layer_types = self.config.layer_types

        # RoPE cos/sin arrive precomputed from the runner as the combined sliding+global
        # table rows ``(1, q_len, sliding_hd + global_hd)``. Slice the sliding sub-range
        # ``[:head_dim]`` and the global sub-range ``[head_dim:]``.
        head_dim = self.config.head_dim
        sliding_cos = rope_cos[..., :head_dim]
        sliding_sin = rope_sin[..., :head_dim]
        global_cos = rope_cos[..., head_dim:]
        global_sin = rope_sin[..., head_dim:]

        if ple_embeddings is None:
            raise ValueError(
                "ple_embeddings is required: Gemma 4 always takes per-layer embeddings"
            )
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self._compute_per_layer_inputs(ple_embeddings, h)

        for i, layer in enumerate(self.layers):
            per_layer_input = None
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, :, i : i + 1, :]

            if layer_types[i] == "sliding_attention":
                rope_cos, rope_sin = sliding_cos, sliding_sin
                cache, attn_mask = sliding_cache, sliding_causal_mask
                write_offset = sliding_in_step
            else:
                rope_cos, rope_sin = global_cos, global_sin
                cache, attn_mask = global_cache, causal_mask
                write_offset = in_step

            h = layer(
                h,
                rope_cos,
                rope_sin,
                write_offset,
                attn_mask,
                per_layer_input=per_layer_input,
                cache=cache,
            )

        return self.norm(h)


class Gemma4Extend(nn.Module):
    """Gemma4 iOS extend module: transformer + lm_head with merged KV cache."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Gemma4Model(config)
        self.embed_scale = config.hidden_size**0.5
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.ple_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.ple_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.prefill_mode = False

        # E2B/E4B tie word embeddings: the output projection reuses the embedding table.
        if not config.tie_word_embeddings:
            raise NotImplementedError("Gemma 4 iOS supports tied word embeddings only")

        # Two compacted KV caches, both FLAT (single dynamic write offset, so each
        # attention region has ≤1 dynamic-offset slice):
        #  - global: full-context flat cache `[n_global_storing, 1, C_g, 1, ctx]`; the
        #    chunked flash `BlockedSDPA` walks the flat slot in block_size chunks.
        #  - sliding: small ring (static seq dim S), n_kv * head_dim channels.
        n_kv_heads = config.num_key_value_heads
        global_channels = n_kv_heads * config.global_head_dim
        sliding_channels = n_kv_heads * config.head_dim
        self.global_cache = KVCacheHandler(self.model.n_global_storing, global_channels)
        self.sliding_cache = KVCacheHandler(self.model.n_sliding_storing, sliding_channels)

    def forward(
        self,
        transformer_input: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        sliding_in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        sliding_key_cache: torch.Tensor,
        sliding_value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.global_cache.register_kv_cache(key_cache, value_cache)
        self.sliding_cache.register_kv_cache(sliding_key_cache, sliding_value_cache)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape

        # Scale embeddings by sqrt(hidden_size)
        transformer_input = transformer_input * torch.tensor(
            self.embed_scale, dtype=transformer_input.dtype
        )

        # Dequantize PLE if INT8
        if ple_embeddings is not None and ple_embeddings.dtype == torch.int8:
            ple_embeddings = dequantize_per_tensor(
                ple_embeddings, self.ple_scale, self.ple_zero_point, transformer_input.dtype
            )

        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            sliding_in_step,
            causal_mask,
            sliding_causal_mask,
            global_cache=self.global_cache,
            sliding_cache=self.sliding_cache,
            ple_embeddings=ple_embeddings,
        )

        if self.prefill_mode:
            # Prefill output is unused (the runner only wants the cache writes); return a
            # width-1 value that still depends on both global cache writes so they aren't
            # dead-code-eliminated. Use rank-preserving width-1 slices on EVERY dim (shape
            # (1,1,1,1,1)), NOT integer indexing: integer indexing squeezes the leading dims
            # and lowers to a rank-1 `(ctx,)` reshape whose ctx dim exceeds the accelerator's
            # per-dim size limit at large ctx (e.g. 131072). Slicing keeps rank 5.
            return (
                self.global_cache.k_cache[:1, :1, :1, :1, :1]
                + self.global_cache.v_cache[:1, :1, :1, :1, :1]
            )

        if embedding_table is None:
            raise ValueError("embedding_table is required for the tied output projection")

        table = embedding_table
        if table.dtype == torch.int8:
            table = dequantize_per_tensor(table, self.emb_scale, self.emb_zero_point, out.dtype)

        table = table.reshape(table.shape[1], table.shape[0], table.shape[2])
        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)

        return (table @ out).transpose(-1, -2)


class Gemma4ForCausalLMForiOS(BaseForCausalLMForiOS):
    """Gemma4 iOS text-only CausalLM (extracted from multimodal HF model)."""

    _HF_MODEL_CLASS = Gemma4ForConditionalGeneration

    #: Sliding-attention window of the supported checkpoints (E2B, E4B).
    SLIDING_WINDOW = 512

    #: Sliding-cache ring depth. A class constant because export_hardware_constraints
    #: gets no config; build_reference_inputs checks the config agrees.
    SLIDING_RING_SIZE = sliding_ring_size(SLIDING_WINDOW, MAX_QUERY_LEN)

    def _init_model(self, config) -> None:
        self.extend = Gemma4Extend(config)

    # ------------------------------------------------------------------
    # Export contract
    #
    # Gemma 4 keeps the base's four entrypoints but widens the transformer's I/O
    # with precomputed RoPE rows, a sliding-window mask/step/cache pair, and
    # `ple_embeddings`. Each hook describes one ladder rung; `models/gemma4/export.py`
    # calls them once per (context bucket, query length).
    # ------------------------------------------------------------------

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (
                TOKEN_IDS_INPUT_NAME,
                EMBEDDING_TABLE_INPUT_NAME,
            ),
            EXTEND_FUNCTION_NAME: (
                TRANSFORMER_INPUT_NAME,
                ROPE_COS_INPUT_NAME,
                ROPE_SIN_INPUT_NAME,
                IN_STEP_INPUT_NAME,
                SLIDING_IN_STEP_INPUT_NAME,
                CAUSAL_MASK_INPUT_NAME,
                SLIDING_CAUSAL_MASK_INPUT_NAME,
                EMBEDDING_TABLE_INPUT_NAME,
                PLE_EMBEDDINGS_INPUT_NAME,
            ),
        }

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (),
            EXTEND_FUNCTION_NAME: (
                KEY_CACHE_INPUT_NAME,
                VALUE_CACHE_INPUT_NAME,
                SLIDING_KEY_CACHE_INPUT_NAME,
                SLIDING_VALUE_CACHE_INPUT_NAME,
            ),
        }

    @classmethod
    @override
    def export_state_output_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (),
            EXTEND_FUNCTION_NAME: (
                KEY_CACHE_OUTPUT_NAME,
                VALUE_CACHE_OUTPUT_NAME,
                SLIDING_KEY_CACHE_OUTPUT_NAME,
                SLIDING_VALUE_CACHE_OUTPUT_NAME,
            ),
        }

    @override
    def build_reference_inputs(
        self, config, target_dtype: torch.dtype, spec: TraceSpec
    ) -> dict[str, dict[str, Any]]:
        """Reference inputs for one ladder rung, traced at fully static shapes.

        ``BlockedSDPA`` unrolls its block loop, so the op count depends on the context
        length: each (context bucket, query length) pair is its own program, with the
        caches at ``spec.cache_seq_len`` and ``spec.query_len`` query positions.
        """
        if self.disable_embedding_quantization:
            raise ValueError(
                "The iOS graph takes INT8 ple_embeddings; disable_embedding_quantization "
                "is for eager tests"
            )
        if config.sliding_window != self.SLIDING_WINDOW:
            raise ValueError(
                f"sliding_window={config.sliding_window}, but the export assumes "
                f"{self.SLIDING_WINDOW} (see SLIDING_RING_SIZE)."
            )
        ctx, query_len = spec.cache_seq_len, spec.query_len
        head_dim, global_head_dim = config.head_dim, config.global_head_dim
        n_kv = config.num_key_value_heads
        ring = self.SLIDING_RING_SIZE
        rope_width = head_dim + global_head_dim

        embedding_table = self.load_embeddings.embedding_table
        input_ids = torch.randint(1, config.vocab_size, (1, query_len), dtype=torch.int32)
        # One graph's reference input is produced by running another.
        transformer_input = self.gather_embeddings(input_ids, embedding_table)

        key_cache = torch.zeros(
            self.extend.model.n_global_storing,
            1,
            n_kv * global_head_dim,
            1,
            ctx,
            dtype=torch.float16,
        )
        sliding_key_cache = torch.zeros(
            self.extend.model.n_sliding_storing,
            1,
            n_kv * head_dim,
            1,
            ring,
            dtype=torch.float16,
        )
        ple_width = config.num_hidden_layers * config.hidden_size_per_layer_input

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                "input_ids": input_ids,
                "embedding_table": embedding_table,
            },
            # Exact signature order of Gemma4Extend.forward.
            EXTEND_FUNCTION_NAME: {
                TRANSFORMER_INPUT_NAME: transformer_input,
                ROPE_COS_INPUT_NAME: torch.zeros(1, query_len, rope_width, dtype=torch.float16),
                ROPE_SIN_INPUT_NAME: torch.zeros(1, query_len, rope_width, dtype=torch.float16),
                IN_STEP_INPUT_NAME: torch.zeros((1,), dtype=torch.int32),
                SLIDING_IN_STEP_INPUT_NAME: torch.zeros((1,), dtype=torch.int32),
                CAUSAL_MASK_INPUT_NAME: torch.zeros(1, ctx, 1, query_len, dtype=torch.float16),
                SLIDING_CAUSAL_MASK_INPUT_NAME: torch.zeros(
                    1, ring, 1, query_len, dtype=torch.float16
                ),
                KEY_CACHE_INPUT_NAME: key_cache,
                VALUE_CACHE_INPUT_NAME: key_cache.clone(),
                SLIDING_KEY_CACHE_INPUT_NAME: sliding_key_cache,
                SLIDING_VALUE_CACHE_INPUT_NAME: sliding_key_cache.clone(),
                EMBEDDING_TABLE_INPUT_NAME: embedding_table,
                PLE_EMBEDDINGS_INPUT_NAME: torch.randint(
                    -128, 127, (1, query_len, 1, ple_width), dtype=torch.int8
                ),
            },
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        # The transformer rungs are fully static (see build_reference_inputs); the gather
        # is dynamic in the query length and specialized, as for every iOS model.
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: None,
            GATHER_EMBEDDINGS_FUNCTION_NAME: super().build_dynamic_shapes(config, spec)[
                GATHER_EMBEDDINGS_FUNCTION_NAME
            ],
            EXTEND_FUNCTION_NAME: None,
        }

    @classmethod
    @override
    def export_static_shape_configs(cls, config, max_context_length: int) -> dict:
        # Only the gather is specialized (over IOS_STATIC_QUERY_LENS, as for every iOS
        # model); the transformer rungs are traced at fully static shapes, one per
        # context bucket (passed here as max_context_length).
        base = super().export_static_shape_configs(config, max_context_length)
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: base[GATHER_EMBEDDINGS_FUNCTION_NAME],
            EXTEND_FUNCTION_NAME: {},
        }

    @classmethod
    @override
    def export_hardware_constraints(
        cls, max_context_length: int
    ) -> dict[str, dict[str, HardwareConstraints]]:
        """The base constraints for one rung, whose global cache is
        ``max_context_length`` (the rung's context bucket) long, with the sliding
        ring's caches aligned to the ring depth instead."""
        constraints = super().export_hardware_constraints(max_context_length)
        sliding_cache = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, cls.KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, cls.KV_CACHE_INTERLEAVE_FACTOR * cls.SLIDING_RING_SIZE, 1],
        )
        for name in (
            SLIDING_KEY_CACHE_INPUT_NAME,
            SLIDING_VALUE_CACHE_INPUT_NAME,
            SLIDING_KEY_CACHE_OUTPUT_NAME,
            SLIDING_VALUE_CACHE_OUTPUT_NAME,
        ):
            constraints[EXTEND_FUNCTION_NAME][name] = sliding_cache
        return constraints

    def forward(
        self,
        input_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        sliding_in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        sliding_causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        sliding_key_cache: torch.Tensor,
        sliding_value_cache: torch.Tensor,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            rope_cos,
            rope_sin,
            in_step,
            sliding_in_step,
            causal_mask,
            sliding_causal_mask,
            key_cache,
            value_cache,
            sliding_key_cache,
            sliding_value_cache,
            self.load_embeddings.embedding_table,
            ple_embeddings,
        )

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        """Extract text config from the multimodal Gemma4Config."""
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            if hasattr(text_config, "layer_types") and text_config.layer_types:
                text_config.layer_types = text_config.layer_types[:num_layers]
            orig_num_layers = text_config.num_hidden_layers
            orig_first_shared = orig_num_layers - text_config.num_kv_shared_layers
            new_shared = max(0, num_layers - orig_first_shared)
            text_config.num_kv_shared_layers = new_shared
            text_config.num_hidden_layers = num_layers

        return text_config

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Transform HF state dict for iOS Gemma4 model."""
        # The checkpoint is multimodal; keep only the text decoder, as `model.*`.
        prefix = "model.language_model."
        if any(k.startswith(prefix) for k in state_dict):
            text = {
                "model." + k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            state_dict.clear()
            state_dict.update(text)

        config = self.config
        num_layers = config.num_hidden_layers
        first_kv_shared_idx = num_layers - config.num_kv_shared_layers
        ple_dim = config.hidden_size_per_layer_input

        # Extract PLE embedding weight (externalized from graph)
        if ple_dim == 0:
            raise NotImplementedError(
                "Gemma 4 iOS supports the per-layer-embedding (E2B/E4B) variants only"
            )
        ple_key = "model.embed_tokens_per_layer.weight"
        if ple_key not in state_dict:
            raise KeyError(f"checkpoint has no per-layer embedding table ({ple_key})")
        ple_weight = state_dict.pop(ple_key)
        expected_dim = num_layers * ple_dim
        if ple_weight.shape[1] > expected_dim:
            ple_weight = ple_weight[:, :expected_dim].contiguous()
        self._ple_weight = ple_weight

        # PLE quant params, used to dequantize the INT8 ``ple_embeddings`` graph input.
        # With embedding quantization disabled (eager parity tests only; the export
        # rejects it) the scale is identity, so the tests can feed fp rows.
        if not self.disable_embedding_quantization:
            ple_scale = _ple_quant_scale(ple_weight, ple_dim**0.5)
        else:
            ple_scale = torch.tensor(1.0)
        state_dict["extend.ple_scale"] = ple_scale.to(ple_weight.dtype)
        state_dict["extend.ple_zero_point"] = torch.tensor(0, dtype=torch.int8)

        # Truncate per_layer_model_projection if needed
        proj_key = "model.per_layer_model_projection.weight"
        if proj_key in state_dict:
            full_proj = state_dict[proj_key]
            expected_out = num_layers * ple_dim
            if full_proj.shape[0] > expected_out:
                state_dict[proj_key] = full_proj[:expected_out, :]

        # Reshape weights for Conv2d and handle shared layers
        for i in range(num_layers):
            is_kv_shared = i >= first_kv_shared_idx and config.num_kv_shared_layers > 0

            for proj in ["q_proj", "o_proj"]:
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            if is_kv_shared:
                for proj in ["k_proj", "v_proj"]:
                    key = f"model.layers.{i}.self_attn.{proj}.weight"
                    state_dict.pop(key, None)
                state_dict.pop(f"model.layers.{i}.self_attn.k_norm.weight", None)
            else:
                for proj in ["k_proj", "v_proj"]:
                    weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                    if weight_key in state_dict:
                        state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            for proj in ["gate_proj", "up_proj", "down_proj"]:
                weight_key = f"model.layers.{i}.mlp.{proj}.weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            if ple_dim > 0:
                for proj in ["per_layer_input_gate", "per_layer_projection"]:
                    weight_key = f"model.layers.{i}.{proj}.weight"
                    if weight_key in state_dict:
                        state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

        if ple_dim > 0:
            proj_key = "model.per_layer_model_projection.weight"
            if proj_key in state_dict:
                state_dict[proj_key] = state_dict[proj_key].unsqueeze(-1).unsqueeze(-1)

        # Wrap model weights under extend.model.* prefix
        new_state_dict = {}
        keys_to_pop = set()
        for k in state_dict:
            if k.startswith("model.") and "embed_tokens" not in k:
                new_state_dict[f"extend.{k}"] = state_dict[k]
                keys_to_pop.add(k)
        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        # Handle embeddings: quantize and set up for iOS layout
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

        state_dict.pop("lm_head.weight", None)

    def dump_ple_embedding(self, output_path: str, model_name: str) -> str:
        """Dump the externalized PLE embedding table as a quantized INT8 safetensors file."""
        import os

        if self.disable_embedding_quantization:
            raise ValueError(
                "The PLE sidecar is INT8; disable_embedding_quantization is for eager tests"
            )
        ple_weight = self._ple_weight
        ple_dim = self.config.hidden_size_per_layer_input
        embed_scale = str(ple_dim**0.5)
        ple_scale = _ple_quant_scale(ple_weight, float(embed_scale))

        # quantize_per_tensor's rounding, a block of rows at a time so the table never
        # has a full fp32 copy.
        ple_q = torch.empty(ple_weight.shape, dtype=torch.int8)
        for start in range(0, ple_weight.shape[0], 8192):
            rows = ple_weight[start : start + 8192].float() * float(embed_scale)
            ple_q[start : start + 8192] = torch.clamp(torch.round(rows / ple_scale), -128, 127).to(
                torch.int8
            )

        os.makedirs(output_path, exist_ok=True)
        ple_path = os.path.join(output_path, f"{model_name}_ple.safetensors")

        save_file(
            {"embed_tokens_per_layer": ple_q},
            ple_path,
            metadata={
                "embed_scale": embed_scale,
                "ple_scale": str(float(ple_scale)),
                "ple_zero_point": "0",
                "dtype": "SI8",
            },
        )
        return ple_path
