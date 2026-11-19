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

from typing import Any, cast

# Registers `torch.ops.coreai.lut_to_dense` and `torch.ops.coreai.constexpr_blockwise_shift_scale`,
# used by the native quantized weight modules.
import coreai_torch._compression.custom_layers  # noqa: F401
import torch
import torch.nn as nn
from coreai.authoring.types import AllocationType, HardwareConstraints
from safetensors.torch import save_file
from transformers import AutoConfig, Gemma4ForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
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
from coreai_models.hub import BackendName, resolve_local
from coreai_models.models.base import (
    BaseForCausalLMForiOS,
    TraceSpec,
    _is_layer_key_beyond,
    move_model_to_disk,
)
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import (
    NATIVE_NBITS,
    dequantize_native,
    dequantize_native_dense,
    dequantize_per_tensor,
    quantize_per_tensor,
    unpack_signed_subbyte,
    unpack_unsigned_subbyte,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import apply_rope
from coreai_models.primitives.ios.sdpa import SDPA, BlockedSDPA

# The "-qat-mobile-transformers" checkpoints ship pre-quantized (Google's own QAT mixed
# INT2/4/8 recipe): `from_pretrained` replaces `nn.Linear`/`nn.Embedding` with these wrapper
# modules and loads packed weights + per-channel scales into them. Optional for the same
# reason as `HFGemma4ForCausalLM` above -- only consumed when a quantized checkpoint is
# actually loaded.
try:
    from transformers.integrations.gemma_quant import (  # type: ignore[import-not-found]
        QuantizedEmbedding,
        QuantizedLinear,
    )
except ImportError:  # pragma: no cover - depends on the installed transformers
    QuantizedEmbedding = None  # type: ignore[assignment,misc]
    QuantizedLinear = None  # type: ignore[assignment,misc]

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
# Companion per-row scale of the native-quantized (QAT) embedding table: a second output of
# `load_embeddings` and a third input of `gather_embeddings`, present for those checkpoints only.
EMBEDDING_SCALE_INPUT_NAME = "embedding_scale"
EMBEDDING_SCALE_OUTPUT_NAME = "embedding_scale"

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


class NativeQuantConv1x1(nn.Module):
    """1x1-Conv2d-shaped weight held in CoreAI's native quantized format.

    Drop-in replacement for a plain ``nn.Conv2d(..., kernel_size=1, bias=False)`` weight,
    used in place of it for quantization-aware-trained (QAT) checkpoints.
    ``quantized_data`` holds the checkpoint's literal signed per-element integers and
    ``scale`` its literal per-output-channel scale -- dequantized at forward time via
    ``dequantize_native``, which the converter lowers straight to CoreAI's native
    ``coreai.blockwise_shift_scale``, carrying those exact values into the exported graph
    with no dequantize/requantize round-trip.
    """

    def __init__(self, in_channels: int, out_channels: int, nbits: int) -> None:
        super().__init__()
        self.nbits = nbits
        self.register_buffer(
            "quantized_data", torch.zeros(out_channels, in_channels, 1, 1, dtype=torch.int8)
        )
        self.register_buffer("scale", torch.ones(out_channels, 1, 1, 1, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_native(self.quantized_data, self.scale, self.nbits, x.dtype)
        return nn.functional.conv2d(x, weight)


class NativeLutQuantConv1x1(nn.Module):
    """1x1-Conv2d-shaped weight held via CoreAI's native LUT (palettization) format.

    ``NativeQuantConv1x1``'s affine ``blockwise_shift_scale`` has no execution path on the
    static-shape engine below INT4. Google's own QAT recipe ships some Gemma4 E2B MLP
    projections at genuine 2 bits though -- the double-wide, shared-KV layers (see
    ``TransformerBlock``) -- and rather than clamp those to INT4 like the affine path
    would (losing density Google already trained for), this decodes through
    ``coreai.lut_to_dense`` with a *canonical* (not clustered) LUT that reproduces the
    checkpoint's exact uniformly-spaced levels bit-for-bit -- e.g. ``[-2, -1, 0, 1]`` for
    2 bits -- then applies the real per-output-channel scale via the same
    ``constexpr_blockwise_shift_scale`` op the affine path already uses (validated
    running on the static-shape engine there). ``indices`` gets promoted to a genuine
    packed sub-byte constant the same way affine weights do (see ``coreai_torch``'s
    ``inject_subbyte_tensors``), so this is bit-exact *and* genuinely dense on disk -- not a
    re-quantization tradeoff.
    """

    def __init__(self, in_channels: int, out_channels: int, nbits: int) -> None:
        super().__init__()
        self.nbits = nbits
        self.register_buffer(
            "indices", torch.zeros(out_channels, in_channels, 1, 1, dtype=torch.uint8)
        )
        self.register_buffer("scale", torch.ones(out_channels, 1, 1, 1, dtype=torch.float16))
        # Canonical identity levels (not learned, not checkpoint data) -- persistent=False
        # keeps it out of the state dict entirely; every instance gets the same one. Built
        # under an explicit CPU device: `from_hf` constructs this module under
        # `model_device="meta"`, and a non-persistent buffer is never rescued by
        # `load_state_dict` afterward the way `indices`/`scale` are, so without this it
        # would stay a meta (no real data) tensor forever -- the same fix
        # `Gemma4CombinedRoPE` already applies to its cos/sin cache for the same reason.
        with torch.device("cpu"):
            bias = 1 << (nbits - 1)
            levels = torch.arange(2**nbits, dtype=torch.int8) - bias
            lut = levels.reshape(1, 1, 1, 1, 2**nbits, 1)
        self.register_buffer("lut", lut, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = torch.ops.coreai.lut_to_dense(self.indices, self.lut, 0)
        weight = dequantize_native_dense(dense, self.scale, x.dtype)
        return nn.functional.conv2d(x, weight)


class NativeQuantLinear(nn.Module):
    """``nn.Linear``-shaped weight held in CoreAI's native quantized format.

    See ``NativeQuantConv1x1``; used for ``lm_head``, which is a plain matmul rather than
    a 1x1 conv. ``lm_head``'s output dimension is the vocab size (262144), which exceeds
    the static-shape engine's per-op tensor size limit for a single dequantize+matmul: left
    whole, the untied lm_head's dequantize and the matmul consuming it fall back off the
    engine, the only ops of the decode graph to do so. ``block_size`` chunks the
    dequantize+matmul over output blocks so each stays under that limit, unrolled at trace
    time -- the same fix ``BlockedSDPA`` (see its docstring) already applies to the
    attention side for the same kind of per-dimension ceiling.

    Each block gets its own ``quantized_data``/``scale`` buffer pair (a
    ``NativeQuantLinearBlock``) rather than slicing one big buffer per block: the
    automatic promotion to a genuine packed sub-byte constant (see
    ``coreai_torch``'s ``inject_subbyte_tensors``) only fires when the tensor feeding
    ``constexpr_blockwise_shift_scale`` is a direct, unsliced placeholder -- slicing a
    single shared buffer at forward time defeats that, silently widening every block
    back out to a full-byte-per-element container.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nbits: int,
        block_size: int | None = None,
    ) -> None:
        super().__init__()
        self.nbits = nbits
        self.out_features = out_features
        self.block_size = block_size
        if block_size is None or out_features <= block_size:
            self.register_buffer(
                "quantized_data", torch.zeros(out_features, in_features, dtype=torch.int8)
            )
            self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float16))
            self.blocks: nn.ModuleList | None = None
        else:
            self.quantized_data = None
            self.scale = None
            self.blocks = nn.ModuleList(
                NativeQuantLinearBlock(in_features, min(block_size, out_features - lo), nbits)
                for lo in range(0, out_features, block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blocks is None:
            weight = dequantize_native(self.quantized_data, self.scale, self.nbits, x.dtype)
            return nn.functional.linear(x, weight)
        return torch.cat([block(x) for block in self.blocks], dim=-1)


class NativeQuantLinearBlock(nn.Module):
    """One output-channel block of a blocked ``NativeQuantLinear`` (see there for why)."""

    def __init__(self, in_features: int, out_features: int, nbits: int) -> None:
        super().__init__()
        self.nbits = nbits
        self.register_buffer(
            "quantized_data", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_native(self.quantized_data, self.scale, self.nbits, x.dtype)
        return nn.functional.linear(x, weight)


def _make_proj_conv(config, in_channels: int, out_channels: int, nbits: int) -> nn.Module:
    """Build a 1x1-conv projection: plain float, or CoreAI-native quantized for QAT
    checkpoints (see ``config.native_quantized``, set in ``Gemma4ForCausalLMForiOS.from_hf``).

    ``nbits < 4`` routes through the LUT/palettization path
    (``NativeLutQuantConv1x1``) instead of the affine one (``NativeQuantConv1x1``), since
    CoreAI's affine ``blockwise_shift_scale`` has no static-shape-engine execution path
    below INT4 -- see ``NativeLutQuantConv1x1``'s docstring. Only Gemma4 E2B's shared-KV MLP layers
    actually ship below 4 bits (see ``_native_mlp_nbits``); every other caller passes 4
    or 8 and gets the existing affine path unchanged.

    A native projection is 2, 4 or 8 bits (``NATIVE_NBITS``), the widths the QAT checkpoints
    ship; any other width is refused here, as the module is built, rather than when its
    weights are unpacked or traced.
    """
    if getattr(config, "native_quantized", False):
        if nbits not in NATIVE_NBITS:
            raise ValueError(
                f"A native-quantized projection is one of {NATIVE_NBITS} bits, got {nbits}."
            )
        if nbits < 4:
            return NativeLutQuantConv1x1(in_channels, out_channels, nbits)
        return NativeQuantConv1x1(in_channels, out_channels, nbits)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def _native_mlp_nbits(config, layer_idx: int) -> int:
    """Per-layer MLP bit-width for QAT checkpoints.

    Gemma4 E2B's checkpoint quantizes `mlp.*` to genuine 2 bits on its shared-KV,
    double-wide layers (see ``Gemma4ForCausalLMForiOS.from_hf``, which builds
    ``config.native_mlp_nbits`` from the checkpoint's actual per-module ``num_bits``); 4
    bits everywhere else, including for non-native (standard) checkpoints, where the
    plan is never set.
    """
    plan = getattr(config, "native_mlp_nbits", None)
    return 4 if plan is None else plan[layer_idx]


class GeGLUMLP(nn.Module):
    """MLP with GELU-gated activation (GeGLU) using Conv2d for iOS."""

    def __init__(self, config, dim: int, hidden_dim: int, layer_idx: int) -> None:
        super().__init__()
        nbits = _native_mlp_nbits(config, layer_idx)
        self.gate_proj = _make_proj_conv(config, dim, hidden_dim, nbits=nbits)
        self.up_proj = _make_proj_conv(config, dim, hidden_dim, nbits=nbits)
        self.down_proj = _make_proj_conv(config, hidden_dim, dim, nbits=nbits)

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

        self.q_proj = _make_proj_conv(config, dim, self.n_heads * self.head_dim, nbits=4)
        if not is_kv_shared:
            self.k_proj = _make_proj_conv(config, dim, self.n_kv_heads * self.head_dim, nbits=4)
            self.v_proj = _make_proj_conv(config, dim, self.n_kv_heads * self.head_dim, nbits=4)
        self.o_proj = _make_proj_conv(config, self.n_heads * self.head_dim, dim, nbits=4)

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
        layer_idx: int,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.self_attn = Attention(
            config=config,
            is_sliding=is_sliding,
            is_kv_shared=is_kv_shared,
            slot=slot,
        )
        self.mlp = GeGLUMLP(config, hidden_size, intermediate_size, layer_idx)

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            # Google's QAT recipe quantizes these two to 8 bits (vs. 4 for attn/mlp).
            self.per_layer_input_gate = _make_proj_conv(
                config, hidden_size, self.hidden_size_per_layer_input, nbits=8
            )
            self.per_layer_projection = _make_proj_conv(
                config, self.hidden_size_per_layer_input, hidden_size, nbits=8
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
                    layer_idx=i,
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
        # When set, `forward` returns (logits, final_hidden_state) so a speculative
        # (MTP/EAGLE) drafter can consume the target's post-norm hidden state. Off by
        # default so the standard single-output export is unchanged.
        self.emit_hidden_states = False

        # Tied embeddings reuse the embedding table for the projection, so there is no
        # separate `lm_head` module to hold. Only the native-quantized (QAT) checkpoints are
        # untied; `from_hf` refuses a tied one.
        self.lm_head: nn.Module | None = None
        if not config.tie_word_embeddings:
            if not getattr(config, "native_quantized", False):
                raise NotImplementedError(
                    "Gemma 4 iOS supports an untied lm_head for native-quantized checkpoints only"
                )
            # Google's QAT recipe quantizes `lm_head` to 2 bits, but CoreAI's native
            # affine blockwise_shift_scale has no static-shape-engine execution path
            # below 4-bit (only LUT-based palettization goes narrower). The compiler
            # silently promotes a declared INT2 affine tensor to INT4 regardless, so
            # declaring INT4 here directly keeps the exported asset's size honest
            # without losing any precision (the values only ever span 4 levels either
            # way).
            self.lm_head = NativeQuantLinear(
                config.hidden_size,
                config.vocab_size,
                nbits=4,
                block_size=DEFAULT_KV_BLOCK_SIZE,
            )

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

        # Post-norm final hidden state (batch, seq, hidden). Captured before the output
        # projection reshapes `out`, so a speculative drafter can consume it.
        final_hidden = out.reshape(batch_size, seq_len, hidden_dim)

        if self.lm_head is not None:
            # Untied checkpoints carry their own output projection.
            logits = self.lm_head(out.transpose(-2, -3))
        else:
            if embedding_table is None:
                raise ValueError("embedding_table is required for the tied output projection")

            table = embedding_table
            if table.dtype == torch.int8:
                table = dequantize_per_tensor(table, self.emb_scale, self.emb_zero_point, out.dtype)

            table = table.reshape(table.shape[1], table.shape[0], table.shape[2])
            proj = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
            logits = (table @ proj).transpose(-1, -2)

        if self.emit_hidden_states:
            return logits, final_hidden
        return logits


def is_native_quantized_config(hf_config) -> bool:
    """Whether a Hugging Face config declares Google's own quantization (the QAT
    ``-qat-mobile-transformers`` checkpoints declare ``quant_method: gemma``)."""
    quantization = getattr(hf_config, "quantization_config", None)
    if isinstance(quantization, dict):
        return quantization.get("quant_method") == "gemma"
    return getattr(quantization, "quant_method", None) == "gemma"


def _is_quantized_hf_model(hf_model: nn.Module) -> bool:
    """Whether ``hf_model``'s text decoder was loaded through HF's Gemma QAT quantizer.

    Only the modules this loader reads count (the text decoder and ``lm_head``): a checkpoint
    that quantizes only its vision or audio tower has a float text decoder, which the native
    modules cannot load.
    """
    if QuantizedLinear is None:
        return False
    return any(
        isinstance(module, (QuantizedLinear, QuantizedEmbedding))
        for name, module in hf_model.named_modules()
        if name.startswith("model.language_model.") or name == "lm_head"
    )


def _native_module_name(hf_name: str, hf_prefix: str) -> str:
    """This model's name for the HF text-decoder module ``hf_name``."""
    if hf_name.startswith(hf_prefix):
        return "extend.model." + hf_name[len(hf_prefix) :]
    return f"extend.{hf_name}"


def _check_native_nbits(checkpoint_nbits: dict[str, int]) -> None:
    """Fail if a checkpoint module is quantized to a width other than 2, 4 or 8 bits.

    Those are the widths the QAT checkpoints ship and the native modules, the sub-byte
    unpacking and ``dequantize_native`` handle (``NATIVE_NBITS``). Checked before any packed
    weight is unpacked, so the error names the module. ``_check_native_widths`` then checks
    each supported width against the width the module tree declares for it.
    """
    unsupported = {
        name: nbits for name, nbits in checkpoint_nbits.items() if nbits not in NATIVE_NBITS
    }
    if unsupported:
        listed = ", ".join(f"{name} ({nbits} bits)" for name, nbits in sorted(unsupported.items()))
        raise NotImplementedError(
            f"Native-quantized modules are one of {NATIVE_NBITS} bits; the checkpoint quantizes "
            f"{listed}."
        )


def _check_native_widths(model: nn.Module, checkpoint_nbits: dict[str, int]) -> None:
    """Fail if a native-quantized module declares fewer bits than its checkpoint module.

    The projection widths are fixed by the module tree (4 bits for attention, 8 for the PLE
    projections, the per-layer plan for the MLP), so a checkpoint quantized differently would
    otherwise load values that do not fit the declared width. A wider declaration is fine: the
    ``lm_head`` is declared 4 bits for a 2-bit checkpoint (see ``Gemma4Extend``). The widths
    themselves are limited to the supported ones first, by ``_check_native_nbits``.
    """
    modules = dict(model.named_modules())
    for name, nbits in checkpoint_nbits.items():
        declared = getattr(modules.get(name), "nbits", None)
        if declared is not None and nbits > declared:
            raise NotImplementedError(
                f"{name} is quantized to {nbits} bits in the checkpoint, but the export "
                f"declares it as {declared} bits."
            )


def _native_ple_table(
    ple_module: nn.Module, config, truncate: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The PLE table's packed rows, per-block scales and bit width, as the sidecar stores them.

    2- and 4-bit rows are packed as unsigned fields holding the value plus ``2^(nbits - 1)``,
    which is what the runner unpacks. The checkpoint stores 8-bit rows as plain signed bytes,
    so those are moved to the same offset form. ``truncate`` (``--num-layers``) cuts each row
    to the remaining layers.
    """
    packed = ple_module.embedding_quantized.clone()
    scale = ple_module.embedding_scale.to(torch.float16).clone()
    nbits = ple_module.num_bits
    if nbits == 8 and packed.dtype == torch.int8:
        packed = (packed.to(torch.int16) + 128).to(torch.uint8)
    if truncate:
        expected_dim = config.num_hidden_layers * config.hidden_size_per_layer_input
        values_per_byte = 8 // nbits
        expected_blocks = expected_dim // (ple_module.embedding_dim // scale.shape[1])
        if scale.shape[1] > expected_blocks:
            scale = scale[:, :expected_blocks].contiguous()
        expected_packed = -(-expected_dim // values_per_byte)  # ceil div
        if packed.shape[1] > expected_packed:
            packed = packed[:, :expected_packed].contiguous()
    return packed, scale, nbits


def _native_quantized_hf_state_dict(hf_model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict for a QAT-quantized HF Gemma4 model, preserving its native quantization.

    This never reconstructs float weights: every
    quantized Linear/Embedding module's packed data is unpacked to one value per element
    and kept alongside its literal scale. Most are keyed as ``<name>.quantized_data`` /
    ``<name>.scale`` (signed, one value per int8 -- see ``unpack_signed_subbyte``) for the
    affine path (``NativeQuantConv1x1`` / ``NativeQuantLinear`` / the native embedding
    modules). ``mlp.*`` modules the checkpoint quantizes below 4 bits (Gemma4 E2B's
    shared-KV layers) are keyed as ``<name>.indices`` / ``<name>.scale`` instead (unsigned
    LUT indices -- see ``unpack_unsigned_subbyte``) for the palettization path
    (``NativeLutQuantConv1x1``); everything else stays on the affine path regardless of
    its own checkpoint bit-width (``lm_head``/``embed_tokens`` are deliberately widened to
    4 bits elsewhere -- see ``Gemma4Extend.__init__``'s ``lm_head`` comment -- so they're
    excluded from the ``.mlp.`` check here even though the checkpoint also ships them
    below 4 bits). ``Gemma4ForCausalLMForiOS._mutate_native_quantized_state_dict``
    reshapes either key set for Conv2d.

    ``embed_tokens_per_layer`` (the PLE table) is excluded here: it's never loaded into
    the model, only externalized by ``dump_ple_embedding`` -- see
    ``Gemma4ForCausalLMForiOS.from_hf``, which reads it directly off ``hf_model`` before
    this function's caller discards it.
    """
    quantized_modules = {
        name: module
        for name, module in hf_model.named_modules()
        if isinstance(module, (QuantizedLinear, QuantizedEmbedding))
        and not name.endswith("embed_tokens_per_layer")
    }
    state_dict = {
        k: v
        for k, v in hf_model.state_dict().items()
        if k.rsplit(".", 1)[0] not in quantized_modules and "embed_tokens_per_layer" not in k
    }
    for name, module in quantized_modules.items():
        if isinstance(module, QuantizedLinear):
            if module.num_bits < 4 and ".mlp." in name:
                state_dict[f"{name}.indices"] = unpack_unsigned_subbyte(
                    module.weight, module.num_bits, module.in_features
                )
            else:
                state_dict[f"{name}.quantized_data"] = unpack_signed_subbyte(
                    module.weight, module.num_bits, module.in_features
                )
            state_dict[f"{name}.scale"] = module.weight_scale.to(torch.float16)
        else:
            unpacked = unpack_signed_subbyte(
                module.embedding_quantized, module.num_bits, module.embedding_dim
            )
            state_dict[f"{name}.quantized_data"] = unpacked
            state_dict[f"{name}.scale"] = module.embedding_scale.to(torch.float16)
    return state_dict


class Gemma4NativeQuantLoadEmbeddings(nn.Module):
    """Loads the native-quantized embedding table + its per-row scale.

    Replaces the standard ``LoadEmbeddings`` for QAT checkpoints. Two outputs instead of
    one: the table (in the same ``(vocab, 1, hidden)`` INT8 layout as the standard table)
    and a per-vocab-row scale, gathered together downstream by
    ``Gemma4NativeQuantGatherEmbeddings``. Google's checkpoint quantizes this to 2 bits;
    the values are kept exactly, one per INT8 element, because the table is a runtime
    input of the gather and the runtime cannot gather from a sub-byte tensor.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Parameter(
            torch.zeros(vocab_size, 1, hidden_size, dtype=torch.int8), requires_grad=False
        )
        self.embedding_scale = nn.Parameter(
            torch.ones(vocab_size, 1, 1, dtype=torch.float16), requires_grad=False
        )

    def forward(self):
        return self.embedding_table, self.embedding_scale


class Gemma4NativeQuantGatherEmbeddings(nn.Module):
    """Gathers embedding rows, dequantizing via CoreAI's native ``blockwise_shift_scale``.

    Unlike the standard ``GatherEmbeddings`` (per-tensor INT8 scale), the scale here is
    gathered per-row alongside the table -- Google's QAT embeddings use a per-vocab-row
    scale, not per-tensor.
    """

    def __init__(self, nbits: int) -> None:
        super().__init__()
        self.nbits = nbits

    def forward(
        self,
        input_ids: torch.Tensor,
        embedding_table: torch.Tensor,
        embedding_scale: torch.Tensor,
    ) -> torch.Tensor:
        gathered = embedding_table[input_ids]
        gathered_scale = embedding_scale[input_ids]
        return dequantize_native(gathered, gathered_scale, self.nbits, torch.float16)


class Gemma4ForCausalLMForiOS(BaseForCausalLMForiOS):
    """Gemma4 iOS text-only CausalLM (extracted from multimodal HF model)."""

    _HF_MODEL_CLASS = Gemma4ForConditionalGeneration

    #: Sliding-attention window of the supported checkpoints (E2B, E4B).
    SLIDING_WINDOW = 512

    #: Sliding-cache ring depth. A class constant because export_hardware_constraints
    #: gets no config; build_reference_inputs checks the config agrees.
    SLIDING_RING_SIZE = sliding_ring_size(SLIDING_WINDOW, MAX_QUERY_LEN)

    def __init__(
        self, config, model_device: str = "cpu", disable_embedding_quantization: bool = False
    ) -> None:
        # INT8 embedding table by default (the iOS default); routes the token
        # gather through the fused_dequant_gather composite. Tests pass
        # ``disable_embedding_quantization=True`` to isolate logic from quant noise.
        super().__init__(
            config, model_device, disable_embedding_quantization=disable_embedding_quantization
        )
        if getattr(config, "native_quantized", False):
            # Google's QAT recipe quantizes `embed_tokens` to 2 bits with a per-row
            # (not per-tensor) scale, so it needs its own load/gather pair -- see
            # `Gemma4NativeQuantGatherEmbeddings`. Held as INT8, not packed: the table is a
            # runtime input of the gather, and the runtime cannot gather from a sub-byte tensor.
            # Built on `model_device` (meta under `from_hf`), as the transformer is, so the
            # table is not allocated only to be replaced by `load_state_dict`.
            with torch.device(model_device):
                self.load_embeddings = Gemma4NativeQuantLoadEmbeddings(
                    config.vocab_size, config.hidden_size
                )
            self.gather_embeddings = Gemma4NativeQuantGatherEmbeddings(nbits=8)

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

    @property
    def _native_quantized(self) -> bool:
        """Whether this model was built from a native-quantized (QAT) checkpoint."""
        return bool(getattr(self.config, "native_quantized", False))

    @override
    def validate_export_contract(
        self, reference_inputs: dict[str, dict[str, Any]], dynamic_shapes: dict[str, Any]
    ) -> None:
        # The name hooks are classmethods, so they describe the standard contract. A
        # native-quantized gather also takes the per-row embedding scale, which `export.py` adds
        # to the names it converts with; the rest of the contract is checked as it is.
        if self._native_quantized:
            gather = GATHER_EMBEDDINGS_FUNCTION_NAME
            reference_inputs = {
                **reference_inputs,
                gather: {
                    name: tensor
                    for name, tensor in reference_inputs[gather].items()
                    if name != EMBEDDING_SCALE_INPUT_NAME
                },
            }
        super().validate_export_contract(reference_inputs, dynamic_shapes)

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
        native = self._native_quantized
        if self.disable_embedding_quantization and not native:
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
        if native:
            embedding_scale = self.load_embeddings.embedding_scale
            # One graph's reference input is produced by running another.
            transformer_input = self.gather_embeddings(input_ids, embedding_table, embedding_scale)
        else:
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
        if native:
            # The runner unpacks and dequantizes native PLE rows itself, so the graph takes plain
            # float16 rows rather than INT8 ones.
            ple_embeddings = torch.zeros(1, query_len, 1, ple_width, dtype=torch.float16)
        else:
            ple_embeddings = torch.randint(
                -128, 127, (1, query_len, 1, ple_width), dtype=torch.int8
            )

        gather_inputs = {"input_ids": input_ids, "embedding_table": embedding_table}
        if native:
            gather_inputs[EMBEDDING_SCALE_INPUT_NAME] = embedding_scale

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: gather_inputs,
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
                PLE_EMBEDDINGS_INPUT_NAME: ple_embeddings,
            },
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        # The transformer rungs are fully static (see build_reference_inputs); the gather
        # is dynamic in the query length and specialized, as for every iOS model. A
        # native-quantized gather is traced at one static shape per query length instead:
        # the blockwise dequantization cannot build its zero-offset constant when the
        # per-row scale it gathers carries a dynamic dimension.
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: None,
            GATHER_EMBEDDINGS_FUNCTION_NAME: (
                None
                if self._native_quantized
                else super().build_dynamic_shapes(config, spec)[GATHER_EMBEDDINGS_FUNCTION_NAME]
            ),
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
        if isinstance(self.gather_embeddings, Gemma4NativeQuantGatherEmbeddings):
            token_embeddings = self.gather_embeddings(
                input_ids,
                self.load_embeddings.embedding_table,
                self.load_embeddings.embedding_scale,
            )
        else:
            token_embeddings = self.gather_embeddings(
                input_ids, self.load_embeddings.embedding_table
            )
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
        text_config.native_quantized = False

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

    @classmethod
    @override
    def from_hf(
        cls,
        huggingface_model_id_or_path: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        disable_embedding_quantization: bool = False,
        use_attention_mask: bool = False,
        backend: BackendName = "huggingface",
    ) -> "Gemma4ForCausalLMForiOS":
        """Load from HF multimodal model, extracting the text decoder only.

        A standard checkpoint loads through the base loader. A QAT checkpoint loads with
        quantized modules, whose packed weights, per-layer bit widths and PLE table are read
        off those modules before the HF model is discarded.
        """
        load_id = resolve_local(
            huggingface_model_id_or_path,
            backend,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "generation_config.json",
            ],
        )
        if not is_native_quantized_config(AutoConfig.from_pretrained(load_id)):
            return super().from_hf(
                load_id,
                max_context_length=max_context_length,
                target_dtype=target_dtype,
                mmap_path=mmap_path,
                num_layers=num_layers,
                disable_embedding_quantization=disable_embedding_quantization,
                use_attention_mask=use_attention_mask,
                backend=backend,
            )

        hf_model = cast(PreTrainedModel, cls._HF_MODEL_CLASS).from_pretrained(
            load_id, dtype=target_dtype
        )
        if not _is_quantized_hf_model(hf_model):
            raise ValueError(
                f"{huggingface_model_id_or_path} declares Gemma quantization but its text decoder "
                "loaded without quantized modules; the installed transformers may predate it."
            )

        config = cls._get_reauthored_config(
            hf_model.config, max_context_length, num_layers=num_layers
        )
        config.native_quantized = True
        if config.tie_word_embeddings:
            raise NotImplementedError(
                "A native-quantized Gemma 4 export needs an untied lm_head: a tied projection "
                "would reuse the low-bit embedding table without its per-row scale."
            )

        modules_by_name = dict(hf_model.named_modules())
        hf_prefix = "model.language_model."
        # The checkpoint's bit width of each quantized text-decoder module, by its name in this
        # model: checked against the supported widths now, and against the widths the modules
        # below declare once they exist.
        checkpoint_nbits = {
            _native_module_name(name, hf_prefix): module.num_bits
            for name, module in modules_by_name.items()
            if isinstance(module, (QuantizedLinear, QuantizedEmbedding))
            and (name.startswith(hf_prefix) or name == "lm_head")
        }
        _check_native_nbits(checkpoint_nbits)

        # Per-layer MLP bit width, read off the checkpoint's quantized modules (see
        # `_native_mlp_nbits`). Set before `cls(config, ...)` builds the module tree.
        config.native_mlp_nbits = []
        for i in range(config.num_hidden_layers):
            widths = {
                getattr(modules_by_name.get(f"{hf_prefix}layers.{i}.mlp.{proj}"), "num_bits", 4)
                for proj in ("gate_proj", "up_proj", "down_proj")
            }
            if len(widths) > 1:
                raise NotImplementedError(
                    f"layer {i} quantizes its MLP projections to mixed widths {sorted(widths)}; "
                    "one width per layer is supported."
                )
            config.native_mlp_nbits.append(widths.pop())

        # The PLE table is never loaded into the model (it is externalized by
        # `dump_ple_embedding`), so its packed data is kept as the checkpoint ships it.
        ple_native = None
        ple_module = modules_by_name.get(f"{hf_prefix}embed_tokens_per_layer")
        if ple_module is not None:
            ple_native = _native_ple_table(ple_module, config, truncate=num_layers is not None)

        full_sd = _native_quantized_hf_state_dict(hf_model)
        text_sd: dict[str, torch.Tensor] = {}
        for k, v in full_sd.items():
            if k.startswith(hf_prefix):
                text_sd["model." + k[len(hf_prefix) :]] = v
            elif k in ("lm_head.quantized_data", "lm_head.scale"):
                # The untied `lm_head` sits outside the `model.language_model.` prefix.
                text_sd[k] = v

        del hf_model, full_sd, modules_by_name, ple_module

        if num_layers is not None:
            text_sd = {k: v for k, v in text_sd.items() if not _is_layer_key_beyond(k, num_layers)}

        model = cls(
            config,
            model_device="meta",
            disable_embedding_quantization=disable_embedding_quantization,
        )
        model.to(dtype=target_dtype)
        _check_native_widths(model, checkpoint_nbits)

        model._mutate_native_quantized_state_dict(text_sd)
        if ple_native is not None:
            model._ple_native = ple_native

        for k, v in text_sd.items():
            if (
                v.dtype != target_dtype
                and v.is_floating_point()
                and "embedding_table" not in k
                and "zero_point" not in k
            ):
                raise ValueError(
                    f"tensor {k} in incorrect dtype {v.dtype}. Expected {target_dtype}."
                )

        strict = num_layers is None
        model.load_state_dict(text_sd, assign=True, strict=strict)

        if mmap_path is not None:
            move_model_to_disk(model, path=mmap_path)

        return model

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

    def _mutate_native_quantized_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """``_mutate_state_dict``'s native-quantized counterpart.

        Same reshaping/pruning shape as ``_mutate_state_dict``, but every Linear-shaped
        param carries a ``(quantized_data, scale)`` or, for the palettization path (see
        ``_native_quantized_hf_state_dict``), an ``(indices, scale)`` pair instead of a
        single ``.weight``, and the embedding table keeps its literal per-row scale rather
        than being requantized to a single per-tensor one. The PLE table isn't here at all
        -- ``from_hf`` externalizes it directly off the HF module before this ever runs.
        """
        config = self.config
        num_layers = config.num_hidden_layers
        first_kv_shared_idx = num_layers - config.num_kv_shared_layers
        ple_dim = config.hidden_size_per_layer_input

        def _reshape_for_conv(base_key: str) -> None:
            scale_key = f"{base_key}.scale"
            for data_key in (f"{base_key}.quantized_data", f"{base_key}.indices"):
                if data_key in state_dict:
                    state_dict[data_key] = state_dict[data_key].unsqueeze(-1).unsqueeze(-1)
                    state_dict[scale_key] = state_dict[scale_key].unsqueeze(-1).unsqueeze(-1)
                    return

        def _drop(base_key: str) -> None:
            state_dict.pop(f"{base_key}.quantized_data", None)
            state_dict.pop(f"{base_key}.indices", None)
            state_dict.pop(f"{base_key}.scale", None)

        # The runner dequantizes the PLE rows itself and feeds the graph a float input, so the
        # graph's own PLE dequantization is an identity here.
        state_dict["extend.ple_scale"] = torch.ones([], dtype=torch.float16)
        state_dict["extend.ple_zero_point"] = torch.zeros([], dtype=torch.int8)

        if ple_dim > 0:
            # Truncate per_layer_model_projection if needed (unquantized -- Google's own
            # recipe leaves it float too, see `modules_to_not_convert`).
            proj_key = "model.per_layer_model_projection.weight"
            if proj_key in state_dict:
                full_proj = state_dict[proj_key]
                expected_out = num_layers * ple_dim
                if full_proj.shape[0] > expected_out:
                    state_dict[proj_key] = full_proj[:expected_out, :]

        # Reshape quantized weights for Conv2d and handle shared layers
        for i in range(num_layers):
            is_kv_shared = i >= first_kv_shared_idx and config.num_kv_shared_layers > 0

            for proj in ("q_proj", "o_proj"):
                _reshape_for_conv(f"model.layers.{i}.self_attn.{proj}")

            if is_kv_shared:
                for proj in ("k_proj", "v_proj"):
                    _drop(f"model.layers.{i}.self_attn.{proj}")
                state_dict.pop(f"model.layers.{i}.self_attn.k_norm.weight", None)
            else:
                for proj in ("k_proj", "v_proj"):
                    _reshape_for_conv(f"model.layers.{i}.self_attn.{proj}")

            for proj in ("gate_proj", "up_proj", "down_proj"):
                _reshape_for_conv(f"model.layers.{i}.mlp.{proj}")

            if ple_dim > 0:
                for proj in ("per_layer_input_gate", "per_layer_projection"):
                    _reshape_for_conv(f"model.layers.{i}.{proj}")

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

        # Embeddings: keep the literal per-row scale, reshaped to this module's
        # `(vocab, 1, hidden)` / `(vocab, 1, 1)` layout convention.
        state_dict["load_embeddings.embedding_table"] = state_dict.pop(
            "model.embed_tokens.quantized_data"
        ).unsqueeze(1)
        state_dict["load_embeddings.embedding_scale"] = state_dict.pop(
            "model.embed_tokens.scale"
        ).unsqueeze(1)

        # The projection is untied (`from_hf` refuses a tied QAT checkpoint), so
        # `emb_scale`/`emb_zero_point` are never read at runtime; keep them as identity values
        # since `Gemma4Extend` still declares the parameters.
        state_dict["extend.emb_scale"] = torch.tensor(1.0, dtype=torch.float16)
        state_dict["extend.emb_zero_point"] = torch.tensor(0, dtype=torch.int8)

        # Handle lm_head: split into the same output-channel blocks
        # `NativeQuantLinear` builds internally (see its docstring for why each block
        # needs its own direct-placeholder buffer pair, not a shared sliced one).
        if not config.tie_word_embeddings:
            full_data = state_dict["lm_head.quantized_data"]
            full_scale = state_dict["lm_head.scale"]
            vocab_size = full_data.shape[0]
            block_size = DEFAULT_KV_BLOCK_SIZE
            if vocab_size <= block_size:
                state_dict["extend.lm_head.quantized_data"] = full_data
                state_dict["extend.lm_head.scale"] = full_scale
            else:
                for i, lo in enumerate(range(0, vocab_size, block_size)):
                    hi = min(lo + block_size, vocab_size)
                    state_dict[f"extend.lm_head.blocks.{i}.quantized_data"] = full_data[
                        lo:hi
                    ].contiguous()
                    state_dict[f"extend.lm_head.blocks.{i}.scale"] = full_scale[lo:hi].contiguous()
        state_dict.pop("lm_head.quantized_data", None)
        state_dict.pop("lm_head.scale", None)

    def dump_ple_embedding(self, output_path: str, model_name: str) -> str:
        """Dump the externalized PLE embedding table as a safetensors sidecar.

        Native-quantized checkpoints (``self._ple_native`` set by ``from_hf``) keep the
        checkpoint's own packed low-bit bytes and per-row-per-block scale exactly as shipped,
        with the architectural ``sqrt(ple_dim)`` embed scale folded into the scale so the
        runner's plain ``int_value * scale[block]`` directly produces the value the graph
        expects. The runner (``PerLayerEmbeddings.swift``) unpacks and dequantizes these rows
        itself and feeds the graph a plain float ``ple_embeddings`` input, so the graph never
        sees a quantized PLE tensor in that format.

        Every other checkpoint gets a quantized INT8 table, which the graph dequantizes.
        """
        import os

        config = self.config
        ple_dim = config.hidden_size_per_layer_input
        embed_scale = ple_dim**0.5

        os.makedirs(output_path, exist_ok=True)
        ple_path = os.path.join(output_path, f"{model_name}_ple.safetensors")

        if hasattr(self, "_ple_native"):
            packed, scale, nbits = self._ple_native
            native_scale = (scale.float() * embed_scale).to(torch.float16)
            save_file(
                {
                    "embed_tokens_per_layer": packed.contiguous(),
                    "embed_tokens_per_layer_scale": native_scale.contiguous(),
                },
                ple_path,
                metadata={
                    "nbits": str(nbits),
                    "ple_total_dim": str(config.num_hidden_layers * ple_dim),
                    "num_blocks": str(scale.shape[1]),
                    "dtype": "native_quantized",
                },
            )
            return ple_path

        if self.disable_embedding_quantization:
            raise ValueError(
                "The PLE sidecar is INT8; disable_embedding_quantization is for eager tests"
            )
        ple_weight = self._ple_weight
        ple_scale = _ple_quant_scale(ple_weight, embed_scale)

        # quantize_per_tensor's rounding, a block of rows at a time so the table never
        # has a full fp32 copy.
        ple_q = torch.empty(ple_weight.shape, dtype=torch.int8)
        for start in range(0, ple_weight.shape[0], 8192):
            rows = ple_weight[start : start + 8192].float() * embed_scale
            ple_q[start : start + 8192] = torch.clamp(torch.round(rows / ple_scale), -128, 127).to(
                torch.int8
            )

        save_file(
            {"embed_tokens_per_layer": ple_q},
            ple_path,
            metadata={
                "embed_scale": str(embed_scale),
                "ple_scale": str(float(ple_scale)),
                "ple_zero_point": "0",
                "dtype": "SI8",
            },
        )
        return ple_path
