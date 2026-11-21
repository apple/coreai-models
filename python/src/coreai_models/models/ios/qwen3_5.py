# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 (text backbone) for iOS.

Hybrid decoder: ``3 × (Gated-DeltaNet → FFN) + 1 × (Attention → FFN)`` repeated.
DeltaNet layers keep a fixed-size recurrent state (O(1) in context); only the
full-attention layers keep a growing KV cache. Exported by
``models/qwen3_5/export.py``; see ``models/qwen3_5/README.md``.
"""

from typing import Any

import torch
import torch.nn as nn
from typing_extensions import override

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    POSITION_IDS_INPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS, EagerState
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import BlockedSDPA
from coreai_models.primitives.ios.ssm_cache import MAX_QUERY_LEN, SSMStateHandler

# Optional so this module imports on a transformers build without qwen3_5.
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (  # type: ignore[import-not-found]
        Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
    )
except ImportError:  # pragma: no cover - depends on the installed transformers
    HFQwen3_5ForCausalLM = None  # type: ignore[assignment,misc]

# Graph I/O the DeltaNet layers add on top of the shared iOS contract. `conv_state` and
# `recurrent_state` are mutable in-graph state like the KV cache: passed in, updated in
# place, rebound by the runtime, with no separate `new_*` output.
#
# `ssm_update_flag`: 1.0 for slots holding a new token. Re-sent and padding slots (0.0) must
# leave every state as it was: the recurrent state, the conv ring and the KV cache.
#
# `conv_step`: the block's start modulo the ring width, where this call writes.
# `conv_seed_selector` is the one-hot that reads the block's whole conv window back out of
# the ring; see `SSMStateHandler.build_seed_selector`.
SSM_UPDATE_FLAG_INPUT_NAME = "ssm_update_flag"
CONV_STATE_INPUT_NAME = "conv_state"
CONV_STEP_INPUT_NAME = "conv_step"
CONV_SEED_SELECTOR_INPUT_NAME = "conv_seed_selector"
RECURRENT_STATE_INPUT_NAME = "recurrent_state"


# Sub-chunk width for the chunk-parallel delta rule. Bounds the triangular inverse to C×C;
# a length that isn't a multiple of it ends in a shorter chunk.
_CHUNK_SIZE = 16

# Query length each family is traced at: decode pads one token to 8, prefill submits a full chunk.
DECODE_QUERY_LEN = 8
PREFILL_QUERY_LEN = MAX_QUERY_LEN


def _rotary_dim(config) -> int:  # type: ignore[no-untyped-def]
    """Rotary dimension, which Qwen3.5 may make a fraction of the head dim."""
    head_dim = BaseForCausalLMForiOS._head_dim(config)
    rope_parameters = getattr(config, "rope_parameters", None)
    factor = (
        rope_parameters.get("partial_rotary_factor", 1.0)
        if isinstance(rope_parameters, dict)
        else 1.0
    )
    return int(head_dim * factor)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (linear-attention SSM).

    A causal depthwise conv over the projected Q/K/V, then the gated delta rule
    unrolled over the (static) query block. The recurrent state ``S`` is a
    per-head ``head_k_dim × head_v_dim`` matrix carried across steps. Scalars and
    ``exp`` are kept in the input dtype (see :func:`_scalar`, :func:`_exp`).
    """

    def __init__(self, config, delta_idx: int) -> None:
        super().__init__()
        self.delta_idx = delta_idx
        hidden = config.hidden_size

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.n_rep = self.num_v_heads // self.num_k_heads
        self._eps = config.rms_norm_eps

        # Projections as 1x1 Conv2d.
        self.in_proj_qkv = nn.Conv2d(hidden, self.conv_dim, 1, bias=False)
        self.in_proj_z = nn.Conv2d(hidden, self.value_dim, 1, bias=False)
        self.in_proj_b = nn.Conv2d(hidden, self.num_v_heads, 1, bias=False)
        self.in_proj_a = nn.Conv2d(hidden, self.num_v_heads, 1, bias=False)

        # Depthwise conv taps, (K, conv_dim, 1, 1); see `_depthwise_conv`.
        self.conv_weight = nn.Parameter(torch.zeros(self.conv_kernel, self.conv_dim, 1, 1))

        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))

        # Gated RMSNorm over head_v_dim (1-centered weight; loaded as-is).
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim))

        self.out_proj = nn.Conv2d(self.value_dim, hidden, 1, bias=False)

    def _to_heads(self, x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        # (B, C, 1, S) -> (B, S, n_heads, head_dim)
        b, _, _, s = x.shape
        return x.squeeze(2).transpose(1, 2).reshape(b, s, n_heads, head_dim)

    def _depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv as a shift-multiply-accumulate.

        Written as ``K`` shifted elementwise multiply-adds, which is exact for a depthwise
        conv. ``x``: (B, conv_dim, 1, K-1+S) -> (B, conv_dim, 1, S)."""
        k = self.conv_kernel
        s = x.shape[-1] - (k - 1)
        out = self.conv_weight[0] * x[:, :, :, 0:s]
        for j in range(1, k):
            out = out + self.conv_weight[j] * x[:, :, :, j : j + s]
        return out

    def forward(
        self,
        x: torch.Tensor,
        delta_state: SSMStateHandler,
        ssm_update_flag: torch.Tensor,
        conv_step: torch.IntTensor,
        conv_seed_selector: torch.Tensor,
    ):
        """
        Args:
            x: (B, S, 1, hidden) block embeddings.
            delta_state: conv/recurrent state handler. This layer's slot is read
                at the start and its update committed in place via ``delta_state``.
            ssm_update_flag: (B, S) 1.0 for genuinely-new positions, else 0.0.
            conv_step: (1,) int32 — the block's start modulo the conv ring width.
            conv_seed_selector: (1, ring_width, K-1+S) one-hot reading this block's
                conv window out of the ring.

        Returns:
            ``out`` (B, S, 1, hidden); the conv/recurrent state is mutated in place
            through ``delta_state`` (no explicit state return).
        """
        b, s, _, _ = x.shape

        recurrent_state = delta_state.recurrent_slot(self.delta_idx)  # (B, H, dk, dv)

        xt = x.transpose(-3, -1)  # (B, hidden, 1, S)
        qkv = self.in_proj_qkv(xt)  # (B, conv_dim, 1, S)
        z = self.in_proj_z(xt)  # (B, value_dim, 1, S)
        a = self.in_proj_a(xt)  # (B, num_v_heads, 1, S)
        bproj = self.in_proj_b(xt)  # (B, num_v_heads, 1, S)

        # Read the causal window out of the ring through the one-hot selector before this
        # block is written: (1, conv_dim, 1, W) @ (1, W, K-1+S) -> (1, conv_dim, 1, K-1+S).
        # Its last S columns are what the ring holds at the block's own positions.
        window = delta_state.conv_slot(self.delta_idx) @ conv_seed_selector
        kernel_window = window.shape[-1] - s
        # Re-sent and padding slots (flag=0) keep the ring's value, so the write is a no-op
        # for them and the conv sees the inputs the call that first wrote them computed.
        flag = ssm_update_flag.reshape(b, 1, 1, s)
        qkv = flag * qkv + (1 - flag) * window[..., kernel_window:]
        delta_state.update_conv(self.delta_idx, conv_step, qkv, s)
        # The ring is at least K-1 + MAX_QUERY_LEN wide, so the block's columns never overlap
        # the window's left context, which the write therefore leaves unchanged.
        conv_in = torch.cat([window[..., :kernel_window], qkv], dim=-1)
        conv_out = self._depthwise_conv(conv_in)  # (B, conv_dim, 1, S)
        conv_out = torch.nn.functional.silu(conv_out)

        query = self._to_heads(conv_out[:, : self.key_dim], self.num_k_heads, self.head_k_dim)
        key = self._to_heads(
            conv_out[:, self.key_dim : self.key_dim * 2], self.num_k_heads, self.head_k_dim
        )
        value = self._to_heads(conv_out[:, self.key_dim * 2 :], self.num_v_heads, self.head_v_dim)

        # Gates (per V-head).
        beta = bproj.squeeze(2).transpose(1, 2).sigmoid()  # (B, S, num_v_heads)
        a = a.squeeze(2).transpose(1, 2)  # (B, S, num_v_heads)
        g = -_exp(self.A_log).to(a.dtype) * _softplus(a + self.dt_bias)  # (B,S,H), < 0

        # L2-normalize q/k per head. The customary `1/sqrt(dk)` on q is *not* applied here —
        # see the scaling of `core` below.
        query = _l2norm(query)
        key = _l2norm(key)

        # Broadcast K/V-shared query/key heads up to num_v_heads.
        if self.n_rep > 1:
            query = query.repeat_interleave(self.n_rep, dim=2)
            key = key.repeat_interleave(self.n_rep, dim=2)

        # Flag-gate decay and update: re-sent/padding positions (flag=0) get zero log-decay
        # (no forgetting) and zero beta (no write), so they are exact no-ops for the state.
        g_eff = (g * ssm_update_flag.unsqueeze(-1)).transpose(1, 2)  # (B, H, S)
        beta_eff = (beta * ssm_update_flag.unsqueeze(-1)).transpose(1, 2)  # (B, H, S)
        core, new_recurrent_state = _chunk_gated_delta_rule(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            g_eff,
            beta_eff,
            recurrent_state,
        )
        # Apply q's 1/sqrt(dk) to the output (exact, since core is linear in q) to avoid fp16
        # underflow.
        core = core * _scalar(self.head_k_dim**-0.5, core)
        core = core.transpose(1, 2)  # (B, S, H, dv)

        # Gated RMSNorm over head_v_dim with gate z, then out_proj.
        z_heads = self._to_heads(z, self.num_v_heads, self.head_v_dim)  # (B,S,H,dv)
        inv = torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + _scalar(self._eps, core))
        core = core * inv * self.norm_weight
        core = core * torch.nn.functional.silu(z_heads)
        core = core.reshape(b, s, 1, self.value_dim)

        out = self.out_proj(core.transpose(-3, -1)).transpose(-3, -1)  # (B,S,1,hidden)

        # Overwrite this layer's recurrent slot in place (mutable state); the conv ring
        # was already written above.
        delta_state.update_recurrent(self.delta_idx, new_recurrent_state)
        return out


def _scalar(value: float, like: torch.Tensor) -> torch.Tensor:
    """A scalar constant in ``like``'s dtype.

    Built at the use site because a non-persistent buffer is not materialised when loading
    from meta."""
    return torch.full((), value, dtype=like.dtype, device=like.device)


def _exp(x: torch.Tensor) -> torch.Tensor:
    """``exp``, cast back to the input dtype."""
    return torch.exp(x).to(x.dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + _scalar(eps, x))


def _tri_inverse_apply(a_mat: torch.Tensor, target: torch.Tensor, length: int) -> torch.Tensor:
    """Apply ``(I − A)⁻¹`` to ``target``; ``A`` strictly-lower ⇒ nilpotent, so
    ``(I − A)⁻¹ = Σ_{k<length} Aᵏ``. Horner: ``u ← target + A·u`` repeated ``length − 1`` times.
    """
    u = target
    for _ in range(length - 1):
        u = target + a_mat @ u
    return u


def _delta_chunk_core(
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    g_h: torch.Tensor,
    beta_h: torch.Tensor,
    s0: torch.Tensor,
    tri_incl: torch.Tensor,
    strict: torch.Tensor,
    csum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form gated delta rule over ONE (sub)chunk, carrying state ``s0``.

    Rank-agnostic in the leading axes: the same body serves one head at ``(B, C, *)`` and every
    head at once at ``(B, H, C, *)``, which is what lets the caller batch the H axis. All
    tensors are sliced to the chunk's ``C`` positions; masks are the constant ``C×C``
    triangular masks. Returns ``(core, new_state)``. Exact for any ``C`` — chaining sub-chunks
    reproduces the sequential scan.
    """
    length = k_h.shape[-2]

    g_cum = g_h @ csum  # (B, C) cumulative log-decay as a matmul, in the input dtype
    diff = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tri_incl  # exponent ≤ 0 ⇒ fp16-safe
    decay_mask = _exp(diff) * tri_incl

    v_beta = v_h * beta_h.unsqueeze(-1)
    k_beta = k_h * beta_h.unsqueeze(-1)
    a_mat = -((k_beta @ k_h.transpose(-1, -2)) * decay_mask) * strict  # strictly-lower, nilpotent

    decay = _exp(g_cum).unsqueeze(-1)
    # (I-A)^-1 (v_beta - exp(g_cum) * (k_beta @ s0)): one dv-wide inverse, with decay applied
    # after the dk reduction.
    v_new = _tri_inverse_apply(a_mat, v_beta - (k_beta @ s0) * decay, length)

    attn = (q_h @ k_h.transpose(-1, -2)) * decay_mask  # causal incl self
    # Decay is per query row, so it is applied after the reduction to avoid fp16 underflow.
    inter = (q_h @ s0) * decay  # inter-chunk carry from state
    core_h = inter + attn @ v_new

    g_last = g_cum[..., -1:]
    decay_to_end = _exp(g_last - g_cum).unsqueeze(-1)  # exponent ≤ 0
    new_state_h = s0 * _exp(g_last).unsqueeze(-1) + ((k_h * decay_to_end).transpose(-1, -2) @ v_new)
    return core_h, new_state_h


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule over one static block in closed form, in sub-chunks of ``_CHUNK_SIZE``.

    The state is threaded across sub-chunks, bounding the triangular inverse to ``C×C``: its
    series grows geometrically with the chunk length and overflows fp16 well before 64. A
    block shorter than ``C`` (e.g. decode q=8) is a single chunk; a length that isn't a
    multiple of ``C`` ends in a shorter chunk. Heads are batched on the H axis.

    Args: query/key (B, H, L, dk) L2-normed, q at unit norm — its `1/√dk` is applied to the
    returned core by the caller; value (B, H, L, dv);
    g/beta (B, H, L) flag-gated; initial_state (B, H, dk, dv).
    Returns core (B, H, L, dv), new_state (B, H, dk, dv).
    """
    dtype, device = query.dtype, query.device
    length = key.shape[-2]

    csz = min(_CHUNK_SIZE, length)

    ones = torch.ones(csz, csz, dtype=dtype, device=device)
    tri_incl = torch.tril(ones)  # i ≥ j
    strict = torch.tril(ones, diagonal=-1)  # i > j
    csum = torch.triu(ones)  # csum[i, t] = 1 if i ≤ t (cumulative sum as a matmul)

    state = initial_state  # (B, H, dk, dv), threaded across sub-chunks for every head at once
    sub_cores: list[torch.Tensor] = []
    for start in range(0, length, csz):
        sl = slice(start, start + csz)
        masks = (tri_incl, strict, csum)
        n = length - start
        if n < csz:  # a shorter last chunk; the exported block lengths never take this path
            masks = tuple(mask[:n, :n] for mask in masks)
        core_c, state = _delta_chunk_core(
            query[:, :, sl],
            key[:, :, sl],
            value[:, :, sl],
            g[..., sl],
            beta[..., sl],
            state,
            *masks,
        )
        sub_cores.append(core_c)

    core = sub_cores[0] if len(sub_cores) == 1 else torch.cat(sub_cores, dim=2)
    return core, state  # (B, H, L, dv), (B, H, dk, dv)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """Softplus as ``relu(x) + log1p(exp(-|x|))``, kept in the input dtype."""
    one = torch.ones((), dtype=x.dtype, device=x.device)
    return torch.relu(x) + torch.log(_exp(-x.abs()) + one)


class Attention(nn.Module):
    """Full attention with grouped-query heads, partial-rotary RoPE, output gate."""

    def __init__(self, config, attn_idx: int) -> None:
        super().__init__()
        self.attn_idx = attn_idx

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = BaseForCausalLMForiOS._head_dim(config)
        self.rotary_dim = _rotary_dim(config)

        # q_proj emits query + fused output gate (2x).
        self.q_proj = nn.Conv2d(dim, self.n_heads * self.head_dim * 2, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, dim, 1, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Blocked (online-softmax) SDPA over the KV cache, in 16384-key blocks.
        self.sdpa = BlockedSDPA(head_dim=self.head_dim, block_size=16384)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        ssm_update_flag: torch.Tensor,
        cache: KVCacheHandler | None,
    ) -> torch.Tensor:
        b, q, _, _ = x.shape
        n_heads, n_kv_heads, hd = self.n_heads, self.n_kv_heads, self.head_dim

        xt = x.transpose(-3, -1)
        qg = self.q_proj(xt).transpose(-3, -1).reshape(b, q, n_heads, hd * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(b, q, n_heads * hd)

        key = self.k_proj(xt).transpose(-3, -1).reshape(b, q, n_kv_heads, hd)
        value = self.v_proj(xt).transpose(-3, -1).reshape(b, q, n_kv_heads, hd)

        query = self.q_norm(query).transpose(-2, -3)  # (B, n_heads, q, hd)
        key = self.k_norm(key).transpose(-2, -3)  # (B, n_kv_heads, q, hd)
        value = value.transpose(-2, -3)  # (B, n_kv_heads, q, hd) — heads-first like q/k

        query = _apply_partial_rope(query, rope_cos, rope_sin, self.rotary_dim)
        key = _apply_partial_rope(key, rope_cos, rope_sin, self.rotary_dim)

        # To iOS cache layout (B, n_kv_heads*hd, 1, q).
        query = query.transpose(-2, -3).reshape(b, q, 1, n_heads * hd).transpose(-3, -1)
        key = key.transpose(-2, -3).reshape(b, q, 1, n_kv_heads * hd).transpose(-3, -1)
        value = value.transpose(-2, -3).reshape(b, q, 1, n_kv_heads * hd).transpose(-3, -1)

        if cache is not None:
            # Slots with flag 0 keep the cached K/V, so the fetched row attention reads holds
            # the values the call that first wrote those positions computed.
            key, value = cache.update_and_fetch(
                self.attn_idx, in_step, key, value, q, update_flag=ssm_update_flag
            )

        output = self.sdpa(query, key, value, causal_mask)  # (B, n_heads*hd, 1, q)
        gate_t = gate.reshape(b, q, 1, n_heads * hd).transpose(-3, -1)  # (B, n_heads*hd, 1, q)
        output = output * torch.sigmoid(gate_t)
        return self.o_proj(output).transpose(-3, -1)


def _apply_partial_rope(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = apply_rope(x_rot, rope_cos, rope_sin)
    return torch.cat([x_rot, x_pass], dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, attn_idx: int, delta_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, delta_idx)
        else:
            self.self_attn = Attention(config, attn_idx)
        self.mlp = MLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        ssm_update_flag: torch.Tensor,
        conv_step: torch.IntTensor,
        conv_seed_selector: torch.Tensor,
        cache: KVCacheHandler | None,
        delta_state: SSMStateHandler | None,
    ):
        h = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            r = self.linear_attn(h, delta_state, ssm_update_flag, conv_step, conv_seed_selector)
        else:
            r = self.self_attn(h, rope_cos, rope_sin, in_step, causal_mask, ssm_update_flag, cache)
        x = x + r
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


def _layer_maps(config) -> tuple[list[int], list[int], int, int]:
    """Return (attn_idx_per_layer, delta_idx_per_layer, n_attn, n_delta)."""
    attn_idx = [-1] * config.num_hidden_layers
    delta_idx = [-1] * config.num_hidden_layers
    na = nd = 0
    for i, t in enumerate(config.layer_types):
        if t == "linear_attention":
            delta_idx[i] = nd
            nd += 1
        else:
            attn_idx[i] = na
            na += 1
    return attn_idx, delta_idx, na, nd


class Qwen3_5Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        attn_idx, delta_idx, self.n_attn, self.n_delta = _layer_maps(config)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, i, attn_idx[i], delta_idx[i])
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        ssm_update_flag: torch.Tensor,
        conv_step: torch.IntTensor,
        conv_seed_selector: torch.Tensor,
        cache: KVCacheHandler | None,
        delta_state: SSMStateHandler | None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                rope_cos,
                rope_sin,
                in_step,
                causal_mask,
                ssm_update_flag,
                conv_step,
                conv_seed_selector,
                cache,
                delta_state,
            )
        return self.norm(x)


class Qwen3_5Extend(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.prefill_mode = False

        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.kv_cache = KVCacheHandler(
            self.model.n_attn, Qwen3_5ForCausalLMForiOS._kv_embed_size(config)
        )

        dims = Qwen3_5ForCausalLMForiOS._deltanet_dims(config)
        self.delta_state = SSMStateHandler(
            n_delta=dims["n_delta"],
            conv_dim=dims["conv_dim"],
            ring_width=dims["ring_width"],
            num_v_heads=dims["num_v_heads"],
            head_k_dim=dims["head_k_dim"],
            head_v_dim=dims["head_v_dim"],
        )

        rotary_dim = _rotary_dim(config)
        self.rope = RoPECache(
            rotary_dim, config.max_position_embeddings, resolve_rope_theta(config)
        )

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        ssm_update_flag: torch.Tensor,
        conv_step: torch.IntTensor,
        conv_seed_selector: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        self.delta_state.register_state(conv_state, recurrent_state)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        b, s, _, hidden = transformer_input.shape
        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            ssm_update_flag,
            conv_step,
            conv_seed_selector,
            self.kv_cache,
            self.delta_state,
        )

        if self.prefill_mode:
            # Prefill skips the LM head; touch all four states so their in-place updates are kept.
            return (
                self.kv_cache.k_cache[0, 0, 0, 0, 0]
                + self.kv_cache.v_cache[0, 0, 0, 0, 0]
                + self.delta_state.recurrent_state[0, 0, 0, 0]
                + self.delta_state.conv_state[0, 0, 0, 0]
            )

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )
        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(b, 1, hidden, s)
        logits = (embedding_table @ out).transpose(-2, -1)
        return logits


_NORM_PLUS_ONE = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


class Qwen3_5ForCausalLMForiOS(BaseForCausalLMForiOS):
    # `Qwen3_5ForCausalLM` loads the text backbone straight from the VLM checkpoint
    # (transformers remaps the `model.language_model.` prefix), so the vision tower is
    # never instantiated.
    _HF_MODEL_CLASS = HFQwen3_5ForCausalLM

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        # Unwrap the text backbone from the multimodal config.
        tc = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        if not getattr(tc, "tie_word_embeddings", True):
            # The iOS graph computes logits from the int8 embedding table and drops lm_head.
            raise ValueError("Qwen3.5 on iOS requires tied input/output embeddings.")
        if max_context_length is not None:
            tc.max_position_embeddings = max_context_length
        if num_layers is not None:
            tc.num_hidden_layers = num_layers
            tc.layer_types = list(tc.layer_types[:num_layers])
        return tc

    @classmethod
    @override
    def from_hf_memory_efficient(  # type: ignore[override]
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = "",
        disable_embedding_quantization: bool = False,
    ) -> "Qwen3_5ForCausalLMForiOS":
        """Same as :meth:`from_hf`.

        The layer-by-layer loader would hand ``_mutate_state_dict`` one layer at a time, but
        it quantizes the tied embedding and rewrites the norms in a single pass over the
        whole checkpoint. ``from_hf`` already unwraps the multimodal config and prefix, so
        the registry's ``hf_config_attr`` / ``hf_state_dict_prefix`` are not needed here.
        """
        return cls.from_hf(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            mmap_path=mmap_path,
            num_layers=num_layers,
            disable_embedding_quantization=disable_embedding_quantization,
        )

    def _init_model(self, config) -> None:
        self.extend = Qwen3_5Extend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        ssm_update_flag: torch.Tensor,
        conv_step: torch.IntTensor,
        conv_seed_selector: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            ssm_update_flag,
            conv_step,
            conv_seed_selector,
            key_cache,
            value_cache,
            conv_state,
            recurrent_state,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # Keep only the text backbone; strip the VLM prefix.
        prefix = "model.language_model."
        renamed: dict[str, torch.Tensor] = {}
        for k, v in list(state_dict.items()):
            if k.startswith(prefix):
                renamed["model." + k[len(prefix) :]] = v
            elif k.startswith("model.") and not k.startswith(("model.visual", "model.mtp")):
                renamed[k] = v  # e.g. model.embed_tokens.weight, model.norm.weight
            # drop lm_head (tied), visual, mtp, everything else
        state_dict.clear()
        state_dict.update(renamed)

        conv2d_suffixes = (
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".o_proj.weight",
            ".gate_proj.weight",
            ".up_proj.weight",
            ".down_proj.weight",
            ".in_proj_qkv.weight",
            ".in_proj_z.weight",
            ".in_proj_b.weight",
            ".in_proj_a.weight",
            ".out_proj.weight",
        )
        for k in list(state_dict.keys()):
            if k.endswith(conv2d_suffixes):
                state_dict[k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)
            elif k.endswith(".conv1d.weight"):
                # HF holds the taps as an nn.Conv1d weight (conv_dim, 1, K); we hold them as
                # one broadcastable tap per kernel position, (K, conv_dim, 1, 1).
                taps = state_dict.pop(k).permute(2, 0, 1).unsqueeze(-1)
                state_dict[k[: -len(".conv1d.weight")] + ".conv_weight"] = taps
            elif k.endswith(_NORM_PLUS_ONE) or k == "model.norm.weight":
                state_dict[k] = state_dict[k] + 1.0
            elif k.endswith(".linear_attn.norm.weight"):
                # Gated RMSNorm weight (1-centered); rename to module param.
                state_dict[k[: -len(".norm.weight")] + ".norm_weight"] = state_dict.pop(k)

        # int8-quantize the tied embedding table.
        emb = state_dict.pop("model.embed_tokens.weight").unsqueeze(1)
        if not self.disable_embedding_quantization:
            emb, scale, zero_point = quantize_per_tensor(emb, nbits=8, symmetric=True)
        else:
            scale = torch.tensor(1.0, dtype=emb.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)
        state_dict["load_embeddings.embedding_table"] = emb
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        # Everything under model.* lives inside extend.model.*
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict["extend." + k] = state_dict.pop(k)

    # ------------------------------------------------------------------
    # Export contract
    #
    # Qwen3.5 keeps the base's four entrypoints but widens the transformer's I/O with the
    # DeltaNet layers' `ssm_update_flag`, `conv_step` and `conv_seed_selector` inputs and
    # their two states. `models/qwen3_5/export.py` reads its name lists through these hooks
    # so the recipe cannot drift from `Qwen3_5Extend.forward`, whose signature order the
    # converter zips them against.
    #
    # All ladder rungs share the transformer signature and differ only in shape, so the
    # recipe registers every bucket's pair of entrypoints against the single EXTEND entry.
    # The shapes themselves are per-bucket and live in the recipe.
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
                POSITION_IDS_INPUT_NAME,
                IN_STEP_INPUT_NAME,
                CAUSAL_MASK_INPUT_NAME,
                SSM_UPDATE_FLAG_INPUT_NAME,
                CONV_STEP_INPUT_NAME,
                CONV_SEED_SELECTOR_INPUT_NAME,
                EMBEDDING_TABLE_INPUT_NAME,
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
                CONV_STATE_INPUT_NAME,
                RECURRENT_STATE_INPUT_NAME,
            ),
        }

    @classmethod
    @override
    def export_state_output_names(cls) -> dict[str, tuple[str, ...]]:
        # The converter names a state's output after its input, so none of the four carries a
        # separate `new_*` alias; the base's KV-only pair would also undercount ours by two.
        return cls.export_state_names()

    @classmethod
    @override
    def export_state_classification(cls) -> dict[str, str]:
        """How the runner must treat each state.

        Only the DeltaNet states are pinned. They are fixed-size and cannot be truncated at
        all, since a partial reset would leave a recurrent state matching no prefix.

        KV caches are left to the runner's footprint heuristic.
        """
        return {
            CONV_STATE_INPUT_NAME: "fixed",
            RECURRENT_STATE_INPUT_NAME: "fixed",
        }

    @override
    def build_reference_inputs(self, config, target_dtype, spec) -> dict[str, dict]:
        raise NotImplementedError(
            "Qwen3.5 builds reference inputs per context bucket; export through "
            "models/qwen3_5/export.py."
        )

    @override
    def build_dynamic_shapes(self, config, spec) -> dict:
        raise NotImplementedError(
            "Qwen3.5 traces one fully static program per context bucket; there are no "
            "dynamic shapes to declare. Export through models/qwen3_5/export.py."
        )

    # ------------------------------------------------------------------
    # Shapes the export recipe needs
    #
    # The DeltaNet layers add three inputs and two states, and only the
    # full-attention layers keep a KV cache, so the caches are sized by `n_attn`.
    # `models/qwen3_5/export.py` reads these to build its reference
    # inputs; they live here because they are derived from the reauthored config.
    # ------------------------------------------------------------------

    @classmethod
    def _deltanet_dims(cls, config) -> dict[str, int]:
        """Layer counts and DeltaNet state dims, read off the HF config.

        Counts come from :func:`_layer_maps`, the same place the model gets them, so the state
        buffers cannot be sized for a different set of layers than the graph writes.
        """
        _, _, n_attn, n_delta = _layer_maps(config)
        nk, nv = config.linear_num_key_heads, config.linear_num_value_heads
        dk, dv = config.linear_key_head_dim, config.linear_value_head_dim
        return {
            "n_attn": n_attn,
            "n_delta": n_delta,
            "num_v_heads": nv,
            "head_k_dim": dk,
            "head_v_dim": dv,
            "conv_dim": dk * nk * 2 + dv * nv,
            "kernel_window": config.linear_conv_kernel_dim - 1,
            "ring_width": SSMStateHandler.ring_width(config.linear_conv_kernel_dim),
        }

    @classmethod
    def _kv_embed_size(cls, config) -> int:
        return config.num_key_value_heads * cls._head_dim(config)

    @classmethod
    def _conv_state(cls, config, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        dims = cls._deltanet_dims(config)
        return SSMStateHandler.get_conv_state(
            dims["n_delta"], dims["conv_dim"], dims["ring_width"], dtype=dtype
        )

    @classmethod
    def _recurrent_state(cls, config, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        dims = cls._deltanet_dims(config)
        return SSMStateHandler.get_recurrent_state(
            dims["n_delta"],
            dims["num_v_heads"],
            dims["head_k_dim"],
            dims["head_v_dim"],
            dtype=dtype,
        )

    @classmethod
    def _seed_selector(
        cls, config, query_len: int, dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        dims = cls._deltanet_dims(config)
        return SSMStateHandler.build_seed_selector(
            dims["ring_width"], dims["kernel_window"], 0, query_len, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Eager evaluation
    #
    # The graph takes three DeltaNet inputs and two extra states on top of the standard iOS
    # contract, and the conv ring bounds how many tokens one call may write. `new_eager_state`
    # and `eager_step`, at the end, let a harness score or generate without knowing either.
    # ------------------------------------------------------------------

    def fresh_states(
        self, context_length: int, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zeroed ``(key_cache, value_cache, conv_state, recurrent_state)``."""
        config = self.config
        dims = self._deltanet_dims(config)
        key_cache = torch.zeros(
            dims["n_attn"], 1, self._kv_embed_size(config), 1, context_length, dtype=dtype
        )
        return (
            key_cache.to(device),
            key_cache.clone().to(device),
            self._conv_state(config, dtype).to(device),
            self._recurrent_state(config, dtype).to(device),
        )

    def block_forward_args(
        self, input_ids: torch.Tensor, states: tuple[torch.Tensor, ...], offset: int = 0
    ) -> tuple[torch.Tensor, ...]:
        """``forward``'s arguments for one block ``(1, q)`` starting at position ``offset``.

        The block must not cross a multiple of ``MAX_QUERY_LEN``, so its conv-ring write
        doesn't straddle the wrap.
        """
        key_cache, value_cache, conv_state, recurrent_state = states
        dtype, device = key_cache.dtype, key_cache.device
        query_len = input_ids.shape[1]
        stop = offset + query_len
        if query_len > MAX_QUERY_LEN - offset % MAX_QUERY_LEN:
            raise ValueError(f"block [{offset}, {stop}) crosses a multiple of {MAX_QUERY_LEN}")
        if stop > key_cache.shape[-1]:
            raise ValueError(f"{stop} tokens exceed the {key_cache.shape[-1]}-token cache")
        dims = self._deltanet_dims(self.config)
        keys = torch.arange(key_cache.shape[-1], device=device).view(1, -1, 1, 1)
        queries = torch.arange(offset, stop, device=device).view(1, 1, 1, -1)
        causal_mask = torch.where(
            keys <= queries,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), -40000.0, dtype=dtype, device=device),
        )
        return (
            input_ids,
            torch.arange(offset, stop, device=device).to(torch.uint16).unsqueeze(0),
            torch.tensor([offset], dtype=torch.int32, device=device),
            causal_mask,
            torch.ones(1, query_len, dtype=dtype, device=device),
            torch.tensor([offset % dims["ring_width"]], dtype=torch.int32, device=device),
            SSMStateHandler.build_seed_selector(
                dims["ring_width"], dims["kernel_window"], offset, query_len, dtype=dtype
            ).to(device),
            key_cache,
            value_cache,
            conv_state,
            recurrent_state,
        )

    @torch.no_grad()
    def forward_sequence(
        self, input_ids: torch.Tensor, states: tuple[torch.Tensor, ...], offset: int = 0
    ) -> torch.Tensor:
        """Run ``input_ids`` ``(1, n)`` from absolute position ``offset``, updating ``states``.

        Tokens go through in blocks of at most ``MAX_QUERY_LEN`` that never cross a multiple
        of it. Returns logits ``(n, vocab)``.
        """
        end = offset + input_ids.shape[1]
        logits: list[torch.Tensor] = []
        start = offset
        while start < end:
            stop = min(end, (start // MAX_QUERY_LEN + 1) * MAX_QUERY_LEN)
            block = input_ids[:, start - offset : stop - offset]
            out = self(*self.block_forward_args(block, states, start))
            logits.append(out.reshape(stop - start, -1))
            start = stop
        return torch.cat(logits)

    @override
    def new_eager_state(
        self, *, dtype: torch.dtype, device: torch.device | str, max_seq_len: int | None = None
    ) -> EagerState:
        """The four states, sized to the whole context like every iOS cache: ``max_seq_len``
        is ignored."""
        return EagerState(self.fresh_states(self.config.max_position_embeddings, dtype, device))

    @override
    def eager_forward_args(self, input_ids: torch.Tensor, state: EagerState) -> tuple[Any, ...]:
        return self.block_forward_args(input_ids, state.tensors, state.offset)

    @override
    def eager_step(self, input_ids: torch.Tensor, state: EagerState) -> torch.Tensor:
        """Logits ``[1, query_len, vocab]``, run through ``forward_sequence``."""
        logits = self.forward_sequence(input_ids, state.tensors, state.offset)
        state.offset += input_ids.shape[1]
        return logits.unsqueeze(0)
