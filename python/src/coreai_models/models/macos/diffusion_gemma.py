# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""DiffusionGemma-26B-A4B text encoder/decoder for Core AI export.

Architecture summary
--------------------
  * Encoder-decoder with shared transformer weights (30 layers).
  * Encoder pass  - autoregressive, causal mask, builds KV cache from prompt.
  * Decoder pass  - block diffusion over a fixed-length canvas; bidirectional
                    (no causal mask); cross-attends to encoder KV via prepended
                    per-layer K/V.
  * Per-layer scalar differentiates encoder / decoder modes.
  * Each layer runs a dense MLP and a 128-expert top-8 MoE MLP in parallel;
    their outputs are summed under a shared post-feedforward norm.
  * Asymmetric attention heads:
      - sliding layers (25/30): 16 Q heads @ head_dim=256, 8 KV @ 256, window=1024
      - full layers   (every 6th): 16 Q heads @ head_dim=512, 2 KV @ 512, no window
  * Full-attention layers reuse K as V (no v_proj weight in the checkpoint).
  * Full-attention layers use partial ("proportional") RoPE.
  * MoE: 128 experts / top-8 active; weights are pre-stacked in the checkpoint.
  * Final logit softcapping at 30.0.

Cache lowering
--------------
The encoder writes and fetches its KV cache through the shared
``KVCache.update_and_fetch`` primitive, which lowers to the fused
``coreai::mutable_cache_update_and_fetch`` op and produces a clean functional
cache output. A single unified cache sized to the maximum head count (8) and
head_dim (512) across layer types is used; per-layer K/V are zero-padded to
those dimensions on write and narrowed back to native dimensions on read. The
decoder consumes the encoder's cache output as per-layer ``encoder_k`` /
``encoder_v`` inputs for cross-attention.

Two exported models
-------------------
  DiffusionGemmaEncoder  - causal, KV-cache, encoder scalars
  DiffusionGemmaDecoder  - bidirectional, canvas denoiser, decoder scalars,
                                    self-conditioning folded in
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.diffusion_gemma_config import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

# ---------------------------------------------------------------------------
# Unified KV-cache write via the fused mutable_cache_update_and_fetch op.
# ---------------------------------------------------------------------------


def _encoder_cache_update(
    cache: KVCache,
    layer_idx: int,
    offset: int,
    k: torch.Tensor,  # [B, n_kv, T, hd]  native per-layer dims
    v: torch.Tensor,
    seq_len: int,
    n_kv: int,
    hd: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write native-sized K/V into the unified cache and fetch the populated prefix.

    The unified cache is sized ``[L, 1, cache_n_kv, ctx, cache_hd]`` with
    ``cache_n_kv = max(8, 2)`` and ``cache_hd = max(256, 512)``. Layers with
    fewer KV heads / smaller head_dim have their K/V zero-padded up to those
    dimensions before the write, and the fetched tensors are narrowed back to the
    native ``(n_kv, hd)`` for attention. The write/fetch goes through the shared
    ``KVCache.update_and_fetch``, which lowers to the fused
    ``coreai::mutable_cache_update_and_fetch`` op (a clean functional cache
    output rather than a handle-based state write).
    """
    cache_n_kv = cache._k_cache.size(2)
    cache_hd = cache._k_cache.size(4)
    T = k.shape[-2]

    if n_kv < cache_n_kv or hd < cache_hd:
        # F.pad pads from the last dim backwards: (hd_lo, hd_hi, T_lo, T_hi, head_lo, head_hi).
        pad = (0, cache_hd - hd, 0, 0, 0, cache_n_kv - n_kv)
        k = F.pad(k, pad)
        v = F.pad(v, pad)

    k_out, v_out = cache.update_and_fetch(layer_idx, offset, k, v, seq_len=seq_len, query_len=T)
    # k_out / v_out: [B, cache_n_kv, seq_len, cache_hd] -> narrow to native dims.
    k_out = k_out.narrow(1, 0, n_kv).narrow(-1, 0, hd)
    v_out = v_out.narrow(1, 0, n_kv).narrow(-1, 0, hd)
    return k_out, v_out


# ---------------------------------------------------------------------------
# Config accessors that work with either our DiffusionGemmaTextConfig or the
# stock transformers config (the latter lacks our convenience methods), so the
# model builds from whatever config the registry/AutoConfig hands it.
# ---------------------------------------------------------------------------


def _is_full_attention_layer(config, layer_idx: int) -> bool:
    return config.layer_types[layer_idx] == "full_attention"


def _cache_head_dim(config) -> int:
    return max(config.head_dim, config.global_head_dim)


def _cache_num_key_value_heads(config) -> int:
    return max(config.num_key_value_heads, config.num_global_key_value_heads)


def _rope_params(config, *, full: bool) -> dict:
    return config.rope_parameters.get("full_attention" if full else "sliding_attention", {})


# ---------------------------------------------------------------------------
# RMSNorm — plain (no +1 offset), computed in float32 to match reference numerics.
# ---------------------------------------------------------------------------


class DiffusionGemmaRMSNorm(nn.Module):
    """Plain RMSNorm with no +1 weight offset, computed in float32.

    normed = x.float() * (mean(x.float()**2, -1) + eps) ** -0.5
    out    = normed * weight.float()       (only when with_scale=True)
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_sq, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


# ---------------------------------------------------------------------------
# Rotary position embedding via the Core AI native `rope` composite op.
# ---------------------------------------------------------------------------


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply RoPE to ``x`` [B, T, n_heads, head_dim] via the native rope op.

    cos/sin are HALF-dim [B, T, head_dim/2]; unsqueeze at ``unsqueeze_dim`` (=2)
    so they broadcast over the head axis. The native `rope` op computes the
    standard split-half rotation (y1 = cos*x1 - sin*x2; y2 = sin*x1 + cos*x2).
    """
    from coreai_torch.composite_ops._rope import rope as _native_rope

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _native_rope(x, cos=cos, sin=sin)


class ScaledEmbedding(nn.Embedding):
    """nn.Embedding that scales outputs by ``embed_scale`` (= sqrt(hidden_size))."""

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(
            self.embed_scale, dtype=self.weight.dtype, device=self.weight.device
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DiffusionGemmaAttention(nn.Module):
    """Shared attention block for both sliding and full-attention layers.

    Separate q_norm / k_norm / v_norm (v_norm has ``with_scale=False``).
    Order per head: q = rope(q_norm(q_proj(x))); k = rope(k_norm(k_proj(x)));
    v = v_norm(value) (no rope). Full-attention layers have no v_proj: value
    reuses the raw (pre-norm, pre-rope) K projection, then v_norm is applied.
    scaling = 1.0. Two SDPA modes are baked in at construction (is_encoder
    selects causal vs bidirectional) so the export graph carries no branch.
    """

    def __init__(
        self, config: DiffusionGemmaTextConfig, layer_idx: int, is_encoder: bool = True
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = _is_full_attention_layer(config, layer_idx)
        self.is_encoder = is_encoder

        self.n_heads = config.num_attention_heads  # 16 always
        dim = config.hidden_size  # 2816

        if self.is_full:
            self.n_kv_heads = config.num_global_key_value_heads  # 2
            self.head_dim = config.global_head_dim  # 512
            rope_full = _rope_params(config, full=True)
            partial_rotary_factor = rope_full.get("partial_rotary_factor", 0.25)
            rope_base = rope_full.get("rope_theta", 1_000_000.0)
            sdpa_window = 0
            inv_freq = self._proportional_inv_freq(self.head_dim, rope_base, partial_rotary_factor)
        else:
            self.n_kv_heads = config.num_key_value_heads  # 8
            self.head_dim = config.head_dim  # 256
            rope_base = _rope_params(config, full=False).get("rope_theta", 10_000.0)
            sdpa_window = config.sliding_window  # 1024
            inv_freq = self._default_inv_freq(self.head_dim, rope_base)

        n_h = self.n_heads
        n_kv = self.n_kv_heads
        hd = self.head_dim

        self.q_proj = nn.Linear(dim, n_h * hd, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv * hd, bias=config.attention_bias)
        self.v_proj = (
            None if self.is_full else nn.Linear(dim, n_kv * hd, bias=config.attention_bias)
        )
        self.o_proj = nn.Linear(n_h * hd, dim, bias=config.attention_bias)

        self.q_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps, with_scale=False)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if is_encoder:
            self.sdpa = SDPA(scale=1.0, window_size=sdpa_window, is_causal=True)
        else:
            self.sdpa = SDPA(scale=1.0, window_size=0, is_causal=False)

    @staticmethod
    def _default_inv_freq(head_dim: int, base: float) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))

    @staticmethod
    def _proportional_inv_freq(
        head_dim: int, base: float, partial_rotary_factor: float
    ) -> torch.Tensor:
        """Only the first ``partial_rotary_factor`` of head_dim is rotated; the
        remaining frequencies are zero (identity rotation)."""
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        nope_angles = head_dim // 2 - rope_angles
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim)
        )
        return torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=inv_freq_rotated.dtype)])

    def _cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-dim cos/sin [B, T, head_dim/2] from position_ids [B, T]."""
        inv_freq = self.inv_freq
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, T, hd/2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        n_h, n_kv = self.n_heads, self.n_kv_heads
        hd = self.head_dim

        seq_len = position_ids.shape[-1]
        torch._check_is_size(T)
        torch._check_is_size(seq_len)
        offset = seq_len - T
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, T)  # [B, T]

        cos, sin = self._cos_sin(rope_positions, x.dtype)

        query = self.q_proj(x).reshape(B, T, n_h, hd)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)  # [B, n_h, T, hd]

        key_raw = self.k_proj(x).reshape(B, T, n_kv, hd)
        # Full-attention layers reuse the raw (pre-norm, pre-rope) K as the value.
        value_raw = self.v_proj(x).reshape(B, T, n_kv, hd) if self.v_proj is not None else key_raw

        key = self.k_norm(key_raw)
        key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
        key = key.transpose(1, 2)  # [B, n_kv, T, hd]

        value = self.v_norm(value_raw)
        value = value.transpose(1, 2)  # [B, n_kv, T, hd]

        if cache is not None and is_causal:
            # Encoder pass: write into the unified cache, fetch the populated prefix.
            key, value = _encoder_cache_update(
                cache, self.layer_idx, offset, key, value, seq_len=seq_len, n_kv=n_kv, hd=hd
            )
        elif encoder_k is not None and encoder_v is not None:
            # Decoder pass: prepend per-layer encoder context to the canvas K/V.
            enc_k_layer = encoder_k[:, :n_kv, :, :hd]
            enc_v_layer = encoder_v[:, :n_kv, :, :hd]
            key = torch.cat([enc_k_layer, key], dim=-2)
            value = torch.cat([enc_v_layer, value], dim=-2)

        out = self.sdpa(query=query, key=key, value=value)
        out = out.permute(0, 2, 1, 3).reshape(B, T, n_h * hd)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------


class DiffusionGemmaDenseMLP(nn.Module):
    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE MLP (pre-stacked expert weights)
# ---------------------------------------------------------------------------


class _GeGLUTanh(nn.Module):
    """GeGLU activation for SwitchGLU: gelu_tanh(gate) * up."""

    def forward(self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate="tanh") * up


class DiffusionGemmaMoEMLP(nn.Module):
    """Sparse Mixture-of-Experts MLP built on the SwitchGLU primitive.

    Router (operates on the raw residual):
      h      = router_norm(residual)          # RMSNorm, with_scale=False
      h      = h * router_scale * hidden_size**-0.5
      scores = softmax(router_proj(h), dim=-1, dtype=float32)
      w, idx = topk(scores, k=top_k)
      w      = w / w.sum(-1, keepdim=True)
      w      = w * per_expert_scale[idx]       # per-expert scale folded in HERE

    Experts (operate on pre_feedforward_layernorm_2(residual)):
      y = SwitchGLU(x, idx); out = sum_e w_e * y_e
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        self.n_experts = config.num_experts  # 128
        self.top_k = config.top_k_experts  # 8
        self.moe_int = config.moe_intermediate_size  # 704
        h = config.hidden_size  # 2816
        self.scalar_root_size = h**-0.5

        self.switch_glu = SwitchGLU(
            hidden_size=h,
            moe_intermediate_size=self.moe_int,
            num_experts=self.n_experts,
            bias=False,
            activation=_GeGLUTanh(),
        )

        self.router_norm = DiffusionGemmaRMSNorm(h, eps=config.rms_norm_eps, with_scale=False)
        self.router_proj = nn.Linear(h, self.n_experts, bias=False)
        self.router_scale = nn.Parameter(torch.ones(h))
        self.per_expert_scale = nn.Parameter(torch.ones(self.n_experts))

    def forward(self, routing_input: torch.Tensor, experts_input: torch.Tensor) -> torch.Tensor:
        h = self.router_norm(routing_input)
        h = h * self.router_scale * self.scalar_root_size
        scores = F.softmax(self.router_proj(h).float(), dim=-1)  # [B, T, E] float32

        top_w, top_idx = torch.topk(scores, self.top_k, dim=-1)  # [B, T, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w * self.per_expert_scale[top_idx]  # fold per-expert scale in once
        top_w = top_w.to(experts_input.dtype)
        # uint16 indices: required by SwitchGLU's GatherMM for correct Core AI lowering.
        top_idx = top_idx.to(torch.uint16)

        y = self.switch_glu(experts_input, top_idx)  # [B, T, k, H]
        out = (y * top_w.unsqueeze(-1)).sum(dim=2)  # [B, T, H]
        return out.to(experts_input.dtype)


# ---------------------------------------------------------------------------
# Self-conditioning MLP (applied between diffusion steps)
# ---------------------------------------------------------------------------


class DiffusionGemmaSelfConditioning(nn.Module):
    """Injects the previous denoising step's prediction into the canvas embeddings.

    normed   = pre_norm(self_conditioning_signal)
    sc       = down(gelu_tanh(gate(normed)) * up(normed))
    combined = inputs_embeds + sc
    return   post_norm(combined)            # post_norm has with_scale=False
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size  # 2112
        self.pre_norm = DiffusionGemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.post_norm = DiffusionGemmaRMSNorm(h, eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(
        self, inputs_embeds: torch.Tensor, self_conditioning_signal: torch.Tensor
    ) -> torch.Tensor:
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(
            F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed)
        )
        combined = inputs_embeds + sc_signal
        return self.post_norm(combined)


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------


class DiffusionGemmaLayer(nn.Module):
    """One transformer block with parallel dense MLP + sparse MoE feedforward paths."""

    def __init__(
        self, config: DiffusionGemmaTextConfig, layer_idx: int, is_encoder: bool = True
    ) -> None:
        super().__init__()
        h = config.hidden_size
        eps = config.rms_norm_eps

        self.self_attn = DiffusionGemmaAttention(config, layer_idx, is_encoder=is_encoder)
        self.mlp = DiffusionGemmaDenseMLP(config)
        self.moe = DiffusionGemmaMoEMLP(config)

        self.input_layernorm = DiffusionGemmaRMSNorm(h, eps=eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(h, eps=eps)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(h, eps=eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(h, eps=eps)

        # Separate encoder / decoder per-layer scalars (distinct checkpoint values).
        self.decoder_scalar = nn.Parameter(torch.ones(1))
        self.encoder_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # -- Attention sub-layer ---------------------------------------------
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, position_ids, cache, encoder_k, encoder_v, is_causal)
        h = self.post_attention_layernorm(h)
        h = residual + h  # first residual

        # -- Feedforward sub-layer (dense + MoE in parallel) -----------------
        residual = h  # post-attn output
        dense_in = self.pre_feedforward_layernorm(h)
        h1 = self.post_feedforward_layernorm_1(self.mlp(dense_in))  # dense path

        experts_in = self.pre_feedforward_layernorm_2(residual)
        h2 = self.moe(residual, experts_in)  # router reads RAW residual
        h2 = self.post_feedforward_layernorm_2(h2)

        h = h1 + h2  # combine dense + moe
        h = self.post_feedforward_layernorm(h)  # OUTER norm on the sum
        out = residual + h  # second residual

        layer_scalar = self.encoder_scalar if is_causal else self.decoder_scalar
        return out * layer_scalar


# ---------------------------------------------------------------------------
# Shared transformer backbone
# ---------------------------------------------------------------------------


class DiffusionGemmaSharedTransformer(nn.Module):
    """30-layer shared transformer backbone used by both encoder and decoder passes."""

    def __init__(self, config: DiffusionGemmaTextConfig, is_encoder: bool = True) -> None:
        super().__init__()
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [
                DiffusionGemmaLayer(config, i, is_encoder=is_encoder)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k_cache: torch.Tensor | None = None,
        encoder_v_cache: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        if input_ids_or_embeds.dtype in (torch.int32, torch.int64):
            h = self.embed_tokens(input_ids_or_embeds)
        else:
            h = input_ids_or_embeds  # decoder canvas pass: already embedded

        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            torch._check_is_size(i)
            torch._check(i < n_layers)
            enc_k = (
                encoder_k_cache.narrow(0, i, 1).squeeze(0) if encoder_k_cache is not None else None
            )
            enc_v = (
                encoder_v_cache.narrow(0, i, 1).squeeze(0) if encoder_v_cache is not None else None
            )
            h = layer(h, position_ids, cache, enc_k, enc_v, is_causal)

        return self.norm(h)


# ---------------------------------------------------------------------------
# Encoder export model (autoregressive, builds KV cache from prompt)
# ---------------------------------------------------------------------------


class DiffusionGemmaEncoder(BaseForCausalLM):
    """Autoregressive encoder pass: causal LM that fills the KV cache.

    Exported forward signature:
        (input_ids[B, T], position_ids[B, seq_len],
         k_cache[n_layers, 1, cache_n_kv, max_ctx, cache_hd],
         v_cache[n_layers, 1, cache_n_kv, max_ctx, cache_hd])
        -> logits[B, T, vocab_size]
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config: DiffusionGemmaTextConfig) -> None:
        self.model = DiffusionGemmaSharedTransformer(config, is_encoder=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Cache-sizing attributes (unified cache: max heads / max head_dim).
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = _cache_num_key_value_heads(config)
        self.head_dim = _cache_head_dim(config)
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache=cache, is_causal=True)
        logits = self.lm_head(out)
        cap = getattr(self.config, "final_logit_softcapping", 30.0)
        if cap and cap > 0.0:
            logits = torch.tanh(logits / cap) * cap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        _mutate_diffusion_gemma_state_dict(state_dict, self)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Decoder export model (bidirectional block-diffusion canvas denoiser)
# ---------------------------------------------------------------------------


class DiffusionGemmaDecoder(BaseForCausalLM):
    """Bidirectional block-diffusion decoder over a fixed-length canvas.

    Exported forward signature:
        (decoder_input_ids[B, canvas]            int32 canvas token ids
         prev_soft_embeds[B, canvas, H]          soft-cond signal (zeros on step 0)
         position_ids[B, canvas]                 int32 absolute positions
         encoder_k[n_layers, 1, cache_n_kv, enc_len, cache_hd]
         encoder_v[n_layers, 1, cache_n_kv, enc_len, cache_hd]
         temperature[1])                         per-step diffusion temperature
        -> (logits[B, canvas, vocab]             processed logits (softcapped / temp)
            soft_embeds[B, canvas, H])           signal for the NEXT step
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config: DiffusionGemmaTextConfig) -> None:
        self.model = DiffusionGemmaSharedTransformer(config, is_encoder=False)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = _cache_num_key_value_heads(config)
        self.head_dim = _cache_head_dim(config)
        self.vocab_size = config.vocab_size

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        prev_soft_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.model.embed_tokens
        inputs_embeds = embed(decoder_input_ids)  # [B, canvas, H]
        fused = self.self_conditioning(inputs_embeds, prev_soft_embeds)

        out = self.model(
            fused,
            position_ids,
            cache=None,
            encoder_k_cache=encoder_k,
            encoder_v_cache=encoder_v,
            is_causal=False,
        )
        raw_logits = self.lm_head(out)
        cap = getattr(self.config, "final_logit_softcapping", 30.0)
        if cap and cap > 0.0:
            raw_logits = torch.tanh(raw_logits / cap) * cap

        processed = raw_logits / temperature

        soft = torch.matmul(
            processed.softmax(dim=-1, dtype=torch.float32).to(embed.weight.dtype),
            embed.weight,
        ) * float(embed.embed_scale)

        if processed.dtype == torch.bfloat16:
            processed = processed.to(torch.float16)

        return processed, soft

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        _mutate_diffusion_gemma_state_dict(state_dict, self)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Weight-key mutation helper
# ---------------------------------------------------------------------------


def _mutate_diffusion_gemma_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: DiffusionGemmaEncoder | DiffusionGemmaDecoder,
) -> None:
    """Remap HF checkpoint keys to our model's parameter namespace in-place.

    Shared transformer weights live under ``model.decoder.layers.{i}.*`` in the
    checkpoint; the encoder scalar lives under
    ``model.encoder.language_model.layers.{i}.layer_scalar``. q/k/v projections
    and q/k norms are kept UNFUSED. Full-attention layers have no v_proj weight.
    """
    _has_self_cond = hasattr(model, "self_conditioning")
    for k in list(state_dict.keys()):
        drop_vision = k.startswith("model.encoder.vision_tower") or k.startswith(
            "model.encoder.embed_vision"
        )
        drop_self_cond = k.startswith("model.decoder.self_conditioning.") and not _has_self_cond
        if drop_vision or drop_self_cond:
            del state_dict[k]

    _top_renames = [
        ("model.decoder.embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.decoder.norm.weight", "model.norm.weight"),
    ]
    for src, dst in _top_renames:
        if src in state_dict:
            state_dict[dst] = state_dict.pop(src)

    for k in list(state_dict.keys()):
        if k.startswith("model.decoder.self_conditioning."):
            new_k = k.replace("model.decoder.self_conditioning.", "model.self_conditioning.")
            state_dict[new_k] = state_dict.pop(k)

    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        hf_pfx = f"model.decoder.layers.{i}"
        our_pfx = f"model.layers.{i}"

        _layer_subs = [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ]
        for sub in _layer_subs:
            old_k = f"{hf_pfx}.{sub}.weight"
            new_k = f"{our_pfx}.{sub}.weight"
            if old_k in state_dict:
                state_dict[new_k] = state_dict.pop(old_k)

        old_dec = f"{hf_pfx}.layer_scalar"
        if old_dec in state_dict:
            state_dict[f"{our_pfx}.decoder_scalar"] = state_dict.pop(old_dec)

        old_enc = f"model.encoder.language_model.layers.{i}.layer_scalar"
        if old_enc in state_dict:
            state_dict[f"{our_pfx}.encoder_scalar"] = state_dict.pop(old_enc)

        gu_old = f"{hf_pfx}.experts.gate_up_proj"
        dn_old = f"{hf_pfx}.experts.down_proj"
        if gu_old in state_dict:
            gu = state_dict.pop(gu_old)  # [E, 2*moe_int, H]
            moe_int = gu.shape[1] // 2
            state_dict[f"{our_pfx}.moe.switch_glu.gate_proj.weight"] = (
                gu[:, :moe_int, :].unsqueeze(0).contiguous()
            )
            state_dict[f"{our_pfx}.moe.switch_glu.up_proj.weight"] = (
                gu[:, moe_int:, :].unsqueeze(0).contiguous()
            )
        if dn_old in state_dict:
            dn = state_dict.pop(dn_old)  # [E, H, moe_int]
            state_dict[f"{our_pfx}.moe.switch_glu.down_proj.weight"] = dn.unsqueeze(0).contiguous()

        rp = f"{hf_pfx}.router.proj.weight"
        rs = f"{hf_pfx}.router.scale"
        re = f"{hf_pfx}.router.per_expert_scale"
        if rp in state_dict:
            state_dict[f"{our_pfx}.moe.router_proj.weight"] = state_dict.pop(rp)
        if rs in state_dict:
            state_dict[f"{our_pfx}.moe.router_scale"] = state_dict.pop(rs)
        if re in state_dict:
            state_dict[f"{our_pfx}.moe.per_expert_scale"] = state_dict.pop(re)


# ---------------------------------------------------------------------------
# Convenience loading functions
# ---------------------------------------------------------------------------


def _load_state_dict_from_hub(
    hf_model_id: str, target_dtype: torch.dtype, num_layers: int | None
) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download

    from coreai_models.models.base import (
        _build_safetensors_key_index,
        _load_tensors_for_keys,
        _resolve_safetensors_files,
    )

    model_dir = snapshot_download(
        hf_model_id, allow_patterns=["*.safetensors", "*.safetensors.index.json"]
    )
    sf_files = _resolve_safetensors_files(model_dir)
    per_layer, shared = _build_safetensors_key_index(
        sf_files, num_layers=num_layers, hf_state_dict_prefix=""
    )
    sd: dict[str, torch.Tensor] = _load_tensors_for_keys(shared, target_dtype)
    for layer_idx in sorted(per_layer.keys()):
        sd.update(_load_tensors_for_keys(per_layer[layer_idx], target_dtype))
    return sd


def _assert_state_dict_loaded(name: str, info) -> None:
    """Fail closed on a mis-renamed or absent weight.

    With ``assign=True`` a missing key silently keeps its random ``nn.Linear``
    init and export would ship a corrupt bundle. lm_head is tied to the
    embedding after loading (not present under this name in the checkpoint), so
    it is the only permitted missing key; anything else missing is an error.
    Extra checkpoint tensors (e.g. the vision tower) are harmless and ignored.
    """
    missing = [k for k in getattr(info, "missing_keys", []) if not k.endswith("lm_head.weight")]
    if missing:
        raise RuntimeError(
            f"{name}: {len(missing)} weight(s) missing after load (would ship as "
            f"random init): {missing[:10]}"
        )


def load_diffusion_gemma_encoder(
    hf_model_id: str,
    target_dtype: torch.dtype = torch.bfloat16,
    max_context_length: int | None = None,
    num_layers: int | None = None,
    mmap_path: str | None = None,
) -> DiffusionGemmaEncoder:
    """Load the encoder model with HF checkpoint weights."""
    cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id).text_config
    if max_context_length is not None:
        cfg.max_position_embeddings = max_context_length
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers

    model = DiffusionGemmaEncoder(cfg)
    model.to(dtype=target_dtype)

    sd = _load_state_dict_from_hub(hf_model_id, target_dtype, num_layers)
    _mutate_diffusion_gemma_state_dict(sd, model)
    _assert_state_dict_loaded("encoder", model.load_state_dict(sd, strict=False, assign=True))
    del sd
    gc.collect()

    if mmap_path is not None:
        from coreai_models.models.base import move_model_to_disk

        move_model_to_disk(model, path=mmap_path)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def load_diffusion_gemma_decoder(
    hf_model_id: str,
    target_dtype: torch.dtype = torch.bfloat16,
    num_layers: int | None = None,
) -> DiffusionGemmaDecoder:
    """Load the canvas denoiser model with HF checkpoint weights."""
    cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id).text_config
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers

    model = DiffusionGemmaDecoder(cfg)
    model.to(dtype=target_dtype)

    sd = _load_state_dict_from_hub(hf_model_id, target_dtype, num_layers)
    _mutate_diffusion_gemma_state_dict(sd, model)
    _assert_state_dict_loaded("decoder", model.load_state_dict(sd, strict=False, assign=True))
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
