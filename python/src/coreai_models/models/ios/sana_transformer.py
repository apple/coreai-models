# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 Sana Sprint transformer, channels-first (BC1S) conv layout.

Same inputs and outputs as diffusers' ``SanaTransformer2DModel`` behind ``SanaTransformerWrapper``.
Tokens live in ``(1, C, 1, N)``; linears are 1x1 ``Conv2d``. Exact rewrites keep float16
activations in range:

- All norms are max-abs (``Σx²`` overflows float16).
- Linear attention scales ``K`` by ``_K_SCALE / N`` (folded into ``norm_k``) and divides each
  token's per-head ``q`` by its max. The output is a ratio in both, so both cancel; they keep
  ``Σ v kᵀ`` and ``(Σ v kᵀ) q`` in range and the denominator well above the float16-tiny floor.
- Cross-attention scales ``to_q`` by ``_Q_SCALE`` and ``norm_q``'s eps by its square, which leaves
  the RMSNorm unchanged and keeps ``to_q``'s output in range.
- SiLU is :class:`TanhSiLU`, more precise than the Neural Engine's fused SiLU.
- ``x * (1 + scale)`` folds the ``1`` into ``scale_shift_table``, and the cross-attention
  ``1/sqrt(d)`` folds into ``norm_q``, so the graph has no float32 literals.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.ios.gelu import gelu_ane
from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm, MaxAbsRMSNormNoScale
from coreai_models.primitives.ios.silu import TanhSiLU

_FP16_TINY = torch.finfo(torch.float16).tiny
# Keeps the linear-attention numerator in float16 range and the denominator above the floor.
_K_SCALE = 32.0
# A power of two, so the fold is exact.
_Q_SCALE = 1 / 16
_MASK_BIAS = -10000.0  # diffusers' cross-attention mask bias


def conv1x1(cin: int, cout: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, bias=bias)


def _load_linear(conv: nn.Conv2d, linear: Any) -> None:
    conv.weight.data.copy_(linear.weight.detach().float()[:, :, None, None])
    if conv.bias is not None:
        conv.bias.data.copy_(linear.bias.detach().float())


class LayerNormNoAffine(nn.Module):
    """Non-affine LayerNorm over channels: RMSNorm of the centered input (float16-safe)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.rms = MaxAbsRMSNormNoScale(eps, axis=-3, floor="add")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms(x - x.mean(-3, keepdim=True))


class TimestepEmbedding(nn.Module):
    """``Timesteps(256, flip_sin_to_cos=True)`` + ``TimestepEmbedding``: [1] -> (1, C, 1, 1).

    ``freq_scale`` folds a constant input multiplier (the guidance embeds scale) into the table.
    """

    freqs: torch.Tensor

    def __init__(self, dim: int, freq_scale: float = 1.0) -> None:
        super().__init__()
        half = 128
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
        self.register_buffer("freqs", (freqs * freq_scale).float().reshape(1, half, 1, 1))
        self.linear_1 = conv1x1(2 * half, dim)
        self.linear_2 = conv1x1(dim, dim)
        self.silu = TanhSiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        arg = t.reshape(1, 1, 1, 1) * self.freqs
        emb = torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)
        return self.linear_2(self.silu(self.linear_1(emb)))


def linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tiny: torch.Tensor
) -> torch.Tensor:
    """``(Σₙ v kᵀ) q / (Σₙ k · q + tiny)`` for non-negative ``q``, ``k``, all ``(B, H, d, N)``."""
    kv = torch.einsum("bhdn,bhen->bhde", v, k)
    num = torch.einsum("bhde,bhen->bhdn", kv, q)
    den = (k.sum(-1, keepdim=True) * q).sum(-2, keepdim=True)
    return num / (den + tiny)


class LinearAttention(nn.Module):
    """ReLU linear self-attention with qk RMSNorm across heads; ``(1, C, 1, N)``."""

    tiny: torch.Tensor

    def __init__(self, dim: int, heads: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = conv1x1(dim, dim, bias=False)
        self.to_k = conv1x1(dim, dim, bias=False)
        self.to_v = conv1x1(dim, dim, bias=False)
        self.norm_q = MaxAbsRMSNorm(dim, eps, axis=-3, floor="add")
        self.norm_k = MaxAbsRMSNorm(dim, eps, axis=-3, floor="add")
        self.to_out = conv1x1(dim, dim)
        # Additive floors: the Neural Engine compiler rejects clamp/reciprocal of an amax here.
        self.register_buffer("tiny", torch.tensor(_FP16_TINY), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, n = x.shape
        q = F.relu(self.norm_q(self.to_q(x))).reshape(b, self.heads, -1, n)
        q = q / (q.amax(2, keepdim=True) + self.tiny)
        k = F.relu(self.norm_k(self.to_k(x))).reshape(b, self.heads, -1, n)
        v = self.to_v(x).reshape(b, self.heads, -1, n)
        out = linear_attention(q, k, v, self.tiny)
        return self.to_out(out.reshape(b, c, 1, n))


class CrossAttention(nn.Module):
    """Softmax cross-attention to the caption tokens.

    Queries are ``(1, C, 1, N)``, keys ``(1, C, 1, L)``.
    """

    def __init__(self, dim: int, heads: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = conv1x1(dim, dim)
        self.to_k = conv1x1(dim, dim)
        self.to_v = conv1x1(dim, dim)
        # RMSNorm(s * x) with eps * s^2 is RMSNorm(x), for ``to_q`` scaled by s = _Q_SCALE.
        self.norm_q = MaxAbsRMSNorm(dim, eps * _Q_SCALE**2, axis=-3, floor="add")
        self.norm_k = MaxAbsRMSNorm(dim, eps, axis=-3, floor="add")
        self.to_out = conv1x1(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, c, _, n = x.shape
        q = self.norm_q(self.to_q(x)).reshape(b, self.heads, -1, n)
        k = self.norm_k(self.to_k(ctx)).reshape(b, self.heads, -1, ctx.shape[-1])
        v = self.to_v(ctx).reshape(b, self.heads, -1, ctx.shape[-1])
        scores = torch.einsum("bhdk,bhdq->bhkq", k, q) + bias  # bias: (1, 1, L, 1)
        attn = torch.softmax(scores, dim=2)
        out = torch.einsum("bhkq,bhdk->bhdq", attn, v)
        return self.to_out(out.reshape(b, c, 1, n))


class GLUMBConv(nn.Module):
    """1x1 -> SiLU -> depthwise 3x3 -> gated SiLU -> 1x1, on the ``(1, C, H, W)`` token grid."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.conv_inverted = nn.Conv2d(dim, 2 * hidden, 1)
        self.conv_depth = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.conv_point = nn.Conv2d(hidden, dim, 1, bias=False)
        self.silu = TanhSiLU()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, c, _, n = x.shape
        h = self.silu(self.conv_inverted(x)).reshape(b, -1, height, width)
        h, gate = self.conv_depth(h).reshape(b, -1, 1, n).chunk(2, dim=1)
        return self.conv_point(h * self.silu(gate))


class SanaBlock(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.norm1 = LayerNormNoAffine(cfg.norm_eps)
        self.attn1 = LinearAttention(dim, cfg.num_attention_heads, 1e-5)
        self.attn2 = CrossAttention(dim, cfg.num_cross_attention_heads, 1e-5)
        self.norm2 = LayerNormNoAffine(cfg.norm_eps)
        self.ff = GLUMBConv(dim, int(cfg.mlp_ratio * dim))
        # (6, C, 1, 1): shift / scale+1 / gate for attention, then for the feed-forward.
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        bias: torch.Tensor,
        t: torch.Tensor,
        hw: tuple[int, int],
    ) -> torch.Tensor:
        mods = self.scale_shift_table + t  # t: (6, C, 1, 1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=0)
        x = x + gate_msa * self.attn1(self.norm1(x) * scale_msa + shift_msa)
        x = x + self.attn2(x, ctx, bias)
        return x + gate_mlp * self.ff(self.norm2(x) * scale_mlp + shift_mlp, *hw)


class SanaTransformer(nn.Module):
    """``(latent [1,Cin,H,W], text [1,L,Dtxt], text mask [1,L], σ [1], guidance [1]) -> velocity``.

    ``guidance`` is the raw scale; ``guidance_embeds_scale`` is folded into its frequency table.
    """

    one: torch.Tensor
    mask_bias: torch.Tensor

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        assert cfg.patch_size == 1 and cfg.guidance_embeds and cfg.interpolation_scale is None
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.dim = dim
        # The token count the linear-attention K scale is folded for.
        self.n_tokens = cfg.sample_size * cfg.sample_size
        self.patch_embed = conv1x1(cfg.in_channels, dim)
        self.time_embedder = TimestepEmbedding(dim)
        self.guidance_embedder = TimestepEmbedding(dim, freq_scale=cfg.guidance_embeds_scale)
        self.time_linear = conv1x1(dim, 6 * dim)
        self.caption_linear_1 = conv1x1(cfg.caption_channels, dim)
        self.caption_linear_2 = conv1x1(dim, dim)
        self.caption_norm = MaxAbsRMSNorm(dim, 1e-5, axis=-3, floor="add")
        self.blocks = nn.ModuleList([SanaBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_out = LayerNormNoAffine(1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, dim, 1, 1))
        self.proj_out = conv1x1(dim, cfg.out_channels)
        self.silu = TanhSiLU()
        self.register_buffer("one", torch.ones(1))
        self.register_buffer("mask_bias", torch.tensor(_MASK_BIAS))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        b, cin, height, width = hidden_states.shape
        n = height * width
        assert n == self.n_tokens, f"built for {self.n_tokens} latent tokens, got {n}"
        x = self.patch_embed(hidden_states.reshape(b, cin, 1, n))

        # Float32 after `to_neural_engine_dtypes`: the conditioning modulates every block, so it
        # runs in float32, off the Neural Engine.
        cond = self.time_embedder(timestep) + self.guidance_embedder(guidance)  # (1, C, 1, 1)
        t = self.time_linear(self.silu(cond)).reshape(6, self.dim, 1, 1).to(x.dtype)
        cond = cond.to(x.dtype)

        ctx = encoder_hidden_states.transpose(1, 2).unsqueeze(2)  # (1, Dtxt, 1, L)
        ctx = self.caption_linear_2(gelu_ane(self.caption_linear_1(ctx)))
        ctx = self.caption_norm(ctx)
        bias = ((self.one - encoder_attention_mask) * self.mask_bias).reshape(1, 1, -1, 1)

        for block in self.blocks:
            x = block(x, ctx, bias, t, (height, width))

        shift, scale = (self.scale_shift_table + cond).chunk(2, dim=0)
        x = self.norm_out(x) * scale + shift
        return self.proj_out(x).reshape(b, -1, height, width)

    def to_neural_engine_dtypes(self) -> SanaTransformer:
        """Float16, except the float32 timestep/guidance conditioning and its inputs."""
        self.half()
        for module in (self.time_embedder, self.guidance_embedder, self.time_linear, self.silu):
            module.float()
        return self

    @classmethod
    def from_source_model(cls, src: Any) -> SanaTransformer:
        """Build from a diffusers ``SanaTransformer2DModel`` (any dtype); folds run in float32."""
        cfg = src.config
        m = cls(cfg).float()
        dim = m.dim
        with torch.no_grad():
            m.patch_embed.load_state_dict(src.patch_embed.proj.float().state_dict())
            te = src.time_embed
            for embedder, emb in (
                (m.time_embedder, te.timestep_embedder),
                (m.guidance_embedder, te.guidance_embedder),
            ):
                _load_linear(embedder.linear_1, emb.linear_1)
                _load_linear(embedder.linear_2, emb.linear_2)
            _load_linear(m.time_linear, te.linear)
            _load_linear(m.caption_linear_1, src.caption_projection.linear_1)
            _load_linear(m.caption_linear_2, src.caption_projection.linear_2)
            m.caption_norm.weight.copy_(src.caption_norm.weight.float().reshape(-1, 1, 1))
            for block, blk in zip(m.blocks, src.transformer_blocks, strict=True):
                dst = cast(SanaBlock, block)
                a1, a2 = blk.attn1, blk.attn2
                for name in ("to_q", "to_k", "to_v"):
                    _load_linear(getattr(dst.attn1, name), getattr(a1, name))
                    _load_linear(getattr(dst.attn2, name), getattr(a2, name))
                dst.attn2.to_q.weight.mul_(_Q_SCALE)
                dst.attn2.to_q.bias.mul_(_Q_SCALE)
                _load_linear(dst.attn1.to_out, a1.to_out[0])
                _load_linear(dst.attn2.to_out, a2.to_out[0])
                dst.attn1.norm_q.weight.copy_(a1.norm_q.weight.float().reshape(-1, 1, 1))
                k_scale = _K_SCALE / m.n_tokens
                dst.attn1.norm_k.weight.copy_(
                    (a1.norm_k.weight.float() * k_scale).reshape(-1, 1, 1)
                )
                head_dim = dim // cfg.num_cross_attention_heads
                q_weight = a2.norm_q.weight.float() / math.sqrt(head_dim)
                dst.attn2.norm_q.weight.copy_(q_weight.reshape(-1, 1, 1))
                dst.attn2.norm_k.weight.copy_(a2.norm_k.weight.float().reshape(-1, 1, 1))
                for name in ("conv_inverted", "conv_depth", "conv_point"):
                    src_conv = getattr(blk.ff, name).float()
                    getattr(dst.ff, name).load_state_dict(src_conv.state_dict())
                table = blk.scale_shift_table.float().clone()
                table[[1, 4]] += 1.0
                dst.scale_shift_table.copy_(table.reshape(6, dim, 1, 1))
            table = src.scale_shift_table.float().clone()
            table[1] += 1.0
            m.scale_shift_table.copy_(table.reshape(2, dim, 1, 1))
            _load_linear(m.proj_out, src.proj_out)
        return m.eval()
