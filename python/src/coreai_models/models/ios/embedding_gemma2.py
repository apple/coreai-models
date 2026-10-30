# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 EmbeddingGemma 2 text encoder, channels-first (BC1S) conv layout.

What: the macOS encoder math as 1x1 Conv2d projections with static shapes.
Why: ScaledRMSNorm divides by the row max-abs before squaring, keeping float16 in range.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
    rope_inv_freq,
)
from coreai_models.models.macos.embedding_gemma2 import (
    load_embedding_gemma2_text_encoder as load_macos_text_encoder,
)
from coreai_models.primitives.ios.sdpa import SDPA

# Smallest positive normal float16; floors the row max-abs and the squared norm so all-zero rows
# stay finite.
_FP16_TINY = 6.103515625e-05
MASK_NEG = -30000.0


class ScaledRMSNorm(nn.Module):
    """RMSNorm over ``dim`` that normalizes by the row max-abs first, so squares stay at most 1."""

    def __init__(
        self, channels: int, eps: float = 1e-6, with_scale: bool = True, dim: int = 1
    ) -> None:
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.dim = dim
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        m = x.abs().amax(d, keepdim=True).clamp(min=_FP16_TINY)
        r = torch.reciprocal(m)
        xs = x * r
        # eps * r^2 as (sqrt(eps) * r)^2. r goes up to 1 / _FP16_TINY, and r * r overflows
        # float16 once r > 256, which zeroes the row.
        e = self.eps**0.5 * r
        y = xs * torch.rsqrt((xs * xs).mean(d, keepdim=True) + e * e)
        if self.with_scale:
            shape = [1] * x.dim()
            shape[d] = self.channels
            y = y * self.weight.to(y.dtype).reshape(shape)
        return y


def rotate_half_channels(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """NeoX rotate-half over the per-head channel axis of ``(B, H*hd, 1, L)``."""
    b, c, one, length = x.shape
    x = x.reshape(b, c // head_dim, head_dim, length)
    x1, x2 = x[:, :, : head_dim // 2], x[:, :, head_dim // 2 :]
    return torch.cat((-x2, x1), dim=2).reshape(b, c, one, length)


def apply_rope_bc1s(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """Apply rotary embedding; ``cos``/``sin``: ``(1, 1, hd, L)``; ``x``: ``(B, H*hd, 1, L)``."""
    b, c, one, length = x.shape
    xr = x.reshape(b, c // head_dim, head_dim, length)
    out = (
        xr * cos
        + rotate_half_channels(x, head_dim).reshape(b, c // head_dim, head_dim, length) * sin
    )
    return out.reshape(b, c, one, length)


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    """Bias-free 1x1 ``Conv2d``, the BC1S form of a bias-free ``Linear``."""
    return nn.Conv2d(cin, cout, kernel_size=1, bias=False)


class EmbeddingGemma2Attention(nn.Module):
    """Bidirectional multi-head attention over ``(B, C, 1, L)``."""

    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv = config.layer_num_key_value_heads(layer_idx)
        self.head_dim = hd = config.layer_head_dim(layer_idx)
        dim, eps = config.hidden_size, config.rms_norm_eps
        self.q_proj = conv1x1(dim, self.n_heads * hd)
        self.k_proj = conv1x1(dim, self.n_kv * hd)
        self.v_proj = conv1x1(dim, self.n_kv * hd)
        self.o_proj = conv1x1(self.n_heads * hd, dim)
        self.q_norm = ScaledRMSNorm(hd, eps, dim=2)
        self.k_norm = ScaledRMSNorm(hd, eps, dim=2)
        self.v_norm = ScaledRMSNorm(hd, eps, with_scale=False, dim=2)
        self.sdpa = SDPA(head_dim=hd, scale=1.0)

    def _per_head(self, x: torch.Tensor, n: int, norm: nn.Module) -> torch.Tensor:
        b, c, one, length = x.shape
        return norm(x.reshape(b, n, self.head_dim, length)).reshape(b, c, one, length)

    def forward(self, x, cos, sin, bias):
        hd = self.head_dim
        q = self._per_head(self.q_proj(x), self.n_heads, self.q_norm)
        k = self._per_head(self.k_proj(x), self.n_kv, self.k_norm)
        v = self._per_head(self.v_proj(x), self.n_kv, self.v_norm)
        q = apply_rope_bc1s(q, cos, sin, hd)
        k = apply_rope_bc1s(k, cos, sin, hd)
        return self.o_proj(self.sdpa(q, k, v, bias))


class EmbeddingGemma2EncoderLayer(nn.Module):
    """One pre-norm encoder block with the per-layer-input mixing."""

    def __init__(self, config: EmbeddingGemma2TextConfig, i: int) -> None:
        super().__init__()
        dim, eps, ple = config.hidden_size, config.rms_norm_eps, config.hidden_size_per_layer_input
        self.is_sliding = config.is_sliding(i)
        self.self_attn = EmbeddingGemma2Attention(config, i)
        self.gate_proj = conv1x1(dim, config.intermediate_size)
        self.up_proj = conv1x1(dim, config.intermediate_size)
        self.down_proj = conv1x1(config.intermediate_size, dim)
        self.per_layer_input_gate = conv1x1(dim, ple)
        self.per_layer_projection = conv1x1(ple, dim)
        self.input_layernorm = ScaledRMSNorm(dim, eps)
        self.post_attention_layernorm = ScaledRMSNorm(dim, eps)
        self.pre_feedforward_layernorm = ScaledRMSNorm(dim, eps)
        self.post_feedforward_layernorm = ScaledRMSNorm(dim, eps)
        self.post_per_layer_input_norm = ScaledRMSNorm(dim, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, ple_i, cos, sin, bias):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cos, sin, bias)
        )
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        x = x + self.post_feedforward_layernorm(h)
        g = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * ple_i
        x = x + self.post_per_layer_input_norm(self.per_layer_projection(g))
        return x * self.layer_scalar.to(x.dtype)


class EmbeddingGemma2TextEncoder(nn.Module):
    """``(inputs_embeds [B,L,H], bias [1,L,1,L], pool_weights [B,L]) -> embedding [B, E]``.

    ``sliding_bias`` optionally replaces ``bias`` on sliding layers. ``pool_weights`` is the
    attention mask divided by its row sum, so pooling is a masked mean.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig) -> None:
        super().__init__()
        self.config = config
        dim, eps, L = config.hidden_size, config.rms_norm_eps, config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input
        self.per_layer_model_projection = conv1x1(dim, L * self.ple_dim)
        self.per_layer_projection_norm = ScaledRMSNorm(self.ple_dim, eps, dim=2)
        self.layers = nn.ModuleList([EmbeddingGemma2EncoderLayer(config, i) for i in range(L)])
        self.norm = ScaledRMSNorm(dim, eps)
        self.embedding_projection = conv1x1(dim, config.embedding_dim)
        self._rope_hd = {"sliding": config.rope_head_dim(True), "full": config.rope_head_dim(False)}

    def rope(
        self, length: int, dtype: torch.dtype, device
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        out = {}
        for kind, theta in (
            ("sliding", self.config.rope_theta_sliding),
            ("full", self.config.rope_theta_full),
        ):
            hd = self._rope_hd[kind]
            inv = rope_inv_freq(hd, theta, device)
            fr = inv[:, None] * torch.arange(length, dtype=torch.float32, device=device)[None, :]
            e = torch.cat([fr, fr], 0).reshape(1, 1, hd, length)
            out[kind] = (e.cos().to(dtype), e.sin().to(dtype))
        return out

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        bias: torch.Tensor,
        pool_weights: torch.Tensor,
        sliding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, length, dim = inputs_embeds.shape
        x = inputs_embeds.transpose(1, 2).unsqueeze(2)  # (B, H, 1, L)
        nl = self.config.num_hidden_layers
        ple = self.per_layer_model_projection(x) * (dim**-0.5)
        ple = self.per_layer_projection_norm(ple.reshape(b, nl, self.ple_dim, length))
        ple = ple.reshape(b, nl * self.ple_dim, 1, length)
        ropes = self.rope(length, x.dtype, x.device)
        sb = bias if sliding_bias is None else sliding_bias
        for i, layer in enumerate(self.layers):
            cos, sin = ropes["sliding" if layer.is_sliding else "full"]
            ple_i = ple[:, i * self.ple_dim : (i + 1) * self.ple_dim]
            x = layer(x, ple_i, cos, sin, sb if layer.is_sliding else bias)
        x = self.norm(x)  # (B, H, 1, L)
        pooled = torch.matmul(x.squeeze(2), pool_weights.to(x.dtype).unsqueeze(-1))  # (B, H, 1)
        y = self.embedding_projection(pooled.unsqueeze(-1)).reshape(b, -1)  # (B, E)
        m = y.abs().amax(-1, keepdim=True).clamp(min=_FP16_TINY)
        ys = y * torch.reciprocal(m)
        return ys * torch.rsqrt((ys * ys).sum(-1, keepdim=True).clamp(min=_FP16_TINY))


def load_conv_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Strict ``load_state_dict`` with each tensor reshaped to the module's shape (e.g. conv)."""
    own = model.state_dict()
    model.load_state_dict(
        {k: v.reshape(own[k].shape) if k in own else v for k, v in state_dict.items()}
    )


def text_encoder_from_state_dict(
    config: EmbeddingGemma2TextConfig,
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> tuple[EmbeddingGemma2TextEncoder, torch.Tensor]:
    """Build the encoder from HF text weights (``language_model.`` prefix stripped).

    Returns ``(encoder, embed_table_fp32)``; the embedding table stays on the host.
    """
    sd = {k: v.float() for k, v in state_dict.items()}
    table = sd.pop("embed_tokens.weight")
    mapped: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        k2 = k.replace("ple.per_layer_model_projection", "per_layer_model_projection")
        k2 = k2.replace("ple.per_layer_projection_norm", "per_layer_projection_norm")
        k2 = k2.replace("mlp.", "").replace("ple_block.", "")
        mapped[k2] = v
    model = EmbeddingGemma2TextEncoder(config)
    load_conv_weights(model, mapped)
    return model.to(dtype).eval(), table


def load_embedding_gemma2_text_encoder(
    model_dir_or_id: str | Path = "google/embeddinggemma-2",
    dtype: torch.dtype = torch.float16,
) -> tuple[EmbeddingGemma2TextEncoder, torch.Tensor]:
    """Load from a local snapshot dir or HF id; returns ``(encoder, embed_table_fp32)``."""
    mac = load_macos_text_encoder(model_dir_or_id, torch.float32)
    return text_encoder_from_state_dict(mac.config, mac.state_dict(), dtype)


__all__ = [
    "MASK_NEG",
    "ScaledRMSNorm",
    "EmbeddingGemma2TextEncoder",
    "conv1x1",
    "load_conv_weights",
    "load_embedding_gemma2_text_encoder",
    "text_encoder_from_state_dict",
]
