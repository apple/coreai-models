# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 EmbeddingGemma 2 text encoder, channels-first (BC1S) conv layout.

The macOS encoder math as 1x1 Conv2d projections with static shapes and max-abs RMSNorms.
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
from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm, MaxAbsRMSNormNoScale
from coreai_models.primitives.ios.sdpa import SDPA

# Smallest positive normal float16; floor for the row max-abs and squared norm in the output L2
# normalization.
_FP16_TINY = 6.103515625e-05
MASK_NEG = -30000.0


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
        self.q_norm = MaxAbsRMSNorm(hd, eps, axis=-2)
        self.k_norm = MaxAbsRMSNorm(hd, eps, axis=-2)
        self.v_norm = MaxAbsRMSNormNoScale(eps, axis=-2)
        self.sdpa = SDPA(head_dim=hd, scale=1.0)

    def forward(self, x, cos, sin, bias):
        """``x``: ``(B, C, 1, L)``; ``cos``/``sin``: ``(1, 1, hd, L)``; ``bias``: additive
        ``(1, L, 1, L)``. Returns ``(B, C, 1, L)``."""
        b, _, _, length = x.shape
        hd, half = self.head_dim, self.head_dim // 2
        q = self.q_norm(self.q_proj(x).reshape(b, self.n_heads, hd, length))
        k = self.k_norm(self.k_proj(x).reshape(b, self.n_kv, hd, length))
        v = self.v_norm(self.v_proj(x).reshape(b, self.n_kv, hd, length))
        q = q * cos + torch.cat((-q[:, :, half:], q[:, :, :half]), dim=2) * sin
        k = k * cos + torch.cat((-k[:, :, half:], k[:, :, :half]), dim=2) * sin
        q, k, v = (t.reshape(b, -1, 1, length) for t in (q, k, v))
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
        self.input_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.post_attention_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.pre_feedforward_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.post_feedforward_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.post_per_layer_input_norm = MaxAbsRMSNorm(dim, eps, axis=-3)
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
    """``(inputs_embeds [B, L, H], bias [1, L, 1, L], pool_weights [B, L]) -> embedding [B, E]``.

    ``H`` is ``hidden_size`` and ``E`` is ``embedding_dim``. ``inputs_embeds`` are token
    embeddings scaled by ``sqrt(hidden_size)``. ``bias`` is additive (0 or ``MASK_NEG``);
    ``sliding_bias`` optionally replaces it on sliding layers. ``pool_weights`` is the attention
    mask divided by its row sum, so pooling is a masked mean. The output has unit L2 norm.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig) -> None:
        super().__init__()
        self.config = config
        dim, eps, L = config.hidden_size, config.rms_norm_eps, config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input
        self.per_layer_model_projection = conv1x1(dim, L * self.ple_dim)
        self.per_layer_projection_norm = MaxAbsRMSNorm(self.ple_dim, eps, axis=-2)
        self.layers = nn.ModuleList([EmbeddingGemma2EncoderLayer(config, i) for i in range(L)])
        self.norm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.embedding_projection = conv1x1(dim, config.embedding_dim)

    def _cos_sin(self, sliding: bool, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``[1, 1, head_dim, L]`` in ``x``'s dtype for the sliding or full layers."""
        head_dim = self.config.rope_head_dim(sliding)
        theta = self.config.rope_theta_sliding if sliding else self.config.rope_theta_full
        length = x.shape[-1]
        inv_freq = rope_inv_freq(head_dim, theta, x.device)[:, None]
        freqs = inv_freq * torch.arange(length, dtype=torch.float32, device=x.device)[None, :]
        emb = torch.cat([freqs, freqs], 0).reshape(1, 1, head_dim, length)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

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
        ropes = {sliding: self._cos_sin(sliding, x) for sliding in (True, False)}
        sb = bias if sliding_bias is None else sliding_bias
        for i, layer in enumerate(self.layers):
            cos, sin = ropes[layer.is_sliding]
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

    Returns ``(encoder, embed_table_fp32)``. The table is the unscaled ``[vocab, hidden_size]``
    token embedding, not part of the encoder.
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
    "EmbeddingGemma2TextEncoder",
    "conv1x1",
    "load_conv_weights",
    "load_embedding_gemma2_text_encoder",
    "text_encoder_from_state_dict",
]
