# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 EmbeddingGemma 2 image encoder, channels-first (BC1S) conv layout.

The vision tower as 1x1 Conv2d projections with max-abs RMSNorms, a static patch budget and
in-graph lookups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.ios.embedding_gemma2 import MASK_NEG, conv1x1, load_conv_weights
from coreai_models.models.macos.embedding_gemma2 import rope_inv_freq
from coreai_models.models.macos.embedding_gemma2_vision import (
    EmbeddingGemma2ImageEncoder as MacEmbeddingGemma2ImageEncoder,
)
from coreai_models.models.macos.embedding_gemma2_vision import EmbeddingGemma2VisionConfig
from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm, MaxAbsRMSNormNoScale


def one_hot_f16(coord: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Exact one-hot of integer ``coord [..., 1]`` over ``grid [G]``; padding (-1) gives zeros."""
    return torch.relu(1.0 - (coord - grid).abs())


def rotate_half_axial(x: torch.Tensor) -> torch.Tensor:
    """Rotate-half inside each spatial half of the head dim; ``x``: ``(B, H, hd, P)``."""
    q = x.shape[2] // 4
    x0, x1, x2, x3 = x[:, :, :q], x[:, :, q : 2 * q], x[:, :, 2 * q : 3 * q], x[:, :, 3 * q :]
    return torch.cat((-x1, x0, -x3, x2), dim=2)


class VisionAttention(nn.Module):
    """Head-batched bidirectional attention, computed over query chunks of ``q_chunk`` rows."""

    def __init__(self, cfg: EmbeddingGemma2VisionConfig, q_chunk: int = 630) -> None:
        super().__init__()
        self.q_chunk = q_chunk
        self.n_heads, self.head_dim = cfg.num_attention_heads, cfg.head_dim
        dim, hd, eps = cfg.hidden_size, cfg.head_dim, cfg.rms_norm_eps
        self.q_proj = conv1x1(dim, self.n_heads * hd)
        self.k_proj = conv1x1(dim, self.n_heads * hd)
        self.v_proj = conv1x1(dim, self.n_heads * hd)
        self.o_proj = conv1x1(self.n_heads * hd, dim)
        self.q_norm = MaxAbsRMSNorm(hd, eps, axis=-2)
        self.k_norm = MaxAbsRMSNorm(hd, eps, axis=-2)
        self.v_norm = MaxAbsRMSNormNoScale(eps, axis=-2)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, p = x.shape
        return x.reshape(b, self.n_heads, self.head_dim, p)

    def forward(self, x, cos, sin, bias):
        b, _, _, p = x.shape
        q = self.q_norm(self._heads(self.q_proj(x)))
        k = self.k_norm(self._heads(self.k_proj(x)))
        v = self.v_norm(self._heads(self.v_proj(x)))
        q = q * cos + rotate_half_axial(q) * sin
        k = k * cos + rotate_half_axial(k) * sin
        c = self.n_heads * self.head_dim
        key_bias = bias.reshape(b, 1, 1, p)
        qt, vt = q.transpose(2, 3), v.transpose(2, 3)  # (B, H, P, hd)
        chunks = []
        for s in range(0, p, self.q_chunk):
            scores = qt[:, :, s : s + self.q_chunk] @ k + key_bias  # (B, H, c, P)
            chunks.append(scores.softmax(-1) @ vt)  # (B, H, c, hd)
        out = torch.cat(chunks, 2) if len(chunks) > 1 else chunks[0]
        return self.o_proj(out.transpose(2, 3).reshape(b, c, 1, p))


class VisionEncoderLayer(nn.Module):
    """Pre-norm block: attention and a GeGLU-tanh MLP, each with a post-norm residual."""

    def __init__(self, cfg: EmbeddingGemma2VisionConfig) -> None:
        super().__init__()
        dim, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = VisionAttention(cfg)
        self.gate_proj = conv1x1(dim, cfg.intermediate_size)
        self.up_proj = conv1x1(dim, cfg.intermediate_size)
        self.down_proj = conv1x1(cfg.intermediate_size, dim)
        self.input_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.post_attention_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.pre_feedforward_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)
        self.post_feedforward_layernorm = MaxAbsRMSNorm(dim, eps, axis=-3)

    def forward(self, x, cos, sin, bias):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cos, sin, bias)
        )
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        return x + self.post_feedforward_layernorm(h)


class EmbeddingGemma2ImageEncoder(nn.Module):
    """``(pixel_values [B, P, patch_dim], position_ids [B, P, 2]) -> [B, L, text_hidden_size]``.

    Padding patches have position ``-1``. ``L`` is ``max_soft_tokens``; the image's soft tokens
    fill the first slots and the rest are zero. Inputs are cast to the model dtype.
    """

    def __init__(self, config: EmbeddingGemma2VisionConfig | None = None):
        super().__init__()
        self.config = cfg = config or EmbeddingGemma2VisionConfig()
        k = cfg.pooling_kernel_size
        self.grid = g = cfg.max_soft_tokens * k  # largest reachable patch coordinate + 1
        dim = cfg.hidden_size
        self.input_proj = conv1x1(cfg.patch_dim, dim)
        self.layers = nn.ModuleList([VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.embedding_pre_projection_norm = MaxAbsRMSNormNoScale(cfg.rms_norm_eps, axis=-3)
        self.embedding_projection = conv1x1(dim, cfg.text_hidden_size)
        # Per-axis lookup tables, filled by ``bake_tables`` (fp32 math, stored in model dtype):
        # [G, hidden] position embedding | [G, hd/4] cos | [G, hd/4] sin.
        self.register_buffer("pos_x", torch.zeros(g, dim), persistent=False)
        self.register_buffer("pos_y", torch.zeros(g, dim), persistent=False)
        q = cfg.head_dim // 4
        self.register_buffer("cos_tab", torch.zeros(g, q), persistent=False)
        self.register_buffer("sin_tab", torch.zeros(g, q), persistent=False)
        c = torch.arange(g, dtype=torch.float32)
        self.register_buffer("grid_ar", c, persistent=False)
        self.register_buffer("block_tab", torch.floor(c / k)[:, None], persistent=False)
        self.register_buffer("width_tab", torch.floor((c + 1) / k)[:, None], persistent=False)
        self.register_buffer(
            "slot_ar", torch.arange(cfg.max_soft_tokens, dtype=torch.float32), persistent=False
        )

    @torch.no_grad()
    def bake_tables(self, position_embedding_table: torch.Tensor) -> None:
        """Fill the position tables from ``position_embedding_table [2, N, hidden_size]`` and the
        RoPE tables from the config.

        Raises:
            ValueError: if ``N`` is smaller than ``self.grid``.
        """
        cfg = self.config
        g = self.grid
        table = position_embedding_table.float()
        if table.shape[1] < g:
            raise ValueError(f"position table has {table.shape[1]} rows; budget needs {g}")
        self.pos_x.copy_(table[0, :g])
        self.pos_y.copy_(table[1, :g])
        inv = rope_inv_freq(cfg.head_dim // 2, cfg.rope_theta)
        ang = torch.arange(g, dtype=torch.float32)[:, None] * inv[None, :]
        self.cos_tab.copy_(ang.cos())
        self.sin_tab.copy_(ang.sin())

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dt = self.input_proj.weight.dtype
        b, p, _ = pixel_values.shape
        pos = position_ids.to(dt)
        ox = one_hot_f16(pos[..., 0:1], self.grid_ar.to(dt))  # [B, P, G]
        oy = one_hot_f16(pos[..., 1:2], self.grid_ar.to(dt))
        valid = ox.sum(-1, keepdim=True)  # [B, P, 1]: 1 real, 0 padding

        # Patch embedding + learned 2-D position embedding (zero for padding rows, as in HF).
        x = 2.0 * (pixel_values.to(dt) - 0.5)
        x = self.input_proj(x.transpose(1, 2).unsqueeze(2))  # (B, C, 1, P)
        pe = ox @ self.pos_x.to(dt) + oy @ self.pos_y.to(dt)  # [B, P, C]
        x = x + pe.transpose(1, 2).unsqueeze(2)

        # Axial RoPE tables [fx, fx, fy, fy] -> (B, 1, hd, P); padding rows get cos = sin = 0.
        cx, sx = ox @ self.cos_tab.to(dt), ox @ self.sin_tab.to(dt)
        cy, sy = oy @ self.cos_tab.to(dt), oy @ self.sin_tab.to(dt)
        cos = torch.cat((cx, cx, cy, cy), -1).transpose(1, 2).unsqueeze(1)
        sin = torch.cat((sx, sx, sy, sy), -1).transpose(1, 2).unsqueeze(1)

        # Additive key-padding bias (B, Pk, 1, 1), broadcast over heads and queries.
        bias = ((valid - 1.0) * -MASK_NEG).reshape(b, p, 1, 1)
        for layer in self.layers:
            x = layer(x, cos, sin, bias)

        # ``k``-by-``k`` block pooling as one matmul with exact one-hot slot weights. It sums where
        # HF takes the mean times ``sqrt(hidden_size)``; the scale-free norm below cancels this.
        bx = ox @ self.block_tab.to(dt)  # [B, P, 1]
        by = oy @ self.block_tab.to(dt)
        max_x = pos[..., 0:1].amax(1, keepdim=True)  # [B, 1, 1]
        blocks_per_row = one_hot_f16(max_x, self.grid_ar.to(dt)) @ self.width_tab.to(dt)
        slot = bx + blocks_per_row * by
        w = one_hot_f16(slot, self.slot_ar.to(dt)) * valid  # [B, P, L]
        pooled = torch.matmul(x.squeeze(2), w)  # (B, C, L)
        y = self.embedding_pre_projection_norm(pooled.unsqueeze(2))  # (B, C, 1, L)
        y = self.embedding_projection(y)  # (B, text_hidden_size, 1, L)
        return y.squeeze(2).transpose(1, 2).contiguous()  # [B, L, text_hidden_size]


def load_image_encoder(
    model_dir: str,
    dtype: torch.dtype = torch.float16,
    max_soft_tokens: int | None = None,
) -> EmbeddingGemma2ImageEncoder:
    """Load from a local EG2 snapshot directory (``config.json`` and its safetensors weights)."""
    mac = MacEmbeddingGemma2ImageEncoder.from_pretrained(model_dir, torch.float32, max_soft_tokens)
    model = load_hf_state_dict(EmbeddingGemma2ImageEncoder(mac.config), mac.state_dict())
    return model.to(dtype).eval()


def load_hf_state_dict(
    model: EmbeddingGemma2ImageEncoder, sd: dict[str, torch.Tensor]
) -> EmbeddingGemma2ImageEncoder:
    """Load checkpoint-named (``vision_tower.*`` / ``embed_vision.*``) or macOS-encoder-named
    tensors into ``model``, bake its lookup tables, and return it.

    Raises:
        RuntimeError: on missing, unexpected, or mis-shaped tensors.
        KeyError: if ``patch_embedder.position_embedding_table`` is absent.
        ValueError: if the position table is shorter than ``model.grid``.
    """
    mapped: dict[str, torch.Tensor] = {}
    for key, v in sd.items():
        k2 = key.replace(".linear.weight", ".weight")
        k2 = k2.removeprefix("vision_tower.").removeprefix("embed_vision.")
        k2 = k2.replace("patch_embedder.input_proj", "input_proj")
        k2 = k2.replace("encoder.layers.", "layers.").replace("mlp.", "")
        mapped[k2] = v
    pos_table = mapped.pop("patch_embedder.position_embedding_table")
    load_conv_weights(model, mapped)
    model.bake_tables(pos_table)
    return model


__all__ = [
    "EmbeddingGemma2ImageEncoder",
    "load_hf_state_dict",
    "load_image_encoder",
    "one_hot_f16",
]
