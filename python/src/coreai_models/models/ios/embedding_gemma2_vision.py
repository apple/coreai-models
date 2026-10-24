# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 EmbeddingGemma 2 image encoder, channels-first (BC1S) conv layout.

What: the vision tower as 1x1 Conv2d projections with a static patch budget and in-graph lookups.
Why: ScaledRMSNorm divides by the row max-abs before squaring, keeping float16 in range.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from coreai_models.models.ios.embedding_gemma2 import MASK_NEG, ScaledRMSNorm, _conv
from coreai_models.models.macos.embedding_gemma2_vision import (
    EmbeddingGemma2VisionConfig,
    axial_inv_freq,
)


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
        self.q_proj = _conv(dim, self.n_heads * hd)
        self.k_proj = _conv(dim, self.n_heads * hd)
        self.v_proj = _conv(dim, self.n_heads * hd)
        self.o_proj = _conv(self.n_heads * hd, dim)
        self.q_norm = ScaledRMSNorm(hd, eps, dim=2)
        self.k_norm = ScaledRMSNorm(hd, eps, dim=2)
        self.v_norm = ScaledRMSNorm(hd, eps, with_scale=False, dim=2)

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
        self.gate_proj = _conv(dim, cfg.intermediate_size)
        self.up_proj = _conv(dim, cfg.intermediate_size)
        self.down_proj = _conv(cfg.intermediate_size, dim)
        self.input_layernorm = ScaledRMSNorm(dim, eps)
        self.post_attention_layernorm = ScaledRMSNorm(dim, eps)
        self.pre_feedforward_layernorm = ScaledRMSNorm(dim, eps)
        self.post_feedforward_layernorm = ScaledRMSNorm(dim, eps)

    def forward(self, x, cos, sin, bias):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cos, sin, bias)
        )
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        return x + self.post_feedforward_layernorm(h)


class EmbeddingGemma2ImageEncoder(nn.Module):
    """``pixel_values [1, P, 768]`` + ``position_ids [1, P, 2]`` (float16) -> ``[1, L, 512]``."""

    def __init__(self, config: EmbeddingGemma2VisionConfig | None = None):
        super().__init__()
        self.config = cfg = config or EmbeddingGemma2VisionConfig()
        k = cfg.pooling_kernel_size
        self.grid = g = cfg.max_soft_tokens * k  # largest reachable patch coordinate + 1
        dim = cfg.hidden_size
        self.input_proj = _conv(cfg.patch_dim, dim)
        self.layers = nn.ModuleList([VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.embedding_pre_projection_norm = ScaledRMSNorm(dim, cfg.rms_norm_eps, with_scale=False)
        self.embedding_projection = _conv(dim, cfg.text_hidden_size)
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
        cfg = self.config
        g = self.grid
        table = position_embedding_table.float()
        if table.shape[1] < g:
            raise ValueError(f"position table has {table.shape[1]} rows; budget needs {g}")
        self.pos_x.copy_(table[0, :g])
        self.pos_y.copy_(table[1, :g])
        inv = axial_inv_freq(cfg)
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

        # Additive key-padding bias (1, Pk, 1, 1), broadcast over heads and queries.
        bias = ((valid - 1.0) * -MASK_NEG).reshape(b, p, 1, 1)
        for layer in self.layers:
            x = layer(x, cos, sin, bias)

        # 3x3 pooling as one matmul with exact one-hot slot weights (sum; the scale cancels in
        # the scale-free norm below, like the dropped x sqrt(hidden)).
        bx = ox @ self.block_tab.to(dt)  # [B, P, 1]
        by = oy @ self.block_tab.to(dt)
        max_x = pos[..., 0:1].amax(1, keepdim=True)  # [B, 1, 1]
        blocks_per_row = one_hot_f16(max_x, self.grid_ar.to(dt)) @ self.width_tab.to(dt)
        slot = bx + blocks_per_row * by
        w = one_hot_f16(slot, self.slot_ar.to(dt)) * valid  # [B, P, L]
        pooled = torch.matmul(x.squeeze(2), w)  # (B, C, L)
        y = self.embedding_pre_projection_norm(pooled.unsqueeze(2))  # (B, C, 1, L)
        y = self.embedding_projection(y)  # (B, 512, 1, L)
        return y.squeeze(2).transpose(1, 2).contiguous()  # [B, L, 512]


def load_image_encoder(
    model_dir: str,
    dtype: torch.dtype = torch.float16,
    max_soft_tokens: int | None = None,
) -> EmbeddingGemma2ImageEncoder:
    """Load from a local EG2 snapshot directory (``config.json`` + ``*.safetensors``)."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = EmbeddingGemma2VisionConfig.from_hf_config(json.load(f), max_soft_tokens)
    model = EmbeddingGemma2ImageEncoder(cfg)
    sd: dict[str, torch.Tensor] = {}
    for fn in sorted(x for x in os.listdir(model_dir) if x.endswith(".safetensors")):
        with safe_open(os.path.join(model_dir, fn), framework="pt") as h:
            for key in h.keys():  # noqa: SIM118
                if key.startswith(("vision_tower.", "embed_vision.")):
                    sd[key] = h.get_tensor(key).float()
    load_hf_state_dict(model, sd)
    return model.to(dtype).eval()


def load_hf_state_dict(
    model: EmbeddingGemma2ImageEncoder, sd: dict[str, torch.Tensor]
) -> EmbeddingGemma2ImageEncoder:
    """Load checkpoint-named ``vision_tower.*`` / ``embed_vision.*`` tensors (fail closed)."""
    mapped: dict[str, torch.Tensor] = {}
    for key, v in sd.items():
        k2 = key.replace(".linear.weight", ".weight")
        k2 = k2.removeprefix("vision_tower.").removeprefix("embed_vision.")
        k2 = k2.replace("patch_embedder.input_proj", "input_proj")
        k2 = k2.replace("encoder.layers.", "layers.").replace("mlp.", "")
        mapped[k2] = v.float()
    pos_table = mapped.pop("patch_embedder.position_embedding_table")
    own = model.state_dict()
    for key, v in list(mapped.items()):
        if key not in own:
            raise RuntimeError(f"unexpected key {key}")
        mapped[key] = v.reshape(own[key].shape)
    missing = set(own) - set(mapped)
    if missing:
        raise RuntimeError(f"missing keys {sorted(missing)[:10]}")
    model.load_state_dict(mapped, strict=True)
    model.bake_tables(pos_table)
    return model


__all__ = [
    "EmbeddingGemma2ImageEncoder",
    "load_hf_state_dict",
    "load_image_encoder",
    "one_hot_f16",
]
