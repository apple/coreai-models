# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""2D axial Rotary Position Embedding (BC1S layout).

SAM3's image encoder uses 2D axial RoPE with pairwise rotation (not the
half-rotation used by most LLMs). This module precomputes cos/sin
buffers in BC1S so they apply directly via element-wise multiply on the
accelerator. Pairwise rotation is implemented with precomputed swap
index/sign buffers so every intermediate stays at rank 4.
"""

import torch
import torch.nn as nn


def rotate_pairwise_bc1s(x: torch.Tensor) -> torch.Tensor:
    """Pairwise rotation on dim=1 (channels) in BC1S layout.

    Reference implementation; produces rank-5 intermediates so it is unsuitable
    for export. ``AxialRoPE2DReauthored`` provides the rank-4-safe path used
    at runtime.
    """
    B, C, one, S = x.shape
    x_pairs = x.reshape(B, C // 2, 2, one, S)
    x1 = x_pairs[:, :, 0:1, :, :]
    x2 = x_pairs[:, :, 1:2, :, :]
    rotated = torch.cat((-x2, x1), dim=2)
    return rotated.reshape(B, C, one, S)


class AxialRoPE2DReauthored(nn.Module):
    """2D axial RoPE in BC1S, fixed grid size, with rank-4 pairwise rotation.

    When ``num_heads > 1`` the buffers are tiled to
    ``(1, num_heads * head_dim, 1, seq_len)`` so the multi-head application is
    a single broadcasted multiply (no per-head Python loop).
    """

    def __init__(
        self,
        head_dim: int,
        grid_h: int,
        grid_w: int,
        num_heads: int = 1,
        rope_theta: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for axial RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.rope_theta = rope_theta
        self.scale = scale

        self._compute_buffers()

    def _compute_buffers(self) -> None:
        dim = self.head_dim
        total_dim = dim * self.num_heads
        seq_len = self.grid_h * self.grid_w

        # Frequency basis: arange(0, dim, 4)[:dim//4]
        freq_indices = torch.arange(0, dim, 4, dtype=torch.float32)[: (dim // 4)]
        freqs = 1.0 / (self.rope_theta ** (freq_indices / dim))

        flattened_indices = torch.arange(seq_len, dtype=torch.long)
        x_positions = (flattened_indices % self.grid_w).float() * self.scale
        y_positions = (
            torch.div(flattened_indices, self.grid_w, rounding_mode="floor").float() * self.scale
        )

        freqs_x = torch.outer(x_positions, freqs)
        freqs_y = torch.outer(y_positions, freqs)
        inv_freq = torch.cat([freqs_x, freqs_y], dim=-1).repeat_interleave(2, dim=-1)

        cos_buf = inv_freq.cos()
        sin_buf = inv_freq.sin()

        cos_bc1s = cos_buf.T.unsqueeze(0).unsqueeze(2)
        sin_bc1s = sin_buf.T.unsqueeze(0).unsqueeze(2)

        if self.num_heads > 1:
            cos_bc1s = cos_bc1s.repeat(1, self.num_heads, 1, 1)
            sin_bc1s = sin_bc1s.repeat(1, self.num_heads, 1, 1)

        self.register_buffer("cos_cached", cos_bc1s.half(), persistent=False)
        self.register_buffer("sin_cached", sin_bc1s.half(), persistent=False)

        # Pairwise swap buffers — keep rotation rank-4.
        swap_idx = torch.arange(total_dim)
        swap_idx[0::2] = torch.arange(1, total_dim, 2)
        swap_idx[1::2] = torch.arange(0, total_dim, 2)
        self.register_buffer("_swap_idx", swap_idx, persistent=False)

        swap_sign = torch.ones(1, total_dim, 1, 1, dtype=torch.float16)
        swap_sign[:, 0::2, :, :] = -1.0
        self.register_buffer("_swap_sign", swap_sign, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached.to(dtype=x.dtype)
        sin = self.sin_cached.to(dtype=x.dtype)
        sign = self._swap_sign.to(dtype=x.dtype)

        x_rotated = sign * x[:, self._swap_idx, :, :]
        return x * cos + x_rotated * sin
