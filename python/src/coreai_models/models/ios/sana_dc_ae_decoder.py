# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine float16 DC-AE decoder (Sana's ``AutoencoderDC.decoder``), NCHW.

Same input and output as diffusers' ``Decoder``: ``z [1, 32, h, w] -> image [1, 3, 32h, 32w]``.
Exact rewrites, all checked against diffusers in float32:

- ``conv3x3(nearest2x(x))`` is a ``ConvTranspose2d(k=4, s=2, p=1)`` with merged taps: no 4x
  upsampled intermediate and 2.25x fewer multiply-adds.
- The upsample shortcut ``pixel_shuffle(repeat_interleave(x))`` and the input shortcut
  ``repeat_interleave(x)`` are fixed 0/1 (transposed) convolutions; ``pixel_shuffle`` decomposes
  to a rank-6 tensor the Neural Engine rejects.
- The multiscale branch's grouped 1x1 conv is a dense block-diagonal one; the Neural Engine
  compiler rejects the grouped form.
- Channel RMSNorms normalize over dim 1 in place of diffusers' ``movedim`` round trips, max-abs
  because ``Σx²`` overflows float16 in every norm.
- Linear attention uses the transformer's rewrite: per-token ``q`` max normalization and a ``K``
  scale folded into the fused qkv weights.
- SiLU is :class:`~coreai_models.primitives.ios.silu.TanhSiLU`.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm
from coreai_models.primitives.ios.silu import TanhSiLU

_FP16_TINY = torch.finfo(torch.float16).tiny
# Linear attention K multiplier over N tokens, as in ``sana_transformer``. With it the decoder has
# no float16 overflow on the Neural Engine over the reference prompts (63.9 dB vs float32).
_K_SCALE = 32.0


class ChannelRMSNorm(nn.Module):
    """Affine RMSNorm with bias over the channels of ``(B, C, H, W)``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = MaxAbsRMSNorm(dim, eps, axis=-3, floor="add")
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias

    def load(self, src: Any) -> None:
        self.norm.weight.data.copy_(src.weight.detach().float().reshape(-1, 1, 1))
        self.bias.data.copy_(src.bias.detach().float().reshape(-1, 1, 1))


def _merged_upsample_conv(conv: nn.Conv2d) -> nn.ConvTranspose2d:
    """``ConvTranspose2d(k=4, s=2, p=1)`` equal to ``conv(nearest2x(x))`` for a 3x3, pad-1 conv."""
    w = conv.weight.detach().float()  # (O, I, 3, 3)
    # Per axis, output 2y takes w0 h[y-1] + (w1+w2) h[y], and output 2y+1 takes
    # (w0+w1) h[y] + w2 h[y+1].
    taps = torch.zeros(4, 3)
    taps[0, 2] = taps[1, 1] = taps[1, 2] = taps[2, 0] = taps[2, 1] = taps[3, 0] = 1.0
    merged = torch.einsum("ka,lb,oiab->iokl", taps, taps, w)  # (I, O, 4, 4)
    up = nn.ConvTranspose2d(conv.in_channels, conv.out_channels, 4, stride=2, padding=1)
    up.weight.data.copy_(merged)
    assert conv.bias is not None and up.bias is not None
    up.bias.data.copy_(conv.bias.detach().float())
    return up


def _shuffle_shortcut(cin: int, cout: int) -> nn.ConvTranspose2d:
    """0/1 ``ConvTranspose2d(k=2, s=2)`` equal to ``pixel_shuffle(repeat_interleave(x, r), 2)``.

    Output channel ``c`` at sub-pixel ``(i, j)`` reads input channel ``(4c + 2i + j) // r``.
    """
    r = cout * 4 // cin
    up = nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False)
    w = torch.zeros(cin, cout, 2, 2)
    for c in range(cout):
        for i in range(2):
            for j in range(2):
                w[(4 * c + 2 * i + j) // r, c, i, j] = 1.0
    up.weight.data.copy_(w)
    up.weight.requires_grad_(False)
    return up


def _dense_block_diagonal(conv: nn.Conv2d) -> nn.Conv2d:
    """A grouped 1x1 conv as a dense one with block-diagonal weights.

    The Neural Engine compiler rejects the grouped conv, and its batched-matmul form after the
    depthwise 5x5; the dense form costs ~7% of the decoder's FLOPs.
    """
    c, g = conv.out_channels, conv.groups
    dense = nn.Conv2d(conv.in_channels, c, 1, bias=False)
    w = torch.zeros(c, conv.in_channels, 1, 1)
    step = c // g
    for i in range(g):
        s = slice(i * step, (i + 1) * step)
        w[s, s] = conv.weight.detach().float()[s]
    dense.weight.data.copy_(w)
    return dense


class UpBlock(nn.Module):
    """``DCUpBlock2d(interpolate=True, shortcut=True)``."""

    def __init__(self, src: Any) -> None:
        super().__init__()
        self.conv = _merged_upsample_conv(src.conv)
        self.shortcut = _shuffle_shortcut(src.conv.in_channels, src.conv.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class ResBlock(nn.Module):
    def __init__(self, src: Any) -> None:
        super().__init__()
        self.conv1 = src.conv1
        self.conv2 = src.conv2
        self.silu = TanhSiLU()
        self.norm = ChannelRMSNorm(src.conv2.out_channels, src.norm.eps)
        self.norm.load(src.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.conv2(self.silu(self.conv1(x))))


class MultiscaleLinearAttention(nn.Module):
    """``SanaMultiscaleLinearAttention`` (ReLU kernel, one multiscale branch, residual)."""

    tiny: torch.Tensor

    def __init__(self, src: Any, n_tokens: int) -> None:
        super().__init__()
        assert len(src.to_qkv_multiscale) == 1 and src.residual_connection
        c = src.to_q.in_features
        self.head_dim = src.attention_head_dim
        self.qkv = nn.Conv2d(c, 3 * src.to_q.out_features, 1, bias=False)
        w = torch.cat([src.to_q.weight, src.to_k.weight, src.to_v.weight]).detach().float()
        # Head groups are consecutive [q, k, v] chunks of head_dim channels across the concatenated
        # qkv; scale every k chunk. The multiscale branch is per chunk, so it inherits the scale.
        chunk = torch.arange(w.shape[0]) // self.head_dim
        w[chunk % 3 == 1] *= _K_SCALE / n_tokens
        self.qkv.weight.data.copy_(w[:, :, None, None])
        ms = src.to_qkv_multiscale[0]
        self.proj_in = ms.proj_in
        self.proj_out = _dense_block_diagonal(ms.proj_out)
        self.to_out = nn.Conv2d(src.to_out.in_features, src.to_out.out_features, 1, bias=False)
        self.to_out.weight.data.copy_(src.to_out.weight.detach().float()[:, :, None, None])
        self.norm_out = ChannelRMSNorm(src.to_out.out_features, src.norm_out.eps)
        self.norm_out.load(src.norm_out)
        self.register_buffer("tiny", torch.tensor(_FP16_TINY), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.qkv(x)
        qkv = torch.cat([qkv, self.proj_out(self.proj_in(qkv))], dim=1)
        qkv = qkv.reshape(b, -1, 3 * self.head_dim, h * w)
        q, k, v = qkv.chunk(3, dim=2)
        q, k = F.relu(q), F.relu(k)
        q = q / (q.amax(2, keepdim=True) + self.tiny)
        kv = torch.einsum("bhdn,bhen->bhde", v, k)
        num = torch.einsum("bhde,bhen->bhdn", kv, q)
        den = (k.sum(-1, keepdim=True) * q).sum(-2, keepdim=True)
        out = (num / (den + self.tiny)).reshape(b, -1, h, w)
        return x + self.norm_out(self.to_out(out))


class GLUMBConv(nn.Module):
    """``GLUMBConv(norm_type="rms_norm", residual_connection=True)``."""

    def __init__(self, src: Any) -> None:
        super().__init__()
        self.conv_inverted = src.conv_inverted
        self.conv_depth = src.conv_depth
        self.conv_point = src.conv_point
        self.silu = TanhSiLU()
        self.norm = ChannelRMSNorm(src.conv_point.out_channels, src.norm.eps)
        self.norm.load(src.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, gate = self.conv_depth(self.silu(self.conv_inverted(x))).chunk(2, dim=1)
        return x + self.norm(self.conv_point(h * self.silu(gate)))


class EfficientViTBlock(nn.Module):
    def __init__(self, src: Any, n_tokens: int) -> None:
        super().__init__()
        self.attn = MultiscaleLinearAttention(src.attn, n_tokens)
        self.conv_out = GLUMBConv(src.conv_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.attn(x))


class SanaDCAEDecoder(nn.Module):
    """``z [1, Cz, h, w] -> image [1, 3, 32h, 32w]`` for a fixed latent size."""

    def __init__(self, src: Any, latent_size: int) -> None:
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_dc import DCUpBlock2d
        from diffusers.models.autoencoders.autoencoder_dc import EfficientViTBlock as SrcEViT
        from diffusers.models.autoencoders.autoencoder_dc import ResBlock as SrcResBlock

        assert src.in_shortcut
        cz, cin = src.conv_in.in_channels, src.conv_in.out_channels
        self.conv_in = src.conv_in
        self.in_shortcut = nn.Conv2d(cz, cin, 1, bias=False)
        w = torch.zeros(cin, cz, 1, 1)
        w[torch.arange(cin), torch.arange(cin) // src.in_shortcut_repeats] = 1.0
        self.in_shortcut.weight.data.copy_(w)
        self.in_shortcut.weight.requires_grad_(False)

        blocks: list[nn.Module] = []
        size = latent_size
        for up_block in reversed(src.up_blocks):
            for m in up_block:
                if isinstance(m, DCUpBlock2d):
                    assert m.interpolate and m.shortcut
                    blocks.append(UpBlock(m))
                    size *= 2
                elif isinstance(m, SrcResBlock):
                    blocks.append(ResBlock(m))
                elif isinstance(m, SrcEViT):
                    blocks.append(EfficientViTBlock(m, size * size))
                else:
                    raise TypeError(type(m))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = ChannelRMSNorm(src.norm_out.weight.shape[0], src.norm_out.eps)
        self.norm_out.load(src.norm_out)
        self.conv_out = src.conv_out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z) + self.in_shortcut(z)
        x = self.blocks(x)
        return self.conv_out(F.relu(self.norm_out(x)))

    @classmethod
    def from_source_model(cls, decoder: Any, latent_size: int = 32) -> SanaDCAEDecoder:
        """Build from diffusers' ``AutoencoderDC.decoder`` (any dtype); folds run in float32."""
        return cls(copy.deepcopy(decoder).float(), latent_size).float().eval()
