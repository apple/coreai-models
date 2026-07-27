# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored SAM3 image encoder backbone in BC1S layout.

32 transformer layers: 28 window attention (24x24 windows) + 4 global
attention at indices [7, 15, 23, 31]. All intermediates in BC1S
(B, C, 1, S) format. Linear projections replaced with Conv2d(1x1).
GELU approximated with sigmoid.

HF reference: ``Sam3ViTModel`` in
``transformers/models/sam3/modeling_sam3.py``.
"""

import torch
import torch.nn as nn

from coreai_models.models.ios.sam3.primitives.rope import AxialRoPE2DReauthored
from coreai_models.models.ios.sam3.primitives.window import (
    window_partition_ane,
    window_unpartition_ane,
)
from coreai_models.primitives.ios.bidirectional_sdpa import BidirectionalSDPA
from coreai_models.primitives.ios.gelu import gelu_ane
from coreai_models.primitives.ios.layer_norm import LayerNormReauthored

# Constants matching the SAM3 image-encoder config defaults.
_HIDDEN_SIZE = 1024
_NUM_HEADS = 16
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS  # 64
_MLP_DIM = 4736
_WINDOW_SIZE = 24
_GLOBAL_ATTN_INDICES = [7, 15, 23, 31]
_PATCH_SIZE = 14
_IMAGE_SIZE = 1008
_GRID_SIZE = _IMAGE_SIZE // _PATCH_SIZE  # 72
_PRETRAIN_GRID = 24  # pretrain_image_size (336) // patch_size (14)
_LAYER_NORM_EPS = 1e-6
_ROPE_THETA = 10000.0


def _linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    """Convert ``nn.Linear`` to ``nn.Conv2d(1x1)`` for BC1S layout."""
    in_features = linear.in_features
    out_features = linear.out_features
    has_bias = linear.bias is not None
    conv = nn.Conv2d(in_features, out_features, 1, bias=has_bias)
    conv.weight.data = linear.weight.data.reshape(out_features, in_features, 1, 1)
    if has_bias:
        conv.bias.data = linear.bias.data
    return conv


class ImageEncoderAttention(nn.Module):
    """Self-attention with 2D axial RoPE in BC1S layout."""

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        head_dim: int = _HEAD_DIM,
        grid_h: int = _WINDOW_SIZE,
        grid_w: int = _WINDOW_SIZE,
        rope_theta: float = _ROPE_THETA,
        rope_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.k_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.v_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.sdpa = BidirectionalSDPA(num_heads=num_heads, head_dim=head_dim)
        self.rope = AxialRoPE2DReauthored(
            head_dim=head_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            num_heads=num_heads,
            rope_theta=rope_theta,
            scale=rope_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_rotated = self.rope(q)
        k_rotated = self.rope(k)
        attn_out = self.sdpa(q_rotated, k_rotated, v)
        return self.o_proj(attn_out)

    @classmethod
    def from_hf_attention(
        cls,
        hf_attn: nn.Module,
        grid_h: int,
        grid_w: int,
        rope_scale: float = 1.0,
        rope_theta: float = _ROPE_THETA,
    ) -> "ImageEncoderAttention":
        hidden_size = hf_attn.hidden_size
        num_heads = hf_attn.num_attention_heads
        head_dim = hf_attn.head_dim

        ane_attn = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            rope_theta=rope_theta,
            rope_scale=rope_scale,
        )

        ane_attn.q_proj = _linear_to_conv2d(hf_attn.q_proj)
        ane_attn.k_proj = _linear_to_conv2d(hf_attn.k_proj)
        ane_attn.v_proj = _linear_to_conv2d(hf_attn.v_proj)
        ane_attn.o_proj = _linear_to_conv2d(hf_attn.o_proj)
        return ane_attn


class ImageEncoderBlock(nn.Module):
    """Image-encoder transformer block in BC1S layout.

    LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual.
    Window blocks partition / unpartition around the attention call.
    """

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        mlp_dim: int = _MLP_DIM,
        is_global: bool = False,
        grid_h: int = _GRID_SIZE,
        grid_w: int = _GRID_SIZE,
        window_size: int = _WINDOW_SIZE,
        layer_norm_eps: float = _LAYER_NORM_EPS,
        rope_theta: float = _ROPE_THETA,
    ) -> None:
        super().__init__()
        self.is_global = is_global
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.window_size = window_size

        self.layer_norm1 = LayerNormReauthored(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        if is_global:
            attn_grid_h, attn_grid_w = grid_h, grid_w
            rope_scale = window_size / grid_h
        else:
            attn_grid_h, attn_grid_w = window_size, window_size
            rope_scale = 1.0

        self.attention = ImageEncoderAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            grid_h=attn_grid_h,
            grid_w=attn_grid_w,
            rope_theta=rope_theta,
            rope_scale=rope_scale,
        )

        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1, bias=True)
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm1(x)

        if not self.is_global:
            h = window_partition_ane(h, self.grid_h, self.grid_w, self.window_size)

        h = self.attention(h)

        if not self.is_global:
            h = window_unpartition_ane(h, self.grid_h, self.grid_w, self.window_size)

        x = residual + h

        residual = x
        h = self.layer_norm2(x)
        h = self.fc1(h)
        h = gelu_ane(h)
        h = self.fc2(h)
        x = residual + h
        return x

    @classmethod
    def from_hf_layer(
        cls,
        hf_layer: nn.Module,
        is_global: bool,
        grid_h: int = _GRID_SIZE,
        grid_w: int = _GRID_SIZE,
        window_size: int = _WINDOW_SIZE,
        layer_norm_eps: float = _LAYER_NORM_EPS,
        rope_theta: float = _ROPE_THETA,
    ) -> "ImageEncoderBlock":
        hidden_size = hf_layer.layer_norm1.normalized_shape[0]
        mlp_dim = hf_layer.mlp.fc1.out_features

        ane_layer = cls(
            hidden_size=hidden_size,
            num_heads=hf_layer.attention.num_attention_heads,
            mlp_dim=mlp_dim,
            is_global=is_global,
            grid_h=grid_h,
            grid_w=grid_w,
            window_size=window_size,
            layer_norm_eps=layer_norm_eps,
            rope_theta=rope_theta,
        )

        ane_layer.layer_norm1 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm1)
        ane_layer.layer_norm2 = LayerNormReauthored.from_torch_layer_norm(hf_layer.layer_norm2)

        if is_global:
            attn_grid_h, attn_grid_w = grid_h, grid_w
            rope_scale = window_size / grid_h
        else:
            attn_grid_h, attn_grid_w = window_size, window_size
            rope_scale = 1.0

        ane_layer.attention = ImageEncoderAttention.from_hf_attention(
            hf_layer.attention,
            grid_h=attn_grid_h,
            grid_w=attn_grid_w,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

        ane_layer.fc1 = _linear_to_conv2d(hf_layer.mlp.fc1)
        ane_layer.fc2 = _linear_to_conv2d(hf_layer.mlp.fc2)
        return ane_layer


class ImageEncoderBackbone(nn.Module):
    """SAM3 image-encoder backbone in BC1S layout.

    1. Patch embed: ``(B, 3, H, W) -> (B, 1024, 1, H/14 * W/14)``
       via two-pass unfold + 1x1 Conv2d (rank-4-safe).
    2. Add learned position embedding (tiled from pretrain grid).
    3. Pre-layer LayerNorm.
    4. 32 transformer blocks (28 window + 4 global).
    """

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        num_heads: int = _NUM_HEADS,
        mlp_dim: int = _MLP_DIM,
        num_layers: int = 32,
        grid_h: int = _GRID_SIZE,
        grid_w: int = _GRID_SIZE,
        window_size: int = _WINDOW_SIZE,
        global_attn_indices: list[int] | None = None,
        patch_size: int = _PATCH_SIZE,
        num_channels: int = 3,
        layer_norm_eps: float = _LAYER_NORM_EPS,
        rope_theta: float = _ROPE_THETA,
    ) -> None:
        super().__init__()
        if global_attn_indices is None:
            global_attn_indices = list(_GLOBAL_ATTN_INDICES)

        self.hidden_size = hidden_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.window_size = window_size

        self.patch_size = patch_size
        patch_flat = num_channels * patch_size * patch_size  # 588 for 3 * 14 * 14
        self.patch_embed = nn.Conv2d(
            patch_flat,
            hidden_size,
            kernel_size=1,
            bias=False,
        )

        seq_len = grid_h * grid_w
        self.position_embedding = nn.Parameter(torch.zeros(1, hidden_size, 1, seq_len))

        self.pre_layer_norm = LayerNormReauthored(hidden_size, eps=layer_norm_eps)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_global = i in global_attn_indices
            self.layers.append(
                ImageEncoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    is_global=is_global,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    window_size=window_size,
                    layer_norm_eps=layer_norm_eps,
                    rope_theta=rope_theta,
                )
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        C = pixel_values.shape[1]
        P = self.patch_size
        pH, pW = self.grid_h, self.grid_w

        # Pass 1: unfold height. Pass 2: unfold width. Each stays at rank <= 5.
        x = pixel_values.reshape(B, C, pH, P, -1)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, C * P, pH, -1)

        x = x.reshape(B, C * P, pH, pW, P)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, C * P * P, pH, pW)

        x = x.reshape(B, C * P * P, 1, pH * pW)
        x = self.patch_embed(x)

        x = x + self.position_embedding
        x = self.pre_layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        return x

    @classmethod
    def from_hf_backbone(
        cls,
        hf_backbone: nn.Module,
        image_size: int = _IMAGE_SIZE,
        patch_size: int = _PATCH_SIZE,
    ) -> "ImageEncoderBackbone":
        config = hf_backbone.config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        mlp_dim = config.intermediate_size
        window_size = config.window_size
        global_attn_indices = config.global_attn_indexes
        layer_norm_eps = config.layer_norm_eps
        rope_theta = config.rope_theta

        grid_h = image_size // patch_size
        grid_w = image_size // patch_size

        ane_backbone = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            grid_h=grid_h,
            grid_w=grid_w,
            window_size=window_size,
            global_attn_indices=global_attn_indices,
            patch_size=patch_size,
            num_channels=config.num_channels,
            layer_norm_eps=layer_norm_eps,
            rope_theta=rope_theta,
        )

        # Patch embedding: HF (1024, 3, 14, 14) -> flat (1024, 588, 1, 1).
        hf_weight = hf_backbone.embeddings.patch_embeddings.projection.weight.data
        flat_weight = hf_weight.reshape(hidden_size, -1, 1, 1)
        ane_backbone.patch_embed.weight.data = flat_weight

        # Position embedding: HF (1, pretrain_patches, hidden) -> tile to runtime grid -> BC1S.
        hf_pos_embed = hf_backbone.embeddings.position_embeddings.data
        pretrain_size = int(hf_pos_embed.shape[1] ** 0.5)

        if pretrain_size == grid_h and pretrain_size == grid_w:
            tiled = hf_pos_embed
        else:
            pos = hf_pos_embed.reshape(1, pretrain_size, pretrain_size, hidden_size)
            pos = pos.permute(0, 3, 1, 2)
            repeat_h = grid_h // pretrain_size + 1
            repeat_w = grid_w // pretrain_size + 1
            pos = pos.tile([1, 1, repeat_h, repeat_w])[:, :, :grid_h, :grid_w]
            tiled = pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, hidden_size)

        ane_backbone.position_embedding.data = tiled.permute(0, 2, 1).unsqueeze(2)

        ane_backbone.pre_layer_norm = LayerNormReauthored.from_torch_layer_norm(
            hf_backbone.layer_norm
        )

        for i, hf_layer in enumerate(hf_backbone.layers):
            is_global = i in global_attn_indices
            ane_backbone.layers[i] = ImageEncoderBlock.from_hf_layer(
                hf_layer,
                is_global=is_global,
                grid_h=grid_h,
                grid_w=grid_w,
                window_size=window_size,
                layer_norm_eps=layer_norm_eps,
                rope_theta=rope_theta,
            )

        return ane_backbone
