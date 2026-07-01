# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored SAM3 FPN neck.

Four FPN levels with scale factors [4.0, 2.0, 1.0, 0.5] producing
``(B, 256, H_i, W_i)`` feature maps. ``GELU`` is replaced with the
sigmoid-based approximation; sinusoidal position encodings are
precomputed buffers.
"""

import math

import torch
import torch.nn as nn

from coreai_models.primitives.ios.gelu import gelu_ane

_BACKBONE_HIDDEN_SIZE = 1024
_FPN_HIDDEN_SIZE = 256
_GRID_SIZE = 72
_SCALE_FACTORS = [4.0, 2.0, 1.0, 0.5]
_NUM_POS_FEATS = _FPN_HIDDEN_SIZE // 2  # 128
_TEMPERATURE = 10000


def _precompute_sine_position_encoding(
    height: int,
    width: int,
    num_pos_feats: int = _NUM_POS_FEATS,
    temperature: int = _TEMPERATURE,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """2D sinusoidal position encoding matching HF ``Sam3SinePositionEmbedding``."""
    scale = 2.0 * math.pi
    eps = 1e-6

    y_embed = torch.arange(1, height + 1, dtype=torch.float64).unsqueeze(1).expand(height, width)
    x_embed = torch.arange(1, width + 1, dtype=torch.float64).unsqueeze(0).expand(height, width)

    y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float64)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
    return pos.to(dtype)


class FPNLayerReauthored(nn.Module):
    """One FPN level with the scale-factor pattern from ``Sam3FPNLayer``."""

    def __init__(
        self,
        in_channels: int = _BACKBONE_HIDDEN_SIZE,
        fpn_dim: int = _FPN_HIDDEN_SIZE,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        self.scale_layers = nn.ModuleList()
        if scale_factor == 4.0:
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
            )
            intermediate_channels = in_channels // 4
            self._has_gelu = True
        elif scale_factor == 2.0:
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            intermediate_channels = in_channels // 2
            self._has_gelu = False
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
            self._has_gelu = False
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
            self._has_gelu = False
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} not supported")

        self.proj1 = nn.Conv2d(intermediate_channels, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.proj1.weight.dtype)

        if self._has_gelu:
            hidden_states = self.scale_layers[0](hidden_states)
            hidden_states = gelu_ane(hidden_states)
            hidden_states = self.scale_layers[1](hidden_states)
        else:
            for layer in self.scale_layers:
                hidden_states = layer(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    @classmethod
    def from_hf_fpn_layer(cls, hf_fpn_layer: nn.Module) -> "FPNLayerReauthored":
        scale_factor = hf_fpn_layer.scale_factor
        if scale_factor in (4.0, 2.0):
            backbone_channels = hf_fpn_layer.scale_layers[0].in_channels
        elif scale_factor in (1.0, 0.5):
            backbone_channels = hf_fpn_layer.proj1.in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor}")

        fpn_dim = hf_fpn_layer.proj1.out_channels

        ane_layer = cls(
            in_channels=backbone_channels,
            fpn_dim=fpn_dim,
            scale_factor=scale_factor,
        )

        if scale_factor == 4.0:
            # HF stores [ConvTranspose2d, GELU, ConvTranspose2d] — we only have the two convs.
            ane_layer.scale_layers[0].weight.data = hf_fpn_layer.scale_layers[0].weight.data.clone()
            ane_layer.scale_layers[0].bias.data = hf_fpn_layer.scale_layers[0].bias.data.clone()
            ane_layer.scale_layers[1].weight.data = hf_fpn_layer.scale_layers[2].weight.data.clone()
            ane_layer.scale_layers[1].bias.data = hf_fpn_layer.scale_layers[2].bias.data.clone()
        elif scale_factor == 2.0:
            ane_layer.scale_layers[0].weight.data = hf_fpn_layer.scale_layers[0].weight.data.clone()
            ane_layer.scale_layers[0].bias.data = hf_fpn_layer.scale_layers[0].bias.data.clone()

        ane_layer.proj1.weight.data = hf_fpn_layer.proj1.weight.data.clone()
        ane_layer.proj1.bias.data = hf_fpn_layer.proj1.bias.data.clone()
        ane_layer.proj2.weight.data = hf_fpn_layer.proj2.weight.data.clone()
        ane_layer.proj2.bias.data = hf_fpn_layer.proj2.bias.data.clone()

        return ane_layer


class FPNNeckReauthored(nn.Module):
    """SAM3 multi-scale FPN neck.

    Reshapes BC1S backbone output to spatial ``(B, C, H, W)``, runs four
    FPN levels, returns spatial feature maps + sinusoidal position
    encodings (one per level).
    """

    def __init__(
        self,
        in_channels: int = _BACKBONE_HIDDEN_SIZE,
        fpn_dim: int = _FPN_HIDDEN_SIZE,
        grid_h: int = _GRID_SIZE,
        grid_w: int = _GRID_SIZE,
        scale_factors: list[float] = None,
    ) -> None:
        super().__init__()
        if scale_factors is None:
            scale_factors = list(_SCALE_FACTORS)

        self.in_channels = in_channels
        self.fpn_dim = fpn_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.scale_factors = scale_factors

        self.fpn_layers = nn.ModuleList(
            [
                FPNLayerReauthored(in_channels=in_channels, fpn_dim=fpn_dim, scale_factor=sf)
                for sf in scale_factors
            ]
        )

        for i, sf in enumerate(scale_factors):
            h = int(grid_h * sf)
            w = int(grid_w * sf)
            pos_enc = _precompute_sine_position_encoding(
                height=h,
                width=w,
                num_pos_feats=fpn_dim // 2,
            )
            self.register_buffer(f"pos_enc_{i}", pos_enc)

    def forward(
        self,
        backbone_output: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        B = backbone_output.shape[0]
        hidden_states = backbone_output.reshape(B, self.in_channels, self.grid_h, self.grid_w)

        fpn_hidden_states: tuple[torch.Tensor, ...] = ()
        fpn_position_encoding: tuple[torch.Tensor, ...] = ()

        for i, fpn_layer in enumerate(self.fpn_layers):
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states += (fpn_output,)

            pos_enc = getattr(self, f"pos_enc_{i}").to(dtype=fpn_output.dtype)
            fpn_position_encoding += (pos_enc,)

        return fpn_hidden_states, fpn_position_encoding

    @classmethod
    def from_hf_fpn(
        cls,
        hf_neck: nn.Module,
        grid_h: int = _GRID_SIZE,
        grid_w: int = _GRID_SIZE,
    ) -> "FPNNeckReauthored":
        config = hf_neck.config
        in_channels = config.backbone_config.hidden_size
        fpn_dim = config.fpn_hidden_size
        scale_factors = config.scale_factors

        ane_neck = cls(
            in_channels=in_channels,
            fpn_dim=fpn_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            scale_factors=scale_factors,
        )

        for i, hf_fpn_layer in enumerate(hf_neck.fpn_layers):
            ane_neck.fpn_layers[i] = FPNLayerReauthored.from_hf_fpn_layer(hf_fpn_layer)
        return ane_neck
