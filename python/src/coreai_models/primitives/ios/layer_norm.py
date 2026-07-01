# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored LayerNorm operating on BC1S tensors."""

import torch
import torch.nn as nn


class LayerNormReauthored(nn.Module):
    """LayerNorm over the channel dim of (B, C, 1, S) tensors.

    Affine parameters are stored in (1, C, 1, 1) shape for direct
    broadcasting; eps is a tensor (not a Python float) to keep the
    exported graph free of f32 constants.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self._eps, dtype=x.dtype, device=x.device)
        mean = x.mean(dim=1, keepdim=True)
        x_centered = x - mean
        var = (x_centered * x_centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + eps)
        return x_centered * inv_std * self.weight + self.bias

    @classmethod
    def from_torch_layer_norm(
        cls, layer_norm: nn.LayerNorm, eps: float | None = None
    ) -> "LayerNormReauthored":
        dim = layer_norm.normalized_shape[0]
        actual_eps = eps if eps is not None else layer_norm.eps
        ane_norm = cls(dim, eps=actual_eps)
        ane_norm.weight.data = layer_norm.weight.data.reshape(1, dim, 1, 1)
        if layer_norm.bias is not None:
            ane_norm.bias.data = layer_norm.bias.data.reshape(1, dim, 1, 1)
        return ane_norm
