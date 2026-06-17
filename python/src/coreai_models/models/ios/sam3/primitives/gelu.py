# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored GELU activation using sigmoid-based tanh approximation.

Uses the identity: 0.5 * x * (1 + tanh(z)) = x * sigmoid(2 * z)
where z = sqrt(2/pi) * (x + 0.044715 * x**3).

PSNR ~92 dB vs exact GELU (compared to ~57 dB for the simpler
x * sigmoid(1.702 * x) approximation). Only sigmoid is used so the
op is safe for on-device execution.
"""

import math

import torch
import torch.nn as nn

# 2 * sqrt(2/pi) ~ 1.5957691; stored as f16 to avoid f32 constants in the graph.
_GELU_COEFF = torch.tensor(2.0 * math.sqrt(2.0 / math.pi), dtype=torch.float16)
_CUBIC_COEFF = torch.tensor(0.044715, dtype=torch.float16)


class GELUReauthored(nn.Module):
    """GELU(x) ~ x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x**3))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = _GELU_COEFF.to(dtype=x.dtype, device=x.device)
        cubic = _CUBIC_COEFF.to(dtype=x.dtype, device=x.device)
        inner = coeff * (x + cubic * x * x * x)
        return x * torch.sigmoid(inner)


def gelu_ane(x: torch.Tensor) -> torch.Tensor:
    """Functional form of the GELU approximation above."""
    coeff = _GELU_COEFF.to(dtype=x.dtype, device=x.device)
    cubic = _CUBIC_COEFF.to(dtype=x.dtype, device=x.device)
    inner = coeff * (x + cubic * x * x * x)
    return x * torch.sigmoid(inner)
