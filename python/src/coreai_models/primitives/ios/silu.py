# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Re-authored SiLU activation using the tanh form.

Uses the identity: x * sigmoid(x) = x * (0.5 * tanh(0.5 * x) + 0.5).

The Neural Engine's fused SiLU is less precise than this form: in Sana Sprint's feed-forward it
costs ~25 dB per block against float32, ~9 dB end to end. The tanh form is slower per element.
"""

import torch
import torch.nn as nn


class TanhSiLU(nn.Module):
    """SiLU(x) = x * (c * tanh(c * x) + c), c = 0.5, with ``c`` a buffer in the module's dtype."""

    c: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.tensor(0.5), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.c * torch.tanh(x * self.c) + self.c)
