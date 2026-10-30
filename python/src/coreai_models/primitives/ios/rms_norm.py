# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch
import torch.nn as nn


# iOS-friendly RMS norm requires that the reduction-mean occurs on dim 1
class RMSNorm(nn.Module):
    """
    Apply root mean square normalization (RMSNorm) to input tensor, with attributes pre-determined.

    The RMSNorm operation is defined as:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
    """

    def __init__(self, dim: int, eps: float = 1e-6, n_heads: int | None = None) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
            self._eps = nn.Buffer(torch.tensor(eps), persistent=False)
        self._use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform root mean square layer normalization on input.

        Args:
            input: Input tensor of shape (batch_size, seq_len, 1, dim)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, 1, dim)
        """
        if self._use_hf_impl:
            input_dtype = input.dtype
            input = input.to(torch.float32)

        square = input * input
        mean_square = square.mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self._eps)
        x_2normalized = input * inv_rms

        if self._use_hf_impl:
            x_2normalized = x_2normalized.to(input_dtype)

        return x_2normalized * self.weight


# Max-abs RMSNorm: a float16-overflow-safe RMSNorm. The input is divided by its max magnitude
# along ``axis``, ``m = amax(|x|)``, before squaring, and ``eps`` is rescaled to match; the
# result equals plain RMSNorm::
#
#     xn = x / m
#     xn * rsqrt(mean(xn^2) + (sqrt(eps) / m)^2) * w  ==  x * rsqrt(mean(x^2) + eps) * w
#
# Plain RMSNorm overflows float16 once ``|x| > ~256`` (``x^2 > 65504``). ``eps / m^2`` is computed
# as ``(sqrt(eps) / m)^2`` to avoid forming ``m^2``. A constant prescale does not work instead:
# the Neural Engine compiler folds it back into ``x^2``. All ops run in the input dtype.


class MaxAbsRMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale, reducing over ``axis`` (counted from the end).

    Float16-overflow-safe and exact; see the comment above. :class:`MaxAbsRMSNorm` adds the scale.
    """

    def __init__(self, eps: float = 1e-6, axis: int = -1) -> None:
        super().__init__()
        self._eps = eps
        self.axis = axis
        self.register_buffer("_eps_sqrt", torch.tensor(eps**0.5), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Max magnitude along ``axis``; the floor keeps an all-zero row at zero instead of NaN.
        m = torch.clamp(x.abs().amax(self.axis, keepdim=True), min=1e-4)
        xn = x / m
        eps_sqrt = self._eps_sqrt.to(dtype=x.dtype)
        eps_term = (eps_sqrt / m) * (eps_sqrt / m)  # eps/m^2, no m^2 overflow
        mean_square = (xn * xn).mean(self.axis, keepdim=True)
        return xn * torch.rsqrt(mean_square + eps_term)


class MaxAbsRMSNorm(MaxAbsRMSNormNoScale):
    """:class:`MaxAbsRMSNormNoScale` followed by a learnable per-channel scale.

    ``axis`` must be negative. The scale has shape ``(dim, 1, ..., 1)``, broadcasting over the dims
    after ``axis``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, axis: int = -1) -> None:
        super().__init__(eps, axis)
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.ones((dim,) + (1,) * (-axis - 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.weight
