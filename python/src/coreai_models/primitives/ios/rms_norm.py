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


# fp16-overflow-safe RMSNorm. RMSNorm is scale-invariant, so we normalize each row
# by its own max magnitude *before* squaring: ``m = amax(|x|)`` (a data-dependent,
# per-row reduction), ``xn = x/m``. Since ``|xn| <= 1``, ``xn^2`` can never overflow
# fp16 (whereas raw ``x^2`` overflows once ``|x|>~256``: ``256^2~=65504``, the fp16 max
# -- the tower's residual stream peaks at ``|x|~=500`` on real images, so raw ``x^2``
# overflows to ``inf`` -> ``rsqrt(inf)=0`` -> zeroed features -> garbage soft tokens).
#
# The result is the *exact* RMSNorm because ``eps`` is folded back at the right scale::
#
#     xn*rsqrt(mean(xn^2) + (sqrt(eps)/m)^2) * w
#       = (x/m)*rsqrt(mean(x^2)/m^2 + eps/m^2) * w
#       = (x/m)*m*rsqrt(mean(x^2) + eps) * w
#       = x*rsqrt(mean(x^2) + eps) * w
#
# ``(sqrt(eps)/m)^2`` computes ``eps/m^2`` without ever forming ``m^2`` (which would
# itself overflow fp16 for ``m~=500``). Crucially this survives the ANE graph
# optimizer: a constant prescale (``x*0.125``) gets reassociated back into ``x^2*c^2``
# (recomputing the overflowing ``x^2``) or stripped as scale-invariant, but
# ``x/amax(|x|)`` is data-dependent and cannot be folded away. All ops
# (``abs``/``amax``/``div``/``mean``/``rsqrt``) are fp16 and ANE-resident -- no fp32
# upcast.


class MaxAbsRMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale, reducing over ``axis`` (counted from the end).

    fp16-overflow-safe via per-row max-normalization before squaring (see the comment
    directly above, and :class:`MaxAbsRMSNorm`, which adds the scale). Exact and ANE-resident (no
    fp32 upcast).
    """

    def __init__(self, eps: float = 1e-6, axis: int = -1) -> None:
        super().__init__()
        self._eps = eps
        self.axis = axis
        # Registered as a buffer (not a plain attribute) so it follows the module's
        # device on `.to(device)` -- a plain `torch.tensor(...)` created in forward()
        # defaults to CPU and raises on a GPU/MPS input (e.g. during parity tests).
        self.register_buffer("_eps_sqrt", torch.tensor(eps**0.5), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-row max magnitude; clamp floor avoids 0/0 on an all-zero row (real rows
        # have m >> floor, so this is a no-op for them).
        m = torch.clamp(x.abs().amax(self.axis, keepdim=True), min=1e-4)
        xn = x / m
        eps_sqrt = self._eps_sqrt.to(dtype=x.dtype)
        eps_term = (eps_sqrt / m) * (eps_sqrt / m)  # eps/m^2, no m^2 overflow
        mean_square = (xn * xn).mean(self.axis, keepdim=True)
        return xn * torch.rsqrt(mean_square + eps_term)


class MaxAbsRMSNorm(MaxAbsRMSNormNoScale):
    """:class:`MaxAbsRMSNormNoScale` followed by a learnable per-channel scale.

    The scale has shape ``(dim, 1, ..., 1)``, broadcasting over the dims after ``axis``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, axis: int = -1) -> None:
        super().__init__(eps, axis)
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.ones((dim,) + (1,) * (-axis - 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.weight
