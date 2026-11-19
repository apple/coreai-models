# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Sine and cosine on ``[0, 2*pi]`` built only from multiply, add and subtract.

The M2-generation Neural Engine has no ``sin``/``cos``. When a graph uses them, MPSGraph
moves those ops, and everything between them, to the GPU, which splits one Neural Engine
region into a Neural Engine -> GPU -> Neural Engine chain. These polynomials keep such
graphs on the Neural Engine.

They are only accurate on ``[0, 2*pi]``, the range of a sine position encoding of a normalized
coordinate (``coord * 2*pi / dim_t`` with ``coord`` in ``[0, 1]`` and ``dim_t >= 1``). They do
no range reduction, so anything outside that range comes back wrong.

Max abs error against ``torch.sin``/``torch.cos`` on the same input is about ``2e-5`` in fp32.
In fp16 it is about ``3e-3``, the same size as the fp16 rounding of the angle itself (half an
ulp of ``x`` near ``2*pi`` is ``2e-3``). Coefficients are minimax fits, rounded to fp16 one at a
time, refitting the rest after each rounding. Like ``gelu.py``, constants are fp16 tensors so no
f32 literals land in the graph. ``cos`` stores its coefficients negated, which folds the
``-cos(x - pi)`` sign into them instead of spending an op on it.
"""

import math

import torch


def _f16(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float16)


def _split_f16(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``value ~ hi + lo`` in fp16, so ``(hi - x) + lo`` keeps ``value``'s low bits."""
    hi = _f16(value)
    return hi, _f16(value - hi.item())


_PI_HI, _PI_LO = _split_f16(math.pi)
_TWO_PI_HI, _TWO_PI_LO = _split_f16(2.0 * math.pi)
# Exact power-of-two scale on the polynomial variable keeps every coefficient a normal fp16 number.
_POLY_SCALE = _f16(0.125)

# sin(x) = (pi - x) * z * R(z / 8),  z = x * (2*pi - x) in [0, pi^2].
_SIN_COEFFS = tuple(
    _f16(c) for c in (0.0506591796875, 0.0308380126953125, 0.0069580078125, 0.0010786056518554688)
)
# cos(x) = -cos(t) = sum_k c_k * (t^2 / 8)^k,  t = x - pi in [-pi, pi]. Negated coefficients.
_NEG_COS_COEFFS = tuple(
    _f16(c) for c in (-1.0, 4.0, -2.666015625, 0.7080078125, -0.09686279296875, 0.00624847412109375)
)


def _horner(w: torch.Tensor, coeffs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    cs = [c.to(dtype=w.dtype, device=w.device) for c in coeffs]
    acc = cs[-1] * w + cs[-2]
    for c in reversed(cs[:-2]):
        acc = acc * w + c
    return acc


def sin_0_2pi(x: torch.Tensor) -> torch.Tensor:
    """``sin(x)`` for ``x`` in ``[0, 2*pi]``.

    ``(pi - x) * x * (2*pi - x)`` carries sin's zeros at 0, pi and 2*pi, so the polynomial is a
    smooth positive factor and small angles keep their relative precision (``sin(x) ~ x``). This
    matters because most channels of a sine encoding take tiny angles.
    """
    pi_hi = _PI_HI.to(dtype=x.dtype, device=x.device)
    pi_lo = _PI_LO.to(dtype=x.dtype, device=x.device)
    two_pi_hi = _TWO_PI_HI.to(dtype=x.dtype, device=x.device)
    two_pi_lo = _TWO_PI_LO.to(dtype=x.dtype, device=x.device)
    scale = _POLY_SCALE.to(dtype=x.dtype, device=x.device)

    z = x * ((two_pi_hi - x) + two_pi_lo)
    return ((pi_hi - x) + pi_lo) * z * _horner(z * scale, _SIN_COEFFS)


def cos_0_2pi(x: torch.Tensor) -> torch.Tensor:
    """``cos(x)`` for ``x`` in ``[0, 2*pi]``, as ``-cos(x - pi)`` with an even polynomial."""
    pi_hi = _PI_HI.to(dtype=x.dtype, device=x.device)
    pi_lo = _PI_LO.to(dtype=x.dtype, device=x.device)
    scale = _POLY_SCALE.to(dtype=x.dtype, device=x.device)

    t = (x - pi_hi) - pi_lo
    return _horner(t * t * scale, _NEG_COS_COEFFS)
