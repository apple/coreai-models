# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the tanh-form SiLU (``primitives/ios/silu.py``)."""

import torch
import torch.nn.functional as F

from coreai_models.primitives.ios.silu import TanhSiLU


def test_tanh_silu_matches_silu() -> None:
    x = torch.linspace(-30, 30, 4001)
    torch.testing.assert_close(TanhSiLU()(x), F.silu(x))


def test_tanh_silu_computes_in_the_module_dtype() -> None:
    x = torch.linspace(-30, 30, 4001)
    got = TanhSiLU().half()(x.half())
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.float(), F.silu(x), rtol=2e-3, atol=2e-3)
