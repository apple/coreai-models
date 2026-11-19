# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the max-abs RMSNorm (``primitives/ios/rms_norm.py``)."""

import pytest
import torch

from coreai_models.primitives.ios.rms_norm import MaxAbsRMSNorm, MaxAbsRMSNormNoScale


@pytest.mark.parametrize("peak", [0.0, 1e-7, 1e-5, 1e-3, 3.9e-3, 4e-3, 1.0, 1e2, 1e4, 6e4])
def test_float16_max_abs_rms_norm_matches_float32_across_magnitudes(peak: float) -> None:
    eps = 1e-6
    x = torch.randn(8, 256, generator=torch.Generator().manual_seed(0))
    x = x / x.abs().amax(1, keepdim=True) * peak
    x16 = x.half()
    want = x16.float() * torch.rsqrt((x16.float() ** 2).mean(1, keepdim=True) + eps)
    with torch.no_grad():
        got = MaxAbsRMSNormNoScale(eps).half()(x16).float()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


def test_scale_over_a_leading_axis_matches_the_last_axis() -> None:
    x = torch.randn(2, 6, 1, 5, generator=torch.Generator().manual_seed(1))
    channels_first, channels_last = MaxAbsRMSNorm(6, axis=-3), MaxAbsRMSNorm(6)
    assert channels_first.weight.shape == (6, 1, 1)
    with torch.no_grad():
        channels_first.weight.copy_(torch.arange(6.0).reshape(6, 1, 1))
        channels_last.weight.copy_(torch.arange(6.0))
        got = channels_first(x)
        want = channels_last(x.transpose(1, 3)).transpose(1, 3)
    torch.testing.assert_close(got, want)
