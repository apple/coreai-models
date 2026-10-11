# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the macOS SDPA sliding-window contiguous-K/V workaround.

Regression guard for bug identified in #311: strided K/V views return NaN on the
fused sliding-window SDPA kernel. The SDPA primitive forces contiguous K/V for
sliding-window layers (``window_size > 0``); global layers are untouched.
"""

import pytest
import torch

try:
    import coreai_torch
    import coreai_torch.composite_ops

    HAS_COREAI = True
except ImportError:
    HAS_COREAI = False


def _strided_kv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a query plus strided (non-contiguous) key/value views of a packed buffer.

    Mirrors the packed-cache ``narrow`` layout from the issue repro: K/V are
    sub-ranges of a wider last dimension, so each view is non-contiguous.
    """
    head_dim = 16
    seq = 8
    packed = torch.randn(1, 2, seq, 2 * head_dim)
    key = packed.narrow(-1, 0, head_dim)
    value = packed.narrow(-1, head_dim, head_dim)
    query = torch.randn(1, 2, seq, head_dim)
    return query, key, value


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestmacOSSDPASlidingWindowContiguous:
    """Lock in the contiguous-K/V workaround scope for the macOS SDPA primitive."""

    def test_sliding_window_forces_contiguous_kv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``window_size > 0`` => K/V handed to the composite kernel are contiguous."""
        from coreai_models.primitives.macos.sdpa import SDPA

        seen: dict[str, bool] = {}

        def spy(self, query, key, value, attn_mask=None, sinks=None):
            seen["key_contiguous"] = key.is_contiguous()
            seen["value_contiguous"] = value.is_contiguous()
            return query  # shape-agnostic stand-in; only inputs are inspected

        monkeypatch.setattr(coreai_torch.composite_ops.SDPA, "forward", spy)

        sdpa = SDPA(is_causal=True, window_size=512)
        query, key, value = _strided_kv()
        assert not key.is_contiguous()
        assert not value.is_contiguous()

        sdpa(query=query, key=key, value=value)

        assert seen["key_contiguous"] is True
        assert seen["value_contiguous"] is True

    def test_global_layer_leaves_kv_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``window_size == 0`` (global layer) => strided K/V passed through unchanged."""
        from coreai_models.primitives.macos.sdpa import SDPA

        seen: dict[str, bool] = {}

        def spy(self, query, key, value, attn_mask=None, sinks=None):
            seen["key_contiguous"] = key.is_contiguous()
            seen["value_contiguous"] = value.is_contiguous()
            return query

        monkeypatch.setattr(coreai_torch.composite_ops.SDPA, "forward", spy)

        sdpa = SDPA(is_causal=True)  # window_size defaults to 0
        query, key, value = _strided_kv()

        sdpa(query=query, key=key, value=value)

        # Global layers are not affected by the kernel bug; the workaround must
        # not silently insert copies there.
        assert seen["key_contiguous"] is False
        assert seen["value_contiguous"] is False
