# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import coreai_torch
import coreai_torch.composite_ops
import torch
from typing_extensions import Self


class SDPA(coreai_torch.composite_ops.SDPA):
    """Apply scaled dot product attention to input tensors, with attributes pre-determined."""

    def __init__(
        self: Self,
        scale: float | None = None,
        is_causal: bool = False,
        window_size: int = 0,
    ) -> None:
        _use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        super().__init__(
            scale=scale,
            is_causal=is_causal,
            window_size=window_size,
            _use_hf_impl=_use_hf_impl,
        )

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Force contiguous K/V on sliding-window layers.

        Workaround for bug identified in #311: strided (non-contiguous) K/V views
        return NaN past ~3584 keys in deep dynamic graphs. `contiguous()` is a
        numerical no-op and only copies when actually strided.
        """
        if self.window_size > 0:
            key = key.contiguous()
            value = value.contiguous()
        return super().forward(
            query,
            key,
            value,
            attn_mask=attn_mask,
            sinks=sinks,
        )
