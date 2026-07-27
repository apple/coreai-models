# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Window partition / unpartition for SAM3 image-encoder window attention.

The HF reference reshapes through rank-6 intermediates
``(B, H//ws, ws, W//ws, ws, C)``, which the on-device compiler rejects.
This pair of helpers stays strictly at rank 4 by working in
channels-last format and folding ``ws*C`` together — two passes (H then
W), each rank 4. Both operate on BC1S tensors.
"""

import torch


def window_partition_ane(
    x: torch.Tensor,
    H: int,
    W: int,
    window_size: int,
) -> torch.Tensor:
    """Partition spatial tokens into non-overlapping windows (BC1S).

    Requires ``H`` and ``W`` divisible by ``window_size``.
    Returns ``(B * num_windows, C, 1, ws*ws)``.
    """
    assert H % window_size == 0, f"H={H} not divisible by window_size={window_size}"
    assert W % window_size == 0, f"W={W} not divisible by window_size={window_size}"

    B, C, one, S = x.shape
    assert one == 1, f"Expected dim 2 to be 1, got {one}"
    assert S == H * W, f"S={S} != H*W={H * W}"

    ws = window_size
    nH = H // ws
    nW = W // ws

    t = x.squeeze(2).permute(0, 2, 1)  # (B, H*W, C)
    t = t.reshape(B, H, W, C)

    t = t.reshape(B * nH, ws, W, C)

    t = t.reshape(B * nH, ws, nW, ws * C)
    t = t.permute(0, 2, 1, 3)
    t = t.reshape(B * nH * nW, ws, ws, C)

    t = t.reshape(B * nH * nW, ws * ws, C)
    t = t.permute(0, 2, 1).unsqueeze(2)
    return t


def window_unpartition_ane(
    x: torch.Tensor,
    H: int,
    W: int,
    window_size: int,
) -> torch.Tensor:
    """Inverse of ``window_partition_ane``."""
    assert H % window_size == 0, f"H={H} not divisible by window_size={window_size}"
    assert W % window_size == 0, f"W={W} not divisible by window_size={window_size}"

    ws = window_size
    nH = H // ws
    nW = W // ws
    num_windows = nH * nW

    BW, C, one, S = x.shape
    assert one == 1, f"Expected dim 2 to be 1, got {one}"
    assert ws * ws == S, f"S={S} != ws*ws={ws * ws}"
    assert BW % num_windows == 0, f"BW={BW} not divisible by num_windows={num_windows}"
    B = BW // num_windows

    t = x.squeeze(2).permute(0, 2, 1)
    t = t.reshape(B * nH * nW, ws, ws, C)

    t = t.reshape(B * nH, nW, ws, ws * C)
    t = t.permute(0, 2, 1, 3)
    t = t.reshape(B * nH, ws, nW * ws, C)

    t = t.reshape(B, nH * ws, W, C)

    t = t.reshape(B, H * W, C)
    t = t.permute(0, 2, 1).unsqueeze(2)
    return t
