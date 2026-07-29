# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared custom ops for Core AI model primitives."""

import torch
from torch import Tensor


@torch.library.custom_op("coreai::mutable_slice_update", mutates_args=["x"])
def mutable_slice_update(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    """
    Mutable slice update operation for cache updates.

    Updates a slice of tensor x with the update tensor using dynamic begin/end indices.
    Begin and end indices are passed as tensors for custom op compatibility.

    Args:
        x: The tensor to update
        update: The update values to insert
        begin: Tensor containing start indices for each dimension
        end: Tensor containing end indices for each dimension

    Returns:
        The updated tensor (clone for torch compatibility)
    """
    # Begin and end indices passed in as tensors for custom op compatibility -> split for slicing
    begin = torch.split(begin, 1, dim=0)  # type: ignore
    end = torch.split(end, 1, dim=0)  # type: ignore
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begin, end, strict=False))
    x[slices] = update
    # Note: Not actually in-place for torch
    return x.clone()


@mutable_slice_update.register_fake
def mutable_slice_update_meta(  # type: ignore
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
):
    """Fake implementation for tracing/meta operations."""
    return torch.empty(x.shape, dtype=x.dtype)


@torch.library.custom_op("coreai::mutable_cache_update_and_fetch", mutates_args=["x"])
def mutable_cache_update_and_fetch(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_dim: int,
    seq_len: int | None,
) -> Tensor:
    """
    Fused KV-cache update-and-fetch operation.

    Writes ``update`` into the slice ``x[begin:end]`` (mutating ``x``), then
    fetches one layer: ``x.narrow(0, layer_idx, 1)`` optionally followed by
    ``narrow(seq_dim, 0, seq_len)``, then ``squeeze(0)``. ``seq_len=None``
    skips the seq-prefix narrow (iOS path returns the full cache row).
    During export ``remove_functionalization`` rewrites this into
    ``unsqueeze`` + ``immutable_slice_update`` + slice (+ optional slice) +
    ``squeeze`` for MLIR lowering.

    Layouts:
      - macOS: x is (L, 1, n_kv_heads, max_seq, head_dim), seq_dim=-2,
        seq_len=offset+query_len. Output: (1, n_kv_heads, seq_len, head_dim).
      - iOS:   x is (L, 1, hidden, 1, max_seq), seq_dim=-1, seq_len=None.
        Output: (1, hidden, 1, max_seq).

    Args:
        x: 5D cache tensor to update.
        update: 4D values to write into the slice (op unsqueezes internally).
        begin: 5-elem tensor of per-dim start indices.
        end: 5-elem tensor of per-dim end indices.
        layer_idx: Layer to fetch after the update.
        seq_dim: Dim along which to truncate the fetched layer (negative idx ok).
        seq_len: Populated seq length to fetch, or None to skip the truncation.
    """
    update = update.unsqueeze(0)
    begin = torch.split(begin, 1, dim=0)  # type: ignore
    end = torch.split(end, 1, dim=0)  # type: ignore
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begin, end, strict=False))
    x[slices] = update
    fetched = x.narrow(0, layer_idx, 1)
    if seq_len is not None:
        fetched = fetched.narrow(seq_dim, 0, seq_len)
    # Clone because the returned slice aliases the mutated cache.
    return fetched.squeeze(0).clone()


@mutable_cache_update_and_fetch.register_fake
def mutable_cache_update_and_fetch_meta(  # type: ignore
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_dim: int,
    seq_len: int | None,
):
    """Fake implementation for tracing/meta operations."""
    out_shape = list(x.shape)
    if seq_len is not None:
        out_shape[seq_dim] = seq_len
    out_shape.pop(0)  # squeeze layer dim
    return torch.empty(out_shape, dtype=x.dtype)
