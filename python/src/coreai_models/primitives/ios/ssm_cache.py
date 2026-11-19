# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import nn
from typing_extensions import Self

from coreai_models.primitives._ops import mutable_slice_update

# Widest query block the iOS export ships; bounds how far back the conv ring must reach.
MAX_QUERY_LEN = 64


class SSMStateHandler:
    """
    Mutable state for gated-DeltaNet decoders.

    ``conv_state`` ``(n_delta, conv_dim, 1, W)`` is a ring buffer of recent conv inputs indexed
    by position mod ``W``; ``recurrent_state`` ``(n_delta, H, dk, dv)`` is the delta-rule state.
    Both are rank 4 and read per layer as one-wide slices.
    """

    _conv_state: torch.Tensor | None
    _recurrent_state: torch.Tensor | None

    @staticmethod
    def ring_width(kernel_size: int, max_query_len: int = MAX_QUERY_LEN) -> int:
        """Width of the conv ring.

        Smallest multiple of ``max_query_len`` that holds ``max_query_len + kernel_size - 1``
        columns, so an aligned block never straddles the wrap.
        """
        needed = max_query_len + kernel_size - 1
        return -(-needed // max_query_len) * max_query_len

    @staticmethod
    def get_conv_state(
        n_delta: int,
        conv_dim: int,
        ring_width: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.zeros((n_delta, conv_dim, 1, ring_width), dtype=dtype)

    @staticmethod
    def get_recurrent_state(
        n_delta: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.zeros((n_delta, num_v_heads, head_k_dim, head_v_dim), dtype=dtype)

    @staticmethod
    def build_seed_selector(
        ring_width: int,
        kernel_window: int,
        block_start: int,
        query_len: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Build the one-hot that reads a block's whole conv window out of the ring.

        Column ``j`` selects position ``block_start - kernel_window + j`` mod ``ring_width``.
        The block's columns are written first, so one matmul reads the whole window. Negative
        positions read zeros on a fresh ring.
        """
        selector = torch.zeros(1, ring_width, kernel_window + query_len, dtype=dtype)
        for j in range(kernel_window + query_len):
            selector[0, (block_start - kernel_window + j) % ring_width, j] = 1.0
        return selector

    def __init__(
        self: Self,
        n_delta: int,
        conv_dim: int,
        ring_width: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ):
        self._conv_state = None
        self._recurrent_state = None
        self._ring_width = ring_width

        # Constant index buffers for the slice bounds. Only the conv ring's column offset
        # is a runtime value; everything else is fixed at construction.
        with torch.device("cpu"):
            self._zero = nn.Buffer(torch.zeros(1, dtype=torch.int32), persistent=False)
            self._one = nn.Buffer(torch.ones(1, dtype=torch.int32), persistent=False)
            self._layer_begin = nn.Buffer(
                torch.arange(n_delta, dtype=torch.int32).unsqueeze(1), persistent=False
            )
            self._layer_end = nn.Buffer(
                torch.arange(1, n_delta + 1, dtype=torch.int32).unsqueeze(1), persistent=False
            )
            self._conv_dim = nn.Buffer(
                torch.tensor([conv_dim], dtype=torch.int32), persistent=False
            )
            self._num_v_heads = nn.Buffer(
                torch.tensor([num_v_heads], dtype=torch.int32), persistent=False
            )
            self._head_k_dim = nn.Buffer(
                torch.tensor([head_k_dim], dtype=torch.int32), persistent=False
            )
            self._head_v_dim = nn.Buffer(
                torch.tensor([head_v_dim], dtype=torch.int32), persistent=False
            )

    def register_state(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor) -> None:
        assert isinstance(conv_state, torch.Tensor) and isinstance(recurrent_state, torch.Tensor)
        # The ring's column arithmetic is taken modulo the width this handler was built with,
        # so a state of a different width would address the wrong columns.
        assert conv_state.shape[-1] == self._ring_width, (
            f"conv_state ring width {conv_state.shape[-1]} does not match the "
            f"{self._ring_width} this handler was built for"
        )
        self._conv_state = conv_state
        self._recurrent_state = recurrent_state

    def conv_update_and_fetch(
        self,
        delta_idx: int,
        conv_step: torch.IntTensor,
        values: torch.Tensor,
        num_token_updates: int,
    ) -> torch.Tensor:
        """Write this layer's block into the conv ring at ``conv_step``, then return the ring.

        ``conv_step`` is the block's start modulo the ring width, computed by the caller.
        ``ring_width`` is a multiple of the widest block, so this write never straddles the
        wrap and stays a single slice.

        Returns the layer's ring after the write, ``(1, conv_dim, 1, ring_width)``, as a
        one-wide slice so the result keeps the state's rank.
        """
        assert self._conv_state is not None, "register_state before conv_update_and_fetch"
        torch._check_is_size(delta_idx, message="int layer index >= 0")
        torch._check(
            delta_idx < self._conv_state.size(0),
            message="layer index < number of linear-attention layers",
        )
        device = conv_step.device
        begin = torch.cat(
            [
                self._layer_begin[delta_idx].to(device),
                self._zero.to(device),
                self._zero.to(device),
                conv_step,
            ]
        )
        end = torch.cat(
            [
                self._layer_end[delta_idx].to(device),
                self._conv_dim.to(device),
                self._one.to(device),
                conv_step + num_token_updates,
            ]
        )
        return mutable_slice_update(self._conv_state, values, begin, end)[delta_idx : delta_idx + 1]

    def recurrent_slot(self, delta_idx: int) -> torch.Tensor:
        """This layer's recurrent state, ``(1, num_v_heads, head_k_dim, head_v_dim)``."""
        assert self._recurrent_state is not None, "register_state before recurrent_slot"
        return self._recurrent_state[delta_idx : delta_idx + 1]

    def update_recurrent(self, delta_idx: int, new_recurrent: torch.Tensor) -> None:
        """Overwrite this layer's recurrent slot in place.

        Whole-slot write at a constant index.

        Args:
            delta_idx: linear-attention layer index (static).
            new_recurrent: ``(1, num_v_heads, head_k_dim, head_v_dim)``.
        """
        assert self._recurrent_state is not None, "register_state before update_recurrent"
        device = new_recurrent.device
        begin = torch.cat(
            [
                self._layer_begin[delta_idx].to(device),
                self._zero.to(device),
                self._zero.to(device),
                self._zero.to(device),
            ]
        )
        end = torch.cat(
            [
                self._layer_end[delta_idx].to(device),
                self._num_v_heads.to(device),
                self._head_k_dim.to(device),
                self._head_v_dim.to(device),
            ]
        )
        mutable_slice_update(x=self._recurrent_state, update=new_recurrent, begin=begin, end=end)

    @property
    def conv_state(self) -> torch.Tensor | None:
        return self._conv_state

    @property
    def recurrent_state(self) -> torch.Tensor | None:
        return self._recurrent_state
