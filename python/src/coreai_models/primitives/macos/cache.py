# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from typing_extensions import Self

from coreai_models.primitives._ops import mutable_cache_update_and_fetch, mutable_slice_update


class KVCache:
    # consts for the HF source model.
    # names start with _ to make sure that the users should ALWAYS
    # interact the caches from the KVCache class APIs.
    HF_K_BUFFER_NAME = "_full_cached_k"
    HF_V_BUFFER_NAME = "_full_cached_v"

    def __init__(
        self: Self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        self._k_cache = k_cache
        self._v_cache = v_cache

    @classmethod
    def seq_len_dim(cls) -> int:
        """
        Get the dimension index for sequence length in the KVCache.
        """
        return 3

    @classmethod
    def create_cache_tensors(
        cls,
        config,
        dtype: torch.dtype = torch.float32,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized KV cache tensors from a model config.

        Args:
            config: Model config supplying the layer/head dimensions.
            dtype: Cache dtype.
            seq_len: Sequence-dim length; defaults to ``config.max_position_embeddings``.
                Pass explicitly to build a trace-sized cache without mutating config.

        Returns:
            (k_cache, v_cache) of shape (n_layers, 1, n_kv_heads, max_seq_len, head_dim).
        """
        n_kv_heads = config.num_key_value_heads
        n_layers = config.num_hidden_layers
        max_seq_len = config.max_position_embeddings if seq_len is None else seq_len
        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        return k_cache, v_cache

    @classmethod
    def from_dimensions(
        cls,
        n_layers: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
    ) -> Self:
        """
        Create a KVCache object with specified dimensions.

        This method creates a standalone KV cache with custom dimensions.
        """
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        return cls(k_cache, v_cache)

    def update_and_fetch(
        self: Self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int | None = None,
        query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # check query size
        if query_len is None:
            query_len = k.shape[-2]
        torch._check_is_size(query_len, message="int query length >= 0")
        torch._check(query_len <= self._k_cache.size(-2), message="query length <= context size")
        torch._check(query_len <= self._v_cache.size(-2), message="query length <= context size")

        # check offset
        torch._check_is_size(offset, message="int offset >= 0")
        torch._check(offset < self._k_cache.size(-2), message="offset < context size")
        torch._check(offset < self._v_cache.size(-2), message="offset < context size")

        # check layer index
        torch._check_is_size(layer_idx, message="int layer index >= 0")
        torch._check(
            layer_idx < self._k_cache.size(0),
            message="layer index < number of transformer layers",
        )
        torch._check(
            layer_idx < self._v_cache.size(0),
            message="layer index < number of transformer layers",
        )

        if seq_len is None:
            seq_len = offset + query_len

        torch._check_is_size(seq_len)
        device = self._k_cache.device

        compute_device = k.device
        cross_device = compute_device != device
        if cross_device:
            k = k.to(device)
            v = v.to(device)

        layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)

        # update k and fetch its populated prefix in a single fused op
        k_out = mutable_cache_update_and_fetch(
            x=self._k_cache,
            update=k,
            begin=torch.concatenate(
                [
                    layer_index,
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((offset,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    torch.tensor((self._k_cache.size(1),), dtype=torch.int32, device=device),
                    torch.tensor((self._k_cache.size(2),), dtype=torch.int32, device=device),
                    torch.tensor((offset + k.size(-2),), dtype=torch.int32, device=device),
                    torch.tensor((self._k_cache.size(4),), dtype=torch.int32, device=device),
                ]
            ),
            layer_idx=layer_idx,
            seq_dim=-2,
            seq_len=seq_len,
        )

        # update v and fetch its populated prefix in a single fused op
        v_out = mutable_cache_update_and_fetch(
            x=self._v_cache,
            update=v,
            begin=torch.cat(
                [
                    layer_index,
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((offset,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    torch.tensor((int(self._v_cache.size(1)),), dtype=torch.int32, device=device),
                    torch.tensor((int(self._v_cache.size(2)),), dtype=torch.int32, device=device),
                    torch.tensor((offset + v.size(-2),), dtype=torch.int32, device=device),
                    torch.tensor((int(self._v_cache.size(4)),), dtype=torch.int32, device=device),
                ]
            ),
            layer_idx=layer_idx,
            seq_dim=-2,
            seq_len=seq_len,
        )

        if cross_device:
            return k_out.to(compute_device), v_out.to(compute_device)
        return k_out, v_out


class SSMState:
    """
    State Space Model (SSM) state cache for managing hidden states across layers.

    This class provides a mechanism to store and update SSM states (e.g., Mamba states)
    across multiple layers in a neural network. It uses a mutable slice update operation
    to efficiently update states for specific layers while maintaining the full state tensor.

    Attributes:
        _states (torch.Tensor): Internal tensor storing SSM states for all layers.
            Shape: (num_layers, batch_size, *state_dims) where:
                - num_layers: Number of transformer layers
                - batch_size: Batch size (typically 1 for inference)
                - *state_dims: Model-specific state dimensions (e.g., state_size, d_inner, etc.)
    """

    def __init__(
        self: Self,
        states: torch.Tensor,
    ) -> None:
        """
        Initialize the SSMState with a pre-allocated state tensor.

        Args:
            states (torch.Tensor): Pre-allocated tensor to store SSM states across layers.
                Shape: (num_layers, batch_size, *state_dims)
                - First dimension must correspond to the number of layers
                - Second dimension is typically batch_size (usually 1 for inference)
                - Remaining dimensions are model-specific state dimensions
        """
        self._states = states

    @property
    def states(self) -> torch.Tensor:
        """
        Get the full SSM state tensor.

        Returns:
            torch.Tensor: The complete state tensor containing states for all layers.
                Shape: (num_layers, batch_size, *state_dims)
        """
        return self._states

    def update_states(
        self: Self,
        layer_idx: int,
        new_state: torch.Tensor,
    ) -> None:
        """
        Update the SSM state for a specific layer.

        This method updates the state cache for a given layer using a mutable slice
        update operation. The update is performed in-place (conceptually) on the
        internal state tensor.

        Args:
            layer_idx (int): Index of the layer to update. Must be >= 0 and < num_layers.
            new_state (torch.Tensor): New state tensor for the specified layer.
                Shape: (batch_size, *state_dims)
                Should match the state dimensions excluding the layer dimension.

        Raises:
            RuntimeError: If layer_idx is out of bounds (>= number of layers).

        Note:
            The update operation uses torch.export-compatible size checking to ensure
            the layer index is valid. The new_state is automatically unsqueezed to add
            the layer dimension before updating.
        """
        cache = self._states

        # size checking for torch.export
        torch._check_is_size(layer_idx)
        torch._check(
            layer_idx < self._states.size(0),
        )

        # use the slice_update to update the cache
        layer_index = torch.tensor((layer_idx,), dtype=torch.int32)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32)

        mutable_slice_update(
            x=cache,
            update=new_state.unsqueeze(0),
            begin=torch.concatenate(
                [
                    layer_index,
                    *[torch.tensor((0,), dtype=torch.int32) for _ in range(cache.dim() - 1)],
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    *[
                        torch.tensor((cache.size(i),), dtype=torch.int32)
                        for i in range(1, 1 + cache.dim() - 2)
                    ],
                ]
            ),
        )


def ring_window_causal_mask(
    query_len: int,
    capacity: int,
    offset: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sliding-window causal mask for a ring-buffer KV cache.

    Reconstructs each slot's absolute position from the ring layout and applies
    causal + window + validity predicates. Companion to RingKVCache.

    Args:
        query_len: number of query positions in this forward call
        capacity: ring buffer size (number of physical slots)
        offset: absolute position of the first query token
        window_size: sliding window size (typically == capacity)
        device: target device

    Returns:
        Boolean mask [query_len, capacity] — True means "attend to this slot"
    """
    row = torch.arange(query_len, device=device)
    q_pos = row.unsqueeze(-1) + offset  # (query_len, 1)

    slot = torch.arange(capacity, device=device)  # (capacity,)
    last_pos = offset + query_len - 1

    # Reconstruct absolute position stored in each slot.
    # Avoids tensor aten.remainder (unsupported in CoreAI MLIR).
    # last_pos % capacity is a scalar symint (sympy Mod) — legal.
    r = last_pos % capacity
    diff = r - slot  # (capacity,), range (-capacity, capacity)
    ring_back = torch.where(diff >= 0, diff, diff + capacity)
    k_pos = last_pos - ring_back  # (capacity,)
    k_pos = k_pos.unsqueeze(0)  # (1, capacity)

    causal = k_pos <= q_pos
    in_window = (q_pos - k_pos) < window_size
    nonneg = k_pos >= 0
    return causal & in_window & nonneg


class RingKVCache:
    """Ring-buffer KV cache for sliding window attention layers.

    Fixed-size cache [n_layers, 1, n_kv_heads, capacity, head_dim] that writes
    at position % capacity and never grows. Use ring_window_causal_mask() to
    build the attention mask.
    """

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self._k_cache = k_cache
        self._v_cache = v_cache

    def capacity(self) -> int:
        return self._k_cache.shape[3]

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write K/V at ring position, return full cache for this layer.

        The write must not wrap around the ring boundary, i.e.
        ``(offset % capacity) + query_len <= capacity`` must hold.
        Decode (query_len=1) always satisfies this. For chunked prefill,
        choose chunk sizes that align to the capacity boundary (e.g.
        capacity // 2) so that no single chunk straddles the wrap point.

        Raises:
            RuntimeError: If the write would wrap around the ring buffer.
        """
        if query_len is None:
            query_len = k.shape[-2]
        torch._check_is_size(query_len)
        torch._check_is_size(layer_idx)
        torch._check(layer_idx < self._k_cache.size(0))

        device = self._k_cache.device
        capacity = self.capacity()

        # Ring slot: offset % capacity (scalar symint, no tensor remainder)
        write_start = offset % capacity

        layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)

        for cache, update in ((self._k_cache, k), (self._v_cache, v)):
            mutable_slice_update(
                x=cache,
                update=update.unsqueeze(0),
                begin=torch.cat(
                    [
                        layer_index,
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((write_start,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                    ]
                ),
                end=torch.cat(
                    [
                        layer_index_end,
                        torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
                        torch.tensor((write_start + query_len,), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
                    ]
                ),
            )

        k_out = self._k_cache.narrow(0, layer_idx, 1).squeeze(0)
        v_out = self._v_cache.narrow(0, layer_idx, 1).squeeze(0)
        return k_out, v_out
