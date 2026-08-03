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

    def capacity(self: Self) -> int:
        """Preallocated cache length along the sequence dimension."""
        return self._k_cache.size(self.seq_len_dim())

    def clamped_len(self: Self, seq_len):
        """Clamp an absolute sequence length to this cache's capacity.

        A bounded cache (see `update_and_fetch_windowed`) only ever physically
        holds up to `capacity()` of the most recent tokens, so any absolute
        position count derived from `offset + query_len` must be clamped
        before being used to size a read from `_k_cache`/`_v_cache` (e.g. the
        KV-sharing donor read in gemma4_text.py). No-op for caches that never
        overflow their capacity.
        """
        return torch.sym_min(seq_len, self.capacity())

    @classmethod
    def create_cache_tensors(
        cls,
        config,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized KV cache tensors from a model config.

        Returns:
            (k_cache, v_cache) tensors of shape (n_layers, 1, n_kv_heads, max_seq_len, head_dim).
        """
        n_kv_heads = config.num_key_value_heads
        n_layers = config.num_hidden_layers
        max_seq_len = config.max_position_embeddings
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
            query_len: int = k.shape[-2]
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

    def update_and_fetch_windowed(
        self: Self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded variant of `update_and_fetch` for sliding-window attention caches.

        Unlike `update_and_fetch`, which assumes the cache is preallocated large
        enough to hold every token ever written (true for full-attention caches
        sized to the model's max context length), this method treats `capacity()`
        as a hard cap: once `offset + query_len` would exceed it, the existing
        window is shifted left (oldest tokens dropped) before the new tokens are
        written, so the cache never grows past its preallocated size. Callers
        should preallocate the cache to exactly `config.sliding_window` (see
        export/macos.py) so this cap coincides with the attention window.

        Returns the full `capacity()`-sized buffer (not narrowed to however many
        tokens are actually valid) so its shape is always static -- narrowing to
        a dynamic `min(capacity(), offset + query_len)` length triggers spurious
        `torch.export` `ConstraintViolationError`s deep in stride/contiguity
        checks (a compound `Min()` expression flowing into a tensor *size*, as
        opposed to plain dynamic sizing, which is fine). Callers must build an
        explicit attention mask from `clamped_len(offset + query_len)` to hide
        the not-yet-written slots during ramp-up (before the window fills);
        SDPA's own window_size masking is a no-op once window_size == capacity()
        so it cannot do this hiding on the caller's behalf.

        Note: for a multi-token query chunk that straddles the window boundary,
        early rows within the chunk may see slightly less history than a strict
        per-row sliding window would allow (up to `query_len - 1` tokens short),
        since the shift is sized for the chunk's last row. This only affects
        prefill/chunked-prefill; decode (query_len == 1) is exact. This is the
        same tradeoff accepted by other chunked sliding-window KV cache
        implementations that size the cache to exactly the window rather than
        window + max_chunk_size.
        """
        if query_len is None:
            query_len: int = k.shape[-2]
        torch._check_is_size(query_len, message="int query length >= 0")

        torch._check_is_size(layer_idx, message="int layer index >= 0")
        torch._check(
            layer_idx < self._k_cache.size(0),
            message="layer index < number of transformer layers",
        )
        torch._check(
            layer_idx < self._v_cache.size(0),
            message="layer index < number of transformer layers",
        )

        torch._check_is_size(offset, message="int offset >= 0")

        device = self._k_cache.device
        seq_dim = self.seq_len_dim()
        capacity = self.capacity()
        torch._check(query_len <= capacity, message=lambda: "query length <= window capacity")

        old_valid = torch.sym_min(offset, capacity)
        new_valid = self.clamped_len(offset + query_len)
        # max(x, 0), expressed via sym_min (not sym_max): coreai_torch's MLIR
        # converter has a registered lowering for aten.sym_min but not
        # aten.sym_max, so max(x, 0) == -min(-x, 0) avoids an unsupported op.
        drop = -torch.sym_min(-(old_valid - (new_valid - query_len)), 0)
        keep = old_valid - drop

        layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)
        shift_idx = torch.clamp(torch.arange(capacity, device=device) + drop, max=capacity - 1)

        for cache, update in ((self._k_cache, k), (self._v_cache, v)):
            shifted = cache.narrow(0, layer_idx, 1).index_select(seq_dim, shift_idx)
            mutable_slice_update(
                x=cache,
                update=shifted,
                begin=torch.cat(
                    [
                        layer_index,
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                    ]
                ),
                end=torch.cat(
                    [
                        layer_index_end,
                        torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
                        torch.tensor((capacity,), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
                    ]
                ),
            )
            mutable_slice_update(
                x=cache,
                update=update.unsqueeze(0),
                begin=torch.cat(
                    [
                        layer_index,
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                        torch.tensor((keep,), dtype=torch.int32, device=device),
                        torch.tensor((0,), dtype=torch.int32, device=device),
                    ]
                ),
                end=torch.cat(
                    [
                        layer_index_end,
                        torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
                        torch.tensor((keep + query_len,), dtype=torch.int32, device=device),
                        torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
                    ]
                ),
            )

        k_out = self._k_cache.narrow(0, layer_idx, 1)
        v_out = self._v_cache.narrow(0, layer_idx, 1)
        return k_out.squeeze(0), v_out.squeeze(0)


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
