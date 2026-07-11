# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import nn
from typing_extensions import Self

from coreai_models.primitives._ops import mutable_slice_update


class KVCacheHandler:
    """
    KV Cache for iOS.

    For iOS, the layout of the KV cache is required to be different than on macOS.
    On iOS we must update on dim 4, whereas on macOS we use dim 3.
    The cache shape is: [n_layers, batch_size, n_kv_heads*head_dim, 1, max_seq_len]
    """

    @staticmethod
    def get_kv_cache_from_hf(
        config, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = config.hidden_size
        n_heads = config.num_attention_heads
        # Some HF configs (e.g. recent MistralConfig) declare `head_dim` but
        # leave it as None when not explicitly set. `getattr`'s default only
        # fires when the attribute is missing, not when it exists-but-is-None,
        # so we use `or` to fall back in both cases.
        head_dim = getattr(config, "head_dim", None) or (dim // n_heads)

        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        max_position_embeddings = config.max_position_embeddings

        key_cache = torch.zeros(
            (
                num_hidden_layers,
                1,
                num_key_value_heads * head_dim,
                1,
                max_position_embeddings,
            ),
            dtype=dtype,
        )
        value_cache = torch.zeros(
            (
                num_hidden_layers,
                1,
                num_key_value_heads * head_dim,
                1,
                max_position_embeddings,
            ),
            dtype=dtype,
        )
        return key_cache, value_cache

    def register_kv_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor):
        assert isinstance(key_cache, torch.Tensor), (
            f"Invalid key_cache type!, expected torch.Tensor, got {type(key_cache)}"
        )
        assert isinstance(value_cache, torch.Tensor), (
            f"Invalid value_cache type!, expected torch.Tensor, got {type(value_cache)}"
        )

        assert key_cache.shape == value_cache.shape, (
            f"key and value cache tensors must have the same shape, "
            f"got key: {key_cache.shape}, value: {value_cache.shape}"
        )

        self._k_cache = key_cache
        self._v_cache = value_cache

    def __init__(self: Self, n_layers: int, hidden_size: int):
        self._k_cache = None
        self._v_cache = None

        # Register constant buffers to the owner of this object for use in indexing the kv cache
        with torch.device("cpu"):
            self._zero = nn.Buffer(torch.zeros(1, dtype=torch.int32), persistent=False)
            self._one = nn.Buffer(torch.ones(1, dtype=torch.int32), persistent=False)
            self._hidden_size = nn.Buffer(
                torch.tensor([hidden_size], dtype=torch.int32), persistent=False
            )
            self._layer_indices = nn.Buffer(
                torch.arange(n_layers, dtype=torch.int32).unsqueeze(1), persistent=False
            )
            self._layer_indices_end = nn.Buffer(
                torch.arange(1, n_layers + 1, dtype=torch.int32).unsqueeze(1),
                persistent=False,
            )

    def gen_slice_args(
        self, layer_idx: int, offset: torch.IntTensor, num_token_updates: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_index = self._layer_indices[layer_idx].to(offset.device)
        layer_index_end = self._layer_indices_end[layer_idx].to(offset.device)
        begin = torch.cat(
            [
                layer_index,
                self._zero.to(offset.device),
                self._zero.to(offset.device),
                self._zero.to(offset.device),
                offset,
            ]
        )
        end = torch.cat(
            [
                layer_index_end,
                self._one.to(offset.device),
                self._hidden_size.to(offset.device),
                self._one.to(offset.device),
                offset + num_token_updates,
            ]
        )
        return begin, end

    def update_and_fetch(
        self: Self,
        layer_idx: int,
        offset: torch.IntTensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_token_updates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache for a specific layer and return the updated cache for that layer.

        Args:
            layer_idx: Index of the transformer layer
            offset: Starting position in the sequence dimension for the update
            k: New key tensor to insert into the cache
            v: New value tensor to insert into the cache
            num_token_updates: Number of tokens being updated

        Returns:
            Tuple of (key_cache, value_cache) for the specified layer
        """

        assert self._k_cache is not None and self._v_cache is not None, (
            "Cannot call update_and_fetch before registering key/value cache!"
        )

        torch._check_is_size(layer_idx, message="int layer index >= 0")
        torch._check(
            layer_idx < self._k_cache.size(0),
            message="layer index < number of transformer layers",
        )
        torch._check(
            layer_idx < self._v_cache.size(0),
            message="layer index < number of transformer layers",
        )

        begin, end = self.gen_slice_args(layer_idx, offset, num_token_updates)

        # update k - note that iOS updates on dimension 4 (the last dimension)
        mutable_slice_update(
            x=self._k_cache,
            update=k.unsqueeze(0),
            begin=begin,
            end=end,
        )

        # update v - note that iOS updates on dimension 4 (the last dimension)
        mutable_slice_update(
            x=self._v_cache,
            update=v.unsqueeze(0),
            begin=begin,
            end=end,
        )

        return self._k_cache[layer_idx], self._v_cache[layer_idx]

    @property
    def k_cache(self) -> torch.Tensor:
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        return self._v_cache
