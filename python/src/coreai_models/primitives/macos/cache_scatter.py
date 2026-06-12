# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Functional KV cache backed by slice_scatter (explicit KV, avoids Metal prefill crash)."""

import torch
from typing_extensions import Self


class KVCache:
    HF_K_BUFFER_NAME = "_full_cached_k"
    HF_V_BUFFER_NAME = "_full_cached_v"

    def __init__(self: Self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self._k_cache = k_cache
        self._v_cache = v_cache

    @classmethod
    def seq_len_dim(cls) -> int:
        return 3

    @classmethod
    def create_cache_tensors(
        cls,
        config,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def update_and_fetch(
        self: Self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int | None = None,
        query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_len is None:
            query_len = k.shape[-2]
        if seq_len is None:
            seq_len = offset + query_len

        layer_k = self._k_cache.narrow(0, layer_idx, 1)
        layer_v = self._v_cache.narrow(0, layer_idx, 1)

        # Use slice_scatter (functional) instead of index_put_ (in-place)
        # to avoid a Metal kernel crash during prefill.
        updated_k = torch.ops.aten.slice_scatter(
            layer_k, k.unsqueeze(0), dim=-2, start=offset, end=offset + query_len, step=1
        )
        updated_v = torch.ops.aten.slice_scatter(
            layer_v, v.unsqueeze(0), dim=-2, start=offset, end=offset + query_len, step=1
        )
        self._k_cache = torch.ops.aten.slice_scatter(
            self._k_cache, updated_k, dim=0, start=layer_idx, end=layer_idx + 1, step=1
        )
        self._v_cache = torch.ops.aten.slice_scatter(
            self._v_cache, updated_v, dim=0, start=layer_idx, end=layer_idx + 1, step=1
        )

        out_k = self._k_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
        out_v = self._v_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
        return out_k.squeeze(0), out_v.squeeze(0)
