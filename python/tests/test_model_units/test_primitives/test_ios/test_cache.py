# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS KVCacheHandler primitive."""

import pytest
import torch

from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.cache import (
    KVCacheHandler as iOSKVCache,
)
from tests._runner_infra.testing_utils import (
    assert_close,
)


class TestiOSKVCacheHandlerBasic:
    """Test KVCacheHandler: create, update, verify values."""

    def _make_cache(self, n_layers=2, n_kv_heads=2, head_dim=8, max_seq_len=16):
        """Create a KVCacheHandler with registered caches."""
        hidden_size = n_kv_heads * head_dim
        handler = KVCacheHandler(n_layers=n_layers, hidden_size=hidden_size)
        k_cache = torch.zeros(n_layers, 1, hidden_size, 1, max_seq_len)
        v_cache = torch.zeros(n_layers, 1, hidden_size, 1, max_seq_len)
        handler.register_kv_cache(k_cache, v_cache)
        return handler, hidden_size

    def test_register_and_shape(self):
        """Cache should have the correct shape after registration."""
        handler, hidden_size = self._make_cache()
        assert handler.k_cache.shape == (2, 1, hidden_size, 1, 16)
        assert handler.v_cache.shape == (2, 1, hidden_size, 1, 16)

    def test_update_and_fetch(self):
        """Values written via update_and_fetch should appear at the correct position."""
        n_layers, n_kv_heads, head_dim, max_seq_len = 2, 2, 8, 16
        handler, hidden_size = self._make_cache(n_layers, n_kv_heads, head_dim, max_seq_len)

        # Insert 3 tokens at offset 0, layer 0
        num_tokens = 3
        k = torch.randn(1, hidden_size, 1, num_tokens)
        v = torch.randn(1, hidden_size, 1, num_tokens)
        offset = torch.tensor([0], dtype=torch.int32)

        k_out, v_out = handler.update_and_fetch(
            layer_idx=0, offset=offset, k=k, v=v, num_token_updates=num_tokens
        )

        # k_out is the full layer cache; verify the updated slice
        assert k_out.shape == (1, hidden_size, 1, max_seq_len)
        torch.testing.assert_close(k_out[:, :, :, :num_tokens], k)
        torch.testing.assert_close(v_out[:, :, :, :num_tokens], v)

    def test_update_flag_keeps_cached_values_for_unflagged_slots(self):
        """Slots with flag 0 must keep the cached K/V, both in the cache and in the fetched
        row; slots with flag 1 are written as usual."""
        handler, hidden_size = self._make_cache(n_layers=2, max_seq_len=16)
        old_k = torch.randn(1, hidden_size, 1, 8)
        old_v = torch.randn(1, hidden_size, 1, 8)
        handler.update_and_fetch(
            layer_idx=1,
            offset=torch.tensor([4], dtype=torch.int32),
            k=old_k,
            v=old_v,
            num_token_updates=8,
        )

        # Rewrite positions 6..9 with only 8 and 9 flagged: 6 and 7 must keep their values.
        new_k = torch.randn(1, hidden_size, 1, 4)
        new_v = torch.randn(1, hidden_size, 1, 4)
        flag = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        k_out, v_out = handler.update_and_fetch(
            layer_idx=1,
            offset=torch.tensor([6], dtype=torch.int32),
            k=new_k,
            v=new_v,
            num_token_updates=4,
            update_flag=flag,
        )

        for out, cache, old, new in (
            (k_out, handler.k_cache[1], old_k, new_k),
            (v_out, handler.v_cache[1], old_v, new_v),
        ):
            for row in (out, cache):
                torch.testing.assert_close(row[..., 4:8], old[..., 0:4])
                torch.testing.assert_close(row[..., 8:10], new[..., 2:4])
                torch.testing.assert_close(row[..., 10:12], old[..., 6:8])
        # The other layer is untouched.
        assert handler.k_cache[0].abs().max() == 0

    def test_sequential_updates(self):
        """Multiple sequential updates should accumulate correctly."""
        n_layers, n_kv_heads, head_dim, max_seq_len = 1, 2, 4, 16
        handler, hidden_size = self._make_cache(n_layers, n_kv_heads, head_dim, max_seq_len)

        # First update: 2 tokens at offset 0
        k1 = torch.ones(1, hidden_size, 1, 2)
        v1 = torch.ones(1, hidden_size, 1, 2)
        handler.update_and_fetch(
            layer_idx=0,
            offset=torch.tensor([0], dtype=torch.int32),
            k=k1,
            v=v1,
            num_token_updates=2,
        )

        # Second update: 1 token at offset 2
        k2 = torch.ones(1, hidden_size, 1, 1) * 2.0
        v2 = torch.ones(1, hidden_size, 1, 1) * 2.0
        k_out, v_out = handler.update_and_fetch(
            layer_idx=0,
            offset=torch.tensor([2], dtype=torch.int32),
            k=k2,
            v=v2,
            num_token_updates=1,
        )

        # Position 2 should have value 2.0
        torch.testing.assert_close(k_out[:, :, :, 2:3], k2, rtol=1e-5, atol=1e-5)

    def test_different_layers(self):
        """Updates to different layers should be independent."""
        n_layers, n_kv_heads, head_dim, max_seq_len = 2, 1, 4, 8
        handler, hidden_size = self._make_cache(n_layers, n_kv_heads, head_dim, max_seq_len)

        k0 = torch.ones(1, hidden_size, 1, 1) * 1.0
        v0 = torch.ones(1, hidden_size, 1, 1) * 1.0
        k1 = torch.ones(1, hidden_size, 1, 1) * 2.0
        v1 = torch.ones(1, hidden_size, 1, 1) * 2.0

        handler.update_and_fetch(
            layer_idx=0,
            offset=torch.tensor([0], dtype=torch.int32),
            k=k0,
            v=v0,
            num_token_updates=1,
        )
        handler.update_and_fetch(
            layer_idx=1,
            offset=torch.tensor([0], dtype=torch.int32),
            k=k1,
            v=v1,
            num_token_updates=1,
        )

        # Layer 0 should have 1.0, layer 1 should have 2.0
        torch.testing.assert_close(
            handler.k_cache[0, :, :, :, 0:1],
            torch.ones(1, hidden_size, 1, 1),
        )
        torch.testing.assert_close(
            handler.k_cache[1, :, :, :, 0:1],
            torch.ones(1, hidden_size, 1, 1) * 2.0,
        )


class TestiOSKVCache:
    @staticmethod
    @pytest.mark.parametrize("with_head_dim", [False, True])
    def test_get_kv_cache_from_hf(with_head_dim: bool) -> None:
        """Test iOSKVCache.get_kv_cache_from_hf with and without head_dim in config."""

        # Mock config class with required attributes - using smaller dimensions for CI
        class MockConfig:
            def __init__(self):
                self.num_hidden_layers = 2
                self.hidden_size = 128
                self.num_attention_heads = 4
                self.num_key_value_heads = 2
                self.max_position_embeddings = 64
                self.head_dim = 32

        # Create config
        config = MockConfig()
        if not with_head_dim:
            delattr(config, "head_dim")

        # Call get_kv_cache_from_hf
        k_cache, v_cache = iOSKVCache.get_kv_cache_from_hf(config)

        # Verify shapes - iOS format: (n_layers, batch_size, n_kv_heads*head_dim, 1, max_seq_len)
        head_dim = 32 if with_head_dim else 32  # 128 // 4 = 32
        expected_shape = (2, 1, 2 * head_dim, 1, 64)

        assert k_cache.shape == expected_shape
        assert v_cache.shape == expected_shape
        assert k_cache.dtype == torch.float32
        assert v_cache.dtype == torch.float32

        # Verify caches are initialized to zeros
        assert torch.all(k_cache == 0.0)
        assert torch.all(v_cache == 0.0)

    @staticmethod
    def test_initialization() -> None:
        """Test iOSKVCache initialization with k_cache and v_cache tensors."""
        n_layers = 4
        batch_size = 1
        n_kv_heads = 4
        head_dim = 32
        max_seq_len = 64

        # iOS cache shape: (n_layers, batch_size, n_kv_heads*head_dim, 1, max_seq_len)
        k_cache = torch.zeros(n_layers, batch_size, n_kv_heads * head_dim, 1, max_seq_len)
        v_cache = torch.zeros(n_layers, batch_size, n_kv_heads * head_dim, 1, max_seq_len)

        ios_kv_cache = iOSKVCache(n_layers, n_kv_heads * head_dim)
        ios_kv_cache.register_kv_cache(k_cache, v_cache)

        assert ios_kv_cache.k_cache.shape == (
            n_layers,
            batch_size,
            n_kv_heads * head_dim,
            1,
            max_seq_len,
        )
        assert ios_kv_cache.v_cache.shape == (
            n_layers,
            batch_size,
            n_kv_heads * head_dim,
            1,
            max_seq_len,
        )
        assert ios_kv_cache.k_cache.dtype == torch.float32
        assert ios_kv_cache.v_cache.dtype == torch.float32

    @staticmethod
    def test_multiple_updates() -> None:
        """Test multiple sequential updates to iOSKVCache."""
        n_layers = 2
        batch_size = 1
        n_kv_heads = 2
        head_dim = 16
        max_seq_len = 32

        k_cache = torch.zeros(n_layers, batch_size, n_kv_heads * head_dim, 1, max_seq_len)
        v_cache = torch.zeros(n_layers, batch_size, n_kv_heads * head_dim, 1, max_seq_len)

        ios_kv_cache = iOSKVCache(n_layers, n_kv_heads * head_dim)
        ios_kv_cache.register_kv_cache(k_cache, v_cache)

        layer_idx = 1

        # First update: tokens 0-9
        offset1 = torch.tensor([0], dtype=torch.int32)
        num_tokens1 = 10
        # iOS cache expects shape: (batch_size, n_kv_heads*head_dim, 1, num_tokens)
        k1 = torch.randn(batch_size, n_kv_heads * head_dim, 1, num_tokens1)
        v1 = torch.randn(batch_size, n_kv_heads * head_dim, 1, num_tokens1)

        k_out1, v_out1 = ios_kv_cache.update_and_fetch(layer_idx, offset1, k1, v1, num_tokens1)

        # Second update: tokens 10-14
        offset2 = torch.tensor([10], dtype=torch.int32)
        num_tokens2 = 5
        k2 = torch.randn(batch_size, n_kv_heads * head_dim, 1, num_tokens2)
        v2 = torch.randn(batch_size, n_kv_heads * head_dim, 1, num_tokens2)

        k_out2, v_out2 = ios_kv_cache.update_and_fetch(layer_idx, offset2, k2, v2, num_tokens2)

        # Verify outputs have correct shape: (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
        assert k_out2.shape == (batch_size, n_kv_heads * head_dim, 1, max_seq_len)
        assert v_out2.shape == (batch_size, n_kv_heads * head_dim, 1, max_seq_len)

        # Verify both updates are in the cache for k_cache
        # Cache stores in shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
        # k1 and k2 are already in the right shape (batch_size, n_kv_heads*head_dim, 1, num_tokens)
        assert_close(ios_kv_cache.k_cache[layer_idx, :, :, :, :num_tokens1], k1)
        assert_close(ios_kv_cache.k_cache[layer_idx, :, :, :, 10:15], k2)

        # Verify both updates are in the cache for v_cache
        assert_close(ios_kv_cache.v_cache[layer_idx, :, :, :, :num_tokens1], v1)
        assert_close(ios_kv_cache.v_cache[layer_idx, :, :, :, 10:15], v2)
