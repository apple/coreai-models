# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS KVCache primitive."""

import pytest
import torch

import coreai_models.primitives.macos.cache as cache_module
from coreai_models.primitives.macos.cache import KVCache, SSMState
from tests._runner_infra.testing_utils import (
    assert_close,
    run_compare_coreai_explicit_kv_cache,
)


class TestmacOSKVCache:
    """Test KVCache: create, update, verify values."""

    def test_from_dimensions(self):
        """Cache should be created with correct shape."""
        n_layers, n_kv_heads, max_seq_len, head_dim = 2, 4, 32, 16
        cache = KVCache.from_dimensions(n_layers, n_kv_heads, max_seq_len, head_dim)
        assert cache._k_cache.shape == (n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        assert cache._v_cache.shape == (n_layers, 1, n_kv_heads, max_seq_len, head_dim)

    def test_update_and_fetch(self):
        """Values written via update_and_fetch should appear in the correct positions."""
        n_layers, n_kv_heads, max_seq_len, head_dim = 2, 4, 32, 16
        cache = KVCache.from_dimensions(n_layers, n_kv_heads, max_seq_len, head_dim)

        # Insert 3 tokens at offset 0, layer 0
        query_len = 3
        k = torch.randn(1, n_kv_heads, query_len, head_dim)
        v = torch.randn(1, n_kv_heads, query_len, head_dim)

        k_out, v_out = cache.update_and_fetch(layer_idx=0, offset=0, k=k, v=v, query_len=query_len)

        # k_out should have seq_len = offset + query_len = 3
        assert k_out.shape == (1, n_kv_heads, query_len, head_dim)
        assert v_out.shape == (1, n_kv_heads, query_len, head_dim)

        # The inserted values should match
        torch.testing.assert_close(k_out[:, :, :query_len, :], k)
        torch.testing.assert_close(v_out[:, :, :query_len, :], v)

    def test_sequential_updates(self):
        """Multiple sequential updates should accumulate correctly."""
        n_layers, n_kv_heads, max_seq_len, head_dim = 1, 2, 16, 8
        cache = KVCache.from_dimensions(n_layers, n_kv_heads, max_seq_len, head_dim)

        # First update: 2 tokens at offset 0
        k1 = torch.ones(1, n_kv_heads, 2, head_dim)
        v1 = torch.ones(1, n_kv_heads, 2, head_dim)
        cache.update_and_fetch(layer_idx=0, offset=0, k=k1, v=v1, query_len=2)

        # Second update: 1 token at offset 2
        k2 = torch.ones(1, n_kv_heads, 1, head_dim) * 2.0
        v2 = torch.ones(1, n_kv_heads, 1, head_dim) * 2.0
        k_out, v_out = cache.update_and_fetch(
            layer_idx=0, offset=2, k=k2, v=v2, query_len=1, seq_len=3
        )

        assert k_out.shape[-2] == 3
        # First 2 positions should be 1.0, third should be 2.0
        torch.testing.assert_close(k_out[:, :, 2:3, :], k2, rtol=1e-5, atol=1e-5)

    def test_seq_len_dim(self):
        """seq_len_dim should return 3 for macOS cache layout."""
        assert KVCache.seq_len_dim() == 3


# =============================================================================
# Functional-parity tests
# =============================================================================
#
# These tests exercise KVCache / SSMState basics plus an Core AI export path
# (via ``run_compare_coreai_explicit_kv_cache``) for explicit-KV-cache models.


class TestKVCache:
    @staticmethod
    def test_seq_len_dim() -> None:
        """Test that seq_len_dim returns the correct dimension index."""
        assert KVCache.seq_len_dim() == 3

    @staticmethod
    @pytest.mark.parametrize("with_head_dim", [0, 1])
    def test_create_cache_tensors(with_head_dim: int) -> None:
        """Test KVCache.create_cache_tensors with and without head_dim in config."""

        # Mock the Gemma3TextConfig class
        class MockGemma3TextConfig:
            def __init__(self):
                self.num_hidden_layers = 12
                self.hidden_size = 768
                self.num_attention_heads = 12
                self.num_key_value_heads = 6
                self.max_position_embeddings = 2048
                self.head_dim = 128

        config = MockGemma3TextConfig()
        if not with_head_dim:
            delattr(config, "head_dim")

        k_cache, v_cache = KVCache.create_cache_tensors(config)

        # Verify that the head_dim from config was used
        head_dim = 128 if with_head_dim else 64
        assert k_cache.shape == (12, 1, 6, 2048, head_dim)
        assert v_cache.shape == (12, 1, 6, 2048, head_dim)

    @staticmethod
    def test_internal_buffer_shape_and_dtype() -> None:
        """Verify KVCache.from_dimensions has correct shape and dtype for internal buffers."""
        n_layers = 12
        n_kv_heads = 8
        max_seq_len = 2048
        head_dim = 64

        kv_cache = KVCache.from_dimensions(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
        )

        assert kv_cache._k_cache.shape == (
            n_layers,
            1,
            n_kv_heads,
            max_seq_len,
            head_dim,
        )
        assert kv_cache._v_cache.shape == (
            n_layers,
            1,
            n_kv_heads,
            max_seq_len,
            head_dim,
        )
        assert kv_cache._k_cache.dtype == torch.float32
        assert kv_cache._v_cache.dtype == torch.float32

    @staticmethod
    def test_update_kv_cache() -> None:
        """Verify update_and_fetch correctly updates cache and returns updated values."""
        n_layers = 12
        n_kv_heads = 8
        max_seq_len = 2048
        head_dim = 64

        kv_cache = KVCache.from_dimensions(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
        )

        # update at layer 3
        seq_len = 10
        idx = 3
        k = torch.randn(1, n_kv_heads, seq_len, head_dim)
        v = torch.randn(1, n_kv_heads, seq_len, head_dim)

        k_cache, v_cache = kv_cache.update_and_fetch(idx, 0, k, v)

        assert k_cache.shape == (1, n_kv_heads, seq_len, head_dim)
        assert v_cache.shape == (1, n_kv_heads, seq_len, head_dim)

        assert_close(kv_cache._k_cache[idx, ..., :seq_len, :], k)
        assert_close(kv_cache._v_cache[idx, ..., :seq_len, :], v)

        # update at layer 2
        seq_len_2 = 5
        idx = 2
        k2 = torch.randn(1, n_kv_heads, seq_len_2, head_dim)
        v2 = torch.randn(1, n_kv_heads, seq_len_2, head_dim)

        k_cache, v_cache = kv_cache.update_and_fetch(idx, 0, k2, v2)

        assert k_cache.shape == (1, n_kv_heads, seq_len_2, head_dim)
        assert v_cache.shape == (1, n_kv_heads, seq_len_2, head_dim)
        assert_close(kv_cache._k_cache[idx, ..., :seq_len_2, :], k2)
        assert_close(kv_cache._v_cache[idx, ..., :seq_len_2, :], v2)

        # second update at layer 3
        seq_len_3 = 4
        idx = 2
        k3 = torch.randn(1, n_kv_heads, seq_len_3, head_dim)
        v3 = torch.randn(1, n_kv_heads, seq_len_3, head_dim)

        k_cache, v_cache = kv_cache.update_and_fetch(idx, seq_len_2, k3, v3)

        assert k_cache.shape == (1, n_kv_heads, seq_len_2 + seq_len_3, head_dim)
        assert v_cache.shape == (1, n_kv_heads, seq_len_2 + seq_len_3, head_dim)
        k_comb = torch.concat([k2, k3], axis=-2)
        v_comb = torch.concat([v2, v3], axis=-2)

        assert_close(kv_cache._k_cache[idx, ..., : seq_len_2 + seq_len_3, :], k_comb)
        assert_close(kv_cache._v_cache[idx, ..., : seq_len_2 + seq_len_3, :], v_comb)

    @staticmethod
    def test_coreai() -> None:
        """Test KVCache with explicit cache inputs through Core AI export and runtime."""

        N_LAYERS = 1
        N_KV_HEADS = 4
        MAX_SEQ = 128
        HEAD_DIM = 64
        HIDDEN = N_KV_HEADS * HEAD_DIM
        VOCAB = 16
        SEQ_LEN = 10

        class KVCacheModel(torch.nn.Module):
            """Minimal model that exercises KVCache with explicit inputs."""

            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
                self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                k_cache: torch.Tensor,
                v_cache: torch.Tensor,
            ) -> torch.Tensor:
                cache = KVCache(k_cache, v_cache)
                h = self.embed(input_ids)
                B, Q, _ = h.shape
                seq_len = position_ids.shape[-1]
                offset = seq_len - Q
                kv = h.reshape(B, Q, N_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
                cache.update_and_fetch(0, offset, kv, kv, seq_len=seq_len, query_len=Q)
                return self.lm_head(h)

        model = KVCacheModel().eval()

        input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), dtype=torch.int32)
        position_ids = torch.arange(SEQ_LEN, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(N_LAYERS, 1, N_KV_HEADS, MAX_SEQ, HEAD_DIM)
        v_cache = torch.zeros(N_LAYERS, 1, N_KV_HEADS, MAX_SEQ, HEAD_DIM)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=(input_ids, position_ids, k_cache, v_cache),
            dynamic_shapes={
                "input_ids": {},
                "position_ids": {},
                "k_cache": {},
                "v_cache": {},
            },
            atol=1e-4,
            rtol=1e-4,
        )

    @staticmethod
    def test_explicit_kv_cache_export_and_runtime() -> None:
        """Test that a model taking KV cache as explicit inputs can be exported and run via Core AI.

        This verifies the pattern used by macOS models (e.g., Qwen2ForCausalLM) where
        forward(input_ids, position_ids, k_cache, v_cache) -> (logits, k_cache, v_cache).
        """
        N_KV_HEADS = 2
        HEAD_DIM = 4
        MAX_SEQ = 32
        HIDDEN = N_KV_HEADS * HEAD_DIM
        VOCAB = 16
        QUERY_LEN = 4

        class TinyKVModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
                self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

            def forward(self, input_ids, position_ids, k_cache, v_cache):
                cache = KVCache(k_cache, v_cache)
                h = self.embed(input_ids)
                B, Q, _ = h.shape
                seq_len = position_ids.shape[-1]
                offset = seq_len - Q
                kv = h.reshape(B, Q, N_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
                cache.update_and_fetch(0, offset, kv, kv, seq_len=seq_len, query_len=Q)
                return self.lm_head(h)

        model = TinyKVModel().eval()

        input_ids = torch.randint(0, VOCAB, (1, QUERY_LEN), dtype=torch.int32)
        position_ids = torch.arange(QUERY_LEN + 2, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(1, 1, N_KV_HEADS, MAX_SEQ, HEAD_DIM)
        v_cache = torch.zeros(1, 1, N_KV_HEADS, MAX_SEQ, HEAD_DIM)
        inputs = (input_ids, position_ids, k_cache, v_cache)

        dynamic_shapes = {
            "input_ids": {1: torch.export.Dim("seq_ids", max=MAX_SEQ - 1)},
            "position_ids": {1: torch.export.Dim("seq_pos", min=1, max=MAX_SEQ - 1)},
            "k_cache": {},
            "v_cache": {},
        }

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
        )


class TestSSMState:
    @staticmethod
    def test_initialization_and_property() -> None:
        """Test SSMState initialization and states property."""
        n_layers = 8
        batch_size = 1
        state_dim = 256

        # Create initial states tensor
        states = torch.zeros(n_layers, batch_size, state_dim)

        # Initialize SSMState
        ssm_state = SSMState(states)

        # Verify property access
        assert ssm_state.states.shape == (n_layers, batch_size, state_dim)
        assert ssm_state.states.dtype == torch.float32
        assert_close(ssm_state.states, states)

    @staticmethod
    def test_update_states() -> None:
        """Test update_states correctly updates the state cache at specific layers."""
        n_layers = 8
        batch_size = 1
        state_dim = 256

        # Create initial states tensor
        states = torch.zeros(n_layers, batch_size, state_dim)
        ssm_state = SSMState(states)

        # Update states at different layers and verify
        indices = range(n_layers)
        res = []
        for i in indices:
            new_state = torch.randn(batch_size, state_dim)
            res.append(new_state)
            ssm_state.update_states(i, new_state)

            # Verify all previously updated states are correct
            for j in range(i + 1):
                assert_close(ssm_state.states[j], res[j])

    @staticmethod
    @pytest.mark.parametrize("state_dims", [(4, 6), (3, 4, 5)])
    def test_update_states_slice_bounds_cover_all_dims(monkeypatch, state_dims) -> None:
        """update_states must build begin/end that span every cache dimension.

        AIProgram.optimize rejects a slice_update whose begin and end ranks
        differ; eager right-pads via zip(strict=False) and never surfaces the
        mismatch, so capture the indices update_states passes down and assert
        the equal-rank invariant directly (plus the written values).
        """
        n_layers, batch_size = 2, 1
        states = torch.zeros(n_layers, batch_size, *state_dims)
        ssm_state = SSMState(states)

        captured: dict[str, torch.Tensor] = {}
        real_slice_update = cache_module.mutable_slice_update

        def spy(*args, **kwargs):
            captured["begin"] = kwargs["begin"]
            captured["end"] = kwargs["end"]
            return real_slice_update(*args, **kwargs)

        monkeypatch.setattr(cache_module, "mutable_slice_update", spy)

        layer_idx = 1
        new_state = torch.randn(batch_size, *state_dims)
        ssm_state.update_states(layer_idx, new_state)

        assert captured["begin"].numel() == states.dim()
        assert captured["end"].numel() == states.dim()

        assert_close(ssm_state.states[layer_idx], new_state)
        assert_close(ssm_state.states[0], torch.zeros(batch_size, *state_dims))
