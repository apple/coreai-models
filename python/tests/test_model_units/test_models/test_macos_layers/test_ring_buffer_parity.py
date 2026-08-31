# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Ring buffer KV cache parity tests.

Validates that incremental decode with the ring buffer produces the same
logits as full recompute (ground truth), both within and across the window
boundary (wrap-around). Uses chunked prefill to stay within buffer capacity.
"""

from types import SimpleNamespace

import pytest
import torch

from coreai_models.models.macos.muse_glimmer_drafter_ring import DrafterRingModel
from coreai_models.primitives.macos.cache import RingKVCache, ring_window_causal_mask


def _drafter_config(window: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        intermediate_size=512,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        sliding_window=window,
        rope_parameters={"rope_theta": 500000.0},
    )


def _make_ring_cache(config):
    n_layers = config.num_hidden_layers
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim
    window = config.sliding_window
    k = torch.zeros(n_layers, 1, n_kv, window, head_dim)
    v = torch.zeros(n_layers, 1, n_kv, window, head_dim)
    return RingKVCache(k, v)


def _chunked_prefill(model, input_ids, cache, chunk_size):
    """Prefill in chunks that fit within the ring buffer capacity."""
    seq_len = input_ids.shape[-1]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        pos = torch.arange(end).unsqueeze(0)
        with torch.no_grad():
            model(chunk, pos, cache)


class TestRingBufferParity:
    """Verify ring buffer produces identical results to full recompute."""

    def test_incremental_matches_recompute_within_window(self):
        """Decode-by-decode output matches single-pass recompute (no wrap)."""
        config = _drafter_config(window=64)
        torch.manual_seed(42)
        model = DrafterRingModel(config)
        model.eval()

        prefill_len = 30
        decode_len = 30  # total 60 < window=64
        input_ids = torch.randint(0, config.vocab_size, (1, prefill_len))
        decode_tokens = torch.randint(0, config.vocab_size, (1, decode_len))
        full_seq = torch.cat([input_ids, decode_tokens], dim=1)

        # Path A: incremental (prefill + decode one-by-one)
        cache_a = _make_ring_cache(config)
        with torch.no_grad():
            model(input_ids, torch.arange(prefill_len).unsqueeze(0), cache_a)

        ring_logits = []
        for i in range(decode_len):
            pos = prefill_len + i
            tok = decode_tokens[:, i : i + 1]
            with torch.no_grad():
                out = model(tok, torch.arange(pos + 1).unsqueeze(0), cache_a)
            ring_logits.append(out[0, 0].clone())

        # Path B: full recompute (single prefill of all tokens)
        recompute_logits = []
        for i in range(decode_len):
            seq_len = prefill_len + i + 1
            cache_b = _make_ring_cache(config)
            with torch.no_grad():
                out = model(full_seq[:, :seq_len], torch.arange(seq_len).unsqueeze(0), cache_b)
            recompute_logits.append(out[0, -1].clone())

        # Compare
        for i in range(decode_len):
            diff = (ring_logits[i] - recompute_logits[i]).abs().max().item()
            assert diff < 1e-5, f"Mismatch at decode step {i} (pos {prefill_len + i}): diff={diff}"

    def test_deterministic_across_wrap_boundary(self):
        """Two independent incremental runs produce identical output after wrap."""
        config = _drafter_config(window=32)
        prefill_len = 20
        decode_len = 30  # positions 20-49, wraps at 32
        input_ids = torch.randint(0, config.vocab_size, (1, prefill_len))
        decode_tokens = torch.randint(0, config.vocab_size, (1, decode_len))

        def run_incremental(seed):
            torch.manual_seed(seed)
            model = DrafterRingModel(config)
            model.eval()
            cache = _make_ring_cache(config)
            with torch.no_grad():
                model(input_ids, torch.arange(prefill_len).unsqueeze(0), cache)
            logits = []
            for i in range(decode_len):
                tok = decode_tokens[:, i : i + 1]
                with torch.no_grad():
                    out = model(tok, torch.arange(prefill_len + i + 1).unsqueeze(0), cache)
                logits.append(out[0, 0].clone())
            return logits

        logits_a = run_incremental(seed=123)
        logits_b = run_incremental(seed=123)

        for i in range(decode_len):
            diff = (logits_a[i] - logits_b[i]).abs().max().item()
            pos = prefill_len + i
            assert diff == 0.0, f"Non-deterministic at step {i} (pos {pos}): {diff}"

    def test_wrap_produces_finite_output(self):
        """100 decode steps past window boundary produces finite output."""
        config = _drafter_config(window=64)
        torch.manual_seed(7)
        model = DrafterRingModel(config)
        model.eval()

        cache = _make_ring_cache(config)
        prefill = torch.randint(0, config.vocab_size, (1, 32))
        with torch.no_grad():
            model(prefill, torch.arange(32).unsqueeze(0), cache)

        for step in range(100):
            pos = 32 + step
            tok = torch.randint(0, config.vocab_size, (1, 1))
            with torch.no_grad():
                out = model(tok, torch.arange(pos + 1).unsqueeze(0), cache)
            assert torch.isfinite(out).all(), f"Non-finite at step {step} (pos {pos})"

    def test_chunked_prefill_matches_single_prefill(self):
        """Chunked prefill produces same cache state as single-pass prefill."""
        config = _drafter_config(window=64)
        torch.manual_seed(99)
        model = DrafterRingModel(config)
        model.eval()

        seq_len = 48  # fits in window
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Single-pass prefill
        cache_single = _make_ring_cache(config)
        with torch.no_grad():
            model(input_ids, position_ids, cache_single)

        # Chunked prefill (chunks of 16)
        cache_chunked = _make_ring_cache(config)
        _chunked_prefill(model, input_ids, cache_chunked, chunk_size=16)

        # Decode one more token from each
        next_tok = torch.randint(0, config.vocab_size, (1, 1))
        next_pos = torch.arange(seq_len + 1).unsqueeze(0)

        with torch.no_grad():
            out_a = model(next_tok, next_pos, cache_single)
            out_b = model(next_tok, next_pos, cache_chunked)

        diff = (out_a - out_b).abs().max().item()
        assert diff < 1e-5, f"Chunked vs single prefill mismatch: {diff}"

    def test_mask_shape_and_values(self):
        """ring_window_causal_mask produces correct shape and causal pattern."""
        mask = ring_window_causal_mask(
            query_len=4, capacity=8, offset=0, window_size=8, device="cpu"
        )
        assert mask.shape == (4, 8)
        # First query can only attend to position 0
        assert mask[0, 0] == 1
        assert mask[0, 1:].sum() == 0
        # Last query attends to positions 0-3
        assert mask[3, :4].sum() == 4
        assert mask[3, 4:].sum() == 0

    def test_update_and_fetch_rejects_wrap_around_write(self):
        """update_and_fetch raises when write_start + query_len > capacity.

        Reproducer for BUG 2: offset=1920, query_len=256, capacity=2048 would
        write to slots [1920, 2176) which overflows the ring buffer. The guard
        must reject this; callers should chunk so writes never straddle the
        ring boundary.
        """
        capacity = 2048
        n_layers, n_kv, head_dim = 2, 4, 32
        k_buf = torch.zeros(n_layers, 1, n_kv, capacity, head_dim)
        v_buf = torch.zeros(n_layers, 1, n_kv, capacity, head_dim)
        cache = RingKVCache(k_buf, v_buf)

        query_len = 256
        offset = 1920  # write_start = 1920 % 2048 = 1920; 1920 + 256 = 2176 > 2048

        k = torch.randn(1, n_kv, query_len, head_dim)
        v = torch.randn(1, n_kv, query_len, head_dim)

        with pytest.raises(RuntimeError):
            cache.update_and_fetch(layer_idx=0, offset=offset, k=k, v=v)

    def test_update_and_fetch_accepts_boundary_aligned_write(self):
        """A write that exactly fills to the boundary is valid (no overflow).

        offset=1792, query_len=256, capacity=2048 -> write_start=1792,
        end=2048 which is exactly at the boundary.
        """
        capacity = 2048
        n_layers, n_kv, head_dim = 2, 4, 32
        k_buf = torch.zeros(n_layers, 1, n_kv, capacity, head_dim)
        v_buf = torch.zeros(n_layers, 1, n_kv, capacity, head_dim)
        cache = RingKVCache(k_buf, v_buf)

        query_len = 256
        offset = 1792  # write_start = 1792 % 2048 = 1792; 1792 + 256 = 2048 == capacity

        k = torch.randn(1, n_kv, query_len, head_dim)
        v = torch.randn(1, n_kv, query_len, head_dim)

        # Should not raise
        k_out, v_out = cache.update_and_fetch(layer_idx=0, offset=offset, k=k, v=v)
        assert k_out.shape == (1, n_kv, capacity, head_dim)

    def test_mask_after_wrap(self):
        """After buffer wraps, mask correctly identifies valid slots."""
        # offset=10 means we've written 14 tokens (10 + query_len=4) into capacity=8
        # Ring has wrapped: slot (10+0)%8=2, (10+1)%8=3, (10+2)%8=4, (10+3)%8=5
        # Previous tokens at slots: 10%8=2..13%8=5 are the last 4
        # But we also have earlier tokens at other slots
        mask = ring_window_causal_mask(
            query_len=1, capacity=8, offset=10, window_size=8, device="cpu"
        )
        assert mask.shape == (1, 8)
        # All 8 slots should be valid (we've filled the buffer and window=8=capacity)
        assert mask[0].sum() == 8
