# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Ring buffer KV cache parity tests.

Validates that incremental decode with the ring buffer produces the same
logits as full recompute (ground truth), both within and across the window
boundary (wrap-around). Uses chunked prefill to stay within buffer capacity.

TestConcatWindowAttention covers the read-before-write path used by
muse_glimmer: fetch_and_concat -> attention -> update, which (unlike
update_and_fetch) is correct for multi-token prefill chunks.
"""

from types import SimpleNamespace

import pytest
import torch

from coreai_models.models.macos.muse_glimmer import MuseGlimmerModel
from coreai_models.models.macos.muse_glimmer_drafter_ring import DrafterRingModel
from coreai_models.primitives.macos.cache import (
    KVCache,
    RingKVCache,
    concat_window_causal_mask,
    ring_window_causal_mask,
)


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


def _simulate_ring_mask(query_len, capacity, offset, window_size):
    """Ground truth for concat_window_causal_mask.

    Replays the ring writes for every position before ``offset`` to learn which
    absolute position each slot physically holds, then applies the sliding
    window predicate directly. Deliberately naive — no modular arithmetic
    tricks to mirror the implementation's.
    """
    slot_pos = [-1] * capacity
    for pos in range(offset):
        slot_pos[pos % capacity] = pos

    mask = torch.zeros(query_len, capacity + query_len, dtype=torch.bool)
    for i in range(query_len):
        q = offset + i
        for slot in range(capacity):
            k = slot_pos[slot]
            mask[i, slot] = k >= 0 and k <= q and (q - k) < window_size
        for j in range(query_len):
            k = offset + j
            mask[i, capacity + j] = k <= q and (q - k) < window_size
    return mask


def _glimmer_config(window: int = 16, n_layers: int = 4) -> SimpleNamespace:
    """All-sliding Muse Glimmer config, so tests isolate the ring path."""
    return SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        vocab_size=128,
        rms_norm_eps=1e-5,
        sliding_window=window,
        layer_types=["sliding_attention"] * n_layers,
        layer_rope_theta=[500000.0] * n_layers,
        qk_scale_factor=1.0,
        max_position_embeddings=256,
    )


def _glimmer_caches(model, config, global_len: int = 256):
    def zeros(n_layers, seq_len):
        return torch.zeros(n_layers, 1, config.num_key_value_heads, seq_len, config.head_dim)

    n_global = max(model.n_global_layers, 1)
    n_sliding = model.n_sliding_layers
    window = config.sliding_window
    return (
        KVCache(zeros(n_global, global_len), zeros(n_global, global_len)),
        RingKVCache(zeros(n_sliding, window), zeros(n_sliding, window)),
    )


class TestConcatWindowAttention:
    """Read-before-write ring path (fetch_and_concat -> attend -> update)."""

    @pytest.mark.parametrize("capacity", [8, 16])
    @pytest.mark.parametrize("window_divisor", [1, 2])
    def test_mask_matches_ring_simulation(self, capacity, window_divisor):
        """Mask matches a naive replay of the ring, at every offset through 3 wraps."""
        window_size = capacity // window_divisor
        for offset in range(0, 3 * capacity + 3):
            for query_len in (1, 2, 3, capacity // 2, capacity):
                got = concat_window_causal_mask(
                    query_len=query_len,
                    capacity=capacity,
                    offset=offset,
                    window_size=window_size,
                    device="cpu",
                )
                want = _simulate_ring_mask(query_len, capacity, offset, window_size)
                assert got.shape == (query_len, capacity + query_len)
                assert torch.equal(got, want), (
                    f"capacity={capacity} window={window_size} "
                    f"offset={offset} query_len={query_len}"
                )

    def test_mask_from_empty_cache_ignores_ring(self):
        """At offset=0 nothing is cached, so only the new-key block is live."""
        mask = concat_window_causal_mask(
            query_len=4, capacity=8, offset=0, window_size=8, device="cpu"
        )
        assert mask.shape == (4, 12)
        assert mask[:, :8].sum() == 0  # ring is entirely unwritten
        # New-key block is plain causal
        assert torch.equal(mask[:, 8:], torch.tril(torch.ones(4, 4, dtype=torch.bool)))

    def test_mask_excludes_slot_about_to_be_overwritten(self):
        """With a full ring, the oldest slot falls outside every query's window."""
        capacity = 8
        mask = concat_window_causal_mask(
            query_len=1, capacity=capacity, offset=16, window_size=capacity, device="cpu"
        )
        # Query at position 16 attends to 9..16: 7 cached slots + itself.
        assert mask[:, :capacity].sum() == capacity - 1
        assert mask[0, capacity] == 1
        # The excluded slot holds position 8 (16 - capacity), written at 8 % 8 = 0.
        assert mask[0, 0] == 0

    def test_fetch_and_concat_does_not_mutate_cache(self):
        """fetch_and_concat is read-only; the write happens in update()."""
        capacity, n_layers, n_kv, head_dim = 8, 2, 2, 4
        k_buf = torch.randn(n_layers, 1, n_kv, capacity, head_dim)
        v_buf = torch.randn(n_layers, 1, n_kv, capacity, head_dim)
        cache = RingKVCache(k_buf.clone(), v_buf.clone())

        k = torch.randn(1, n_kv, 3, head_dim)
        v = torch.randn(1, n_kv, 3, head_dim)
        full_k, full_v = cache.fetch_and_concat(0, k, v)

        assert full_k.shape == (1, n_kv, capacity + 3, head_dim)
        assert torch.equal(cache._k_cache, k_buf), "fetch_and_concat mutated the K cache"
        assert torch.equal(cache._v_cache, v_buf), "fetch_and_concat mutated the V cache"
        # Concatenated result is [cached window, new keys]
        assert torch.equal(full_k[:, :, :capacity], k_buf[0])
        assert torch.equal(full_k[:, :, capacity:], k)
        assert torch.equal(full_v[:, :, capacity:], v)

        # The subsequent update must not disturb the already-returned tensors.
        snapshot = full_k.clone()
        cache.update(0, offset=0, k=k, v=v, query_len=3)
        assert torch.equal(full_k, snapshot), "update() aliased fetch_and_concat output"
        assert torch.equal(cache._k_cache[0, :, :, :3], k)

    @pytest.mark.parametrize("chunk", [2, 4, 8, 16])
    def test_chunked_prefill_matches_decode_past_window(self, chunk):
        """Multi-token prefill matches token-by-token decode, well past the window.

        Regression test for history loss during prefill: a write-first ring lets
        a chunk's own K/V evict the oldest ``chunk - 1`` in-window positions
        before the chunk's earliest queries read them. Decode (query_len=1)
        never hits that, so it is the trusted reference here.

        ``chunk`` divides the window so no write straddles the ring boundary.
        """
        config = _glimmer_config(window=16)
        torch.manual_seed(0)
        model = MuseGlimmerModel(config).eval().to(torch.float32)

        seq_len = 48  # 3 full wraps of the 16-slot ring
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        def run(step):
            global_cache, sliding_cache = _glimmer_caches(model, config)
            outs = []
            for start in range(0, seq_len, step):
                end = min(start + step, seq_len)
                with torch.no_grad():
                    outs.append(
                        model(
                            input_ids[:, start:end],
                            torch.arange(end).unsqueeze(0),
                            global_cache,
                            sliding_cache,
                        )
                    )
            return torch.cat(outs, dim=1)

        reference = run(1)
        diff = (run(chunk) - reference).abs().max().item()
        assert diff < 1e-4, f"chunk={chunk} diverges from decode reference: {diff}"

    @pytest.mark.parametrize("chunk", [2, 4, 8, 16])
    def test_drafter_chunked_prefill_matches_decode_past_window(self, chunk):
        """Same regression check for the drafter, whose layers are all sliding.

        TestRingBufferParity.test_chunked_prefill_matches_single_prefill cannot
        catch this: it prefills 48 tokens into a 64-slot ring, so the buffer
        never fills and no history is evicted. Here the ring wraps three times.
        """
        config = _drafter_config(window=16)
        torch.manual_seed(11)
        model = DrafterRingModel(config)
        model.eval()

        seq_len = 48  # 3 full wraps of the 16-slot ring
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        def run(step):
            cache = _make_ring_cache(config)
            outs = []
            for start in range(0, seq_len, step):
                end = min(start + step, seq_len)
                with torch.no_grad():
                    outs.append(
                        model(input_ids[:, start:end], torch.arange(end).unsqueeze(0), cache)
                    )
            return torch.cat(outs, dim=1)

        reference = run(1)
        diff = (run(chunk) - reference).abs().max().item()
        assert diff < 1e-4, f"chunk={chunk} diverges from decode reference: {diff}"
