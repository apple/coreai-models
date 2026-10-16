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

import math
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
    """Ground truth for concat_window_causal_mask: replay the ring writes, then apply the window."""
    slot_pos = [-1] * capacity
    for pos in range(offset):
        slot_pos[pos % capacity] = pos
    key_pos = slot_pos + [offset + j for j in range(query_len)]

    mask = torch.zeros(query_len, capacity + query_len, dtype=torch.bool)
    for i in range(query_len):
        q = offset + i
        for col, k in enumerate(key_pos):
            mask[i, col] = 0 <= k <= q and (q - k) < window_size
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
                assert torch.equal(got, want), (
                    f"capacity={capacity} window={window_size} "
                    f"offset={offset} query_len={query_len}"
                )

    def test_fetch_and_concat_does_not_mutate_cache(self):
        """fetch_and_concat is read-only, and update() does not alias its output."""
        capacity, n_kv, head_dim = 8, 2, 4
        k_buf = torch.randn(2, 1, n_kv, capacity, head_dim)
        v_buf = torch.randn(2, 1, n_kv, capacity, head_dim)
        cache = RingKVCache(k_buf.clone(), v_buf.clone())
        k = torch.randn(1, n_kv, 3, head_dim)
        v = torch.randn(1, n_kv, 3, head_dim)

        full_k, full_v = cache.fetch_and_concat(0, k, v)
        assert torch.equal(cache._k_cache, k_buf) and torch.equal(cache._v_cache, v_buf)
        assert torch.equal(full_k, torch.cat([k_buf[0], k], dim=2))
        assert torch.equal(full_v[:, :, capacity:], v)

        snapshot = full_k.clone()
        cache.update(0, offset=0, k=k, v=v, query_len=3)
        assert torch.equal(full_k, snapshot)
        assert torch.equal(cache._k_cache[0, :, :, :3], k)

    @pytest.mark.parametrize("drafter", [False, True], ids=["target", "drafter"])
    @pytest.mark.parametrize("chunk", [2, 4, 8, 16])
    def test_chunked_prefill_matches_decode_past_window(self, drafter, chunk):
        """Multi-token prefill matches token-by-token decode across three ring wraps.

        The target picks its ring path by ``prefill_mode`` (concat for prefill,
        write-first for decode); the drafter always uses concat.
        """
        torch.manual_seed(0)
        if drafter:
            config = _drafter_config(window=16)
            model = DrafterRingModel(config).eval()

            def new_caches():
                return (_make_ring_cache(config),)

            def forward(ids, pos, caches, prefill):
                return model(ids, pos, *caches)
        else:
            config = _glimmer_config(window=16)
            model = MuseGlimmerModel(config).eval().to(torch.float32)

            def new_caches():
                return _glimmer_caches(model, config)

            def forward(ids, pos, caches, prefill):
                return model(ids, pos, *caches, prefill_mode=prefill)

        seq_len = 48
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        def run(step):
            caches, outs = new_caches(), []
            for start in range(0, seq_len, step):
                end = min(start + step, seq_len)
                with torch.no_grad():
                    outs.append(
                        forward(
                            input_ids[:, start:end],
                            torch.arange(end).unsqueeze(0),
                            caches,
                            step > 1,
                        )
                    )
            return torch.cat(outs, dim=1)

        diff = (run(chunk) - run(1)).abs().max().item()
        assert diff < 1e-4, f"chunk={chunk} diverges from decode reference: {diff}"


# Primitive-level oracle: one ring layer vs brute-force attention over the true positions.
WINDOW, HEAD_DIM, N_HEADS = 8, 4, 2  # N_HEADS == n_kv keeps the oracle exact (no GQA)


def _masked_attention(q, keys, values, mask):
    """(heads, Q, d) x (heads, L, d) attention with a (Q, L) boolean mask."""
    scores = torch.einsum("hqd,hld->hql", q, keys) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hql,hld->hqd", torch.softmax(scores, dim=-1), values)


def _sliding_window_oracle(q, k_log, v_log, offset):
    """Attend each query over its true last-WINDOW positions from a full K/V log; no ring."""
    end = offset + q.shape[1]
    q_pos = torch.arange(offset, end)[:, None]
    k_pos = torch.arange(end)[None, :]
    mask = (k_pos <= q_pos) & (q_pos - k_pos < WINDOW)
    return _masked_attention(q, k_log[:, :end], v_log[:, :end], mask)


def _ring_attention(cache, q, k, v, offset, write_first):
    """One chunk through the ring: concat (read, attend, write) or write-first (``main``)."""
    q_len = q.shape[1]
    mask_args = dict(query_len=q_len, capacity=WINDOW, offset=offset, window_size=WINDOW)
    if write_first:
        keys, values = cache.update_and_fetch(0, offset, k, v, query_len=q_len)
        mask = ring_window_causal_mask(**mask_args, device="cpu")
        return _masked_attention(q, keys[0], values[0], mask)
    keys, values = cache.fetch_and_concat(0, k, v)
    out = _masked_attention(
        q, keys[0], values[0], concat_window_causal_mask(**mask_args, device="cpu")
    )
    cache.update(0, offset, k, v, query_len=q_len)
    return out


def _oracle_diffs(schedule, seed, write_first_last=False):
    """Run chunks of the given sizes through one ring layer; max diff vs the oracle per chunk."""
    torch.manual_seed(seed)
    total = sum(schedule)
    q_log, k_log, v_log = (torch.randn(N_HEADS, total, HEAD_DIM) for _ in range(3))
    cache = RingKVCache(
        torch.zeros(1, 1, N_HEADS, WINDOW, HEAD_DIM), torch.zeros(1, 1, N_HEADS, WINDOW, HEAD_DIM)
    )
    diffs, offset = [], 0
    for i, size in enumerate(schedule):
        assert offset % WINDOW + size <= WINDOW, "schedule wraps a ring write"
        sl = slice(offset, offset + size)
        q, k, v = q_log[:, sl], k_log[:, sl].unsqueeze(0), v_log[:, sl].unsqueeze(0)
        write_first = write_first_last and i == len(schedule) - 1
        out = _ring_attention(cache, q, k, v, offset, write_first)
        diffs.append((out - _sliding_window_oracle(q, k_log, v_log, offset)).abs().max().item())
        offset += size
    return diffs


def _ring_aligned_plan(token_count, chunk_size, start_offset, ring_capacity):
    """Mirror of the Swift ``prefillChunkSizes`` with a ring: chunks end at ring boundaries."""
    sizes, position, remaining = [], start_offset, token_count
    while remaining > 0:
        chunk = min(chunk_size, remaining, ring_capacity - position % ring_capacity)
        sizes.append(chunk)
        remaining -= chunk
        position += chunk
    return sizes


def _two_turn_plan(first_turn, second_turn, chunk_size):
    return _ring_aligned_plan(first_turn, chunk_size, 0, WINDOW) + _ring_aligned_plan(
        second_turn, chunk_size, first_turn, WINDOW
    )


class TestRingOracle:
    """The bug only shows for a multi-token chunk once the ring is full; the cases target that."""

    @pytest.mark.parametrize(
        "schedule",
        [
            pytest.param([8, 8, 1], id="decode-after-wrap"),
            pytest.param([8, 8, 8], id="window-chunk-after-wrap"),
            pytest.param([8, 4, 4], id="read-window-straddles-wrap"),
            pytest.param([8, 8, 2], id="Q2-after-wrap"),
            pytest.param([8, 8, 3], id="Q3-after-wrap"),
            pytest.param([8, 8, 4], id="Q4-after-wrap"),
            pytest.param(_two_turn_plan(37, 0, 2 * WINDOW), id="2W-chunk-split-at-ring"),
            *(
                pytest.param(_two_turn_plan(t, 23, WINDOW), id=f"turn2-starts-at-slot-{t}")
                for t in (3, 5, 11, 13)
            ),
        ],
    )
    def test_concat_matches_oracle(self, schedule):
        diffs = _oracle_diffs(schedule, seed=sum(schedule))
        assert max(diffs) < 1e-5, f"schedule={schedule}: {diffs}"

    def test_write_first_matches_oracle_at_decode(self):
        assert _oracle_diffs([8, 8, 1], seed=1, write_first_last=True)[-1] < 1e-5

    @pytest.mark.parametrize("chunk_len", [2, 4, 8])
    def test_write_first_diverges_for_multitoken_after_wrap(self, chunk_len):
        """The regime the Swift guard refuses: write-first evicts history the chunk still needs."""
        diff = _oracle_diffs([8, 8, chunk_len], seed=7, write_first_last=True)[-1]
        assert diff > 1e-2, f"write-first should diverge at Q={chunk_len}, diff={diff}"

    def test_wrapping_write_is_rejected_eagerly(self):
        cache = RingKVCache(
            torch.zeros(1, 1, N_HEADS, WINDOW, HEAD_DIM),
            torch.zeros(1, 1, N_HEADS, WINDOW, HEAD_DIM),
        )
        k = torch.randn(1, N_HEADS, WINDOW, HEAD_DIM)
        with pytest.raises(RuntimeError):
            cache.update(0, 5, k, k, query_len=WINDOW)


class TestConcatWriteIndexIsShapeSymint:
    """The ring write index must trace to a shape symint, never tensor data.

    The slice_update kernel crashes on the GPU backend when its write column
    comes from a runtime data tensor.
    """

    def test_write_index_derives_from_shape_not_data(self):
        capacity, n_kv, head_dim, query_len = 8, 1, 4, 4

        class _ConcatUpdate(torch.nn.Module):
            def forward(self, k, v, k_cache, v_cache, position_ids):
                seq_len, q_len = position_ids.shape[-1], k.shape[-2]
                torch._check_is_size(seq_len)
                torch._check_is_size(q_len)
                offset = seq_len - q_len
                torch._check_is_size(offset)
                cache = RingKVCache(k_cache, v_cache)
                full_k, full_v = cache.fetch_and_concat(0, k, v)
                cache.update(0, offset, k, v, query_len=q_len)
                return full_k + 0.0, full_v + 0.0

        kv = torch.zeros(1, n_kv, query_len, head_dim)
        kv_cache = torch.zeros(1, 1, n_kv, capacity, head_dim)
        position_ids = torch.arange(6, dtype=torch.int32)[None]
        seq = torch.export.Dim("seq", min=query_len, max=capacity * 4)
        exported = torch.export.export(
            _ConcatUpdate(),
            (kv, kv, kv_cache, kv_cache.clone(), position_ids),
            dynamic_shapes={
                "k": None,
                "v": None,
                "k_cache": None,
                "v_cache": None,
                "position_ids": {1: seq},
            },
        )

        nodes = {n.name: n for n in exported.graph.nodes}

        def ancestor_targets(name, seen=None):
            seen = set() if seen is None else seen
            if name not in seen and name in nodes:
                seen.add(name)
                for inp in nodes[name].all_input_nodes:
                    ancestor_targets(inp.name, seen)
            return seen

        writes = [n for n in exported.graph.nodes if "mutable_slice_update" in str(n.target)]
        assert writes, "concat path produced no ring write"
        for node in writes:
            begin = node.args[2]  # (x, update, begin, end)
            targets = {
                str(nodes[a].target)
                for a in ancestor_targets(begin.name)
                if nodes[a].op == "call_function"
            }
            assert any("sym_size" in t for t in targets), f"not shape-derived: {sorted(targets)}"
            assert not [t for t in targets if "_local_scalar_dense" in t or ".item" in t]
