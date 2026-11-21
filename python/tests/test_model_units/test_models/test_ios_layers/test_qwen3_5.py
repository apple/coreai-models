# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit and parity tests for the Qwen3.5 iOS hybrid decoder.

The conv state is a ring buffer read back through a one-hot selector, so its correctness
depends on three values the runner computes (`in_step`, `conv_step` and the selector's
`block_start`) agreeing with each other on every call. The parity tests here check that.

Every parity assertion is paired with a mutation check that perturbs the value under test
and asserts the comparison then fails.
"""

import functools
import inspect
import math
import pathlib

import pytest
import torch

pytest.importorskip("transformers")

try:  # transformers>=5.5 (qwen3_5)
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
except Exception:  # pragma: no cover - requires transformers>=5.5
    Qwen3_5Config = None
    Qwen3_5TextConfig = None
    Qwen3_5ForCausalLM = None

from coreai_models._constants import EXTEND_FUNCTION_NAME  # noqa: E402
from coreai_models.export import pipeline as export_pipeline  # noqa: E402
from coreai_models.llm.export import _load_compression_config_object  # noqa: E402
from coreai_models.models.ios.qwen3_5 import (  # noqa: E402
    DECODE_QUERY_LEN,
    PREFILL_QUERY_LEN,
    Qwen3_5Extend,
    Qwen3_5ForCausalLMForiOS,
    _chunk_gated_delta_rule,
    _l2norm,
    _layer_maps,
)
from coreai_models.primitives.ios.cache import KVCacheHandler  # noqa: E402
from coreai_models.primitives.ios.ssm_cache import (  # noqa: E402
    MAX_QUERY_LEN,
    SSMStateHandler,
)

pytestmark = pytest.mark.skipif(
    Qwen3_5ForCausalLM is None, reason="qwen3_5 requires transformers>=5.5"
)

CTX = 128
QUERY_LEN = 8
# Not a multiple of QUERY_LEN, so the last block is padded.
N_TOKENS = 21


def tiny_config(n_layers: int = 4, ctx: int = CTX) -> Qwen3_5TextConfig:
    """A 4-layer Qwen3.5 whose shapes are small enough to run on CPU in a test.

    Built from the checkpoint's config class, so the test needs no network and no HF cache.
    """
    cfg = Qwen3_5TextConfig()
    cfg.num_hidden_layers = n_layers
    cfg.layer_types = list(cfg.layer_types)[:n_layers]
    cfg.vocab_size = 512
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.linear_num_key_heads = 2
    cfg.linear_num_value_heads = 4
    cfg.linear_key_head_dim = 16
    cfg.linear_value_head_dim = 16
    cfg.max_position_embeddings = ctx
    # The checkpoints tie the LM head to the embedding table, and our decoder assumes it —
    # a default-constructed config does not, which would give a comparison model an
    # independent random head.
    cfg.tie_word_embeddings = True
    return cfg


def _ios_export_config(monkeypatch, **kwargs) -> "export_pipeline.ExportConfig":
    """An iOS export of Qwen3.5 whose checkpoint config is the tiny one, read without network."""
    hf_config = Qwen3_5Config(text_config=tiny_config().to_dict())
    monkeypatch.setattr(export_pipeline.AutoConfig, "from_pretrained", lambda *_, **__: hf_config)
    return export_pipeline.ExportConfig(
        hf_model_id="Qwen/Qwen3.5-2B", variant="iOS", compression="none", **kwargs
    )


def randomize(model: torch.nn.Module) -> None:
    """Replace every weight with noise so no logit is structurally zero."""
    for p in model.parameters():
        if p.data.numel() <= 1:
            continue  # emb_scale / emb_zero_point: leave the quantization params alone
        if p.data.is_floating_point():
            p.data = torch.randn_like(p.data) * 0.05
        else:
            # The tied embedding table is int8; an all-zero table makes every logit zero
            # and any comparison against it vacuous.
            p.data = torch.randint(-8, 9, p.data.shape, dtype=p.data.dtype)


def fresh_state(cfg, ctx: int = CTX):
    dims = Qwen3_5ForCausalLMForiOS._deltanet_dims(cfg)
    kv = torch.zeros(dims["n_attn"], 1, Qwen3_5ForCausalLMForiOS._kv_embed_size(cfg), 1, ctx)
    conv = torch.zeros(dims["n_delta"], dims["conv_dim"], 1, dims["ring_width"])
    recurrent = torch.zeros(
        dims["n_delta"], dims["num_v_heads"], dims["head_k_dim"], dims["head_v_dim"]
    )
    return kv, kv.clone(), conv, recurrent, dims


def causal_mask(ctx: int, block_start: int, query_len: int) -> torch.Tensor:
    """(1, ctx, 1, q) additive mask: query i (absolute `block_start + i`) sees key j <= it."""
    mask = torch.full((1, ctx, 1, query_len), -40000.0)
    for i in range(query_len):
        mask[0, : block_start + i + 1, 0, i] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Layer-index maps
# ---------------------------------------------------------------------------


class TestLayerMaps:
    def test_every_layer_gets_exactly_one_index(self):
        cfg = tiny_config(n_layers=8)
        attn_idx, delta_idx, n_attn, n_delta = _layer_maps(cfg)
        assert n_attn + n_delta == cfg.num_hidden_layers
        for layer, kind in enumerate(cfg.layer_types):
            if kind == "linear_attention":
                assert delta_idx[layer] >= 0 and attn_idx[layer] == -1
            else:
                assert attn_idx[layer] >= 0 and delta_idx[layer] == -1

    def test_indices_are_dense_and_ordered(self):
        """Each family's indices must be 0..n-1 in layer order — they index state slots."""
        cfg = tiny_config(n_layers=8)
        attn_idx, delta_idx, n_attn, n_delta = _layer_maps(cfg)
        assert [i for i in attn_idx if i >= 0] == list(range(n_attn))
        assert [i for i in delta_idx if i >= 0] == list(range(n_delta))

    def test_matches_the_3_to_1_pattern(self):
        cfg = tiny_config(n_layers=8)
        _, _, n_attn, n_delta = _layer_maps(cfg)
        assert (n_delta, n_attn) == (6, 2)


# ---------------------------------------------------------------------------
# Conv-ring seed selector
# ---------------------------------------------------------------------------


class TestSeedSelector:
    RING, KW, Q = 16, 3, 8

    def test_shape_is_window_wide_and_ring_tall(self):
        sel = SSMStateHandler.build_seed_selector(self.RING, self.KW, 0, self.Q)
        assert sel.shape == (1, self.RING, self.KW + self.Q)

    def test_exactly_one_hot_per_column(self):
        sel = SSMStateHandler.build_seed_selector(self.RING, self.KW, 5, self.Q)
        assert torch.equal(sel.sum(dim=1).flatten(), torch.ones(self.KW + self.Q))

    @pytest.mark.parametrize("block_start", [0, 1, 5, 8, 15, 16, 31])
    def test_column_j_selects_position_block_start_minus_window_plus_j(self, block_start):
        """The window is the KW causal left-context positions, then the block's own."""
        sel = SSMStateHandler.build_seed_selector(self.RING, self.KW, block_start, self.Q)
        for j in range(self.KW + self.Q):
            expected_row = (block_start - self.KW + j) % self.RING
            assert sel[0, expected_row, j] == 1.0

    def test_wraps_past_the_end_of_the_ring(self):
        """A window straddling the ring's end must select both the tail and the head."""
        sel = SSMStateHandler.build_seed_selector(self.RING, self.KW, self.RING - 2, self.Q)
        rows = sel[0].argmax(dim=0).tolist()
        assert max(rows) > min(rows) + self.Q, "expected a wrap, got a contiguous window"
        assert set(rows) == {
            (self.RING - 2 - self.KW + j) % self.RING for j in range(self.KW + self.Q)
        }

    def test_negative_positions_select_the_ring_tail(self):
        """Position 0's left context is negative; it must read the tail, which is zero
        on a fresh ring — exactly the padding the first token should see."""
        sel = SSMStateHandler.build_seed_selector(self.RING, self.KW, 0, self.Q)
        for j in range(self.KW):
            assert sel[0, (-self.KW + j) % self.RING, j] == 1.0


# ---------------------------------------------------------------------------
# Whole-decoder parity: blocked calls vs one shot
# ---------------------------------------------------------------------------


def run_one_shot(model, ids, cfg, n_tokens):
    kv, v, conv, recurrent, dims = fresh_state(cfg)
    with torch.no_grad():
        out = model(
            ids,
            torch.arange(n_tokens).to(torch.uint16).unsqueeze(0),
            torch.zeros((1,), dtype=torch.int32),
            causal_mask(CTX, 0, n_tokens),
            torch.ones(1, n_tokens),
            torch.zeros((1,), dtype=torch.int32),
            SSMStateHandler.build_seed_selector(
                dims["ring_width"], dims["kernel_window"], 0, n_tokens, dtype=torch.float32
            ),
            kv,
            v,
            conv,
            recurrent,
        )
    last = out.reshape(n_tokens, -1)[-1]
    return last, recurrent.clone()


def run_blocked(model, ids, cfg, n_tokens, query_len=QUERY_LEN, conv_step_bias=0):
    """Walk the sequence in `query_len`-wide blocks, carrying state across calls.

    `conv_step_bias` exists for the mutation check.
    """
    kv, v, conv, recurrent, dims = fresh_state(cfg)
    ring_w, kw = dims["ring_width"], dims["kernel_window"]
    processed, out, last_start = 0, None, 0
    while processed < n_tokens:
        start = (processed // query_len) * query_len
        block = torch.zeros(1, query_len, dtype=torch.int32)
        n_real = min(query_len, n_tokens - start)
        block[0, :n_real] = ids[0, start : start + n_real]
        # 1.0 only for genuinely-new positions: re-sent tokens must be exact no-ops for
        # the recurrent state, and padding beyond the sequence must be too.
        flag = torch.tensor(
            [[1.0 if (start + i >= processed and i < n_real) else 0.0 for i in range(query_len)]]
        )
        with torch.no_grad():
            out = model(
                block,
                (torch.arange(query_len) + start).to(torch.uint16).unsqueeze(0),
                torch.tensor([start], dtype=torch.int32),
                causal_mask(CTX, start, query_len),
                flag,
                torch.tensor([(start + conv_step_bias) % ring_w], dtype=torch.int32),
                SSMStateHandler.build_seed_selector(
                    ring_w, kw, start, query_len, dtype=torch.float32
                ),
                kv,
                v,
                conv,
                recurrent,
            )
        last_start, processed = start, start + n_real

    idx = (n_tokens - 1) - last_start
    last = out.reshape(query_len, -1)[idx]
    return last, recurrent.clone()


def run_prefill_then_decode(model, ids, cfg, n_prefill, query_len=QUERY_LEN, flag_override=None):
    """Prefill `n_prefill` tokens in one call, then take one decode step.

    This is the only walk that re-sends: the decode block's start is aligned
    down to a multiple of `query_len`, so when `n_prefill` is not a multiple of it the
    block re-sends the tail of the prompt. Those positions must be exact no-ops for the
    recurrent state, which is what `ssm_update_flag` is for. A walk that only ever starts
    on a multiple of `query_len` never exercises this.

    `flag_override` exists for the mutation check.
    """
    kv, v, conv, recurrent, dims = fresh_state(cfg)
    ring_w, kw = dims["ring_width"], dims["kernel_window"]

    with torch.no_grad():
        model(
            ids[:, :n_prefill],
            torch.arange(n_prefill).to(torch.uint16).unsqueeze(0),
            torch.zeros((1,), dtype=torch.int32),
            causal_mask(CTX, 0, n_prefill),
            torch.ones(1, n_prefill),
            torch.zeros((1,), dtype=torch.int32),
            SSMStateHandler.build_seed_selector(ring_w, kw, 0, n_prefill, dtype=torch.float32),
            kv,
            v,
            conv,
            recurrent,
        )

    processed = n_prefill
    total = n_prefill + 1
    start = (processed // query_len) * query_len
    assert start < processed, "test must re-send: pick n_prefill off a query_len boundary"

    n_real = min(query_len, total - start)
    block = torch.zeros(1, query_len, dtype=torch.int32)
    block[0, :n_real] = ids[0, start : start + n_real]
    if flag_override is not None:
        flag = flag_override(query_len)
    else:
        flag = torch.tensor(
            [[1.0 if (start + i >= processed and i < n_real) else 0.0 for i in range(query_len)]]
        )

    with torch.no_grad():
        out = model(
            block,
            (torch.arange(query_len) + start).to(torch.uint16).unsqueeze(0),
            torch.tensor([start], dtype=torch.int32),
            causal_mask(CTX, start, query_len),
            flag,
            torch.tensor([start % ring_w], dtype=torch.int32),
            SSMStateHandler.build_seed_selector(ring_w, kw, start, query_len, dtype=torch.float32),
            kv,
            v,
            conv,
            recurrent,
        )

    idx = (total - 1) - start
    last = out.reshape(query_len, -1)[idx]
    return last, recurrent.clone()


@pytest.fixture(scope="module")
def parity_setup():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Qwen3_5ForCausalLMForiOS(cfg, torch.device("cpu")).eval()
    randomize(model)
    ids = torch.randint(1, cfg.vocab_size, (1, N_TOKENS), dtype=torch.int32)
    reference, reference_state = run_one_shot(model, ids, cfg, N_TOKENS)
    # A vacuous comparison (all-zero logits) would pass every assertion below.
    assert reference.abs().max().item() > 1e-3
    assert reference_state.abs().max().item() > 1e-6
    return model, ids, cfg, reference, reference_state


@pytest.fixture(scope="module")
def resend_setup():
    """Prefill 21 then decode: the decode block starts at 16 and re-sends 5 tokens."""
    torch.manual_seed(2)
    cfg = tiny_config()
    model = Qwen3_5ForCausalLMForiOS(cfg, torch.device("cpu")).eval()
    randomize(model)
    n_prefill = N_TOKENS
    ids = torch.randint(1, cfg.vocab_size, (1, n_prefill + 1), dtype=torch.int32)
    reference, reference_state = run_one_shot(model, ids, cfg, n_prefill + 1)
    assert reference.abs().max().item() > 1e-3
    assert reference_state.abs().max().item() > 1e-6
    return model, ids, cfg, n_prefill, reference, reference_state


@pytest.mark.parametrize("length", [8, 57, 64])
def test_delta_rule_stays_finite_in_fp16_for_any_block_length(length):
    """Correlated keys make the triangular inverse's series grow with chunk length; a
    57-long single chunk overflows fp16. Every length must be sub-chunked."""
    torch.manual_seed(0)
    b, h, dk, dv = 1, 4, 128, 128
    base = torch.randn(b, h, 1, dk)
    key = _l2norm(0.9 * base + 0.1 * torch.randn(b, h, length, dk))
    query = _l2norm(0.9 * base + 0.1 * torch.randn(b, h, length, dk))
    value = torch.randn(b, h, length, dv)
    g = -torch.rand(b, h, length) * 0.05
    beta = torch.full((b, h, length), 0.95)
    state = torch.zeros(b, h, dk, dv)
    inputs = (query, key, value, g, beta, state)
    expected, _ = _chunk_gated_delta_rule(*(t.double() for t in inputs))
    got, new_state = _chunk_gated_delta_rule(*(t.half() for t in inputs))
    assert torch.isfinite(got).all() and torch.isfinite(new_state).all()
    assert (got.double() - expected).abs().max().item() < 0.05


class TestConvRingParity:
    # A mutation must exceed the baseline error by this factor.
    MUTATION_RATIO = 100.0

    def test_blocked_matches_one_shot(self, parity_setup):
        """Walking the sequence in query-length blocks, carrying ring state across calls,
        must not change the final token's logits or the recurrent state."""
        model, ids, cfg, reference, reference_state = parity_setup
        got, state = run_blocked(model, ids, cfg, N_TOKENS)
        scale = reference.abs().max().item()
        assert (got - reference).abs().max().item() < 1e-3 * scale
        state_scale = reference_state.abs().max().item()
        assert (state - reference_state).abs().max().item() < 1e-3 * state_scale

    def test_resend_after_prefill_matches_one_shot(self, resend_setup):
        """A decode block that re-sends the tail of the prompt across a
        ring-buffer write produces the same logits as one uninterrupted call."""
        model, ids, cfg, n_prefill, reference, _ = resend_setup
        got, _ = run_prefill_then_decode(model, ids, cfg, n_prefill)
        scale = reference.abs().max().item()
        assert (got - reference).abs().max().item() < 1e-3 * scale

    # -- mutation checks: prove the assertions above can actually fail --

    def test_wrong_conv_step_breaks_blocked_parity(self, parity_setup):
        """Mutation check: writing the conv window at the wrong ring offset must show up.
        If this passes, `test_blocked_matches_one_shot` is inert.

        A block's own conv inputs do not round-trip through the ring, so a misplaced write
        only corrupts the left context the next block reads. The final token's window lies
        inside its block, so this is asserted on the recurrent state, which every earlier
        token feeds.
        """
        model, ids, cfg, _, reference_state = parity_setup
        baseline = (run_blocked(model, ids, cfg, N_TOKENS)[1] - reference_state).abs().max().item()
        mutated = (
            (run_blocked(model, ids, cfg, N_TOKENS, conv_step_bias=1)[1] - reference_state)
            .abs()
            .max()
            .item()
        )
        assert mutated > self.MUTATION_RATIO * baseline

    def test_resend_leaves_the_recurrent_state_matching_one_shot(self, resend_setup):
        """Re-sent positions must leave the recurrent state unchanged, so the state after
        prefill+decode must match one uninterrupted call."""
        model, ids, cfg, n_prefill, _, reference_state = resend_setup
        _, state = run_prefill_then_decode(model, ids, cfg, n_prefill)
        scale = reference_state.abs().max().item()
        assert (state - reference_state).abs().max().item() < 1e-3 * scale

    def test_all_ones_update_flag_corrupts_the_recurrent_state(self, resend_setup):
        """Mutation check for the flag gate. With the re-sent positions unmasked their
        recurrent update is applied twice, so the state diverges.

        Asserted on the state because the logit change is too small to detect.
        """
        model, ids, cfg, n_prefill, _, reference_state = resend_setup
        _, good = run_prefill_then_decode(model, ids, cfg, n_prefill)
        _, bad = run_prefill_then_decode(
            model, ids, cfg, n_prefill, flag_override=lambda q: torch.ones(1, q)
        )
        baseline = (good - reference_state).abs().max().item()
        mutated = (bad - reference_state).abs().max().item()
        assert mutated > self.MUTATION_RATIO * baseline


class TestConvRingWrap:
    """Ring wrap at the primitive level.

    The model's ring is `MAX_QUERY_LEN`-derived (128 wide), which no sequence short enough
    to run in a unit test can wrap. So drive `SSMStateHandler` directly with a small ring.
    """

    KERNEL, MAX_Q = 4, 8

    def test_ring_width_is_a_multiple_of_max_query_len(self):
        """A block of width b starting at a multiple of b must never straddle the wrap,
        because a straddling write cannot be expressed as one slice."""
        width = SSMStateHandler.ring_width(self.KERNEL, self.MAX_Q)
        assert width % self.MAX_Q == 0
        assert width >= self.MAX_Q + self.KERNEL - 1

    def test_write_then_read_round_trips_across_a_wrap(self):
        """Write successive blocks around the ring and read each one's window back through
        the selector; the values must survive the wrap."""
        width = SSMStateHandler.ring_width(self.KERNEL, self.MAX_Q)
        conv_dim, kernel_window = 6, self.KERNEL - 1
        handler = SSMStateHandler(1, conv_dim, width, num_v_heads=1, head_k_dim=1, head_v_dim=1)
        handler.register_state(torch.zeros(1, conv_dim, 1, width), torch.zeros(1, 1, 1, 1))

        # Walk past the end of the ring so at least one block wraps. Each block goes through
        # the model's write primitive, so its begin/end arithmetic is what is tested.
        for start in range(0, 2 * width, self.MAX_Q):
            block = torch.full((1, conv_dim, 1, self.MAX_Q), float(start + 1))
            conv_step = torch.tensor([start % width], dtype=torch.int32)
            handler.update_conv(0, conv_step, block, self.MAX_Q)
            ring = handler.conv_slot(0)

            selector = SSMStateHandler.build_seed_selector(
                width, kernel_window, start, self.MAX_Q, dtype=torch.float32
            )
            # The read is a matmul against the one-hot: (1, C, 1, W) @ (1, W, K-1+Q).
            window = ring.squeeze(2) @ selector
            own = window[:, :, kernel_window:]
            assert torch.allclose(own, block.squeeze(2)), f"lost the block written at {start}"
            # The left context is the previous block's tail; for a block at offset 0 it is
            # read from the end of the ring.
            if start:
                assert torch.all(window[:, :, :kernel_window] == start - self.MAX_Q + 1)


# ---------------------------------------------------------------------------
# Re-sent slots
# ---------------------------------------------------------------------------
#
# The engine aligns a decode block's start down to a multiple of its query length, so the
# block re-sends the tail of what is already cached with `ssm_update_flag=0`. Those slots
# hold values recomputed against a recurrent state that already includes them, so writing
# them would corrupt every state. Run in float64 with large weights, so the DeltaNet output
# is not drowned out and the comparisons can be tight.

F64 = torch.float64


def _float64_model(seed: int) -> Qwen3_5ForCausalLMForiOS:
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = Qwen3_5ForCausalLMForiOS(cfg, torch.device("cpu")).eval().double()
    for p in model.parameters():  # weights large enough that DeltaNet output matters
        if p.numel() <= 1:
            continue
        if p.is_floating_point():
            p.data = torch.randn_like(p) * 0.3
        else:  # int8 embedding table
            p.data = torch.randint(-8, 9, p.shape, dtype=p.dtype)
    return model


def _float64_state(cfg):
    return [t.double() if torch.is_tensor(t) else t for t in fresh_state(cfg)]


def _block_call(model, state, ids, start, flag) -> torch.Tensor:
    """One graph call on the block ``ids`` ``(1, q)`` at ``start``; returns ``(q, vocab)``."""
    kv, v, conv, rec, dims = state
    q = ids.shape[1]
    with torch.no_grad():
        out = model(
            ids,
            (torch.arange(q) + start).to(torch.uint16).unsqueeze(0),
            torch.tensor([start], dtype=torch.int32),
            causal_mask(CTX, start, q).to(F64),
            flag.to(F64).unsqueeze(0),
            torch.tensor([start % dims["ring_width"]], dtype=torch.int32),
            SSMStateHandler.build_seed_selector(
                dims["ring_width"], dims["kernel_window"], start, q, dtype=F64
            ),
            kv,
            v,
            conv,
            rec,
        )
    return out.reshape(q, -1)


def _engine_decode(model, state, ids, n_cached, query_len, flag_override=None) -> torch.Tensor:
    """Decode token ``n_cached`` the way the static-shape engine does: the block starts at
    ``n_cached`` aligned down to ``query_len``, re-sends the cached tail with flag 0, and
    pads the rest with flag 0. Returns the new token's logits.

    `flag_override` exists for the mutation checks: it receives the engine's flag and the
    new token's slot, and returns the flag to send instead.
    """
    start = (n_cached // query_len) * query_len
    block = torch.zeros(1, query_len, dtype=torch.int32)
    block[0, : n_cached + 1 - start] = ids[0, start : n_cached + 1]
    flag = torch.zeros(query_len)
    flag[n_cached - start] = 1.0
    if flag_override is not None:
        flag = flag_override(flag, n_cached - start)
    return _block_call(model, state, block, start, flag)[n_cached - start]


def _resent_slots_as_new(flag: torch.Tensor, new_slot: int) -> torch.Tensor:
    """Mutation: mark the re-sent slots (before the new one) as new, so they write state."""
    flag = flag.clone()
    flag[:new_slot] = 1.0
    return flag


def _turn_off_kv_gate(monkeypatch) -> None:
    """Mutation: make the K/V write ignore `update_flag`, as a broken gate in
    `KVCacheHandler.update_and_fetch` would."""
    write = KVCacheHandler.update_and_fetch

    def ungated(self, *args, update_flag=None, **kwargs):
        return write(self, *args, **kwargs)

    monkeypatch.setattr(KVCacheHandler, "update_and_fetch", ungated)


def _resent_state_errors(model, n_prefill=21, query_len=8):
    """Prefill ``n_prefill`` tokens, decode one engine-style, and return the max change of
    the K cache, the V cache and the conv ring at the re-sent positions, plus the change of
    the K cache at the new token's position."""
    cfg = model.config
    ids = torch.randint(1, cfg.vocab_size, (1, n_prefill + 1), dtype=torch.int32)
    state = _float64_state(cfg)
    kv, v, conv, _, dims = state

    # Prefill in one call: every slot is new.
    _block_call(model, state, ids[:, :n_prefill], 0, torch.ones(n_prefill))
    assert kv.abs().max() > 0 and conv.abs().max() > 0, "prefill wrote no state"
    kv0, v0, conv0 = kv.clone(), v.clone(), conv.clone()

    _engine_decode(model, state, ids, n_prefill, query_len)

    start = (n_prefill // query_len) * query_len
    resent = list(range(start, n_prefill))
    cols = [p % dims["ring_width"] for p in resent]
    kv_err = (kv[..., resent] - kv0[..., resent]).abs().max().item()
    v_err = (v[..., resent] - v0[..., resent]).abs().max().item()
    conv_err = max(
        (conv[i][..., cols] - conv0[i][..., cols]).abs().max().item() for i in range(conv.shape[0])
    )
    new_slot_change = (kv[..., n_prefill] - kv0[..., n_prefill]).abs().max().item()
    return kv_err, v_err, conv_err, new_slot_change


def test_resent_slots_do_not_modify_kv_or_conv_state():
    """A q=8 decode after a 21-token prefill starts at 16 and re-sends 16..20 with flag 0.
    Those slots must leave the K/V cache and the conv ring as prefill wrote them."""
    kv_err, v_err, conv_err, new_slot_change = _resent_state_errors(_float64_model(seed=2))
    assert kv_err < 1e-9, f"re-sent slots overwrote K cache (max diff {kv_err:.3e})"
    assert v_err < 1e-9, f"re-sent slots overwrote V cache (max diff {v_err:.3e})"
    assert conv_err < 1e-9, f"re-sent slots overwrote conv ring (max diff {conv_err:.3e})"
    # The new slot itself must still be written.
    assert new_slot_change > 1e-6


def test_ungated_kv_write_breaks_the_resent_state_check(monkeypatch):
    """Mutation check: with the K/V gate off, re-sent slots overwrite the K and V cache, so
    `test_resent_slots_do_not_modify_kv_or_conv_state` would fail."""
    _turn_off_kv_gate(monkeypatch)
    kv_err, v_err, _, _ = _resent_state_errors(_float64_model(seed=2))
    assert kv_err > 1e-6 and v_err > 1e-6, f"K err {kv_err:.3e}, V err {v_err:.3e}"


@pytest.fixture(scope="module")
def resend_float64_setup():
    model = _float64_model(seed=5)
    cfg = model.config
    n_max = 32
    ids = torch.randint(1, cfg.vocab_size, (1, n_max), dtype=torch.int32)
    # Every position's logits from one uninterrupted call.
    reference = _block_call(model, _float64_state(cfg), ids, 0, torch.ones(n_max))
    return model, ids, reference


class TestResentSlotEquivalence:
    """The new token's logits from an engine-style block, which re-sends the cached tail,
    must match decoding that token with no re-sent slots at all."""

    # Relative to the largest reference value. Float64 leaves only rounding error, which the
    # differently chunked delta rule and the large weights amplify to ~1e-9; the bug is ~1e-1.
    TOLERANCE = 1e-7
    DECODE_STEPS = 3
    # A mutation must exceed the baseline error by this factor, and the tolerance as well.
    MUTATION_RATIO = 100.0

    def _one_shot_error(self, setup, n_prefill, query_len, flag_override=None):
        """Max logit error, relative to the reference's scale, over the decode steps."""
        model, ids, reference = setup
        state = _float64_state(model.config)
        _block_call(model, state, ids[:, :n_prefill], 0, torch.ones(n_prefill))
        worst = 0.0
        for pos in range(n_prefill, n_prefill + self.DECODE_STEPS):
            got = _engine_decode(model, state, ids, pos, query_len, flag_override)
            worst = max(
                worst,
                (got - reference[pos]).abs().max().item() / reference[pos].abs().max().item(),
            )
        return worst

    @pytest.mark.parametrize("query_len", [DECODE_QUERY_LEN, PREFILL_QUERY_LEN])
    @pytest.mark.parametrize("n_prefill", [17, 21, 23])
    def test_engine_decode_matches_one_shot(self, resend_float64_setup, n_prefill, query_len):
        """Prefill, then decode several tokens engine-style. Each new token's logits must
        match one uninterrupted call over the whole sequence."""
        model, ids, reference = resend_float64_setup
        assert n_prefill % query_len, "test must re-send: pick n_prefill off a block boundary"
        state = _float64_state(model.config)
        _block_call(model, state, ids[:, :n_prefill], 0, torch.ones(n_prefill))
        for pos in range(n_prefill, n_prefill + self.DECODE_STEPS):
            got = _engine_decode(model, state, ids, pos, query_len)
            scale = reference[pos].abs().max().item()
            err = (got - reference[pos]).abs().max().item()
            assert err < self.TOLERANCE * scale, f"token {pos}: max diff {err:.3e}"

    @pytest.mark.parametrize("query_len", [DECODE_QUERY_LEN, PREFILL_QUERY_LEN])
    @pytest.mark.parametrize("n_prefill", [17, 21, 23])
    def test_engine_decode_matches_single_slot_decode(
        self, resend_float64_setup, n_prefill, query_len
    ):
        """The same decode as a one-slot call at the new token's own position, so no slot
        is re-sent; both start from the same prefill."""
        model, ids, _ = resend_float64_setup
        engine, single = _float64_state(model.config), _float64_state(model.config)
        for state in (engine, single):
            _block_call(model, state, ids[:, :n_prefill], 0, torch.ones(n_prefill))
        got = _engine_decode(model, engine, ids, n_prefill, query_len)
        expected = _block_call(
            model, single, ids[:, n_prefill : n_prefill + 1], n_prefill, torch.ones(1)
        )[0]
        scale = expected.abs().max().item()
        assert (got - expected).abs().max().item() < self.TOLERANCE * scale
        # Every state must agree as well, at the positions both calls own.
        kv_e, v_e, conv_e, rec_e, dims = engine
        kv_s, v_s, conv_s, rec_s, _ = single
        written = list(range(n_prefill + 1))
        cols = [p % dims["ring_width"] for p in written]
        for a, b in (
            (kv_e[..., written], kv_s[..., written]),
            (v_e[..., written], v_s[..., written]),
            (conv_e[..., cols], conv_s[..., cols]),
            (rec_e, rec_s),
        ):
            assert (a - b).abs().max().item() < self.TOLERANCE * b.abs().max().item()

    # -- mutation checks: prove the tolerances above catch a broken gate --

    @pytest.mark.parametrize("query_len", [DECODE_QUERY_LEN, PREFILL_QUERY_LEN])
    def test_resent_slots_flagged_new_break_one_shot_parity(self, resend_float64_setup, query_len):
        """Mutation check: re-sent slots sent with flag 1 rewrite every state, and the
        logit error must exceed both `TOLERANCE` and the baseline by `MUTATION_RATIO`."""
        baseline = self._one_shot_error(resend_float64_setup, 21, query_len)
        mutated = self._one_shot_error(
            resend_float64_setup, 21, query_len, flag_override=_resent_slots_as_new
        )
        assert mutated > self.MUTATION_RATIO * self.TOLERANCE, f"mutated error {mutated:.3e}"
        assert mutated > self.MUTATION_RATIO * baseline, f"{mutated:.3e} vs {baseline:.3e}"

    @pytest.mark.parametrize("query_len", [DECODE_QUERY_LEN, PREFILL_QUERY_LEN])
    def test_ungated_kv_write_breaks_one_shot_parity(
        self, resend_float64_setup, query_len, monkeypatch
    ):
        """Mutation check for the K/V gate alone: with `update_and_fetch` ignoring
        `update_flag`, the re-sent slots overwrite the K/V cache and the logit error must
        exceed both `TOLERANCE` and the baseline by `MUTATION_RATIO`."""
        baseline = self._one_shot_error(resend_float64_setup, 21, query_len)
        _turn_off_kv_gate(monkeypatch)
        mutated = self._one_shot_error(resend_float64_setup, 21, query_len)
        assert mutated > self.MUTATION_RATIO * self.TOLERANCE, f"mutated error {mutated:.3e}"
        assert mutated > self.MUTATION_RATIO * baseline, f"{mutated:.3e} vs {baseline:.3e}"


# ---------------------------------------------------------------------------
# Export contract
# ---------------------------------------------------------------------------


@functools.cache
def _recipe():
    """Load ``models/qwen3_5/export.py`` by path.

    It is a script under ``models/``, not an importable package, so there is no module name to
    import it by. Cached: executing it imports the converter, which is slow.
    """
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[5] / "models" / "qwen3_5" / "export.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_export_recipe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExportRecipe:
    def test_graph_io_names_match_the_forward_signature(self):
        """Input and state names must match Qwen3_5Extend.forward's parameter order; the
        converter binds them positionally."""
        params = list(inspect.signature(Qwen3_5Extend.forward).parameters)[1:]  # drop self
        inputs = list(Qwen3_5ForCausalLMForiOS.export_input_names()[EXTEND_FUNCTION_NAME])
        states = list(Qwen3_5ForCausalLMForiOS.export_state_names()[EXTEND_FUNCTION_NAME])

        # `embedding_table` is last in the signature but is an input, not a state, so the
        # declared order is inputs-minus-embedding-table, then states, then embedding_table.
        expected = inputs[:-1] + states + [inputs[-1]]
        assert params == expected

    def test_ladder_is_the_shipping_buckets_and_collapses_for_small_caps(self):
        recipe = _recipe()
        assert recipe.context_ladder(32768) == [256, 1024, 8192, 32768]
        assert recipe.context_ladder(1024) == [256, 1024]
        assert recipe.context_ladder(256) == [256]
        # Not a power of two: rounds up to the next one and keeps the rungs below it.
        assert recipe.context_ladder(5000) == [256, 1024, 8192]

    def test_query_lengths_fit_the_conv_ring(self):
        """The ring is sized from MAX_QUERY_LEN, which bounds the re-sent tail."""
        recipe = _recipe()
        assert recipe.PREFILL_QUERY_LEN <= MAX_QUERY_LEN
        assert recipe.DECODE_QUERY_LEN <= MAX_QUERY_LEN

    def test_deltanet_states_are_pinned_fixed(self):
        """The runner must not infer these from shape: a DeltaNet state cannot be truncated,
        so a partial reset would leave it matching no prefix."""
        recipe = _recipe()
        states = recipe._ios_metadata_extras()["states"]
        assert states == {
            "conv_state": "fixed",
            "recurrent_state": "fixed",
        }

    def test_kv_caches_are_not_pinned(self):
        """KV caches are left to the runner's footprint heuristic."""
        recipe = _recipe()
        states = recipe._ios_metadata_extras()["states"]
        assert "key_cache" not in states and "value_cache" not in states
        # Everything that is pinned must still be a real state name.
        declared = set(Qwen3_5ForCausalLMForiOS.export_state_names()[EXTEND_FUNCTION_NAME])
        assert set(states) <= declared

    def test_kv_seq_alignment_is_the_buckets_own_ctx(self):
        """Each bucket's KV seq stride is aligned to its own ctx."""
        recipe = _recipe()
        interleave = Qwen3_5ForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR
        for ctx in recipe.context_ladder(32768):
            alignments = recipe.forward_constraints(ctx)["key_cache"].alignments
            assert alignments[4] == interleave * ctx, (
                f"ctx {ctx} aligns the seq stride to {alignments[4]}, "
                f"expected {interleave * ctx} (its own ctx)"
            )

    def test_registry_points_ios_export_at_the_recipe(self):
        """`coreai.llm.export --variant iOS` must refuse Qwen3.5 up front and name the recipe,
        so the registry entry has to point at a script that exists."""
        from coreai_models.models.registry import get_model_entry

        entry = get_model_entry("qwen3_5")
        assert entry.ios_export_script == "models/qwen3_5/export.py"
        repo_root = pathlib.Path(__file__).resolve().parents[5]
        assert (repo_root / entry.ios_export_script).is_file()

    def test_evaluation_resolves_the_ios_class(self, monkeypatch):
        """Evaluation resolves the iOS class through `resolve_model`, so only export may refuse."""
        resolved = export_pipeline.resolve_model(_ios_export_config(monkeypatch))
        assert resolved.model_class is Qwen3_5ForCausalLMForiOS

    def test_export_points_at_the_recipe_before_loading_weights(self, monkeypatch):
        def load(*_, **__):
            raise AssertionError("export loaded the weights before refusing")

        monkeypatch.setattr(Qwen3_5ForCausalLMForiOS, "from_hf", classmethod(load))
        with pytest.raises(ValueError, match="uv run models/qwen3_5/export.py"):
            export_pipeline.export_model(_ios_export_config(monkeypatch))

    def test_build_model_palettizes_with_eager_forward_args(self, monkeypatch):
        """The palettizer traces `forward` on what `build_model` hands it, and Qwen3.5's takes
        the DeltaNet inputs and states on top of the KV pair."""
        cfg = tiny_config()
        monkeypatch.setattr(
            Qwen3_5ForCausalLMForiOS,
            "from_hf",
            classmethod(lambda cls, *_, **__: cls(cfg, torch.device("cpu")).to(torch.float16)),
        )
        logits = []

        def palettize(model, example_inputs, *_, **__):
            logits.append(model(*example_inputs))
            return model

        monkeypatch.setattr(export_pipeline, "palettize_pytorch_model", palettize)
        recipe = pathlib.Path(__file__).resolve().parents[5] / "models/qwen3_5"
        config = _ios_export_config(
            monkeypatch,
            compression_config_object=_load_compression_config_object(
                recipe / "qwen3_5_2b_4bit_palettized.yaml", "iOS"
            ),
        )
        export_pipeline.build_model(export_pipeline.resolve_model(config))
        assert logits[0].shape[-1] == cfg.vocab_size

    def test_constraints_name_real_graph_io(self):
        """Constraints are looked up by I/O name, so a name the graph does not have is
        silently ignored. Every constrained name must be an input, state or state output."""
        recipe = _recipe()
        cls = Qwen3_5ForCausalLMForiOS
        io_names = (
            set(cls.export_input_names()[EXTEND_FUNCTION_NAME])
            | set(cls.export_state_names()[EXTEND_FUNCTION_NAME])
            | set(cls.export_state_output_names()[EXTEND_FUNCTION_NAME])
        )
        constraints = recipe.forward_constraints(1024)
        assert set(constraints) <= io_names, sorted(set(constraints) - io_names)
        for name in ("key_cache", "value_cache", "conv_state", "recurrent_state"):
            assert name in constraints, f"{name} has no layout constraint"

    def test_both_deltanet_states_interleave_the_channel_axis(self):
        """Rank 4, so the channel axis is index 1."""
        recipe = _recipe()
        constraints = recipe.forward_constraints(1024)
        factor = Qwen3_5ForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR
        for name in ("conv_state", "recurrent_state"):
            assert tuple(constraints[name].interleave) == (1, factor, 1, 1)

    def test_kv_caches_are_sized_by_attention_layers_not_total_layers(self):
        """Only the full-attention layers keep a KV cache; sizing by `num_hidden_layers`
        would over-allocate by 4x."""
        cfg = tiny_config(n_layers=8)
        n_attn = Qwen3_5ForCausalLMForiOS._deltanet_dims(cfg)["n_attn"]
        assert n_attn == cfg.layer_types.count("full_attention")
        assert n_attn < cfg.num_hidden_layers


# ---------------------------------------------------------------------------
# HuggingFace parity
# ---------------------------------------------------------------------------


def _psnr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB between two logit tensors."""
    error = (reference - actual).pow(2).mean().item()
    if error == 0.0:
        return float("inf")
    peak = reference.abs().max().item() ** 2
    return 10.0 * math.log10(peak / error)


@pytest.fixture(scope="module")
def hf_pair():
    """The same weights loaded into HuggingFace's decoder and ours.

    Built from a tiny random checkpoint, so this needs no network. Embedding quantization is
    off, since int8 embeddings would dominate the error.
    """
    torch.manual_seed(3)
    cfg = tiny_config()
    hf = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()

    ios = Qwen3_5ForCausalLMForiOS(
        cfg, torch.device("cpu"), disable_embedding_quantization=True
    ).eval()
    state_dict = dict(hf.state_dict())
    ios._mutate_state_dict(state_dict)
    ios.load_state_dict(state_dict, assign=True, strict=True)
    return hf, ios, cfg


class TestHuggingFaceParity:
    """Our reauthored decoder against ``transformers``' own, on identical weights.

    The self-consistency tests above pin our graph against itself under different block
    walks; only this one pins it against the reference implementation.
    """

    def test_single_shot_prefill_matches_hf(self, hf_pair):
        hf, ios, cfg = hf_pair
        ids = torch.randint(1, cfg.vocab_size, (1, N_TOKENS), dtype=torch.int32)
        with torch.no_grad():
            expected = hf(ids.to(torch.long)).logits[0, -1].to(torch.float32)
        actual, _ = run_one_shot(ios, ids, cfg, N_TOKENS)
        assert expected.abs().max().item() > 1e-3
        assert _psnr(expected, actual.to(torch.float32)) > 40.0
        assert int(expected.argmax()) == int(actual.argmax())

    def test_forward_sequence_matches_hf_at_every_position(self, hf_pair):
        """The eval hook walks past a ``MAX_QUERY_LEN`` boundary; every position must match."""
        hf, ios, cfg = hf_pair
        n = MAX_QUERY_LEN + 36
        ids = torch.randint(1, cfg.vocab_size, (1, n), dtype=torch.int32)
        with torch.no_grad():
            expected = hf(ids.to(torch.long)).logits[0].to(torch.float32)
        actual = ios.forward_sequence(ids, ios.fresh_states(CTX, torch.float32))
        assert actual.shape == expected.shape
        assert _psnr(expected, actual) > 40.0
        assert torch.equal(expected.argmax(-1), actual.argmax(-1))

    def test_forward_sequence_continues_from_an_unaligned_offset(self, hf_pair):
        """Scoring a prefix, then the rest from its end, matches one call (decode's shape)."""
        _, ios, cfg = hf_pair
        n, split = MAX_QUERY_LEN + 36, 37
        ids = torch.randint(1, cfg.vocab_size, (1, n), dtype=torch.int32)
        whole = ios.forward_sequence(ids, ios.fresh_states(CTX, torch.float32))
        states = ios.fresh_states(CTX, torch.float32)
        head = ios.forward_sequence(ids[:, :split], states)
        tail = ios.forward_sequence(ids[:, split:], states, offset=split)
        assert _psnr(whole, torch.cat([head, tail])) > 100.0

    def test_eager_step_matches_hf_across_calls(self, hf_pair):
        """The scorers' entry point: two calls, the second from an unaligned offset and past a
        ``MAX_QUERY_LEN`` boundary, give HF's logits at every position."""
        hf, ios, cfg = hf_pair
        n, split = MAX_QUERY_LEN + 36, 37
        ids = torch.randint(1, cfg.vocab_size, (1, n), dtype=torch.int32)
        with torch.no_grad():
            expected = hf(ids.to(torch.long)).logits.to(torch.float32)
        state = ios.new_eager_state(dtype=torch.float32, device="cpu")
        head = ios.eager_step(ids[:, :split], state)
        tail = ios.eager_step(ids[:, split:], state)
        actual = torch.cat([head, tail], dim=1)
        assert state.offset == n
        assert _psnr(expected, actual) > 40.0
        assert torch.equal(expected.argmax(-1), actual.argmax(-1))

    def test_blocked_walk_matches_hf(self, hf_pair):
        """A blocked walk must match HF."""
        hf, ios, cfg = hf_pair
        ids = torch.randint(1, cfg.vocab_size, (1, N_TOKENS), dtype=torch.int32)
        with torch.no_grad():
            expected = hf(ids.to(torch.long)).logits[0, -1].to(torch.float32)
        actual, _ = run_blocked(ios, ids, cfg, N_TOKENS)
        assert _psnr(expected, actual.to(torch.float32)) > 40.0
        assert int(expected.argmax()) == int(actual.argmax())
