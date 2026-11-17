# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS Gemma4 two-cache sliding-window attention parity with HuggingFace.

The iOS Gemma4 decoder uses two compacted KV caches — a full-context global
cache and a small sliding-window ring (depth S) — and applies windowed attention
via a runner-built ``sliding_causal_mask``. These tests mimic the Swift runner's
chunked prefill against persistent caches and compare per-position logits to a
single HF forward (which applies sliding-window attention internally). The key
case is a prompt longer than both W and S so the ring wraps.
"""

import tempfile

import pytest
import torch
from safetensors import safe_open

pytest.importorskip("transformers")

try:  # transformers>=5.5 (gemma4)
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4 import Gemma4ForCausalLM
except Exception:  # pragma: no cover - requires transformers>=5.5
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None

from coreai_models._constants import EXTEND_FUNCTION_NAME  # noqa: E402
from coreai_models.models.base import TraceSpec, _is_layer_key_beyond  # noqa: E402
from coreai_models.models.ios.gemma4_text import (  # noqa: E402
    DEFAULT_KV_BLOCK_SIZE,
    GeGLUMLP,
    Gemma4Extend,
    Gemma4ForCausalLMForiOS,
    NativeLutQuantConv1x1,
    NativeQuantConv1x1,
    NativeQuantLinear,
    QuantizedLinear,
    _check_native_nbits,
    _check_native_widths,
    _compute_kv_layout,
    _is_quantized_hf_model,
    _make_proj_conv,
    _native_mlp_nbits,
    _native_module_name,
    _native_ple_table,
    sliding_ring_size,
)
from coreai_models.primitives.ios.quantization import (  # noqa: E402
    dequantize_native,
    quantize_per_tensor,
    unpack_signed_subbyte,
    unpack_unsigned_subbyte,
)
from coreai_models.primitives.ios.rope import RoPECache  # noqa: E402
from tests._runner_infra._deps import _hf_hub_reachable  # noqa: E402

DTYPE = torch.float32
NEG = float("-inf")

pytestmark = pytest.mark.skipif(
    Gemma4ForCausalLM is None, reason="gemma4 requires transformers>=5.5"
)


# Numerical reference for the combined sliding+global RoPE rows the runner builds.
class Gemma4CombinedRoPE(RoPECache):
    """Single RoPE cache for Gemma4's dual head dims.

    Gemma4 has two RoPE variants — standard for sliding-attention layers (head_dim 256)
    and proportional (0.25 rotary) for global layers (head_dim 512). Both variants' cos/sin
    tables are concatenated along the head dim into a single
    ``[max_pos, sliding_hd + global_hd]`` cache and gathered once. Callers slice
    ``[:sliding_hd]`` for sliding layers and ``[sliding_hd:]`` for global layers.
    """

    def __init__(
        self,
        sliding_head_dim: int,
        global_head_dim: int,
        max_cache_size: int,
        sliding_base: float,
        global_base: float,
        partial_rotary_factor: float = 0.25,
    ) -> None:
        self._sliding_head_dim = sliding_head_dim
        self._global_head_dim = global_head_dim
        self._sliding_base = sliding_base
        self._global_base = global_base
        self._partial_rotary_factor = partial_rotary_factor
        super().__init__(sliding_head_dim + global_head_dim, max_cache_size, sliding_base)

    @staticmethod
    def _emb(theta: torch.Tensor, max_cache_size: int) -> torch.Tensor:
        seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)
        freqs = seq_idx[:, None] * theta
        return torch.concatenate((freqs, freqs), dim=-1)

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        with torch.device("cpu"):
            # Sliding (standard RoPE).
            s_theta = 1.0 / (
                self._sliding_base
                ** (
                    torch.arange(0, self._sliding_head_dim, 2, dtype=torch.float32)
                    / self._sliding_head_dim
                )
            )
            s_emb = self._emb(s_theta, self._max_cache_size)

            # Global (proportional RoPE: only partial_rotary_factor of dims rotate).
            hd = self._global_head_dim
            rope_angles = int(self._partial_rotary_factor * hd // 2)
            nope_angles = hd // 2 - rope_angles
            inv_freq = 1.0 / (
                self._global_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / hd)
            )
            if nope_angles > 0:
                g_theta = torch.cat(
                    [inv_freq, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
                )
            else:
                g_theta = inv_freq
            g_emb = self._emb(g_theta, self._max_cache_size)

            cos = torch.cat([torch.cos(s_emb), torch.cos(g_emb)], dim=-1)
            sin = torch.cat([torch.sin(s_emb), torch.sin(g_emb)], dim=-1)
            self.cos_cached = torch.nn.Buffer(cos.to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(sin.to(dtype=dtype), persistent=False)


def _make_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        sliding_window=8,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rms_norm_eps=1e-6,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
    )


def _build_ios_model(cfg, hf_sd):
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(cfg, model_device="cpu", disable_embedding_quantization=True)
    model.to(DTYPE).eval()
    model._mutate_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)
    return model


def _build_quantized_ios_model(cfg, hf_sd):
    """The model as exported: INT8 embedding table and PLE path."""
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(cfg, model_device="cpu", disable_embedding_quantization=False)
    model.to(DTYPE).eval()
    model._mutate_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)
    return model


def _ple_input_fp(hf_model, cfg, token_ids):
    # Build the fp ``ple_embeddings`` graph input the way export quantizes it
    # (ple_weight[token] * sqrt(ple_dim)), but kept fp to avoid INT8 noise.
    w = hf_model.state_dict()["model.embed_tokens_per_layer.weight"]
    ple_dim = cfg.hidden_size_per_layer_input
    total = cfg.num_hidden_layers * ple_dim
    rows = w[:, :total][token_ids].to(DTYPE) * (float(ple_dim) ** 0.5)
    return rows.reshape(1, len(token_ids), 1, total)


def _global_mask(ctx, q_len, aligned_step):
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        m[0, : aligned_step + i + 1, 0, i] = 0.0
    return m


def _sliding_mask(S, q_len, aligned_step, window):
    m = torch.full((1, S, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = aligned_step + i
        for pos in range(max(0, p - window + 1), p + 1):
            m[0, pos % S, 0, i] = 0.0
    return m


def _combined_rope(cfg, max_ctx: int, dtype: torch.dtype) -> Gemma4CombinedRoPE:
    """The dual (sliding + global) RoPE table the runner precomputes and feeds in
    as ``rope_cos``/``rope_sin`` rows."""
    return Gemma4CombinedRoPE(
        sliding_head_dim=cfg.head_dim,
        global_head_dim=cfg.global_head_dim,
        max_cache_size=max_ctx,
        sliding_base=cfg.rope_parameters["sliding_attention"]["rope_theta"],
        global_base=cfg.rope_parameters["full_attention"]["rope_theta"],
        partial_rotary_factor=cfg.rope_parameters["full_attention"].get(
            "partial_rotary_factor", 0.25
        ),
    ).to(dtype)


def _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx):
    n_kv = cfg.num_key_value_heads
    sliding_storing, global_storing, _ = _compute_kv_layout(cfg)
    seq = len(token_ids)
    rope = _combined_rope(cfg, ctx, DTYPE)

    key_cache = torch.zeros(len(global_storing), 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE)
    value_cache = key_cache.clone()
    skey_cache = torch.zeros(len(sliding_storing), 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
    svalue_cache = skey_cache.clone()

    out_logits = torch.zeros(seq, cfg.vocab_size, dtype=DTYPE)
    for start in range(0, seq, q_len):
        chunk = token_ids[start : start + q_len]
        ids = chunk.reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        rope_cos, rope_sin = rope.gather_cos_sin(pos)
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        with torch.no_grad():
            out = ios(
                ids,
                rope_cos,
                rope_sin,
                in_step,
                sliding_in_step,
                _global_mask(ctx, q_len, start),
                _sliding_mask(S, q_len, start, cfg.sliding_window),
                key_cache,
                value_cache,
                skey_cache,
                svalue_cache,
                _ple_input_fp(hf, cfg, chunk),
            )
        out_logits[start : start + q_len] = out.reshape(q_len, cfg.vocab_size)
    return out_logits


def test_export_contract_describes_one_rung():
    """The export hooks describe one fully static (context bucket, query length) rung."""
    cfg = _make_config()
    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW
    model = _build_quantized_ios_model(cfg, Gemma4ForCausalLM(cfg).state_dict())
    ctx, q_len, ring = 1024, 8, Gemma4ForCausalLMForiOS.SLIDING_RING_SIZE
    spec = TraceSpec(max_context_length=4 * ctx, cache_seq_len=ctx, query_len=q_len)

    refs = model.build_reference_inputs(cfg, torch.float16, spec)
    model.validate_export_contract(refs, model.build_dynamic_shapes(cfg, spec))
    extend = refs[EXTEND_FUNCTION_NAME]
    assert extend["key_cache"].shape[-1] == ctx
    assert extend["sliding_key_cache"].shape[-1] == ring
    assert extend["causal_mask"].shape == (1, ctx, 1, q_len)
    assert extend["sliding_causal_mask"].shape == (1, ring, 1, q_len)

    factor = Gemma4ForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR
    constraints = Gemma4ForCausalLMForiOS.export_hardware_constraints(ctx)[EXTEND_FUNCTION_NAME]
    for name in ("key_cache", "value_cache"):
        assert constraints[name].alignments[4] == factor * ctx
    for name in ("sliding_key_cache", "sliding_value_cache"):
        assert constraints[name].alignments[4] == factor * ring

    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW // 2
    with pytest.raises(ValueError, match="sliding_window"):
        model.build_reference_inputs(cfg, torch.float16, spec)


def test_export_rejects_unquantized_ple():
    """The graph and the runner take INT8 ple_embeddings, so an fp PLE path can't export."""
    cfg = _make_config()
    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW
    model = _build_ios_model(cfg, dict(Gemma4ForCausalLM(cfg).state_dict()))
    spec = TraceSpec(max_context_length=1024, cache_seq_len=1024, query_len=8)
    with pytest.raises(ValueError, match="INT8"):
        model.build_reference_inputs(cfg, torch.float16, spec)
    with pytest.raises(ValueError, match="INT8"):
        model.dump_ple_embedding(tempfile.mkdtemp(), "m")


def test_ple_sidecar_matches_graph_scale():
    """The sidecar's INT8 rows and scale are what the graph dequantizes with."""
    torch.manual_seed(0)
    cfg = _make_config()
    model = _build_quantized_ios_model(cfg, Gemma4ForCausalLM(cfg).state_dict())
    embed_scale = cfg.hidden_size_per_layer_input**0.5
    ref_q, ref_scale, _ = quantize_per_tensor(
        model._ple_weight.float() * embed_scale, nbits=8, symmetric=True
    )
    with (
        tempfile.TemporaryDirectory() as tmp,
        safe_open(model.dump_ple_embedding(tmp, "m"), "pt") as f,
    ):
        rows = f.get_tensor("embed_tokens_per_layer")
        sidecar_scale = float(f.metadata()["ple_scale"])
    assert torch.equal(rows, ref_q)
    assert sidecar_scale == float(ref_scale)
    assert model.extend.ple_scale.item() == float(ref_scale.to(model.extend.ple_scale.dtype))


def test_rejects_checkpoints_without_per_layer_embeddings():
    cfg = _make_config()
    cfg.hidden_size_per_layer_input = 0
    model = Gemma4ForCausalLMForiOS(cfg, model_device="cpu", disable_embedding_quantization=False)
    with pytest.raises(NotImplementedError, match="per-layer-embedding"):
        model._mutate_state_dict({})


@pytest.mark.parametrize(
    "key, beyond",
    [
        ("model.layers.3.self_attn.q_proj.weight", True),
        ("layers.3.self_attn.q_proj.weight", True),
        ("model.language_model.layers.1.mlp.up_proj.weight", False),
        ("layers.1.mlp.up_proj.weight", False),
        ("model.embed_tokens.weight", False),
        ("model.sublayers.9.weight", False),
    ],
)
def test_is_layer_key_beyond_with_and_without_prefix(key, beyond):
    """A stripped ``layers.N.`` key is filtered like a prefixed one."""
    assert _is_layer_key_beyond(key, 2) is beyond


def test_kv_layout_dead_slot_compaction():
    """Layout keeps only storing layers; shared layers reuse the source slot."""
    cfg = _make_config()
    sliding_storing, global_storing, layout = _compute_kv_layout(cfg)
    assert sliding_storing == [0, 1, 3]
    assert global_storing == [2]
    # Shared sliding layer 4 -> source 3 -> sliding slot 2; shared global 5 -> global slot 0.
    assert layout[4] == (True, True, 2)
    assert layout[5] == (False, True, 0)


@pytest.mark.parametrize(
    "kv_block_size",
    [None, 8, 12],
    ids=["one global block", "4 global blocks", "3 ragged global blocks"],
)
def test_sliding_parity_with_ring_wrap(kv_block_size):
    """Chunked prefill over a prompt longer than both W and S matches HF, with the
    global attention's flash loop running over one or several blocks."""
    torch.manual_seed(0)
    cfg = _make_config()
    cfg.kv_block_size = kv_block_size
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = sliding_ring_size(cfg.sliding_window, q_len)  # 12
    ctx = 32
    seq = 24  # > S (ring wraps) and > W (windowing active)
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)

    assert (ios_logits.argmax(-1) == hf_logits.argmax(-1)).all()
    torch.testing.assert_close(ios_logits, hf_logits, atol=1e-3, rtol=1e-3)


def test_final_logit_softcap_left_to_runner():
    """The iOS forward emits *uncapped* logits; the runner applies the cap on CPU.

    ``tanh`` is best run on the CPU rather than in the graph, so the iOS decoder no longer applies
    ``c·tanh(logits/c)`` (see ``models/ios/gemma4_text.py``); the Swift runner does it
    instead, between reading ``out_logits`` and sampling (``LogitSoftcap``). This pins
    both halves of that split: our forward reproduces HF's *pre-cap* logits, and capping
    our output afterwards reproduces HF's capped logits.
    """
    torch.manual_seed(2)
    cfg = _make_config()
    # The synthetic model's logits are all |x| < 0.5, so the released cap (30.0) would be
    # indistinguishable from the identity here. Pick a cap small enough that ``tanh``
    # actually bends this data — the guard below fails if it doesn't.
    cap = 0.1
    cfg.final_logit_softcapping = cap
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = sliding_ring_size(cfg.sliding_window, q_len)
    ctx = 32
    seq = 12
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)

    # Guard: the cap must actually bite on this data, or the assertions below are vacuous.
    assert not torch.allclose(ios_logits, hf_logits, atol=1e-2), (
        "cap had no measurable effect — pick a smaller cap or different seed"
    )

    # Our forward leaves the logits uncapped: applying the cap ourselves lands on HF.
    # (Had the forward still capped, this would be a double cap and would not match.)
    torch.testing.assert_close(torch.tanh(ios_logits / cap) * cap, hf_logits, atol=1e-3, rtol=1e-3)

    # The cap is monotonic, so it cannot move an argmax — which is why greedy sampling
    # is unaffected by *where* it runs, and only the logit values need the runner's pass.
    assert (ios_logits.argmax(-1) == hf_logits.argmax(-1)).all()


def test_emit_hidden_states_adds_the_final_hidden_output():
    """With ``emit_hidden_states`` the decode forward returns ``(logits, final_hidden)``.

    The logits are unchanged, and the hidden state is the post-norm activation the tied
    projection consumes, one row per query position.
    """
    torch.manual_seed(3)
    cfg = _make_config()
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len, ctx = 4, 32
    S = sliding_ring_size(cfg.sliding_window, q_len)
    n_kv = cfg.num_key_value_heads
    sliding_storing, global_storing, _ = _compute_kv_layout(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (q_len,))
    rope_cos, rope_sin = _combined_rope(cfg, ctx, DTYPE).gather_cos_sin(
        torch.arange(q_len, dtype=torch.int32).reshape(1, q_len)
    )

    def forward():
        key_cache = torch.zeros(
            len(global_storing), 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE
        )
        skey_cache = torch.zeros(len(sliding_storing), 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
        with torch.no_grad():
            return ios(
                token_ids.reshape(1, q_len),
                rope_cos,
                rope_sin,
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                _global_mask(ctx, q_len, 0),
                _sliding_mask(S, q_len, 0, cfg.sliding_window),
                key_cache,
                key_cache.clone(),
                skey_cache,
                skey_cache.clone(),
                _ple_input_fp(hf, cfg, token_ids),
            )

    plain = forward()
    assert isinstance(plain, torch.Tensor)

    ios.extend.emit_hidden_states = True
    logits, hidden = forward()
    torch.testing.assert_close(logits, plain)
    assert hidden.shape == (1, q_len, cfg.hidden_size)


# ---------------------------------------------------------------------------
# Native quantization primitives (QAT "-qat-mobile-transformers" checkpoints).
#
# These exercise the CoreAI-native weight-holder modules directly with synthetic
# data -- no real checkpoint or network needed, since the modules only care about
# their own buffers.
# ---------------------------------------------------------------------------


def test_native_quant_conv1x1_matches_manual_dequant():
    """Affine dequant (scale * quantized_data) matches a plain broadcasted multiply."""
    torch.manual_seed(0)
    out_c, in_c, nbits = 16, 8, 4
    mod = NativeQuantConv1x1(in_c, out_c, nbits=nbits)
    q = torch.randint(-8, 8, (out_c, in_c, 1, 1), dtype=torch.int8)
    scale = torch.rand(out_c, 1, 1, 1, dtype=torch.float16) * 0.01
    with torch.no_grad():
        mod.quantized_data.copy_(q)
        mod.scale.copy_(scale)
    mod.eval()

    x = torch.randn(1, in_c, 1, 4, dtype=torch.float16)
    with torch.no_grad():
        out = mod(x)

    expected_weight = q.to(torch.float16) * scale
    expected = torch.nn.functional.conv2d(x, expected_weight)
    torch.testing.assert_close(out, expected)


def test_native_lut_quant_conv1x1_matches_manual_lookup():
    """LUT decode + scale matches a plain ``(index - bias) * scale`` lookup.

    Also pins the meta-device bug: ``lut`` is a non-persistent buffer built under
    ``model_device="meta"`` construction, so nothing rescues it via
    ``load_state_dict`` afterward -- without the explicit ``with torch.device("cpu")``
    fix, it silently stays uninitialized meta data forever (caught during
    development via a large numeric mismatch, not a crash).
    """
    torch.manual_seed(0)
    out_c, in_c, nbits = 16, 8, 2
    mod = NativeLutQuantConv1x1(in_c, out_c, nbits=nbits)

    assert mod.lut.device.type != "meta"
    assert mod.lut.flatten().tolist() == [-2, -1, 0, 1]

    indices = torch.randint(0, 2**nbits, (out_c, in_c, 1, 1), dtype=torch.uint8)
    scale = torch.rand(out_c, 1, 1, 1, dtype=torch.float16) * 0.01
    with torch.no_grad():
        mod.indices.copy_(indices)
        mod.scale.copy_(scale)
    mod.eval()

    x = torch.randn(1, in_c, 1, 4, dtype=torch.float16)
    with torch.no_grad():
        out = mod(x)

    bias = 1 << (nbits - 1)
    expected_weight = (indices.to(torch.int16) - bias).to(torch.float16) * scale
    expected = torch.nn.functional.conv2d(x, expected_weight)
    torch.testing.assert_close(out, expected)


def test_native_quant_linear_blocking_matches_unblocked():
    """Output-dimension-blocked ``NativeQuantLinear`` matches an unblocked one bit-exact.

    Pins the sliced-shared-buffer bug: each block must own its own
    ``quantized_data``/``scale`` buffers (a ``NativeQuantLinearBlock``), not a slice of
    one shared buffer, or CoreAI's converter silently stops promoting them to genuine
    packed sub-byte constants at export time. This test only covers the numeric
    behavior (identical either way in eager mode); the packing regression itself needs
    an export-time check.
    """
    torch.manual_seed(0)
    out_f, in_f, nbits, block = 32, 8, 4, 8
    q = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
    scale = torch.rand(out_f, 1, dtype=torch.float16) * 0.01

    blocked = NativeQuantLinear(in_f, out_f, nbits=nbits, block_size=block)
    assert blocked.blocks is not None
    assert len(blocked.blocks) == out_f // block
    with torch.no_grad():
        for i, blk in enumerate(blocked.blocks):
            blk.quantized_data.copy_(q[i * block : (i + 1) * block])
            blk.scale.copy_(scale[i * block : (i + 1) * block])

    unblocked = NativeQuantLinear(in_f, out_f, nbits=nbits, block_size=None)
    assert unblocked.blocks is None
    with torch.no_grad():
        unblocked.quantized_data.copy_(q)
        unblocked.scale.copy_(scale)

    blocked.eval()
    unblocked.eval()
    x = torch.randn(1, 4, 1, in_f, dtype=torch.float16)
    with torch.no_grad():
        out_blocked = blocked(x)
        out_unblocked = unblocked(x)
    torch.testing.assert_close(out_blocked, out_unblocked)


@pytest.mark.parametrize("nbits", [2, 4, 8])
def test_unpack_subbyte_signed_is_unsigned_minus_offset(nbits):
    """Both unpackers read the same fields; the signed one subtracts 2**(nbits-1)."""
    packed = torch.randint(0, 256, (3, 5), dtype=torch.uint8)
    width = 5 * (8 // nbits) - 1  # a partial final byte keeps only the leading fields
    unsigned = unpack_unsigned_subbyte(packed, nbits, width)
    signed = unpack_signed_subbyte(packed, nbits, width)
    assert unsigned.shape == signed.shape == (3, width)
    bias = 0 if nbits == 8 else 1 << (nbits - 1)
    expected = unsigned.to(torch.int16) - bias if nbits != 8 else packed.to(torch.int8)[:, :width]
    assert torch.equal(signed.to(torch.int16), expected.to(torch.int16))


@pytest.mark.parametrize("nbits", [1, 3, 16])
def test_native_quant_helpers_reject_unsupported_nbits(nbits):
    with pytest.raises(ValueError, match="nbits must be one of"):
        unpack_signed_subbyte(torch.zeros(1, 1, dtype=torch.uint8), nbits, 1)
    with pytest.raises(ValueError, match="nbits must be one of"):
        unpack_unsigned_subbyte(torch.zeros(1, 1, dtype=torch.uint8), nbits, 1)
    with pytest.raises(ValueError, match="nbits must be one of"):
        dequantize_native(torch.zeros(1, 1), torch.ones(1, 1), nbits, torch.float16)


@pytest.mark.parametrize("nbits", [1, 3, 6, 16])
def test_native_projections_reject_unsupported_nbits(nbits):
    """A native projection is built at 2, 4 or 8 bits only; a standard one ignores the width."""
    cfg = _make_config()
    cfg.native_quantized = True
    with pytest.raises(ValueError, match=r"one of \(2, 4, 8\) bits"):
        _make_proj_conv(cfg, 4, 8, nbits=nbits)
    cfg.native_quantized = False
    assert isinstance(_make_proj_conv(cfg, 4, 8, nbits=nbits), torch.nn.Conv2d)


def test_checkpoint_widths_must_be_supported():
    """A checkpoint module at a width other than 2, 4 or 8 bits is refused, by name, before
    anything is unpacked."""
    ple = "extend.model.embed_tokens_per_layer"
    q_proj = "extend.model.layers.0.self_attn.q_proj"
    _check_native_nbits({q_proj: 4, ple: 2, "extend.lm_head": 2, "extend.model.embed_tokens": 8})
    with pytest.raises(NotImplementedError, match=rf"{ple} \(3 bits\)"):
        _check_native_nbits({q_proj: 4, ple: 3})


def test_make_proj_conv_dispatches_on_nbits_and_native_flag():
    """``_make_proj_conv`` picks LUT vs affine by bit-width, plain Conv2d if not native."""
    cfg = _make_config()

    cfg.native_quantized = True
    assert isinstance(_make_proj_conv(cfg, 4, 8, nbits=4), NativeQuantConv1x1)
    assert isinstance(_make_proj_conv(cfg, 4, 8, nbits=8), NativeQuantConv1x1)
    assert isinstance(_make_proj_conv(cfg, 4, 8, nbits=2), NativeLutQuantConv1x1)

    cfg.native_quantized = False
    assert isinstance(_make_proj_conv(cfg, 4, 8, nbits=2), torch.nn.Conv2d)


def test_native_mlp_nbits_reads_per_layer_plan():
    """``_native_mlp_nbits`` defaults to 4 with no plan, else reads the per-layer entry."""
    cfg = _make_config()
    assert _native_mlp_nbits(cfg, 0) == 4  # no plan set (standard, non-QAT checkpoint)

    cfg.native_mlp_nbits = [4, 4, 2, 2, 4, 4]
    assert _native_mlp_nbits(cfg, 0) == 4
    assert _native_mlp_nbits(cfg, 2) == 2
    assert _native_mlp_nbits(cfg, 3) == 2


def test_untied_lm_head_needs_a_native_quantized_checkpoint():
    """A standard untied checkpoint has no loader for its ``lm_head``, so it is refused."""
    cfg = _make_config()
    cfg.tie_word_embeddings = False
    with pytest.raises(NotImplementedError, match="untied lm_head"):
        Gemma4Extend(cfg)


def test_native_widths_must_cover_the_checkpoint():
    """A module declaring fewer bits than its checkpoint module is refused; more is fine."""
    cfg = _make_config()
    cfg.native_quantized = True
    model = Gemma4ForCausalLMForiOS(cfg, model_device="meta")
    q_proj = "extend.model.layers.0.self_attn.q_proj"
    _check_native_widths(model, {q_proj: 4, "extend.model.embed_tokens": 2})
    _check_native_widths(model, {q_proj: 2})
    with pytest.raises(NotImplementedError, match="8 bits in the checkpoint"):
        _check_native_widths(model, {q_proj: 8})


def test_native_ple_table_stores_8_bit_rows_offset():
    """8-bit PLE rows ship as signed bytes; the sidecar stores them offset by 128, as the
    runner reads every width."""
    from types import SimpleNamespace

    values = torch.tensor([[-128, -1, 0, 127]], dtype=torch.int8)
    module = SimpleNamespace(
        embedding_quantized=values,
        embedding_scale=torch.ones(1, 1),
        num_bits=8,
        embedding_dim=4,
    )
    packed, scale, nbits = _native_ple_table(module, _make_config(), truncate=False)
    assert nbits == 8 and packed.dtype == torch.uint8
    assert packed.tolist() == [[0, 127, 128, 255]]
    assert (packed.to(torch.int16) - 128).tolist() == values.tolist()

    # Sub-byte rows are already offset-packed and pass through unchanged.
    packed4 = torch.tensor([[0x8F]], dtype=torch.uint8)
    module4 = SimpleNamespace(
        embedding_quantized=packed4, embedding_scale=torch.ones(1, 1), num_bits=4, embedding_dim=2
    )
    assert torch.equal(_native_ple_table(module4, _make_config(), truncate=False)[0], packed4)


def test_native_ple_table_truncates_to_the_remaining_layers():
    """``--num-layers`` cuts each packed row and its per-block scales to the kept layers."""
    from types import SimpleNamespace

    cfg = _make_config()  # 6 layers x 16 PLE dims; the checkpoint below has 8 layers
    ple_dim, nbits, block = cfg.hidden_size_per_layer_input, 4, 32
    full_dim = 8 * ple_dim
    module = SimpleNamespace(
        embedding_quantized=torch.arange(3 * full_dim // 2, dtype=torch.uint8).reshape(3, -1),
        embedding_scale=torch.arange(3 * full_dim // block, dtype=torch.float32).reshape(3, -1),
        num_bits=nbits,
        embedding_dim=full_dim,
    )
    packed, scale, _ = _native_ple_table(module, cfg, truncate=True)
    kept = cfg.num_hidden_layers * ple_dim
    assert packed.shape == (3, kept // 2)
    assert scale.shape == (3, kept // block)
    assert torch.equal(packed, module.embedding_quantized[:, : kept // 2])
    assert torch.equal(scale, module.embedding_scale[:, : kept // block].to(torch.float16))
    assert packed.is_contiguous() and scale.is_contiguous()

    # Without truncation the rows ship whole.
    packed, scale, _ = _native_ple_table(module, cfg, truncate=False)
    assert packed.shape == (3, full_dim // 2) and scale.shape == (3, full_dim // block)


def _native_config(mlp_nbits: list[int]) -> Gemma4TextConfig:
    """A tiny native-quantized config: untied ``lm_head`` wider than one output block, so it is
    split, and the given per-layer MLP widths."""
    cfg = _make_config()
    cfg.vocab_size = DEFAULT_KV_BLOCK_SIZE + 8
    cfg.tie_word_embeddings = False
    cfg.native_quantized = True
    cfg.native_mlp_nbits = mlp_nbits
    return cfg


def _fake_native_hf_state_dict(cfg) -> dict[str, torch.Tensor]:
    """A text decoder's state dict in the form ``_native_quantized_hf_state_dict`` produces it:
    each quantized projection and the embedding table as an unpacked ``(quantized_data, scale)``
    pair, a sub-4-bit MLP projection as ``(indices, scale)``, everything else as is, and no PLE
    table (``from_hf`` externalizes it)."""
    torch.manual_seed(0)
    out: dict[str, torch.Tensor] = {}
    for key, value in Gemma4ForCausalLM(cfg).state_dict().items():
        base, leaf = key.rsplit(".", 1)
        if "embed_tokens_per_layer" in base:
            continue
        quantized = leaf == "weight" and (
            (".self_attn." in base and base.endswith("_proj"))
            or ".mlp." in base
            or base.endswith((".per_layer_input_gate", ".per_layer_projection"))
            or base in ("model.embed_tokens", "lm_head")
        )
        if not quantized:
            out[key] = value
            continue
        nbits = 8
        if ".mlp." in base:
            nbits = cfg.native_mlp_nbits[int(base.split(".")[2])]
        if nbits < 4:
            out[f"{base}.indices"] = torch.randint(0, 2**nbits, value.shape, dtype=torch.uint8)
        else:
            lo, hi = -(2 ** (nbits - 1)), 2 ** (nbits - 1)
            out[f"{base}.quantized_data"] = torch.randint(lo, hi, value.shape, dtype=torch.int8)
        out[f"{base}.scale"] = torch.rand(value.shape[0], 1).to(torch.float16)
    return out


def test_native_quantized_state_dict_loads_strictly():
    """``_mutate_native_quantized_state_dict`` maps every checkpoint tensor onto the native
    model: the shared-KV layers' ``k``/``v`` dropped, LUT vs affine MLP keys, the embedding
    table and scale moved to ``load_embeddings``, and ``lm_head`` split into output blocks."""
    cfg = _native_config([4, 4, 4, 4, 2, 2])  # layers 4 and 5 share KV and ship at 2 bits
    hf_sd = _fake_native_hf_state_dict(cfg)
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(cfg, model_device="meta")
    model.to(dtype=torch.float16)
    model._mutate_native_quantized_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)

    layers = model.extend.model.layers
    assert isinstance(layers[0].mlp.down_proj, NativeQuantConv1x1)
    assert isinstance(layers[5].mlp.down_proj, NativeLutQuantConv1x1)
    assert not hasattr(layers[5].self_attn, "k_proj")
    assert torch.equal(
        layers[5].mlp.down_proj.indices.squeeze(-1).squeeze(-1),
        hf_sd["model.layers.5.mlp.down_proj.indices"],
    )
    assert torch.equal(
        layers[0].self_attn.k_proj.quantized_data.squeeze(-1).squeeze(-1),
        hf_sd["model.layers.0.self_attn.k_proj.quantized_data"],
    )
    assert torch.equal(
        model.load_embeddings.embedding_table.squeeze(1),
        hf_sd["model.embed_tokens.quantized_data"],
    )
    assert torch.equal(
        model.load_embeddings.embedding_scale.squeeze(1), hf_sd["model.embed_tokens.scale"]
    )

    blocks = model.extend.lm_head.blocks
    assert [b.quantized_data.shape[0] for b in blocks] == [DEFAULT_KV_BLOCK_SIZE, 8]
    assert torch.equal(
        torch.cat([b.quantized_data for b in blocks]), hf_sd["lm_head.quantized_data"]
    )
    assert torch.equal(torch.cat([b.scale for b in blocks]), hf_sd["lm_head.scale"])


def test_native_quantized_reads_the_config_flag():
    """A built model reports the flag its modules were built from; no flag is standard."""
    cfg = _make_config()
    assert not hasattr(cfg, "native_quantized")
    standard = Gemma4ForCausalLMForiOS(cfg, model_device="meta")
    assert not standard.native_quantized
    assert isinstance(standard.extend.model.layers[0].self_attn.q_proj, torch.nn.Conv2d)

    native = Gemma4ForCausalLMForiOS(_native_config([4] * 6), model_device="meta")
    assert native.native_quantized
    assert isinstance(native.extend.model.layers[0].self_attn.q_proj, NativeQuantConv1x1)


def test_native_embedding_tables_follow_the_model_device():
    """The native embedding table and its scale are built on ``model_device``, so ``from_hf``
    (which builds on meta) does not allocate the table before loading it."""
    model = Gemma4ForCausalLMForiOS(_native_config([4] * 6), model_device="meta")
    assert model.load_embeddings.embedding_table.is_meta
    assert model.load_embeddings.embedding_scale.is_meta


@pytest.mark.skipif(QuantizedLinear is None, reason="needs transformers' Gemma QAT quantizer")
def test_quantized_detection_reads_the_text_decoder_only():
    """A quantized vision or audio tower alone does not make the text decoder native."""
    hf_model = torch.nn.Module()
    hf_model.model = torch.nn.Module()
    hf_model.model.vision_tower = QuantizedLinear(4, 4, num_bits=8)
    hf_model.model.language_model = torch.nn.Module()
    hf_model.model.language_model.q_proj = torch.nn.Linear(4, 4)
    assert not _is_quantized_hf_model(hf_model)

    hf_model.model.language_model.q_proj = QuantizedLinear(4, 4, num_bits=4)
    assert _is_quantized_hf_model(hf_model)


# ---------------------------------------------------------------------------
# Real QAT checkpoint: native-quantized module dispatch + bit-exactness.
#
# The tests above cover the standard (dequantize-and-palettize) checkpoints. This
# one loads the real "-qat-mobile-transformers" E2B checkpoint through ``from_hf``
# and checks that (1) each MLP projection routes to the module and bit width its
# checkpoint calls for (affine vs LUT), and (2) both paths dequantize bit-exact to
# HF's own reference dequantization -- no dequantize/requantize round-trip
# anywhere. It is heavy (downloads and loads a ~5B QAT checkpoint) and needs
# network + weights + transformers>=5.5, so it is opt-in via ``-m slow``, same as
# ``test_e2b_full_parity_hf_vs_torch``.
# ---------------------------------------------------------------------------

E2B_QAT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"


@pytest.mark.slow
@pytest.mark.flaky(reruns=0)
def test_e2b_qat_native_quantization_dispatch_and_bit_exactness():
    """Real E2B QAT checkpoint: per-layer affine/LUT dispatch and bit-exact dequant."""
    if not _hf_hub_reachable(E2B_QAT_MODEL_ID):
        pytest.skip(f"HuggingFace Hub unreachable for {E2B_QAT_MODEL_ID!r}")
    from huggingface_hub.errors import GatedRepoError
    from transformers import Gemma4ForConditionalGeneration

    from coreai_models.primitives.ios.quantization import dequantize_native_dense

    try:
        hf_model = Gemma4ForConditionalGeneration.from_pretrained(
            E2B_QAT_MODEL_ID, dtype=torch.float16
        )
    except (OSError, GatedRepoError) as exc:  # download or gating failures only
        pytest.skip(f"could not download E2B QAT weights: {exc}")
    # A load failure past the download is a bug, not a skip.
    model = Gemma4ForCausalLMForiOS.from_hf(
        E2B_QAT_MODEL_ID, max_context_length=1024, target_dtype=torch.float16
    ).eval()

    assert model.config.native_quantized is True
    hf_mods = dict(hf_model.named_modules())

    # Every projection's dispatch and width is checked (cheap); the numeric compare runs
    # for the first layer of each kind, since the module-level tests above cover the
    # arithmetic exhaustively.
    compared: set[bool] = set()
    for i, layer in enumerate(model.extend.model.layers):
        is_lut = None
        for proj in ("gate_proj", "up_proj", "down_proj"):
            hf_proj = hf_mods[f"model.language_model.layers.{i}.mlp.{proj}"]
            ours = getattr(layer.mlp, proj)
            lut = hf_proj.num_bits < 4
            assert is_lut in (None, lut), f"layer {i} mixes LUT and affine MLP projections"
            is_lut = lut
            assert isinstance(ours, NativeLutQuantConv1x1 if lut else NativeQuantConv1x1)
            assert ours.nbits == hf_proj.num_bits, f"layer {i} {proj}"

            if lut in compared:
                continue
            if lut:
                dense = torch.ops.coreai.lut_to_dense(ours.indices, ours.lut, 0)
                weight = dequantize_native_dense(dense, ours.scale, torch.float16)
            else:
                weight = dequantize_native(
                    ours.quantized_data, ours.scale, ours.nbits, torch.float16
                )
            expected = hf_proj._dequantize_weights(torch.float16)
            torch.testing.assert_close(weight.squeeze(-1).squeeze(-1), expected, atol=0, rtol=0)
        compared.add(is_lut)

    assert True in compared, (
        "expected at least one genuinely sub-4-bit MLP layer in this checkpoint"
    )

    # Every quantized text-decoder projection has a native module at least as wide as the
    # checkpoint's, so `_check_native_widths` saw it; only the shared-KV layers' k/v
    # projections, which this model does not build, have none.
    ours_mods = dict(model.named_modules())
    hf_prefix = "model.language_model."
    covered = set()
    for name, hf_mod in hf_mods.items():
        if not isinstance(hf_mod, QuantizedLinear) or not (
            name.startswith(hf_prefix) or name == "lm_head"
        ):
            continue
        ours = ours_mods.get(_native_module_name(name, hf_prefix))
        if ours is None:
            assert name.endswith((".self_attn.k_proj", ".self_attn.v_proj")), name
            continue
        assert ours.nbits >= hf_mod.num_bits, name
        covered.add(name.rsplit(".", 1)[-1])
    assert {
        "q_proj",
        "o_proj",
        "per_layer_input_gate",
        "per_layer_projection",
        "lm_head",
    } <= covered
    assert False in compared, "expected at least one 4-bit MLP layer in this checkpoint"

    del hf_model, model


def test_native_quant_conv1x1_activation_quant_matches_hf_srq():
    """With activation scales set, the module quantizes its input and output exactly as the
    HF QAT layer does (``apply_srq`` around the matmul); a scale of 0 leaves that side alone."""
    gemma_quant = pytest.importorskip("transformers.integrations.gemma_quant")
    torch.manual_seed(0)
    out_c, in_c = 16, 8
    mod = NativeQuantConv1x1(in_c, out_c, nbits=4)
    q = torch.randint(-8, 8, (out_c, in_c, 1, 1), dtype=torch.int8)
    scale = torch.rand(out_c, 1, 1, 1, dtype=torch.float16) * 0.05
    with torch.no_grad():
        mod.quantized_data.copy_(q)
        mod.scale.copy_(scale)
    x = torch.randn(1, in_c, 1, 6, dtype=torch.float16)
    weight = q.reshape(out_c, in_c).to(torch.float16) * scale.reshape(out_c, 1)

    def reference(s_in: float, s_out: float) -> torch.Tensor:
        xs = x.squeeze(2).transpose(1, 2)  # (1, 6, in_c): the HF linear layout
        xs = gemma_quant.apply_srq(xs, torch.tensor(s_in))
        out = gemma_quant.apply_srq(torch.nn.functional.linear(xs, weight), torch.tensor(s_out))
        return out.transpose(1, 2).unsqueeze(2)

    for s_in in (0.0, 0.02):
        mod.input_activation_scale, mod.output_activation_scale = s_in, 0.0
        with torch.no_grad():
            torch.testing.assert_close(mod(x), reference(s_in, 0.0), atol=1e-3, rtol=1e-3)
    # With the output quantized too, the conv and the linear accumulate in a different fp16
    # order, so a value on a rounding boundary can land one step either way.
    mod.input_activation_scale, mod.output_activation_scale = 0.02, 0.01
    with torch.no_grad():
        diff = (mod(x) - reference(0.02, 0.01)).abs()
    assert diff.max() <= 0.01 + 1e-3
    assert (diff > 1e-3).float().mean() < 0.05
    # The scales really do change the result.
    mod.input_activation_scale = mod.output_activation_scale = 0.0
    plain = mod(x)
    mod.input_activation_scale, mod.output_activation_scale = 0.05, 0.05
    assert not torch.allclose(mod(x), plain)


def test_geglu_mlp_puts_tokens_in_the_conv_width():
    """The MLP runs its 1x1 convs with the tokens in their width, (B, D, 1, S), and computes
    exactly what the per-token batch layout, (B*S, D, 1, 1), computes, activation quantization
    included."""
    cfg = _make_config()
    cfg.native_quantized = True
    torch.manual_seed(0)
    mlp = GeGLUMLP(cfg, cfg.hidden_size, cfg.intermediate_size, layer_idx=0).half().eval()
    with torch.no_grad():
        for conv in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            conv.quantized_data.copy_(torch.randint(-8, 8, conv.quantized_data.shape))
            conv.scale.copy_(torch.rand_like(conv.scale) * 0.05)
            conv.input_activation_scale, conv.output_activation_scale = 0.02, 0.03
    shapes = []
    mlp.gate_proj.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    seq, dim = 7, cfg.hidden_size
    x = torch.randn(1, seq, 1, dim, dtype=torch.float16)

    with torch.no_grad():
        out = mlp(x)
        xb = x.reshape(seq, dim, 1, 1)
        gate = torch.nn.functional.gelu(mlp.gate_proj(xb), approximate="tanh")
        reference = mlp.down_proj(mlp.up_proj(xb) * gate).reshape(1, seq, 1, dim)

    assert shapes[0] == (1, dim, 1, seq)
    assert torch.equal(out, reference)
