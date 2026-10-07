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
    Gemma4ForCausalLMForiOS,
    _compute_kv_layout,
    sliding_ring_size,
)
from coreai_models.primitives.ios.quantization import quantize_per_tensor  # noqa: E402
from coreai_models.primitives.ios.rope import RoPECache  # noqa: E402

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
