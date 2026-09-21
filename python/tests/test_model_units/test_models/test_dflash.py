# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for DFlash speculative decoding: drafter shapes, hidden state extraction, fused target."""

import torch
import torch.nn as nn

from coreai_models._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.muse_glimmer import (
    MuseGlimmerForCausalLMWithDrafter,
    MuseGlimmerModelWithDrafter,
)
from coreai_models.models.macos.muse_glimmer_drafter_dflash import (
    MuseGlimmerDFlashDrafterForCausalLM,
    dflash_draft_mask,
)
from coreai_models.primitives.macos.cache import RingKVCache


def _make_small_config():
    """Create a minimal Muse Glimmer config for testing (4 layers, tiny dims)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        vocab_size=200,
        rms_norm_eps=1e-5,
        sliding_window=32,
        rope_parameters={"rope_theta": 500000.0},
        block_size=4,
        mask_token_id=99,
        max_position_embeddings=512,
        output_multiplier=1.0,
        qk_scale_factor=1.0,
        final_logit_softcapping=None,
        tie_word_embeddings=False,
        post_norm_eps=1e-8,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
        target_layer_ids=[0, 2],
    )


class TestDFlashDraftMask:
    """Verify the bidirectional attention mask for DFlash draft mode."""

    def test_mask_shape(self):
        mask = dflash_draft_mask(query_len=4, capacity=32, offset=5, device=torch.device("cpu"))
        assert mask.shape == (4, 32)

    def test_mask_valid_count(self):
        mask = dflash_draft_mask(query_len=4, capacity=32, offset=5, device=torch.device("cpu"))
        assert mask[0].sum().item() == 5 + 4  # n_injected + n_draft

    def test_mask_bidirectional(self):
        mask = dflash_draft_mask(query_len=8, capacity=64, offset=10, device=torch.device("cpu"))
        assert torch.all(mask[0] == mask[-1]), "All rows should be identical (bidirectional)"

    def test_mask_no_stale_slots(self):
        n_inject, K = 5, 4
        dev = torch.device("cpu")
        mask = dflash_draft_mask(query_len=K, capacity=32, offset=n_inject, device=dev)
        assert mask[0, n_inject + K :].sum().item() == 0


class TestDFlashDrafter:
    """Shape and correctness tests for the DFlash drafter."""

    def test_inject_kv_populates_cache(self):
        cfg = _make_small_config()
        model = MuseGlimmerDFlashDrafterForCausalLM(cfg)
        model.eval()

        k = torch.zeros(
            cfg.num_hidden_layers,
            1,
            cfg.num_key_value_heads,
            cfg.sliding_window,
            cfg.head_dim,
        )
        v = torch.zeros_like(k)
        cache = RingKVCache(k, v)

        features = torch.randn(1, 3, cfg.hidden_size)
        pos = torch.arange(3).unsqueeze(0)
        with torch.no_grad():
            model.inject_kv(features, pos, cache)

            assert k[0, 0, 0, 0, :].norm().item() > 0, "Cache should be populated after inject_kv"
            assert k[0, 0, 0, 3, :].norm().item() == 0

    def test_draft_output_shape(self):
        cfg = _make_small_config()
        K = cfg.block_size
        model = MuseGlimmerDFlashDrafterForCausalLM(cfg)
        model.eval()

        k = torch.zeros(
            cfg.num_hidden_layers,
            1,
            cfg.num_key_value_heads,
            cfg.sliding_window,
            cfg.head_dim,
        )
        v = torch.zeros_like(k)
        cache = RingKVCache(k, v)

        features = torch.randn(1, 5, cfg.hidden_size)
        model.inject_kv(features, torch.arange(5).unsqueeze(0), cache)

        draft_ids = torch.tensor([[42] + [cfg.mask_token_id] * (K - 1)])
        full_pos = torch.arange(5 + K).unsqueeze(0)
        with torch.no_grad():
            logits = model.draft(draft_ids, full_pos, cache)

        assert logits.shape == (1, K, cfg.vocab_size)

    def test_draft_no_nan(self):
        cfg = _make_small_config()
        K = cfg.block_size
        model = MuseGlimmerDFlashDrafterForCausalLM(cfg)
        model.eval()

        k = torch.zeros(
            cfg.num_hidden_layers,
            1,
            cfg.num_key_value_heads,
            cfg.sliding_window,
            cfg.head_dim,
        )
        v = torch.zeros_like(k)
        cache = RingKVCache(k, v)

        features = torch.randn(1, 5, cfg.hidden_size)
        model.inject_kv(features, torch.arange(5).unsqueeze(0), cache)

        draft_ids = torch.tensor([[42] + [cfg.mask_token_id] * (K - 1)])
        full_pos = torch.arange(5 + K).unsqueeze(0)
        with torch.no_grad():
            logits = model.draft(draft_ids, full_pos, cache)

        assert not torch.isnan(logits).any(), "Draft logits should not contain NaN"

    def test_no_embed_norm_in_draft(self):
        """Regression test: DFlash drafter must NOT apply embed norm."""
        import inspect

        from coreai_models.models.macos.muse_glimmer_drafter_dflash import DFlashDrafterModel

        src = inspect.getsource(DFlashDrafterModel.draft)
        assert "rsqrt" not in src, (
            "DFlash drafter draft() must NOT apply embed norm (rsqrt). "
            "HF explicitly says: 'The assistant needs embedding without norm'."
        )


class _InjectKVGraphModule(nn.Module):
    """nn.Module wrapper around ``inject_kv_graph`` for torch.export.

    ``torch.export.export`` requires an ``nn.Module``, not a bound method (the
    same reason the exporter wraps these entrypoints in ``export.macos``). The
    forward param names match the graph's input/state names so ``dynamic_shapes``
    keys stay valid.
    """

    def __init__(self, model: MuseGlimmerDFlashDrafterForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, features, position_ids, sliding_k_cache, sliding_v_cache):
        return self.model.inject_kv_graph(features, position_ids, sliding_k_cache, sliding_v_cache)


class _DraftGraphModule(nn.Module):
    """nn.Module wrapper around ``draft_graph`` for torch.export (see
    :class:`_InjectKVGraphModule`)."""

    def __init__(self, model: MuseGlimmerDFlashDrafterForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, sliding_k_cache, sliding_v_cache):
        return self.model.draft_graph(input_ids, position_ids, sliding_k_cache, sliding_v_cache)


def _export_graph(module, kwargs, dyn):
    """torch.export one entrypoint via its nn.Module wrapper (the supported route).

    ``torch.export.export`` demands an ``nn.Module``, so each entrypoint is traced
    through its wrapper -- the same approach the exporter uses in ``export.macos``.
    """
    return torch.export.export(module, args=(), kwargs=kwargs, dynamic_shapes=dyn)


class TestDFlashDrafterExportContract:
    """Two-entrypoint (inject_kv + draft) export contract, trace-level only.

    Runs without the 55GB model: tiny config, hand-built tensors, no coreai
    conversion.
    """

    def _model_and_spec(self):
        cfg = _make_small_config()
        m = MuseGlimmerDFlashDrafterForCausalLM(cfg)
        m.eval()
        # cache_seq_len must not exceed max_context_length (small config uses 512).
        spec = TraceSpec(
            max_context_length=cfg.max_position_embeddings,
            cache_seq_len=min(TRACE_KV_CACHE_SEQ_LEN, cfg.max_position_embeddings),
        )
        return cfg, m, spec

    def test_contract_hooks_agree(self):
        cfg, m, spec = self._model_and_spec()
        ref = m.build_reference_inputs(cfg, torch.float32, spec)
        dyn = m.build_dynamic_shapes(cfg, spec)
        m.validate_export_contract(ref, dyn)  # must not raise

        graphs = {"inject_kv", "draft"}
        assert set(m.export_input_names()) == graphs
        assert set(m.export_state_names()) == graphs
        assert set(m.export_output_names()) == graphs
        assert set(ref) == graphs
        assert set(dyn) == graphs

        assert m.export_output_names()["inject_kv"] == ()
        assert m.export_output_names()["draft"] == ("logits",)
        # Identical state tuple for both graphs => one shared physical ring buffer.
        assert (
            m.export_state_names()["inject_kv"]
            == m.export_state_names()["draft"]
            == ("slidingKeyCache", "slidingValueCache")
        )
        assert m.export_state_classification() == {
            "slidingKeyCache": "sliding_kv_cache",
            "slidingValueCache": "sliding_kv_cache",
        }

    def test_graph_wrapper_shapes(self):
        cfg = _make_small_config()
        m = MuseGlimmerDFlashDrafterForCausalLM(cfg)
        m.eval()

        q, K = 5, cfg.block_size

        sk = torch.zeros(
            cfg.num_hidden_layers,
            1,
            cfg.num_key_value_heads,
            cfg.sliding_window,
            cfg.head_dim,
        )
        sv = torch.zeros_like(sk)

        # inject_kv_graph: offset 0 (position_ids length == features length).
        features = torch.randn(1, q, cfg.hidden_size)
        inject_pos = torch.arange(q).unsqueeze(0)
        with torch.no_grad():
            out = m.inject_kv_graph(features, inject_pos, sk, sv)
        assert out == ()
        assert sk[0, 0, 0, 0, :].norm().item() > 0, "inject_kv_graph must populate the ring cache"

        # draft_graph reads the SAME (shared) buffer and appends at offset == q.
        draft_ids = torch.tensor([[42] + [cfg.mask_token_id] * (K - 1)])
        full_pos = torch.arange(q + K).unsqueeze(0)
        with torch.no_grad():
            logits = m.draft_graph(draft_ids, full_pos, sk, sv)
        assert logits.shape == (1, K, cfg.vocab_size)

    def test_torch_export_traces_both_graphs(self):
        """Guard the shape-derived inject offset (no ``.item()``) actually traces.

        The inject offset is ``seq_len - n_features`` (a shape arithmetic), not
        ``int(position_ids[0, 0])``. If it regressed to reading a value out of the
        tensor, the trace would emit a data-dependent ``aten._local_scalar_dense``
        (``.item()``) node, so we assert the inject_kv graph carries none.
        """
        cfg, m, spec = self._model_and_spec()
        ref = m.build_reference_inputs(cfg, torch.float32, spec)
        dyn = m.build_dynamic_shapes(cfg, spec)

        inject_ep = _export_graph(_InjectKVGraphModule(m), ref["inject_kv"], dyn["inject_kv"])
        draft_ep = _export_graph(_DraftGraphModule(m), ref["draft"], dyn["draft"])

        assert isinstance(inject_ep, torch.export.ExportedProgram)
        assert isinstance(draft_ep, torch.export.ExportedProgram)

        # A shape-derived offset must NOT lower to a data-dependent .item() read.
        data_dependent = {
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.item.default,
        }
        offending = [
            node
            for node in inject_ep.graph_module.graph.nodes
            if node.op == "call_function" and node.target in data_dependent
        ]
        assert not offending, (
            f"inject_kv graph must derive the offset from shapes, not .item(); "
            f"found data-dependent nodes: {[n.target for n in offending]}"
        )


class TestMuseGlimmerWithDrafter:
    """Tests for the fused target model that extracts hidden states."""

    def test_with_drafter_extracts_correct_count(self):
        config = _make_small_config()

        model = MuseGlimmerModelWithDrafter(config, target_layer_ids=[0, 2])
        model.eval()

        from coreai_models.primitives.macos.cache import KVCache

        ids = torch.randint(0, config.vocab_size, (1, 4))
        pos = torch.arange(4).unsqueeze(0)
        ng = sum(1 for t in config.layer_types if t == "full_attention")
        ns = sum(1 for t in config.layer_types if t == "sliding_attention")
        gk = torch.zeros(ng, 1, config.num_key_value_heads, 32, config.head_dim)
        gv = torch.zeros_like(gk)
        sk = torch.zeros(ns, 1, config.num_key_value_heads, config.sliding_window, config.head_dim)
        sv = torch.zeros_like(sk)

        with torch.no_grad():
            hidden, extracted = model(ids, pos, KVCache(gk, gv), RingKVCache(sk, sv))

        assert len(extracted) == 2, f"Expected 2 extracted layers, got {len(extracted)}"
        assert extracted[0].shape == (1, 4, config.hidden_size)
        assert hidden.shape == (1, 4, config.hidden_size)

    def test_fused_target_dual_output(self):
        config = _make_small_config()
        config.target_layer_ids = [0, 1, 2, 3]  # 4 layers to match encoder_fc

        model = MuseGlimmerForCausalLMWithDrafter(config)
        model.eval()

        ids = torch.randint(0, config.vocab_size, (1, 4))
        pos = torch.arange(4).unsqueeze(0)
        ng = sum(1 for t in config.layer_types if t == "full_attention")
        ns = sum(1 for t in config.layer_types if t == "sliding_attention")
        gk = torch.zeros(ng, 1, config.num_key_value_heads, 32, config.head_dim)
        gv = torch.zeros_like(gk)
        sk = torch.zeros(ns, 1, config.num_key_value_heads, config.sliding_window, config.head_dim)
        sv = torch.zeros_like(sk)

        with torch.no_grad():
            logits, features = model(ids, pos, gk, gv, sk, sv)

        assert logits.shape == (1, 4, config.vocab_size)
        assert features.shape == (1, 4, config.hidden_size)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(features).any()
