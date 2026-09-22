# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Stage-2 fused-target export enablement: prefill honoring, tuple-aware logits cast,
speculative bundle metadata, and registry wiring. All unit-testable without the 55GB
model — tiny configs and hand-built inputs only."""

import torch

from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLMWithDrafter
from coreai_models.models.macos.muse_glimmer_drafter_dflash import (
    MuseGlimmerDFlashDrafterForCausalLM,
)

from .test_models.test_dflash import _make_small_config


def _build_caches(config):
    ng = sum(1 for t in config.layer_types if t == "full_attention")
    ns = sum(1 for t in config.layer_types if t == "sliding_attention")
    gk = torch.zeros(ng, 1, config.num_key_value_heads, 32, config.head_dim)
    sk = torch.zeros(ns, 1, config.num_key_value_heads, config.sliding_window, config.head_dim)
    return gk, torch.zeros_like(gk), sk, torch.zeros_like(sk)


class TestFusedTargetPrefill:
    def _model(self):
        cfg = _make_small_config()
        cfg.target_layer_ids = [0, 1, 2, 3]  # match encoder_fc (4 * hidden)
        model = MuseGlimmerForCausalLMWithDrafter(cfg)
        model.eval()
        return cfg, model

    def test_prefill_mode_returns_empty_and_writes_kv(self):
        cfg, model = self._model()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        pos = torch.arange(4).unsqueeze(0)
        gk, gv, sk, sv = _build_caches(cfg)
        model.set_prefill_mode(True)
        with torch.no_grad():
            out = model(ids, pos, gk, gv, sk, sv)
        assert out == ()  # prefill drops logits + drafter_features
        assert gk.abs().sum() > 0  # backbone ran → KV cache written

    def test_prefill_toggle_restores_dual_output(self):
        cfg, model = self._model()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        pos = torch.arange(4).unsqueeze(0)
        gk, gv, sk, sv = _build_caches(cfg)
        model.set_prefill_mode(True)
        with torch.no_grad():
            model(ids, pos, gk, gv, sk, sv)
        model.set_prefill_mode(False)
        with torch.no_grad():
            logits, features = model(ids, pos, gk, gv, sk, sv)
        assert logits.shape == (1, 4, cfg.vocab_size)
        assert features.shape == (1, 4, cfg.hidden_size)

    def test_fused_target_casts_bf16_to_fp16(self):
        cfg, model = self._model()
        model = model.to(torch.bfloat16)
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        pos = torch.arange(4).unsqueeze(0)
        gk, gv, sk, sv = (c.to(torch.bfloat16) for c in _build_caches(cfg))
        with torch.no_grad():
            logits, features = model(ids, pos, gk, gv, sk, sv)
        assert logits.dtype == torch.float16
        assert features.dtype == torch.float16  # tuple element cast too


class TestTupleAwareCast:
    """The cast decorator recurses into tuple/list returns."""

    @staticmethod
    def _wrap(value_fn):
        @BaseForCausalLM.cast_logits_bfloat16_to_float16
        def f():
            return value_fn()

        return f

    def test_bare_tensor(self):
        assert self._wrap(lambda: torch.zeros(2, dtype=torch.bfloat16))().dtype == torch.float16

    def test_tuple_both_cast(self):
        out = self._wrap(
            lambda: (torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.bfloat16))
        )()
        assert isinstance(out, tuple)
        assert all(t.dtype == torch.float16 for t in out)

    def test_mixed_tuple_only_bf16(self):
        out = self._wrap(
            lambda: (torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.float32))
        )()
        assert out[0].dtype == torch.float16
        assert out[1].dtype == torch.float32

    def test_empty_tuple_passthrough(self):
        assert self._wrap(lambda: ())() == ()

    def test_fp16_unchanged(self):
        assert self._wrap(lambda: torch.zeros(2, dtype=torch.float16))().dtype == torch.float16

    def test_disable_env(self, monkeypatch):
        monkeypatch.setenv("DISABLE_BFLOAT16_CAST_FOR_LOGITS", "1")
        assert self._wrap(lambda: torch.zeros(2, dtype=torch.bfloat16))().dtype == torch.bfloat16


class TestSpeculativeMetadata:
    def test_merge_and_key_rename(self):
        from types import SimpleNamespace

        from coreai_models.export.bundle import _speculative_decoding_metadata

        cfg = SimpleNamespace(
            target_layer_ids=[1, 13, 25, 37, 49],
            draft_mask_token_id=201818,
            block_size=16,
            drafter_hidden_size=4096,
            hidden_size=5760,
        )
        block = _speculative_decoding_metadata(cfg, {"num_draft_tokens": 5, "shared_embeddings": True})
        assert block["mask_token_id"] == 201818  # draft_mask_token_id -> mask_token_id
        assert block["block_size"] == 16
        assert block["target_layer_ids"] == [1, 13, 25, 37, 49]
        assert block["drafter_hidden_size"] == 4096
        assert block["num_draft_tokens"] == 5

    def test_drafter_hidden_size_falls_back_to_hidden_size(self):
        from types import SimpleNamespace

        from coreai_models.export.bundle import _speculative_decoding_metadata

        cfg = SimpleNamespace(draft_mask_token_id=99, block_size=4, hidden_size=64)
        assert _speculative_decoding_metadata(cfg, {})["drafter_hidden_size"] == 64

    def test_runtime_knob_overrides_structural(self):
        from types import SimpleNamespace

        from coreai_models.export.bundle import _speculative_decoding_metadata

        cfg = SimpleNamespace(block_size=16, hidden_size=64)
        assert _speculative_decoding_metadata(cfg, {"block_size": 8})["block_size"] == 8


class TestDrafterStructuralValidation:
    """Explicit + validated drafter structural metadata (fail loud on divergence)."""

    @staticmethod
    def _target():
        from types import SimpleNamespace

        return SimpleNamespace(
            vocab_size=202048,
            hidden_size=5760,
            head_dim=128,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=64,
        )

    @staticmethod
    def _drafter(**overrides):
        from types import SimpleNamespace

        base = dict(
            vocab_size=202048,
            hidden_size=5760,
            head_dim=128,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=4,  # intentionally shallower than the target
            target_layer_ids=[1, 13, 25, 37, 49],
            draft_mask_token_id=201818,
            block_size=16,
            drafter_hidden_size=5760,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_structural_metadata_sourced_from_drafter_not_target(self):
        from types import SimpleNamespace

        from coreai_models.export.bundle import _speculative_decoding_metadata

        # Target has NO DFlash structural attrs at all — proving the block is
        # sourced from the drafter, not implicitly from the target.
        target = SimpleNamespace(
            vocab_size=202048,
            hidden_size=5760,
            head_dim=128,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        drafter = self._drafter()
        block = _speculative_decoding_metadata(target, {"drafter_kind": "dflash"}, drafter_config=drafter)
        assert block["block_size"] == 16
        assert block["mask_token_id"] == 201818
        assert block["target_layer_ids"] == [1, 13, 25, 37, 49]
        assert block["drafter_hidden_size"] == 5760
        assert block["drafter_kind"] == "dflash"

    def test_matching_geometry_passes(self):
        from coreai_models.export.bundle import _speculative_decoding_metadata

        # Should not raise.
        _speculative_decoding_metadata(self._target(), {}, drafter_config=self._drafter())

    def test_vocab_divergence_fails_loud(self):
        import pytest

        from coreai_models.export.bundle import _speculative_decoding_metadata

        # The historical crash: drafter 262144 vs target 202048.
        drafter = self._drafter(vocab_size=262144)
        with pytest.raises(ValueError, match=r"vocab_size.*262144.*202048"):
            _speculative_decoding_metadata(self._target(), {}, drafter_config=drafter)

    def test_head_dim_divergence_fails_loud(self):
        import pytest

        from coreai_models.export.bundle import _speculative_decoding_metadata

        drafter = self._drafter(head_dim=64)
        with pytest.raises(ValueError, match=r"head_dim"):
            _speculative_decoding_metadata(self._target(), {}, drafter_config=drafter)

    def test_shallower_drafter_is_not_a_divergence(self):
        from coreai_models.export.bundle import _speculative_decoding_metadata

        # num_hidden_layers differs by design (drafter is shallow) — must NOT raise.
        drafter = self._drafter(num_hidden_layers=2)
        _speculative_decoding_metadata(self._target(), {}, drafter_config=drafter)

    def test_absent_param_on_one_side_is_not_a_divergence(self):
        from types import SimpleNamespace

        from coreai_models.export.bundle import _speculative_decoding_metadata

        # Target omits head_dim; comparison skips it rather than false-positiving.
        target = SimpleNamespace(vocab_size=202048, hidden_size=5760)
        _speculative_decoding_metadata(target, {}, drafter_config=self._drafter())


class TestRegistry:
    def test_fused_target_registered(self):
        from coreai_models.models.registry import _get_registry

        entry = _get_registry()["muse_glimmer_text"]
        assert entry.fused_target_class is MuseGlimmerForCausalLMWithDrafter

    def test_dflash_drafter_registered(self):
        from coreai_models.models.registry import _get_registry

        entry = _get_registry()["muse_glimmer_text"]
        assert entry.dflash_drafter_class is MuseGlimmerDFlashDrafterForCausalLM

    def test_export_output_names_order(self):
        assert MuseGlimmerForCausalLMWithDrafter.export_output_names() == {
            "main": ("logits", "drafter_features")
        }
