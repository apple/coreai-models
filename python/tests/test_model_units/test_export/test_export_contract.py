# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the export contract hooks on ``BaseForCausalLM``.

Cover the graph contract independently of any real architecture, plus the
append-extras path. No hardware or HuggingFace weights required.
"""

from types import SimpleNamespace

import pytest
import torch
from typing_extensions import override

from coreai_models._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models._constants import (
    MAIN_GRAPH_NAME as MAIN,
)
from coreai_models.models.base import BaseForCausalLM, TraceSpec
from coreai_models.primitives.macos.cache import KVCache

MAX_CONTEXT_LENGTH = 8192


def _tiny_config() -> SimpleNamespace:
    """The smallest config the contract hooks read."""
    return SimpleNamespace(
        vocab_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        head_dim=8,
        max_position_embeddings=MAX_CONTEXT_LENGTH,
    )


class _StandardLM(BaseForCausalLM):
    """A model with the default contract: (input_ids, position_ids, k_cache, v_cache)."""

    @override
    def _init_model(self, config) -> None:
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @override
    def _mutate_state_dict(self, state_dict) -> None:
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("shape contract only; never traced in these tests")


class _ExtraStateLM(_StandardLM):
    """A model that appends state beyond the standard KV pair.

    The base KV pair first, then extra state args.
    """

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        extra_state: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError("shape contract only; never traced in these tests")

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN: (*super().export_state_names()[MAIN], "extraState")}

    @override
    def build_reference_inputs(self, config, target_dtype, spec):
        graphs = super().build_reference_inputs(config, target_dtype, spec)
        graphs[MAIN]["extra_state"] = torch.zeros(2, spec.cache_seq_len, dtype=target_dtype)
        return graphs

    @override
    def build_dynamic_shapes(self, config, spec):
        graphs = super().build_dynamic_shapes(config, spec)
        graphs[MAIN]["extra_state"] = None
        return graphs


@pytest.fixture
def config() -> SimpleNamespace:
    return _tiny_config()


@pytest.fixture
def model(config) -> _StandardLM:
    return _StandardLM(config)


class TestMacOSContract:
    """The macOS model describes one graph, keyed ``main``."""

    def _built(self, model, config, spec=None):
        spec = spec or TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        return (
            model.build_reference_inputs(config, torch.float16, spec),
            model.build_dynamic_shapes(config, spec),
        )

    def test_every_hook_describes_exactly_the_main_graph(self, model, config) -> None:
        refs, shapes = self._built(model, config)
        for hook in (
            model.export_input_names(),
            model.export_state_names(),
            model.export_output_names(),
            refs,
            shapes,
        ):
            assert list(hook) == [MAIN]

    def test_names(self, model) -> None:
        assert model.export_input_names()[MAIN] == ("input_ids", "position_ids")
        assert model.export_state_names()[MAIN] == (KEY_CACHE_NAME, VALUE_CACHE_NAME)
        assert model.export_output_names()[MAIN] == ("logits",)

    def test_reference_inputs_are_in_exact_signature_order(self, model, config) -> None:
        """They bind to the traced callable, so order is exact, not relative."""
        import inspect

        refs, _ = self._built(model, config)
        params = list(inspect.signature(model.forward).parameters)
        keys = list(refs[MAIN])
        assert keys == params[: len(keys)]

    def test_reference_input_shapes(self, model, config) -> None:
        refs, _ = self._built(model, config)
        graph = refs[MAIN]
        assert graph["input_ids"].shape == (1, QUANT_TRACE_QUERY_LEN)
        assert graph["input_ids"].dtype == torch.int32
        assert graph["position_ids"].shape == (1, QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET)
        expected = (2, 1, 2, TRACE_KV_CACHE_SEQ_LEN, 8)
        for name in ("k_cache", "v_cache"):
            assert graph[name].shape == expected
            assert graph[name].dtype == torch.float16

    def test_caches_traced_at_cache_seq_len(self, model, config) -> None:
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, cache_seq_len=512)
        refs, _ = self._built(model, config, spec)
        assert refs[MAIN]["k_cache"].shape[KVCache.seq_len_dim()] == 512

    def test_config_is_not_mutated(self, model, config) -> None:
        """Regression: sizing the cache used to mutate and restore the config."""
        self._built(
            model, config, TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, cache_seq_len=512)
        )
        assert config.max_position_embeddings == MAX_CONTEXT_LENGTH

    def test_dynamic_shape_bounds(self, model, config) -> None:
        _, shapes = self._built(model, config)
        graph = shapes[MAIN]
        assert graph["input_ids"][1].max == MAX_CONTEXT_LENGTH - 2
        assert graph["position_ids"][1].min == QUANT_TRACE_QUERY_LEN
        assert graph["position_ids"][1].max == MAX_CONTEXT_LENGTH - 1
        seq_dim = KVCache.seq_len_dim()
        for name in ("k_cache", "v_cache"):
            assert graph[name][seq_dim].min == TRACE_KV_CACHE_SEQ_LEN
            assert graph[name][seq_dim].max == MAX_CONTEXT_LENGTH


class TestSmallContext:
    """Contexts at or below the default trace cache length.

    Regression: the cache dim was built as ``Dim(min=TRACE_KV_CACHE_SEQ_LEN,
    max=max_context_length)`` unconditionally, which raises from inside ``torch.export``
    whenever the context is <= the trace length.
    """

    def test_cache_seq_len_may_equal_the_context(self) -> None:
        assert TraceSpec(max_context_length=512, cache_seq_len=512).cache_seq_len == 512

    def test_cache_seq_len_above_the_context_is_rejected(self) -> None:
        # A cache longer than the context it serves is meaningless, so the spec
        # rejects it rather than quietly shrinking it.
        with pytest.raises(ValueError, match="must not be greater than"):
            TraceSpec(max_context_length=512, cache_seq_len=513)

    def test_larger_context_leaves_trace_length_alone(self) -> None:
        assert TraceSpec(max_context_length=8192).cache_seq_len == TRACE_KV_CACHE_SEQ_LEN

    def test_context_too_small_to_trace_is_rejected(self) -> None:
        limit = TraceSpec(max_context_length=8192).query_len + 2
        TraceSpec(max_context_length=limit, cache_seq_len=limit)
        for bad in (limit - 1, 2, 0, -5):
            with pytest.raises(ValueError, match="too small to trace"):
                TraceSpec(max_context_length=bad)

    @pytest.mark.parametrize("max_ctx", [512, TRACE_KV_CACHE_SEQ_LEN])
    def test_cache_dims_pin_instead_of_raising(self, model, max_ctx) -> None:
        config = _tiny_config()
        config.max_position_embeddings = max_ctx
        spec = TraceSpec(
            max_context_length=max_ctx, cache_seq_len=min(TRACE_KV_CACHE_SEQ_LEN, max_ctx)
        )
        assert spec.caches_are_static
        shapes = model.build_dynamic_shapes(config, spec)[MAIN]
        assert shapes["k_cache"] is None
        assert shapes["v_cache"] is None

    def test_dims_stay_dynamic_when_there_is_room(self, model, config) -> None:
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        assert not spec.caches_are_static
        assert model.build_dynamic_shapes(config, spec)[MAIN]["k_cache"] is not None


class TestValidateExportContract:
    """Cross-checks the five hooks against each other."""

    def _built(self, model, config):
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        return (
            model.build_reference_inputs(config, torch.float16, spec),
            model.build_dynamic_shapes(config, spec),
        )

    def test_accepts_the_default_contract(self, model, config) -> None:
        model.validate_export_contract(*self._built(model, config))

    def test_accepts_appended_state(self, config) -> None:
        m = _ExtraStateLM(config)
        m.validate_export_contract(*self._built(m, config))

    def test_rejects_a_hook_covering_different_graphs(self, model, config) -> None:
        refs, shapes = self._built(model, config)
        refs["extra_graph"] = {}
        with pytest.raises(ValueError, match="must describe the same graphs"):
            model.validate_export_contract(refs, shapes)

    def test_rejects_a_name_count_mismatch(self, config) -> None:
        class _TooFewNames(_StandardLM):
            @classmethod
            @override
            def export_state_names(cls) -> dict[str, tuple[str, ...]]:
                return {MAIN: (KEY_CACHE_NAME,)}

        m = _TooFewNames(config)
        with pytest.raises(ValueError, match="build_reference_inputs supplies"):
            m.validate_export_contract(*self._built(m, config))

    def test_rejects_dynamic_shape_key_mismatch(self, model, config) -> None:
        refs, shapes = self._built(model, config)
        del shapes[MAIN]["position_ids"]
        with pytest.raises(ValueError, match="do not match reference inputs"):
            model.validate_export_contract(refs, shapes)


class TestReferenceInputsAsArgs:
    """Positional conversion for the quantizer, which takes a tuple not kwargs."""

    def _graph(self, model, config):
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        return model.build_reference_inputs(config, torch.float16, spec)[MAIN]

    def test_returns_signature_order(self, model, config) -> None:
        graph = self._graph(model, config)
        args = model.reference_inputs_as_args(graph)
        assert len(args) == 4
        for arg, expected in zip(args, graph.values(), strict=True):
            assert arg is expected

    def test_rejects_reordered_keys(self, model, config) -> None:
        graph = self._graph(model, config)
        swapped = {"position_ids": graph["position_ids"], "input_ids": graph["input_ids"]}
        swapped.update({k: graph[k] for k in ("k_cache", "v_cache")})
        with pytest.raises(ValueError, match="not a contiguous in-order prefix"):
            model.reference_inputs_as_args(swapped)

    def test_sees_through_the_logits_cast_decorator(self, config) -> None:
        """Every real subclass decorates forward; introspection needs functools.wraps."""

        class _Decorated(_StandardLM):
            @BaseForCausalLM.cast_logits_bfloat16_to_float16
            @override
            def forward(self, input_ids, position_ids, k_cache, v_cache):
                raise NotImplementedError

        m = _Decorated(config)
        assert len(m.reference_inputs_as_args(self._graph(m, config))) == 4


class _MaskLM(_StandardLM):
    """Default contract plus an opt-in additive ``attn_mask`` input.

    ``forward`` takes ``attn_mask`` after the KV pair; the extra graph input only
    appears when the model is built with ``use_attention_mask=True``.
    """

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError("shape contract only; never traced in these tests")


class TestMaskContract:
    """The opt-in additive ``attn_mask`` graph input, gated on ``use_attention_mask``."""

    def _built(self, model, config, spec=None):
        spec = spec or TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        return (
            model.build_reference_inputs(config, torch.float16, spec),
            model.build_dynamic_shapes(config, spec),
        )

    def test_mask_mode_declares_attn_mask_input(self, config) -> None:
        m = _MaskLM(config, use_attention_mask=True)
        assert m.export_input_names()[MAIN] == ("input_ids", "position_ids", "attn_mask")

    def test_mask_mode_appends_attn_mask_last(self, config) -> None:
        m = _MaskLM(config, use_attention_mask=True)
        refs, _ = self._built(m, config)
        assert list(refs[MAIN]) == [
            "input_ids",
            "position_ids",
            "k_cache",
            "v_cache",
            "attn_mask",
        ]
        mask = refs[MAIN]["attn_mask"]
        # Scores are [B, heads, query_len, key_len]; the key length is the fetched
        # cache prefix, which equals the position_ids length (query_len + offset).
        assert mask.shape == (
            1,
            1,
            QUANT_TRACE_QUERY_LEN,
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET,
        )
        assert mask.dtype == torch.float16

    def test_mask_mode_dynamic_shapes_reuse_dims(self, config) -> None:
        m = _MaskLM(config, use_attention_mask=True)
        _, shapes = self._built(m, config)
        graph = shapes[MAIN]
        assert "attn_mask" in graph
        # Query axis is the same dim as input_ids; key axis is the attention key
        # length, which is tied to position_ids (not the k_cache tensor's seq dim).
        assert graph["attn_mask"][2] is graph["input_ids"][1]
        assert graph["attn_mask"][3] is graph["position_ids"][1]

    def test_mask_mode_validates_and_converts_to_args(self, config) -> None:
        m = _MaskLM(config, use_attention_mask=True)
        refs, shapes = self._built(m, config)
        m.validate_export_contract(refs, shapes)
        args = m.reference_inputs_as_args(refs[MAIN])
        assert len(args) == 5

    def test_mask_mode_static_cache_still_ties_key_to_position_ids(self, config) -> None:
        config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        spec = TraceSpec(
            max_context_length=TRACE_KV_CACHE_SEQ_LEN, cache_seq_len=TRACE_KV_CACHE_SEQ_LEN
        )
        assert spec.caches_are_static
        m = _MaskLM(config, use_attention_mask=True)
        refs, shapes = self._built(m, config, spec)
        # Caches are pinned, but position_ids stays dynamic, so the mask's key axis
        # does too -- it tracks the attention key length, not the cache tensor.
        assert shapes[MAIN]["k_cache"] is None
        assert shapes[MAIN]["attn_mask"] == {
            2: shapes[MAIN]["input_ids"][1],
            3: shapes[MAIN]["position_ids"][1],
        }
        m.validate_export_contract(refs, shapes)

    def test_flag_off_leaves_the_contract_unchanged(self, config) -> None:
        m = _MaskLM(config)  # flag defaults off
        refs, shapes = self._built(m, config)
        assert m.export_input_names()[MAIN] == ("input_ids", "position_ids")
        assert "attn_mask" not in refs[MAIN]
        assert "attn_mask" not in shapes[MAIN]
        m.validate_export_contract(refs, shapes)

    def test_batched_contract_traces_at_batch_size_with_dim0_pinned(self, config) -> None:
        """batch_size>1 traces reference tensors at B with dim0/cache-dim1 PINNED (static batch)."""
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, batch_size=2)
        m = _MaskLM(config, use_attention_mask=True)
        refs = m.build_reference_inputs(config, torch.float16, spec)[MAIN]
        shapes = m.build_dynamic_shapes(config, spec)[MAIN]

        # Reference tensors carry the fixed batch on dim0 (ids/pos/mask) and cache dim1.
        assert refs["input_ids"].shape[0] == 2
        assert refs["position_ids"].shape[0] == 2
        assert refs["attn_mask"].shape[0] == 2
        assert refs["k_cache"].shape[1] == 2  # cache batch dim is index 1
        assert refs["v_cache"].shape[1] == 2

        # The batch dim is PINNED (static), not a dynamic Dim: dim0 / cache-dim1 absent from shapes.
        assert shapes["input_ids"].get(0) is None
        assert shapes["position_ids"].get(0) is None
        assert shapes["attn_mask"].get(0) is None
        assert (shapes["k_cache"] or {}).get(1) is None
        assert (shapes["v_cache"] or {}).get(1) is None
        m.validate_export_contract({MAIN: refs}, {MAIN: shapes})

    def test_batch_size_one_traces_single_batch(self, config) -> None:
        """Default batch_size=1 keeps the validated single-batch contract untouched."""
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)  # batch_size defaults to 1
        m = _MaskLM(config, use_attention_mask=True)
        refs = m.build_reference_inputs(config, torch.float16, spec)[MAIN]
        shapes = m.build_dynamic_shapes(config, spec)[MAIN]
        assert refs["input_ids"].shape[0] == 1
        assert shapes["input_ids"].get(0) is None  # dim0 pinned, not dynamic
        assert shapes["k_cache"].get(1) is None  # cache batch dim pinned

    def test_dynamic_batch_declares_one_shared_batch_dim(self, config) -> None:
        """max_batch_size declares ONE dynamic ``batch`` Dim on dim0 of ids/pos/mask and
        dim1 of the caches, while reference tensors still trace at the smaller batch_size
        (torch.export rejects a dim whose only legal value is the trace size)."""
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, batch_size=2, max_batch_size=8)
        m = _MaskLM(config, use_attention_mask=True)
        refs = m.build_reference_inputs(config, torch.float16, spec)[MAIN]
        shapes = m.build_dynamic_shapes(config, spec)[MAIN]

        # Reference tensors still trace at batch_size (the trace size), NOT the max.
        assert refs["input_ids"].shape[0] == 2
        assert refs["k_cache"].shape[1] == 2  # cache batch dim is index 1

        batch_dim = shapes["input_ids"][0]
        assert batch_dim.min == 1
        assert batch_dim.max == 8
        # The SAME Dim object on every batch axis, so torch.export ties them together.
        assert shapes["position_ids"][0] is batch_dim
        assert shapes["attn_mask"][0] is batch_dim
        assert shapes["k_cache"][1] is batch_dim
        assert shapes["v_cache"][1] is batch_dim
        m.validate_export_contract({MAIN: refs}, {MAIN: shapes})

    def test_dynamic_batch_static_cache_still_declares_batch_dim(self, config) -> None:
        """With seq-static caches, dim1 still carries the batch Dim (not pinned to None)."""
        config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        spec = TraceSpec(
            max_context_length=TRACE_KV_CACHE_SEQ_LEN,
            cache_seq_len=TRACE_KV_CACHE_SEQ_LEN,
            batch_size=2,
            max_batch_size=8,
        )
        assert spec.caches_are_static
        m = _MaskLM(config, use_attention_mask=True)
        refs = m.build_reference_inputs(config, torch.float16, spec)[MAIN]
        shapes = m.build_dynamic_shapes(config, spec)[MAIN]
        batch_dim = shapes["input_ids"][0]
        # Seq dim pinned, batch dim dynamic: dim1 carries the batch Dim, seq axis absent.
        assert shapes["k_cache"] == {1: batch_dim}
        assert shapes["v_cache"] == {1: batch_dim}
        m.validate_export_contract({MAIN: refs}, {MAIN: shapes})

    def test_dynamic_batch_without_mask_still_declares_batch_dim(self, config) -> None:
        """Batching is independent of the attn_mask flag: ids/pos/caches get the batch Dim
        even with the mask off (there is simply no attn_mask axis to tie)."""
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, batch_size=2, max_batch_size=8)
        m = _MaskLM(config)  # mask flag off
        shapes = m.build_dynamic_shapes(config, spec)[MAIN]
        batch_dim = shapes["input_ids"][0]
        assert batch_dim.min == 1 and batch_dim.max == 8
        assert shapes["position_ids"][0] is batch_dim
        assert shapes["k_cache"][1] is batch_dim
        assert "attn_mask" not in shapes

    def test_max_batch_size_must_exceed_trace_batch(self) -> None:
        """Trace size must be strictly below the max, and >= 2, or torch.export specializes it."""
        with pytest.raises(ValueError, match="max_batch_size"):
            TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, batch_size=2, max_batch_size=2)
        with pytest.raises(ValueError, match="max_batch_size"):
            TraceSpec(max_context_length=MAX_CONTEXT_LENGTH, batch_size=1, max_batch_size=8)


class TestRegisteredModelsSatisfyTheContract:
    """Pins the real registry, so renaming a forward parameter fails here."""

    @staticmethod
    def _macos_entries():
        from coreai_models.models.registry import _get_registry

        return sorted(
            ((mt, e) for mt, e in _get_registry().items() if e.macos_class is not None),
            key=lambda kv: kv[0],
        )

    def test_registry_is_not_empty(self) -> None:
        assert self._macos_entries()

    def test_every_registered_model_validates(self) -> None:
        import inspect

        from transformers import AutoConfig

        # DiffusionGemma is a two-graph (encoder + canvas decoder) block-diffusion
        # export whose MoE/cache config is not fully specified by transformers'
        # *default* diffusion_gemma_text config (moe_intermediate_size, num_experts,
        # and num_global_key_value_heads default to None); it needs a
        # checkpoint-derived config, so this stock-default contract check does not
        # apply. Its export contract is covered by the DiffusionGemma model tests.
        contract_skip = {"diffusion_gemma_text"}

        for model_type, entry in self._macos_entries():
            if model_type in contract_skip:
                continue
            raw = AutoConfig.for_model(model_type)
            cfg = (
                getattr(raw, entry.hf_config_attr)
                if entry.hf_config_attr and hasattr(raw, entry.hf_config_attr)
                else raw
            )
            cfg.num_hidden_layers = 2
            cfg.max_position_embeddings = MAX_CONTEXT_LENGTH
            m = entry.macos_class(cfg, model_device="meta")

            spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
            refs = m.build_reference_inputs(cfg, torch.float16, spec)
            shapes = m.build_dynamic_shapes(cfg, spec)
            m.validate_export_contract(refs, shapes)

            params = list(inspect.signature(m.forward).parameters)
            keys = list(refs[MAIN])
            assert keys == params[: len(keys)], model_type
