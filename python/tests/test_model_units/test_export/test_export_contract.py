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

        for model_type, entry in self._macos_entries():
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


class TestIOSHooksRaise:
    """iOS models must not inherit the macOS contract.

    ``export/ios.py`` never calls these hooks, but the defaults were reachable:
    ``--variant iOS --compression 4bit`` resolves a macOS quantization preset and the
    quantize branch is not variant-gated.
    """

    HOOKS = (
        "export_input_names",
        "export_state_names",
        "export_output_names",
        "build_reference_inputs",
        "build_dynamic_shapes",
    )

    @pytest.fixture
    def ios_model(self):
        from transformers import AutoConfig

        from coreai_models.models.ios.qwen3 import Qwen3ForCausalLMForiOS

        cfg = AutoConfig.for_model("qwen3")
        cfg.num_hidden_layers = 2
        cfg.max_position_embeddings = MAX_CONTEXT_LENGTH
        return Qwen3ForCausalLMForiOS(cfg, model_device="meta")

    def _call(self, model, hook):
        spec = TraceSpec(max_context_length=MAX_CONTEXT_LENGTH)
        args = {
            "build_reference_inputs": (model.config, torch.float16, spec),
            "build_dynamic_shapes": (model.config, spec),
        }.get(hook, ())
        return getattr(model, hook)(*args)

    @pytest.mark.parametrize("hook", HOOKS)
    def test_hook_raises_not_implemented(self, ios_model, hook) -> None:
        with pytest.raises(NotImplementedError) as exc:
            self._call(ios_model, hook)
        message = str(exc.value)
        assert "not yet implemented for iOS models" in message
        assert hook in message, "the error should name the hook that was called"
        assert type(ios_model).__name__ in message

    def test_every_base_hook_is_overridden(self) -> None:
        """Guard against a hook being added to the base and silently inherited."""
        from coreai_models.models.base import BaseForCausalLMForiOS

        for hook in self.HOOKS:
            assert hook in vars(BaseForCausalLMForiOS), (
                f"{hook} is not overridden on BaseForCausalLMForiOS; iOS models would "
                "inherit the macOS default"
            )

    def test_the_reachable_quantization_path_raises_clearly(self, ios_model) -> None:
        """``--variant iOS --compression 4bit`` reaches ``quantize_for_export``."""
        from coreai_models.export.compression import quantize_for_export
        from coreai_models.export.presets import get_preset

        quantization_config = get_preset("4bit").get("torch_quantization_config")
        assert quantization_config is not None
        with pytest.raises(NotImplementedError, match="not yet implemented for iOS models"):
            quantize_for_export(
                ios_model, ios_model.config, torch.float16, dict(quantization_config)
            )
