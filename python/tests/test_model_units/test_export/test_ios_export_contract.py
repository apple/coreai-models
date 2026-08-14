# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the iOS export contract.

iOS emits four entrypoints from three contract entries: the two transformer
entrypoints share their inputs, states and outputs, so ``export/ios.py`` uses the
``extend`` entry for both. These trace a tiny Qwen3 iOS model but stop before the Core
AI converter, so they need no runtime.
"""

import inspect

import pytest
import torch
from transformers import AutoConfig

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    GATHERED_EMBEDDINGS_OUTPUT_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    KEY_CACHE_OUTPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    LOAD_EMBEDDINGS_OUTPUT_NAME,
    OUTPUT_LOGITS_NAME,
    POSITION_IDS_INPUT_NAME,
    PROMPT_OPT_FUNCTION_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    VALUE_CACHE_OUTPUT_NAME,
)
from coreai_models.models.base import TraceSpec
from coreai_models.models.ios.qwen3 import Qwen3ForCausalLMForiOS

MAX_CTX = 256
CONTRACT_GRAPHS = (
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    EXTEND_FUNCTION_NAME,
)


@pytest.fixture
def config():
    cfg = AutoConfig.for_model("qwen3")
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.intermediate_size = 128
    cfg.vocab_size = 64
    cfg.max_position_embeddings = MAX_CTX
    return cfg


@pytest.fixture
def model(config):
    return Qwen3ForCausalLMForiOS(config, model_device="cpu").to(torch.float16).eval()


@pytest.fixture
def spec():
    return TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX)


@pytest.fixture
def built(model, config, spec):
    return (
        model.build_reference_inputs(config, torch.float16, spec),
        model.build_dynamic_shapes(config, spec),
    )


class TestIOSContract:
    def test_three_entries_not_four(self, model, built) -> None:
        """extend and prompt_opt are identical, so the contract carries one entry."""
        refs, shapes = built
        for hook in (
            model.export_input_names(),
            model.export_state_names(),
            model.export_output_names(),
            refs,
            shapes,
        ):
            assert tuple(hook) == CONTRACT_GRAPHS
        assert PROMPT_OPT_FUNCTION_NAME not in refs

    def test_validates(self, model, built) -> None:
        model.validate_export_contract(*built)

    def test_gather_renames_input_ids(self, model, built) -> None:
        """Graph names are not parameter names."""
        refs, _ = built
        assert list(refs[GATHER_EMBEDDINGS_FUNCTION_NAME]) == ["input_ids", "embedding_table"]
        assert model.export_input_names()[GATHER_EMBEDDINGS_FUNCTION_NAME] == (
            TOKEN_IDS_INPUT_NAME,
            EMBEDDING_TABLE_INPUT_NAME,
        )

    def test_load_embeddings_takes_nothing(self, model, built) -> None:
        refs, shapes = built
        assert refs[LOAD_EMBEDDINGS_FUNCTION_NAME] == {}
        assert shapes[LOAD_EMBEDDINGS_FUNCTION_NAME] == {}
        assert model.export_input_names()[LOAD_EMBEDDINGS_FUNCTION_NAME] == ()
        assert model.export_output_names()[LOAD_EMBEDDINGS_FUNCTION_NAME] == (
            LOAD_EMBEDDINGS_OUTPUT_NAME,
        )

    def test_only_the_transformer_has_state(self, model) -> None:
        states = model.export_state_names()
        assert states[LOAD_EMBEDDINGS_FUNCTION_NAME] == ()
        assert states[GATHER_EMBEDDINGS_FUNCTION_NAME] == ()
        assert states[EXTEND_FUNCTION_NAME] == (KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME)

    def test_state_outputs_are_named(self, model) -> None:
        """Hardware constraints attach to the mutated output as well as the input."""
        assert model.export_state_output_names()[EXTEND_FUNCTION_NAME] == (
            KEY_CACHE_OUTPUT_NAME,
            VALUE_CACHE_OUTPUT_NAME,
        )

    def test_reference_inputs_are_in_exact_signature_order(self, model, built) -> None:
        """They bind to the traced callable, so order is exact and interleaved."""
        refs, _ = built
        for graph, module in (
            (GATHER_EMBEDDINGS_FUNCTION_NAME, model.gather_embeddings),
            (EXTEND_FUNCTION_NAME, model.extend),
        ):
            params = list(inspect.signature(module.forward).parameters)
            keys = list(refs[graph])
            assert keys == params[: len(keys)], graph

    def test_names_carry_only_relative_order(self, model, built) -> None:
        """``embedding_table`` is an input declared after both caches in the signature."""
        refs, _ = built
        params = list(refs[EXTEND_FUNCTION_NAME])
        assert params.index("embedding_table") > params.index("value_cache")
        assert model.export_input_names()[EXTEND_FUNCTION_NAME] == (
            TRANSFORMER_INPUT_NAME,
            POSITION_IDS_INPUT_NAME,
            IN_STEP_INPUT_NAME,
            CAUSAL_MASK_INPUT_NAME,
            EMBEDDING_TABLE_INPUT_NAME,
        )

    def test_outputs(self, model) -> None:
        outputs = model.export_output_names()
        assert outputs[EXTEND_FUNCTION_NAME] == (OUTPUT_LOGITS_NAME,)
        assert outputs[GATHER_EMBEDDINGS_FUNCTION_NAME] == (GATHERED_EMBEDDINGS_OUTPUT_NAME,)


class TestIOSStaticShapeConfigs:
    """The static shape ladder each iOS graph is specialized over."""

    @pytest.fixture
    def shapes(self, model, config):
        return model.export_static_shape_configs(config, MAX_CTX)

    def test_keyed_like_the_name_hooks(self, shapes) -> None:
        assert tuple(shapes) == CONTRACT_GRAPHS

    def test_load_embeddings_needs_no_specialization(self, shapes) -> None:
        """It takes no inputs, so there is nothing to specialize over."""
        assert shapes[LOAD_EMBEDDINGS_FUNCTION_NAME] == {}

    def test_gather_is_specialized_over_query_length_alone(self, model, shapes) -> None:
        gather = shapes[GATHER_EMBEDDINGS_FUNCTION_NAME]
        assert tuple(gather) == tuple(f'"{q}"' for q in model.IOS_STATIC_QUERY_LENS)
        for q in model.IOS_STATIC_QUERY_LENS:
            assert gather[f'"{q}"'] == {TOKEN_IDS_INPUT_NAME: (1, q)}

    def test_transformer_ladder_doubles_the_cache_up_to_the_context(self, model, shapes) -> None:
        cache_lens, cl = [], model.IOS_STATIC_MIN_CACHE_LEN
        while cl <= MAX_CTX:
            cache_lens.append(cl)
            cl *= 2
        expected = {f'"{c}_{q}"' for c in cache_lens for q in model.IOS_STATIC_QUERY_LENS}
        assert set(shapes[EXTEND_FUNCTION_NAME]) == expected

    def test_transformer_shapes_agree_with_the_declared_inputs(self, config, shapes) -> None:
        """Every specialized input is one the contract declares as an input or state."""
        declared = set(Qwen3ForCausalLMForiOS.export_input_names()[EXTEND_FUNCTION_NAME]) | set(
            Qwen3ForCausalLMForiOS.export_state_names()[EXTEND_FUNCTION_NAME]
        )
        kv_embed = config.num_key_value_heads * config.head_dim
        for label, entry in shapes[EXTEND_FUNCTION_NAME].items():
            assert set(entry) <= declared, label
            cache_len, q_len = (int(n) for n in label.strip('"').split("_"))
            assert entry[TRANSFORMER_INPUT_NAME] == (1, q_len, 1, config.hidden_size)
            assert entry[POSITION_IDS_INPUT_NAME] == (1, q_len)
            assert entry[CAUSAL_MASK_INPUT_NAME] == (1, cache_len, 1, q_len)
            for name in (KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME):
                assert entry[name] == (config.num_hidden_layers, 1, kv_embed, 1, cache_len)

    def test_a_larger_context_adds_rungs(self, model, config) -> None:
        small = model.export_static_shape_configs(config, MAX_CTX)[EXTEND_FUNCTION_NAME]
        large = model.export_static_shape_configs(config, MAX_CTX * 4)[EXTEND_FUNCTION_NAME]
        assert set(small) < set(large)

    @pytest.mark.parametrize("model_type", ["qwen2", "mixtral"])
    def test_head_dim_is_derived_when_absent(self, model_type) -> None:
        """Configs without an explicit head_dim fall back to hidden_size // heads.

        Not testable on the qwen3 fixture: transformers >= 5 configs are strict
        dataclasses, and qwen3 types ``head_dim`` as ``int``, so assigning None
        raises. The fallback is still live for real checkpoints, in both of the
        shapes ``_head_dim`` handles -- qwen2 declares no ``head_dim`` field at
        all, mixtral declares one that defaults to None.
        """
        config = AutoConfig.for_model(model_type)
        config.num_hidden_layers = 2
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.max_position_embeddings = MAX_CTX
        assert not isinstance(getattr(config, "head_dim", None), int), "no head_dim to fall back on"

        shape_configs = Qwen3ForCausalLMForiOS.export_static_shape_configs(config, MAX_CTX)
        derived = config.hidden_size // config.num_attention_heads
        for entry in shape_configs[EXTEND_FUNCTION_NAME].values():
            assert entry[KEY_CACHE_INPUT_NAME][2] == config.num_key_value_heads * derived


class TestIOSHardwareConstraints:
    """Layout constraints on the buffers the runner shares with the compiled graphs."""

    @pytest.fixture
    def constraints(self, model):
        return model.export_hardware_constraints(MAX_CTX)

    def test_keyed_like_the_name_hooks(self, constraints) -> None:
        assert tuple(constraints) == CONTRACT_GRAPHS

    def test_every_graph_constrains_the_embedding_table(self, constraints) -> None:
        for graph in CONTRACT_GRAPHS:
            assert EMBEDDING_TABLE_INPUT_NAME in constraints[graph]

    def test_caches_are_constrained_on_input_and_output(self, model, constraints) -> None:
        """The mutated output needs the same layout as the input it aliases."""
        transformer = constraints[EXTEND_FUNCTION_NAME]
        for name in (
            *model.export_state_names()[EXTEND_FUNCTION_NAME],
            *model.export_state_output_names()[EXTEND_FUNCTION_NAME],
        ):
            assert name in transformer, name
        assert set(transformer) == {
            EMBEDDING_TABLE_INPUT_NAME,
            KEY_CACHE_INPUT_NAME,
            VALUE_CACHE_INPUT_NAME,
            KEY_CACHE_OUTPUT_NAME,
            VALUE_CACHE_OUTPUT_NAME,
        }

    def test_cache_alignment_scales_with_the_context(self, model) -> None:
        interleave = model.KV_CACHE_INTERLEAVE_FACTOR
        for max_ctx in (MAX_CTX, MAX_CTX * 4):
            cache = model.export_hardware_constraints(max_ctx)[EXTEND_FUNCTION_NAME][
                KEY_CACHE_INPUT_NAME
            ]
            assert list(cache.interleave) == [1, 1, interleave, 1, 1]
            assert list(cache.alignments) == [1, 1, 1, 1, interleave * max_ctx, 1]

    def test_embedding_table_constraints_do_not_depend_on_the_context(self, model) -> None:
        a = model.export_hardware_constraints(MAX_CTX)[LOAD_EMBEDDINGS_FUNCTION_NAME]
        b = model.export_hardware_constraints(MAX_CTX * 4)[LOAD_EMBEDDINGS_FUNCTION_NAME]
        for table in (a[EMBEDDING_TABLE_INPUT_NAME], b[EMBEDDING_TABLE_INPUT_NAME]):
            assert list(table.interleave) == [8, 1, 1]
            assert list(table.alignments) == [1, 1, 1, 1]


class TestIOSTracing:
    @pytest.fixture
    def programs(self, model, built):
        from coreai_models.export.ios import _export_programs

        return _export_programs(model, *built)

    def test_four_entrypoints_from_three_entries(self, programs) -> None:
        assert set(programs) == {
            LOAD_EMBEDDINGS_FUNCTION_NAME,
            GATHER_EMBEDDINGS_FUNCTION_NAME,
            EXTEND_FUNCTION_NAME,
            PROMPT_OPT_FUNCTION_NAME,
        }

    def test_declared_names_match_the_traced_signature(self, model, programs) -> None:
        inputs, states = model.export_input_names(), model.export_state_names()
        outputs = model.export_output_names()
        contract_for = {
            LOAD_EMBEDDINGS_FUNCTION_NAME: LOAD_EMBEDDINGS_FUNCTION_NAME,
            GATHER_EMBEDDINGS_FUNCTION_NAME: GATHER_EMBEDDINGS_FUNCTION_NAME,
            EXTEND_FUNCTION_NAME: EXTEND_FUNCTION_NAME,
            PROMPT_OPT_FUNCTION_NAME: EXTEND_FUNCTION_NAME,
        }
        for entrypoint, graph in contract_for.items():
            sig = programs[entrypoint].graph_signature
            declared = len(inputs[graph]) + len(states[graph])
            assert len(sig.user_inputs) == declared, entrypoint
            assert len(sig.user_outputs) == len(outputs[graph]), entrypoint

    def test_prefill_differs_between_the_two_transformer_programs(self, model, programs) -> None:
        """They share a contract entry but must not be the same program."""
        assert str(programs[EXTEND_FUNCTION_NAME].graph) != str(
            programs[PROMPT_OPT_FUNCTION_NAME].graph
        )
