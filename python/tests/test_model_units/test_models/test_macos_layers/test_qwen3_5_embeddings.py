# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Qwen3.5 inputs_embeds decoder against the token-id decoder it is built on.

Both load one Hugging Face state dict, and the embeddings decoder is fed
``embed_tokens(input_ids)``. Fed that, it has to be the token-id decoder exactly: the same
logits, the same four states after every call, the same prefill entrypoint. Small random
configs only; nothing is downloaded and nothing is converted.
"""

import coreai_torch
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
)

from coreai_models._constants import MAIN_GRAPH_NAME
from coreai_models.export.mlir_ops import remove_functionalization
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM
from coreai_models.models.macos.qwen3_5_embeddings import Qwen3_5ForCausalLMEmbeddings

MAX_CTX = 64
STATE_ARGS = ("k_cache", "v_cache", "conv_states", "recurrent_states")
# Two linear-attention layers around each full-attention one, so every state has more
# than one row and both layer kinds sit in the middle of the stack.
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]


def make_config(tie_word_embeddings: bool) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        vocab_size=128,
        max_position_embeddings=MAX_CTX,
        tie_word_embeddings=tie_word_embeddings,
    )
    config.layer_types = list(LAYER_TYPES)
    return config


def load_pair(config: Qwen3_5TextConfig) -> tuple[Qwen3_5ForCausalLM, Qwen3_5ForCausalLMEmbeddings]:
    """Both decoders, loaded from one Hugging Face state dict the production way."""
    torch.manual_seed(0)
    hf_model = HFQwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            # Norms initialize to a constant, and the default decay empties the recurrent
            # state within a token or two; random values and a slow decay keep every
            # weight, and the state carried between calls, visible in the logits.
            if name.endswith("A_log"):
                param.uniform_(-4, -1)
            else:
                param.normal_(0.0, 0.1)
    hf_state = hf_model.state_dict()

    models = []
    for cls in (Qwen3_5ForCausalLM, Qwen3_5ForCausalLMEmbeddings):
        model = cls(config, model_device="cpu").to(torch.float32).eval()
        state_dict = {k: v.clone() for k, v in hf_state.items()}
        model._mutate_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        models.append(model)
    return models[0], models[1]


def fresh_states(config: Qwen3_5TextConfig) -> dict[str, torch.Tensor]:
    k_cache, v_cache = Qwen3_5ForCausalLM.create_kv_cache_tensors(config, seq_len=MAX_CTX)
    conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(config)
    return dict(zip(STATE_ARGS, (k_cache, v_cache, conv_states, recurrent_states), strict=True))


def run(model, first_input: torch.Tensor, offset: int, states: dict, prefill: bool = False):
    """One call; ``states`` is written in place, so calls chain across a prompt."""
    query_len = first_input.shape[1]
    position_ids = torch.arange(offset + query_len, dtype=torch.int32).unsqueeze(0)
    model.set_prefill_mode(prefill)
    try:
        with torch.no_grad():
            return model(first_input, position_ids, **states)
    finally:
        model.set_prefill_mode(False)


def embed(model: Qwen3_5ForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.model.embed_tokens(input_ids)


def assert_same_states(got: dict, expected: dict) -> None:
    for name in STATE_ARGS:
        # Written, so two zeroed tensors cannot pass.
        assert torch.any(expected[name] != 0), f"{name} was never written"
        torch.testing.assert_close(got[name], expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("tie_word_embeddings", [True, False], ids=["tied", "untied"])
def test_matches_the_token_decoder_across_calls(tie_word_embeddings) -> None:
    """A prompt in two chunks and then two decode steps, each call compared on logits
    and on all four states, so the offset and every carried state are exercised."""
    config = make_config(tie_word_embeddings)
    tokens, embeddings = load_pair(config)
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (1, 12), dtype=torch.int32)

    token_states, embeds_states = fresh_states(config), fresh_states(config)
    for start, end in ((0, 5), (5, 10), (10, 11), (11, 12)):
        ids = input_ids[:, start:end]
        expected = run(tokens, ids, start, token_states)
        got = run(embeddings, embed(embeddings, ids), start, embeds_states)

        torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=f"logits [{start}:{end}]")
        assert_same_states(embeds_states, token_states)


def test_prefill_mode_writes_the_token_decoders_states() -> None:
    """The prefill entrypoint declares no outputs, so the states are all it produces."""
    config = make_config(tie_word_embeddings=True)
    tokens, embeddings = load_pair(config)
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.int32)

    token_states, embeds_states = fresh_states(config), fresh_states(config)
    assert run(tokens, input_ids, 0, token_states, prefill=True) == ()
    assert run(embeddings, embed(embeddings, input_ids), 0, embeds_states, prefill=True) == ()
    assert embeddings.prefill_mode is False
    assert_same_states(embeds_states, token_states)


class TestExportContract:
    @pytest.fixture
    def config(self) -> Qwen3_5TextConfig:
        return make_config(tie_word_embeddings=True)

    @pytest.fixture
    def spec(self) -> TraceSpec:
        return TraceSpec(max_context_length=MAX_CTX, cache_seq_len=32)

    def test_names(self) -> None:
        assert Qwen3_5ForCausalLMEmbeddings.export_input_names() == {
            MAIN_GRAPH_NAME: ("inputs_embeds", "position_ids")
        }
        for hook in ("export_state_names", "export_output_names"):
            assert (
                getattr(Qwen3_5ForCausalLMEmbeddings, hook)() == getattr(Qwen3_5ForCausalLM, hook)()
            )
        assert Qwen3_5ForCausalLMEmbeddings.exports_prefill_graph

    def test_reference_inputs_replace_the_token_ids(self, config, spec) -> None:
        model = Qwen3_5ForCausalLMEmbeddings(config, model_device="meta")
        token_model = Qwen3_5ForCausalLM(config, model_device="meta")
        refs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
        token_refs = token_model.build_reference_inputs(config, torch.float16, spec)[
            MAIN_GRAPH_NAME
        ]

        # Positional conversion for the quantizer reads them in this order.
        model.reference_inputs_as_args(refs)
        embeds = refs["inputs_embeds"]
        assert embeds.shape == (1, spec.query_len, config.hidden_size)
        assert embeds.dtype == torch.float16
        assert list(refs)[1:] == list(token_refs)[1:]
        for name in ("position_ids", *STATE_ARGS):
            assert refs[name].shape == token_refs[name].shape, name
            assert refs[name].dtype == token_refs[name].dtype, name

    def test_dynamic_shapes_move_the_sequence_dim_to_the_embeddings(self, config, spec) -> None:
        model = Qwen3_5ForCausalLMEmbeddings(config, model_device="meta")
        token_model = Qwen3_5ForCausalLM(config, model_device="meta")
        shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
        token_shapes = token_model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]

        assert list(shapes) == ["inputs_embeds", *list(token_shapes)[1:]]
        assert set(shapes["inputs_embeds"]) == {1}
        assert str(shapes["inputs_embeds"][1]) == str(token_shapes["input_ids"][1])
        model.validate_export_contract(
            model.build_reference_inputs(config, torch.float16, spec),
            model.build_dynamic_shapes(config, spec),
        )

    @pytest.mark.parametrize("prefill", [False, True], ids=["main", "prefill"])
    def test_traced_graph_writes_every_state(
        self, config, spec, prefill, disable_hf_impl_for_coreai
    ) -> None:
        """The graph the converter is handed: decomposed, then defunctionalized, the way
        ``export_to_coreai`` stages it."""
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLMEmbeddings(config, model_device="cpu").to(torch.float16).eval()
        refs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
        shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]

        model.set_prefill_mode(prefill)
        try:
            with torch.no_grad():
                program = torch.export.export(model, (), kwargs=refs, dynamic_shapes=shapes)
        finally:
            model.set_prefill_mode(False)
        program = program.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(program)

        mutated = set(program.graph_signature.user_inputs_to_mutate.values())
        assert mutated == set(STATE_ARGS)
        assert program.graph_signature.user_inputs[:2] == ("inputs_embeds", "position_ids")
