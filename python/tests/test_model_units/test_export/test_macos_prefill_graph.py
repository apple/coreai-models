# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the optional ``prefill`` entrypoint on macOS exports.

A model that sets ``exports_prefill_graph`` gets a second entrypoint from the same
signature: traced again with prefill mode on when the model is eager, or the one decode
trace staged again with its non-state outputs trimmed when it arrives flattened. Either
way it must declare no outputs and bind exactly the inputs and states ``main`` does,
because that is the contract the Swift runner validates in ``loadPrefillGraph``. Beyond
that shape contract, the KV cache it writes is its only product, so the tests here
execute both flavours and check that cache numerically against eager torch.

Exports a tiny randomly-initialised model, so no HuggingFace weights are needed, but
it does run a real conversion and therefore needs ``coreai-torch``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coreai_models._constants import (
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    PREFILL_GRAPH_NAME,
    VALUE_CACHE_NAME,
)

try:
    import coreai.runtime as rt
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from coreai_models.export.externalize import patch_model_for_externalization
    from coreai_models.export.macos import export_macos_model
    from coreai_models.models.base import TraceSpec
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
    from coreai_models.primitives.macos.cache import KVCache

    # Imported here rather than at module scope: `testing_utils` imports
    # `coreai_torch` unguarded, so a top-level import would break collection in
    # environments without the toolchain, which the skip below exists to tolerate.
    from tests._runner_infra.testing_utils import assert_close

    HAS_COREAI = True
except ImportError:  # pragma: no cover - depends on the installed toolchain
    HAS_COREAI = False

MAX_CTX = 256

# Prompt length for the parity test. Longer than one token so prefill is doing real
# multi-token work, and >= the traced `position_ids` minimum of QUANT_TRACE_QUERY_LEN.
PREFILL_LEN = 32

pytestmark = pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")


def _tiny_config() -> Qwen3Config:
    """Smallest Qwen3 that still exercises attention, an MLP and a KV cache."""
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=MAX_CTX,
        tie_word_embeddings=True,
    )


def _export(*, opts_in: bool) -> tuple[torch.nn.Module, object]:
    """Export a tiny random model, opting the prefill graph in or out.

    Subclassing to flip the flag keeps the real ``forward`` -- including its
    ``prefill_mode`` guard -- so the opt-out case proves the flag alone decides,
    not the presence of the guard.
    """

    class _Model(Qwen3ForCausalLM):
        exports_prefill_graph = opts_in

    config = _tiny_config()
    # Seeded so the parity test's error margin is reproducible run to run.
    torch.manual_seed(0)
    model = _Model(config).to(torch.float16).eval()
    # `compute_precision` has to agree with the dtype above: the exporter resolves the
    # trace dtype from it, not from the model's parameters.
    export_config = SimpleNamespace(max_context_length=MAX_CTX, compute_precision="float16")
    program = export_macos_model(model, config, export_config)
    return model, program


async def _function_descriptors(program) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Save the program and read back one descriptor per entrypoint.

    Loads through ``asset.executable()`` rather than ``AIModel.load()`` directly: that
    scopes the model's resources to this block, so they are released before the temp
    directory backing them is removed instead of at some later, GC-determined point.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.aimodel"
        asset = program.save_asset(path, rt.AIModelAssetMetadata())
        async with asset.executable() as model:
            return {name: model.load_function(name).desc for name in model.function_names}


def _prefill_inputs(config: Qwen3Config) -> tuple[torch.Tensor, torch.Tensor]:
    """One whole prompt: ``position_ids`` is as long as ``input_ids``, so offset 0."""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        1, config.vocab_size, (1, PREFILL_LEN), dtype=torch.int32, generator=generator
    )
    position_ids = torch.arange(PREFILL_LEN, dtype=torch.int32).unsqueeze(0)
    return input_ids, position_ids


def _zeroed_caches() -> tuple[torch.Tensor, torch.Tensor]:
    """A fresh, empty cache at the exported context length."""
    return KVCache.create_cache_tensors(_tiny_config(), dtype=torch.float16, seq_len=MAX_CTX)


async def _run_entrypoint(
    program,  # type: ignore[no-untyped-def]
    name: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one entrypoint over an empty cache and return the state it wrote.

    The KV cache is the prefill graph's only product -- it declares no outputs -- so
    the state arrays are what there is to compare. They're copied out before the asset
    is torn down, since they may alias memory it owns.
    """
    k_cache, v_cache = _zeroed_caches()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.aimodel"
        asset = program.save_asset(path, rt.AIModelAssetMetadata())
        async with asset.executable() as model:
            function = model.load_function(name)
            state = {
                KEY_CACHE_NAME: rt.NDArray(data=k_cache),
                VALUE_CACHE_NAME: rt.NDArray(data=v_cache),
            }
            await function(
                {
                    "input_ids": rt.NDArray(data=input_ids.contiguous()),
                    "position_ids": rt.NDArray(data=position_ids.contiguous()),
                },
                state=state,
            )
            return (
                state[KEY_CACHE_NAME].numpy().copy(),
                state[VALUE_CACHE_NAME].numpy().copy(),
            )


def _torch_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager reference: run ``forward`` in prefill mode and return the caches it filled.

    This is the same authoring the prefill graph is traced from, in prefill mode, which
    is the only reference available: HuggingFace has no prefill-only forward to compare
    against. So this pins the conversion, not the authoring.
    """
    k_cache, v_cache = _zeroed_caches()
    model.set_prefill_mode(True)
    try:
        with torch.no_grad():
            assert model(input_ids, position_ids, k_cache, v_cache) == ()
    finally:
        model.set_prefill_mode(False)
    return k_cache, v_cache


def test_prefill_graph_is_emitted_when_opted_in() -> None:
    model, program = _export(opts_in=True)
    descs = asyncio.run(_function_descriptors(program))

    assert set(descs) == {MAIN_GRAPH_NAME, PREFILL_GRAPH_NAME}

    prefill, main = descs[PREFILL_GRAPH_NAME], descs[MAIN_GRAPH_NAME]

    # No outputs at all: the KV cache writes are the graph's only product. The runner
    # rejects a prefill graph that declares any, so this is the load-bearing assertion.
    assert list(prefill.output_names) == []
    assert list(main.output_names) == ["logits"]

    # Same bindings as `main`, by name -- the runner feeds both from one code path.
    assert list(prefill.input_names) == list(main.input_names)
    assert set(prefill.state_names) == set(main.state_names)
    assert set(prefill.state_names) == {KEY_CACHE_NAME, VALUE_CACHE_NAME}

    # The exporter must not leave a shared model in prefill mode.
    assert model.prefill_mode is False


def test_prefill_graph_is_absent_when_opted_out() -> None:
    model, program = _export(opts_in=False)
    descs = asyncio.run(_function_descriptors(program))

    assert set(descs) == {MAIN_GRAPH_NAME}
    assert PREFILL_GRAPH_NAME not in descs
    assert list(descs[MAIN_GRAPH_NAME].output_names) == ["logits"]
    assert model.prefill_mode is False


def _export_flattened() -> tuple[torch.nn.Module, object]:
    """Export the way graph-mode quantization does: mark, capture, hand over the graph.

    Returns the eager model -- still the export contract's source of truth, and the
    reference for what prefill should write -- alongside the program.
    """
    config = _tiny_config()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(torch.float16).eval()
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX)
    reference_inputs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
    dynamic_shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
    # Marking before capture is what graph-mode quantization does, and it is what leaves
    # the composite call sites in the flattened graph for the exporter to externalize.
    patch_model_for_externalization(model)
    with torch.no_grad():
        flattened = torch.export.export(
            model, args=tuple(reference_inputs.values()), dynamic_shapes=dynamic_shapes
        ).module()

    assert model.exports_prefill_graph
    export_config = SimpleNamespace(max_context_length=MAX_CTX, compute_precision="float16")
    return model, export_macos_model(flattened, config, export_config, externalized_model=model)


def test_flattened_model_gets_a_trimmed_prefill_graph() -> None:
    """Graph-mode quantization hands the exporter a flattened module, which resolved
    ``prefill_mode`` when it was captured, so there is no second trace to take. The
    exporter stages the one decode trace again with its non-state outputs trimmed, which
    leaves the LM head dead. The result has to satisfy the same runner contract as the
    twice-traced eager one.
    """
    _, program = _export_flattened()

    descs = asyncio.run(_function_descriptors(program))
    assert set(descs) == {MAIN_GRAPH_NAME, PREFILL_GRAPH_NAME}

    prefill, main = descs[PREFILL_GRAPH_NAME], descs[MAIN_GRAPH_NAME]
    assert list(prefill.output_names) == []
    assert list(main.output_names) == ["logits"]
    assert list(prefill.input_names) == list(main.input_names)
    assert set(prefill.state_names) == set(main.state_names) == {KEY_CACHE_NAME, VALUE_CACHE_NAME}


def test_flattened_prefill_graph_kv_writes_match_torch() -> None:
    """Trimming the outputs must not take the cache writes with it.

    Same check as the eager path's, and the same eager reference: the graph declares no
    outputs, so the cache is the only place a mistrimmed graph would show up.
    """
    model, program = _export_flattened()
    input_ids, position_ids = _prefill_inputs(_tiny_config())

    k_coreai, v_coreai = asyncio.run(
        _run_entrypoint(program, PREFILL_GRAPH_NAME, input_ids, position_ids)
    )
    k_torch, v_torch = _torch_prefill(model, input_ids, position_ids)

    written = (slice(None), slice(None), slice(None), slice(0, PREFILL_LEN))
    assert torch.any(k_torch[written] != 0)
    assert_close(k_coreai.astype(np.float32), k_torch.float(), atol=1e-2, rtol=1e-2)
    assert_close(v_coreai.astype(np.float32), v_torch.float(), atol=1e-2, rtol=1e-2)


def test_prefill_graph_kv_writes_match_torch() -> None:
    """The converted prefill graph fills the cache the way eager prefill does.

    The shape assertions above would pass just as well for a prefill graph that dropped
    its cache writes along with the LM head, or wrote them at the wrong offsets, since
    the graph has no outputs to be wrong. This is what catches that.

    The reference is our own authoring in prefill mode, not HuggingFace: HF has no
    prefill-only forward to compare against. So this pins the conversion, not the
    authoring, which is what the rest of ``test_models/`` covers.
    """
    model, program = _export(opts_in=True)
    input_ids, position_ids = _prefill_inputs(_tiny_config())

    k_coreai, v_coreai = asyncio.run(
        _run_entrypoint(program, PREFILL_GRAPH_NAME, input_ids, position_ids)
    )
    k_torch, v_torch = _torch_prefill(model, input_ids, position_ids)

    # Not vacuous: the prompt's positions really were written, so an all-zeros cache on
    # both sides can't pass. Positions past the prompt stay zero, and comparing the full
    # tensor keeps a graph that scribbles beyond its range from passing either.
    written = (slice(None), slice(None), slice(None), slice(0, PREFILL_LEN))
    assert torch.any(k_torch[written] != 0)
    assert torch.any(v_torch[written] != 0)

    # fp32 so the comparison isn't done in the caches' own fp16. Tolerances are set for
    # fp16 rounding differences between the runtime and eager: the observed max abs
    # error is ~5e-3 on values of order 1, and rtol alone won't do because entries near
    # zero carry a large relative error.
    for name, torch_cache, coreai_cache in (
        (KEY_CACHE_NAME, k_torch, k_coreai),
        (VALUE_CACHE_NAME, v_torch, v_coreai),
    ):
        print(f"comparing {name}")
        assert_close(coreai_cache.astype(np.float32), torch_cache.float(), atol=1e-2, rtol=1e-2)
