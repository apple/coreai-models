# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
iOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram for iOS.
The iOS export produces 4 entrypoints:
- load_embeddings: returns the embedding table
- gather_embeddings: token IDs -> embedded representations
- extend: single forward pass (decode mode)
- prompt_opt: forward pass in prefill mode
"""

import logging
from typing import Any

import torch
from coreai.authoring import AIProgram
from coreai_torch import TorchConverter

from coreai_models._constants import (
    DEFAULT_INCLUDE_DEBUG_INFO,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    PROMPT_OPT_FUNCTION_NAME,
)
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.models.base import BaseForCausalLMForiOS, TraceSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _export_programs(
    model: BaseForCausalLMForiOS,
    reference_inputs: dict[str, dict],
    dynamic_shapes: dict[str, dict],
) -> dict[str, torch.export.ExportedProgram]:
    """Trace one program per emitted entrypoint.

    The transformer contract is traced twice, decode then prefill, since ``extend`` and
    ``prompt_opt`` differ only by module state. Which callable belongs to which
    entrypoint, and the prefill toggle, live here rather than in the contract.
    """
    modules = {
        LOAD_EMBEDDINGS_FUNCTION_NAME: model.load_embeddings,
        GATHER_EMBEDDINGS_FUNCTION_NAME: model.gather_embeddings,
        EXTEND_FUNCTION_NAME: model.extend,
    }
    # iOS decomp table: keep silu as-is, it lowers to a single fused op.
    decomp_table = torch.export.default_decompositions()
    decomp_table.pop(torch.ops.aten.silu.default)
    decomp_table.pop(torch.ops.aten.silu.out)

    def trace(graph: str, module: Any) -> torch.export.ExportedProgram:
        logger.info(f"Exporting {graph} module...")
        return torch.export.export(
            module,
            args=(),
            kwargs=reference_inputs[graph],
            dynamic_shapes=dynamic_shapes[graph] or None,
        )

    programs: dict[str, torch.export.ExportedProgram] = {}
    with torch.no_grad():
        for graph in (LOAD_EMBEDDINGS_FUNCTION_NAME, GATHER_EMBEDDINGS_FUNCTION_NAME):
            programs[graph] = trace(graph, modules[graph])

        # The transformer entry is used for both emitted entrypoints.
        for entrypoint, prefill in (
            (EXTEND_FUNCTION_NAME, False),
            (PROMPT_OPT_FUNCTION_NAME, True),
        ):
            model.set_prefill_mode(prefill)
            program = trace(EXTEND_FUNCTION_NAME, modules[EXTEND_FUNCTION_NAME])
            program = program.run_decompositions(decomp_table)
            remove_functionalization(program)
            programs[entrypoint] = program

    return programs


async def _convert_to_coreai(
    model: BaseForCausalLMForiOS,
    programs: dict[str, torch.export.ExportedProgram],
    config,
    max_context_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert the traced programs to one AIProgram with iOS constraints.

    Graph I/O names, static shapes and hardware constraints all come from the model's
    contract. What stays here is export-side: which callable each emitted entrypoint
    traces, and applying the contract to the ``AIProgram``.
    """
    inputs = model.export_input_names()
    states = model.export_state_names()
    outputs = model.export_output_names()

    # Emitted entrypoint -> the contract entry describing it.
    contract_for = {
        LOAD_EMBEDDINGS_FUNCTION_NAME: LOAD_EMBEDDINGS_FUNCTION_NAME,
        GATHER_EMBEDDINGS_FUNCTION_NAME: GATHER_EMBEDDINGS_FUNCTION_NAME,
        EXTEND_FUNCTION_NAME: EXTEND_FUNCTION_NAME,
        PROMPT_OPT_FUNCTION_NAME: EXTEND_FUNCTION_NAME,
    }

    mode = TorchConverter.Mode.DEBUG if include_debug_info else TorchConverter.Mode.RELEASE
    converter = TorchConverter(mode=mode)
    register_custom_torch_lowering(converter)
    for entrypoint, graph in contract_for.items():
        converter.add_exported_program(
            programs[entrypoint],
            input_names=list(inputs[graph]),
            state_names=list(states[graph]),
            output_names=list(outputs[graph]),
            entrypoint_name=entrypoint,
        )

    coreai_program: AIProgram = converter.to_coreai()

    # Both transformer entrypoints share the transformer contract entry, so each gets the
    # same shapes and constraints. An empty entry means the graph needs none.
    static_shapes = model.export_static_shape_configs(config, max_context_length)
    constraints = model.export_hardware_constraints(max_context_length)
    for entrypoint, graph in contract_for.items():
        if static_shapes[graph]:
            coreai_program.set_static_shape_config(entrypoint, static_shapes[graph])
        if constraints[graph]:
            coreai_program.set_hardware_constraints(entrypoint, constraints[graph])

    logger.info("Applying optimization passes...")
    coreai_program.optimize()

    return coreai_program


async def export_ios_model(
    model: BaseForCausalLMForiOS,
    config,
    export_config,
) -> AIProgram:
    """Export an iOS model to an AIProgram.

    The graph I/O names, reference inputs, static shapes and hardware constraints all
    come from the model's contract; this function owns the tracing and conversion.

    Args:
        model: A loaded iOS model, already in the correct dtype.
        config: HuggingFace model config.
        export_config: An ExportConfig instance.

    Returns:
        An optimized AIProgram with four entrypoints, static shape configs, and
        hardware constraints set for iOS.
    """
    requested = getattr(export_config, "max_context_length", None) or getattr(
        config, "max_position_embeddings", None
    )
    max_context_length: int = int(requested) if requested is not None else 2048

    logger.info(
        f"Exporting iOS model (max_context_length={max_context_length}, "
        f"vocab_size={config.vocab_size})"
    )

    target_dtype = next(model.parameters()).dtype
    spec = TraceSpec(max_context_length=max_context_length, cache_seq_len=max_context_length)
    reference_inputs = model.build_reference_inputs(config, target_dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    model.validate_export_contract(reference_inputs, dynamic_shapes)

    programs = _export_programs(model, reference_inputs, dynamic_shapes)

    return await _convert_to_coreai(
        model=model,
        programs=programs,
        config=config,
        max_context_length=max_context_length,
        include_debug_info=getattr(export_config, "include_debug_info", DEFAULT_INCLUDE_DEBUG_INFO),
    )
