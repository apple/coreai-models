# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
macOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram via:
torch.export -> decompose -> defunctionalize -> TorchConverter -> optimize.
"""

import logging
from typing import Any

import coreai_torch
import coreai_torch.composite_ops
import torch
from coreai.authoring import AIProgram

from coreai_models._constants import (
    DEFAULT_INCLUDE_DEBUG_INFO,
    MAIN_GRAPH_NAME,
    TRACE_KV_CACHE_SEQ_LEN,
)
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.models.base import BaseForCausalLM, TraceSpec

logger = logging.getLogger(__name__)

# Composite ops that are externalized (kept as named composites in the MLIR graph
# rather than being inlined/decomposed).
_EXTERNALIZE_SPECS = [
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.GatherMM,
        composite_op_name="gather_mm",
        composite_attrs=["num_batch_axes"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RoPE,
        composite_op_name="rope",
        composite_attrs=["scale", "base", "dims", "interleaved"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.GatedDeltaUpdate,
        composite_op_name="gated_delta_update",
        composite_attrs=[],
    ),
]


def _build_reference_inputs(
    model: BaseForCausalLM,
    config,
    target_dtype: torch.dtype,
    max_context_length: int,
) -> tuple[dict[str, Any], dict]:
    """Reference inputs and dynamic shapes for macOS export.

    Thin wrapper over the model's export-contract hooks, where the per-model variation
    lives. Returns ``(reference_inputs, dynamic_shapes)``.
    """
    # The trace cache length only bounds peak memory, so cap it at the context it serves.
    spec = TraceSpec(
        max_context_length=max_context_length,
        cache_seq_len=min(TRACE_KV_CACHE_SEQ_LEN, max_context_length),
    )
    reference_inputs = model.build_reference_inputs(config, target_dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    model.validate_export_contract(reference_inputs, dynamic_shapes)
    # A macOS model has exactly one graph.
    return reference_inputs[MAIN_GRAPH_NAME], dynamic_shapes[MAIN_GRAPH_NAME]


def export_to_coreai(
    model: torch.nn.Module,
    reference_inputs: dict[str, Any],
    dynamic_shapes: dict | None = None,
    input_names: tuple[str, ...] | None = None,
    output_names: tuple[str, ...] | None = None,
    state_names: tuple[str, ...] | None = None,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Export a stateful macOS model to a AIProgram.

    Low-level building block under `export_macos_model` (text-only LLMs). Use
    that when possible; reach for this directly only when you need
    component-specific input/output names that `export_macos_model`'s
    text-only defaults don't fit.

    This is the core export function that handles:
    1. torch.export with no_grad
    2. Decomposition via coreai_torch decomp table
    3. Defunctionalization (replacing auto-functionalized ops with immutable variants)
    4. TorchConverter with externalized composite modules
    5. Custom MLIR lowering registration

    Args:
        model: The PyTorch model to export (must be in eval mode).
        reference_inputs: Dict of reference input tensors (keyword args to forward).
        dynamic_shapes: Dynamic shape specifications for torch.export.
        input_names: Names for the model inputs in the exported graph. If both
            ``input_names`` and ``state_names`` are ``None``, the names default
            to ``reference_inputs.keys()``.
        output_names: Names for the model outputs in the exported graph.
        state_names: Names of inputs that are state (i.e. mutated in place by
            the forward pass and surfaced via the runtime ``state=`` kwarg
            rather than as regular inputs/outputs).
        include_debug_info: When True, the converter runs in ``DEBUG`` mode and embeds debug
            information in the exported ``.aimodel``. Defaults to ``RELEASE`` mode,
            which embeds minimum debug information and makes the exported asset smaller.

    Returns:
        A AIProgram ready for optimization and compilation.
    """
    # If the caller didn't pass input_names explicitly, derive them from
    # ``reference_inputs.keys()`` while excluding any name the caller declared
    # as state. This keeps the call to ``add_pytorch_module`` predictable
    # regardless of whether ``state_names`` is also set.
    if input_names is None:
        state_names_set = set(state_names or ())
        input_names = tuple(k for k in reference_inputs if k not in state_names_set)

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            aten_exported_program = torch.export.export(
                module,
                args=(),
                kwargs=reference_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        coreai_decomp_table = coreai_torch.get_decomp_table()
        coreaten_exported_program = aten_exported_program.run_decompositions(coreai_decomp_table)
        remove_functionalization(coreaten_exported_program)
        return coreaten_exported_program

    model.eval()
    mode = (
        coreai_torch.TorchConverter.Mode.DEBUG
        if include_debug_info
        else coreai_torch.TorchConverter.Mode.RELEASE
    )
    converter = coreai_torch.TorchConverter(mode=mode)
    converter.add_pytorch_module(
        model,
        export_fn=export_fn,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=input_names,
        output_names=output_names,
        state_names=state_names,
    )
    register_custom_torch_lowering(converter)
    return converter.to_coreai()


def export_macos_model(
    model: BaseForCausalLM,
    config,
    export_config,
) -> AIProgram:
    """Export a macOS model to a AIProgram.

    This is the main entry point for macOS model export. It:
    1. Builds reference inputs and dynamic shapes from the model config
    2. Exports the model through torch.export -> TorchConverter
    3. Optimizes the resulting AIProgram

    Args:
        model: A loaded PyTorch model (already in the correct dtype). Its
            export-contract hooks supply the graph's inputs, states, and names.
        config: HuggingFace model config (used for cache dimensions, vocab size, etc.).
        export_config: An ExportConfig instance (used for max_context_length, etc.).

    Returns:
        An optimized AIProgram ready for MLIR quantization and compilation.
    """
    max_context_length = getattr(export_config, "max_context_length", None)
    if max_context_length is None:
        max_context_length = getattr(config, "max_position_embeddings", 2048)

    # Determine target dtype from the model parameters
    target_dtype = next(model.parameters()).dtype

    logger.info(
        f"Exporting macOS model (dtype={target_dtype}, max_context_length={max_context_length})"
    )

    reference_inputs, dynamic_shapes = _build_reference_inputs(
        model, config, target_dtype, max_context_length
    )

    logger.info("Exporting model to Core AI dialect...")
    coreai_program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=model.export_input_names()[MAIN_GRAPH_NAME],
        output_names=model.export_output_names()[MAIN_GRAPH_NAME],
        state_names=model.export_state_names()[MAIN_GRAPH_NAME],
        include_debug_info=getattr(export_config, "include_debug_info", DEFAULT_INCLUDE_DEBUG_INFO),
    )

    logger.info("Optimizing AIProgram...")
    coreai_program.optimize()

    return coreai_program
