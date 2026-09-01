# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Stateless GPU export for diffusion components.

Each component is a single forward pass. Video models use dynamic shapes
for variable temporal dimensions.
"""

import logging

import coreai_torch
import torch
from coreai.authoring import AIProgram

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO

logger = logging.getLogger(__name__)


def _decomp_empty_permuted(size, physical_layout, **kwargs):
    """Decompose empty_permuted to empty + permute (not in coreai_torch decomp table)."""
    perm = [0] * len(physical_layout)
    for i, p in enumerate(physical_layout):
        perm[p] = i
    return torch.empty([size[p] for p in physical_layout], **kwargs).permute(perm)


def export_stateless(
    wrapper: torch.nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_shapes: tuple[dict[int, torch.export.Dim] | None, ...] | None = None,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Export a stateless model to a Core AI AIProgram.

    Args:
        wrapper: A thin torch.nn.Module that wraps a HF model component.
        dummy_inputs: Reference input tensors (positional) for tracing.
        input_names: Names for the exported model's inputs.
        output_names: Names for the exported model's outputs.
        dynamic_shapes: Per-input dynamic dimension specs (same positional order
            as dummy_inputs). None entries mean that input is fully static.
        include_debug_info: When True, the converter runs in ``DEBUG`` mode and embeds debug
            information in the exported ``.aimodel``. Defaults to ``RELEASE`` mode,
            which embeds minimum debug information and makes the exported asset smaller.

    Returns:
        An optimized AIProgram ready for saving/compilation.

    Raises:
        ValueError: If ``input_names`` or ``dynamic_shapes`` disagrees in length with
            ``dummy_inputs``.
    """
    # This path traces positionally (args=), so all three tuples are bound by order
    # and a length mismatch misbinds silently. A dynamic spec would land on the
    # wrong input, or the last input would go unconstrained.
    if len(input_names) != len(dummy_inputs):
        raise ValueError(
            f"input_names has {len(input_names)} entries but {len(dummy_inputs)} dummy "
            f"inputs were given: {input_names}"
        )
    if dynamic_shapes is not None and len(dynamic_shapes) != len(dummy_inputs):
        raise ValueError(
            f"dynamic_shapes has {len(dynamic_shapes)} entries but {len(dummy_inputs)} "
            "dummy inputs were given. It is a positional tuple, so it needs one entry "
            "per input (None for a fully static one)."
        )

    wrapper.eval()

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            exported = torch.export.export(module, args=dummy_inputs, dynamic_shapes=dynamic_shapes)
        coreai_decomp_table = coreai_torch.get_decomp_table()
        coreai_decomp_table[torch.ops.aten.empty_permuted.default] = _decomp_empty_permuted
        decomposed: torch.export.ExportedProgram = exported.run_decompositions(coreai_decomp_table)
        return decomposed

    converter = coreai_torch.TorchConverter(
        mode=(
            coreai_torch.TorchConverter.Mode.DEBUG
            if include_debug_info
            else coreai_torch.TorchConverter.Mode.RELEASE
        )
    )
    converter.add_pytorch_module(
        wrapper,
        export_fn=export_fn,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    program.optimize()
    return program


def export_multifunction(
    functions: list[tuple[str, torch.nn.Module, tuple[torch.Tensor, ...]]],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Export multiple function variants into a single .aimodel with shared weights.

    Each function is a separate static trace of the same model at different input
    shapes. Weights are shared automatically, so disk size equals one copy.

    Args:
        functions: List of (entrypoint_name, wrapper, dummy_inputs) tuples.
            The wrapper should be the same nn.Module instance (or share weights).
        input_names: Names for the exported model's inputs (same for all functions).
        output_names: Names for the exported model's outputs (same for all functions).
        include_debug_info: When True, the converter runs in ``DEBUG`` mode and embeds debug
            information in the exported ``.aimodel``. Defaults to ``RELEASE`` mode,
            which embeds minimum debug information and makes the exported asset smaller.

    Returns:
        An optimized AIProgram with multiple named functions.
    """
    converter = coreai_torch.TorchConverter(
        mode=(
            coreai_torch.TorchConverter.Mode.DEBUG
            if include_debug_info
            else coreai_torch.TorchConverter.Mode.RELEASE
        )
    )
    coreai_decomp_table = coreai_torch.get_decomp_table()

    for name, wrapper, dummy_inputs in functions:
        wrapper.eval()
        logger.info(f"Tracing function '{name}' (seq dims: {[t.shape for t in dummy_inputs[:2]]})")

        with torch.no_grad():
            exported = torch.export.export(wrapper, args=dummy_inputs)
        decomposed = exported.run_decompositions(coreai_decomp_table)

        converter.add_exported_program(
            decomposed,
            input_names=input_names,
            output_names=output_names,
            entrypoint_name=name,
        )

    program = converter.to_coreai()
    program.optimize()
    return program
