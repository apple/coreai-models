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
    """
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
