# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Convert torch modules to Core AI programs.

``export_stateless`` converts one module into a single-function program. For a program whose
functions share weights, add each function to one converter::

    converter = make_converter()
    add_function(converter, "text_encoder", text_tower, args, input_names, output_names)
    add_function(converter, "image_encoder", image_tower, args, input_names, output_names)
    program = converter.to_coreai()
"""

import coreai_torch
import torch
from coreai.authoring import AIProgram

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO


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

    converter = make_converter(include_debug_info)
    converter.add_pytorch_module(
        wrapper,
        export_fn=export_fn,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    return program


def make_converter(
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> coreai_torch.TorchConverter:
    """Return a ``TorchConverter`` in ``RELEASE`` mode, or ``DEBUG`` with ``include_debug_info``."""
    mode = coreai_torch.TorchConverter.Mode
    return coreai_torch.TorchConverter(mode=mode.DEBUG if include_debug_info else mode.RELEASE)


def add_function(
    converter: coreai_torch.TorchConverter,
    name: str,
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...] | list[str],
    output_names: tuple[str, ...] | list[str],
    *,
    dynamic_shapes: tuple[dict[int, torch.export.Dim] | None, ...] | None = None,
    decompositions: dict | None = None,
) -> None:
    """Add ``module`` to ``converter`` as function ``name``.

    ``args`` are sample inputs for tracing and ``dynamic_shapes`` marks dynamic axes, as in
    ``torch.export.export``. Functions added to one converter share weights. ``decompositions``
    defaults to Core AI's table.
    """
    module.eval()
    with torch.no_grad():
        exported = torch.export.export(module, args=args, dynamic_shapes=dynamic_shapes)
    table = coreai_torch.get_decomp_table() if decompositions is None else decompositions
    converter.add_exported_program(
        exported.run_decompositions(table),
        input_names=input_names,
        output_names=output_names,
        entrypoint_name=name,
    )
