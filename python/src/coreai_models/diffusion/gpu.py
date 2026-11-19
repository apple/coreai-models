# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Multi-function GPU export for diffusion components.

``export_multifunction`` exports one component combining several static shapes into one program
with shared weights. Single-shape export (``export_stateless``) is in ``coreai_models.converter``.
"""

import logging

import torch
from coreai.authoring import AIProgram

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.converter import add_function, make_converter

logger = logging.getLogger(__name__)


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
    converter = make_converter(include_debug_info)
    for name, wrapper, dummy_inputs in functions:
        logger.info(f"Tracing function '{name}' (seq dims: {[t.shape for t in dummy_inputs[:2]]})")
        add_function(converter, name, wrapper, dummy_inputs, input_names, output_names)
    return converter.to_coreai()
