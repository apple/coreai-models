# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Composite-op module externalization.

Externalization keeps selected submodules as named composite ops in the emitted
Core AI graph instead of letting them decompose into primitive aten ops, so
RMSNorm, RoPE, SDPA, etc. map onto fused kernels.
"""

import logging
from collections.abc import Sequence

import coreai_torch
import coreai_torch.composite_ops
import torch
from coreai_torch.externalize import _find_marked_submodules

logger = logging.getLogger(__name__)

# Superset of every composite op any model family might use. Specs match by
# `isinstance`, so one that matches nothing only warns.
EXTERNALIZE_SPECS: list[type | coreai_torch.ExternalizeSpec] = [
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


def patch_model_for_externalization(
    model: torch.nn.Module,
    specs: Sequence[type | coreai_torch.ExternalizeSpec] | None = None,
) -> None:
    """Mark ``model``'s composite-op submodules in place.

    Args:
        model: The eager module to mark. Mutated in place.
        specs: Externalization specs. Defaults to ``EXTERNALIZE_SPECS``.
    """
    specs = EXTERNALIZE_SPECS if specs is None else specs
    coreai_torch._patch_model_for_externalization(model, list(specs))
    logger.info(
        "Marked %d composite-op submodule(s) for externalization",
        len(_find_marked_submodules(model)),
    )


def subexport_and_restore(
    model: torch.nn.Module,
    exported_program: torch.export.ExportedProgram,
) -> list:
    """Sub-export every marked submodule of ``model``, then unpatch ``model``.

    Args:
        model: The module patched with ``patch_model_for_externalization``.
        exported_program: The whole-model program the composites were captured in.

    Returns:
        One entry per externalized call site, for
        ``TorchConverter.add_exported_program(_externalized_exported_programs=...)``.
    """
    marked_count = len(_find_marked_submodules(model))
    externalized_programs = coreai_torch._subexport_and_restore(model, exported_program)

    if marked_count and not externalized_programs:
        logger.warning(
            "None of the %d marked submodule(s) had a call site in the exported program, "
            "so nothing was externalized. The model was most likely captured before "
            "patch_model_for_externalization ran.",
            marked_count,
        )
    else:
        logger.info(
            "Externalized %d composite op call site(s) from %d marked submodule(s)",
            len(externalized_programs),
            marked_count,
        )
    return externalized_programs
