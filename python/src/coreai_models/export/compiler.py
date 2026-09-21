# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MLIR-level quantization helpers for the export pipeline."""

import logging

from coreai.authoring import AIProgram
from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    quantize_weights,
)
from coreai_opt.coreai_utils.common import QScheme

_GRANULARITY_MAP: dict[str, CompressionGranularity] = {
    "per_tensor": CompressionGranularity.PER_TENSOR,
    "per_channel": CompressionGranularity.PER_CHANNEL,
    "per_block": CompressionGranularity.PER_BLOCK,
    "per_grouped_channel": CompressionGranularity.PER_GROUPED_CHANNEL,
}


logger = logging.getLogger(__name__)


async def apply_mlir_quantization(
    coreai_program: AIProgram,
    quantize_config: dict,
) -> AIProgram:
    """
    Apply post-MLIR INT4 weight quantization to a Core AI program.

    Deprecated. Prefer quantizing the PyTorch module before export, by giving the
    component spec a ``quant_target_fn``. This remains for the diffusion pipelines still
    on the post-export path, and goes away with the last of them.

    Args:
        coreai_program: The Core AI program to quantize.
        quantize_config: A coreai-opt ``quantization_config`` dict, as built by
            ``coreai_models.diffusion.presets``. Only the global weight spec is read.

    Returns:
        The (potentially modified) Core AI program.

    Raises:
        ValueError: For any quantization type beyond int4. Earlier revisions warned and
            returned the program untouched, which yielded a full-precision asset that
            the metadata still described as compressed.
    """
    # Presets are written for the torch quantizer, so pull the equivalent arguments back
    # out for this pass. `symmetric_with_clipping` is int4 over [-7, 7], which is what
    # `QScheme.SYMMETRIC` produces here.
    weight_spec = quantize_config["global_config"]["op_state_spec"]["weight"]
    granularity_spec = weight_spec["granularity"]
    quant_type = weight_spec["dtype"]
    symmetric = weight_spec["qscheme"] != "asymmetric"
    granularity = granularity_spec["type"]
    block_size = granularity_spec.get("block_size", 32)

    logger.warning(
        "Post-export MLIR quantization is deprecated. Migrate this pipeline to "
        "pre-export torch quantization by setting `quant_target_fn` on its component "
        "specs."
    )
    logger.info(
        f"Applying {quant_type} quantization with {granularity} granularity "
        f"(block_size={block_size})"
    )

    if quant_type != "int4":
        raise ValueError(
            f"Post-export MLIR quantization only implements int4, got '{quant_type}'. "
            "Pre-export torch quantization supports the other dtypes."
        )

    coreai_program = quantize_weights(
        coreai_program,
        dtype=DType.INT4,
        qscheme=QScheme.SYMMETRIC if symmetric else QScheme.ASYMMETRIC,
        granularity=_GRANULARITY_MAP[granularity],
        block_size=block_size,
        weight_num_threshold=32768,
        in_place=True,
    )
    logger.info("Applied INT4 weight quantization")
    return coreai_program
