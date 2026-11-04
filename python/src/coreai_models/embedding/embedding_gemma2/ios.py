# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""iOS programs of an EmbeddingGemma 2 bundle.

The float16 token table (GPU), the static-length encoder and the image encoder (Neural Engine).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO, MASK_NEG
from coreai_models.embedding.contract import IMAGE_COMPONENT, OUTPUT_NAME, CheckpointFacts
from coreai_models.embedding.embedding_gemma2.common import (
    ENCODER_COMPONENT,
    IMAGE_INPUT_NAMES,
    IMAGE_OUTPUT_NAME,
    TOKEN_TABLE_COMPONENT,
    TOKEN_TABLE_INPUT_NAMES,
    TOKEN_TABLE_OUTPUT_NAME,
    EmbeddingGemma2Family,
    FunctionSpec,
    ImageTowerFacts,
    TokenTableExportWrapper,
    encoder_functions,
    image_reference_inputs,
    output_dtype,
    token_table_trace,
)
from coreai_models.embedding.export_utils import TORCH_DTYPES, save_program

if TYPE_CHECKING:
    from coreai.authoring import AIProgram


ANE_ENCODER_INPUT_NAMES = ("inputs_embeds", "bias", "pool_weights")
# The iOS programs compute in this dtype and output it.
DTYPE = TORCH_DTYPES[output_dtype("iOS")]


def scaled_token_table(table: torch.Tensor, scale: float, dtype: torch.dtype) -> nn.Embedding:
    """Return an ``nn.Embedding`` of ``table * scale`` (computed in float32) as ``dtype``."""
    return nn.Embedding.from_pretrained((table.float() * scale).to(dtype))


def export_token_table_program(
    embed_tokens: nn.Module,
    max_length: int,
    dtype: torch.dtype,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """``token_table``: function ``main``, ids -> scaled embeddings, dynamic length (batch 1)."""
    from coreai_models.converter import export_stateless

    # Placeholder ids look up like any id; the host overwrites those rows with soft tokens.
    args, dynamic_shapes = token_table_trace(max_length)
    return export_stateless(
        TokenTableExportWrapper(embed_tokens, dtype),
        args,
        TOKEN_TABLE_INPUT_NAMES,
        (TOKEN_TABLE_OUTPUT_NAME,),
        dynamic_shapes=dynamic_shapes,
        include_debug_info=include_debug_info,
    )


class EncoderBandWrapper(nn.Module):
    """``(inputs_embeds, bias, pool_weights) -> embedding`` at one static length (Neural Engine).

    When ``length > sliding_window + 1``, the sliding-window layers use ``min(bias, band)``, with
    ``band`` a constant ``|i - j| <= sliding_window`` mask for that length. Every length takes
    the same three inputs.
    """

    def __init__(self, encoder: nn.Module, length: int, sliding_window: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.banded = length > sliding_window + 1
        if self.banded:
            pos = torch.arange(length)
            band = torch.where((pos[:, None] - pos[None, :]).abs() <= sliding_window, 0.0, MASK_NEG)
            dtype = next(encoder.parameters()).dtype
            self.register_buffer("band", band.reshape(1, length, 1, length).to(dtype))

    def forward(
        self, inputs_embeds: torch.Tensor, bias: torch.Tensor, pool_weights: torch.Tensor
    ) -> torch.Tensor:
        sliding = torch.minimum(bias, self.band) if self.banded else None
        return self.encoder(inputs_embeds, bias, pool_weights, sliding)


def encoder_reference_inputs(
    length: int, hidden: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neural Engine encoder trace inputs (all valid tokens): embeds, zero bias, uniform pooling."""
    generator = torch.Generator().manual_seed(length)
    embeds = torch.randn((1, length, hidden), generator=generator).to(DTYPE)
    bias = torch.zeros((1, length, 1, length), dtype=DTYPE)
    pool = torch.full((1, length), 1.0 / length, dtype=DTYPE)
    return embeds, bias, pool


def convert_ane_functions(
    functions: Sequence[
        tuple[str, nn.Module, tuple[torch.Tensor, ...], tuple[str, ...], tuple[str, ...]]
    ],
    *,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert ``(name, module, args, input_names, output_names)`` entries into one Neural Engine
    program with one function per entry.

    Uses ``torch.export.default_decompositions()`` and ``register_custom_torch_lowering``.
    """
    from coreai_models.converter import add_function, make_converter
    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    converter = make_converter(include_debug_info)
    register_custom_torch_lowering(converter)
    decompositions = torch.export.default_decompositions()
    for name, module, args, input_names, output_names in functions:
        add_function(
            converter, name, module, args, input_names, output_names, decompositions=decompositions
        )
    return converter.to_coreai()


def export_ane_encoder_program(
    encoder: nn.Module,
    functions: list[FunctionSpec],
    sliding_window: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Neural Engine ``encoder``: one static batch-1 float16 function per length."""
    hidden = int(encoder.config.hidden_size)
    traces = [
        (
            spec.name,
            EncoderBandWrapper(encoder, spec.length, sliding_window),
            encoder_reference_inputs(spec.length, hidden),
            ANE_ENCODER_INPUT_NAMES,
            (OUTPUT_NAME,),
        )
        for spec in functions
    ]
    return convert_ane_functions(traces, include_debug_info=include_debug_info)


def export_ane_image_encoder_program(
    image_encoder: nn.Module,
    image: ImageTowerFacts,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Neural Engine ``image_encoder``: function ``main`` at the static budget."""
    args = image_reference_inputs(image, DTYPE, DTYPE)
    return convert_ane_functions(
        [("main", image_encoder, args, IMAGE_INPUT_NAMES, (IMAGE_OUTPUT_NAME,))],
        include_debug_info=include_debug_info,
    )


def export_ios_programs(
    family: EmbeddingGemma2Family,
    snapshot: Path,
    out: Path,
    facts: CheckpointFacts,
    *,
    with_image: bool,
    include_debug_info: bool,
) -> None:
    """Write the ``token_table``, ``encoder`` and, if ``with_image``, ``image_encoder`` assets to
    ``out``.
    """
    from coreai_models.models.ios.embedding_gemma2 import (
        load_embedding_gemma2_text_encoder_and_table,
    )

    # The token table runs on the GPU and stores the scaled table in float16, the encoder's input
    # precision.
    # TODO: write the table as a safetensors sidecar that the host gathers from, as the Gemma 4
    # per-layer embeddings do (SafeTensorsReader), instead of a GPU program; lands with the Swift
    # embedding runtime.
    encoder, table = load_embedding_gemma2_text_encoder_and_table(snapshot, dtype=DTYPE)
    token_table = scaled_token_table(table, encoder.config.hidden_size**0.5, DTYPE)
    del table
    save_program(
        export_token_table_program(token_table, facts.max_tokens, DTYPE, include_debug_info),
        out,
        family,
        TOKEN_TABLE_COMPONENT,
    )
    del token_table
    save_program(
        export_ane_encoder_program(
            encoder,
            encoder_functions(family, "iOS", with_image=with_image),
            int(encoder.config.sliding_window),
            include_debug_info,
        ),
        out,
        family,
        ENCODER_COMPONENT,
    )
    del encoder
    if with_image:
        assert family.image_tower is not None
        from coreai_models.models.ios.embedding_gemma2_vision import (
            load_embedding_gemma2_image_encoder,
        )

        image_encoder = load_embedding_gemma2_image_encoder(
            str(snapshot), DTYPE, max_soft_tokens=family.image_tower.max_soft_tokens
        )
        save_program(
            export_ane_image_encoder_program(image_encoder, family.image_tower, include_debug_info),
            out,
            family,
            IMAGE_COMPONENT,
        )
