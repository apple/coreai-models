# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Neural Engine (iOS) programs of an embedding bundle.

The static-length encoder, the float16 token table, and the image encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import (
    ANE_ENCODER_INPUT_NAMES,
    ENCODER_COMPONENT,
    IMAGE_COMPONENT,
    IMAGE_INPUT_NAMES,
    IMAGE_OUTPUT_NAME,
    OUTPUT_NAME,
    TOKEN_TABLE_COMPONENT,
    EmbeddingFamily,
    FunctionSpec,
    ImageTowerFacts,
    encoder_functions,
)
from coreai_models.embedding.export_utils import (
    export_token_table_program,
    image_reference_inputs,
    save_program,
    scaled_token_table,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram


class EncoderBandWrapper(nn.Module):
    """``(inputs_embeds, bias, pool_weights) -> embedding`` at one static length (Neural Engine).

    When ``length > sliding_window + 1`` the sliding layers also need the ``|i - j| <= window``
    band; it is a per-length constant folded in here (``min(bias, band)``), so every length keeps
    the same three inputs.
    """

    def __init__(self, encoder: nn.Module, length: int, sliding_window: int) -> None:
        super().__init__()
        from coreai_models.models.ios.embedding_gemma2 import MASK_NEG

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


def ane_text_reference_inputs(
    length: int, hidden: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neural Engine encoder trace inputs (all valid tokens): embeds, zero bias, uniform pooling."""
    generator = torch.Generator().manual_seed(length)
    embeds = torch.randn((1, length, hidden), generator=generator).to(torch.float16)
    bias = torch.zeros((1, length, 1, length), dtype=torch.float16)
    pool = torch.full((1, length), 1.0 / length, dtype=torch.float16)
    return embeds, bias, pool


def convert_ane_functions(
    functions: Sequence[
        tuple[str, nn.Module, tuple[torch.Tensor, ...], tuple[str, ...], tuple[str, ...]]
    ],
    *,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert each ``(name, module, args, input_names, output_names)`` into one Neural Engine
    program.

    Uses the default decompositions plus the repo's custom lowerings, as ``export/ios.py`` does.
    """
    from coreai_models.diffusion.gpu import add_function, new_converter
    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    converter = new_converter(include_debug_info)
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
            ane_text_reference_inputs(spec.length, hidden),
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
    args = image_reference_inputs(image, torch.float16, torch.float16)
    return convert_ane_functions(
        [("main", image_encoder, args, IMAGE_INPUT_NAMES, (IMAGE_OUTPUT_NAME,))],
        include_debug_info=include_debug_info,
    )


def export_ane_programs(
    family: EmbeddingFamily,
    snapshot: Path,
    out: Path,
    *,
    with_image: bool,
    include_debug_info: bool,
) -> None:
    if family.key != "embeddinggemma-2":
        raise NotImplementedError(f"No Neural Engine export for family {family.key!r}")
    from coreai_models.models.ios.embedding_gemma2 import load_embedding_gemma2_text_encoder

    # The token table runs on the GPU and stores the scaled table in float16, the encoder's input
    # precision.
    # TODO: write the table as a safetensors sidecar that the host gathers from, as the Gemma 4
    # per-layer embeddings do (SafeTensorsReader), instead of a GPU program; lands with the Swift
    # embedding runtime.
    encoder, table = load_embedding_gemma2_text_encoder(snapshot, dtype=torch.float16)
    token_table = scaled_token_table(table, encoder.config.hidden_size**0.5, torch.float16)
    del table
    save_program(
        export_token_table_program(
            token_table, family.model_max_tokens, include_debug_info, dtype=torch.float16
        ),
        out,
        family,
        TOKEN_TABLE_COMPONENT,
    )
    del token_table
    save_program(
        export_ane_encoder_program(
            encoder,
            encoder_functions(family, "iOS"),
            int(encoder.config.sliding_window),
            include_debug_info,
        ),
        out,
        family,
        ENCODER_COMPONENT,
    )
    del encoder
    if with_image:
        assert family.image is not None
        from coreai_models.models.ios.embedding_gemma2_vision import load_image_encoder

        image_encoder = load_image_encoder(
            str(snapshot), torch.float16, max_soft_tokens=family.image.budget
        )
        save_program(
            export_ane_image_encoder_program(image_encoder, family.image, include_debug_info),
            out,
            family,
            IMAGE_COMPONENT,
        )
