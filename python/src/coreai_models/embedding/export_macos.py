# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""GPU (macOS) programs of an embedding bundle.

The text encoder, the shared text / token-table / encoder program, and the image encoder.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import (
    ENCODER_INPUT_NAMES,
    IMAGE_COMPONENT,
    IMAGE_INPUT_NAMES,
    IMAGE_OUTPUT_NAME,
    OUTPUT_NAME,
    TEXT_COMPONENT,
    TEXT_INPUT_NAMES,
    TOKEN_TABLE_INPUT_NAMES,
    TOKEN_TABLE_OUTPUT_NAME,
    EmbeddingFamily,
    FunctionSpec,
    ImageTowerFacts,
    encoder_functions,
    output_dtype,
    text_functions,
)
from coreai_models.embedding.export_utils import (
    TORCH_DTYPES,
    TokenTableExportWrapper,
    image_reference_inputs,
    save_program,
    token_table_trace,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram


OUTPUT_DTYPE = TORCH_DTYPES[output_dtype("macOS")]


class TextEmbeddingExportWrapper(nn.Module):
    """``(input_ids, attention_mask) -> embedding`` as ``OUTPUT_DTYPE``.

    The wrapped encoder owns pooling (masked mean incl. prompt), projection and L2 normalization.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).to(OUTPUT_DTYPE)


class InputsEmbedsEncoderWrapper(nn.Module):
    """``(inputs_embeds, attention_mask) -> embedding`` for merged (e.g. image) inputs.

    Outputs ``OUTPUT_DTYPE``.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        return self.encoder(
            inputs_embeds=inputs_embeds.to(dtype), attention_mask=attention_mask
        ).to(OUTPUT_DTYPE)


class ImageEncoderExportWrapper(nn.Module):
    """``(pixel_values f32, position_ids int32) -> soft_tokens`` as ``OUTPUT_DTYPE``.

    The encoder computes at the module's dtype.
    """

    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype
        return self.image_encoder(pixel_values.to(dtype), position_ids).to(OUTPUT_DTYPE)


@contextmanager
def without_token_table(encoder: nn.Module) -> Iterator[nn.Module]:
    """Temporarily drop ``encoder.embed_tokens`` so an ``inputs_embeds`` trace carries no table."""
    table = encoder.embed_tokens
    encoder.embed_tokens = nn.Identity()
    try:
        yield encoder
    finally:
        encoder.embed_tokens = table


def reference_inputs(spec: FunctionSpec, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace inputs for one function (values do not matter; shapes and dtypes do)."""
    generator = torch.Generator().manual_seed(spec.batch * 100_003 + spec.length)
    ids = torch.randint(3, vocab_size, (spec.batch, spec.length), generator=generator)
    mask = torch.ones(spec.batch, spec.length, dtype=torch.int32)
    return ids.to(torch.int32), mask


def embeds_reference_inputs(spec: FunctionSpec, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace inputs for one ``inputs_embeds`` function (values do not matter)."""
    generator = torch.Generator().manual_seed(spec.length)
    embeds = torch.randn((spec.batch, spec.length, hidden), generator=generator)
    return embeds, torch.ones((spec.batch, spec.length), dtype=torch.int32)


def export_shared_text_program(
    encoder: nn.Module,
    text_fns: list[FunctionSpec],
    encoder_fns: list[FunctionSpec],
    max_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """GPU ``text_encoder``; with ``encoder_fns`` it also holds the ``token_table`` and ``encoder``.

    Text functions take ids at a static (batch, length); the host pads to the next listed batch.
    ``main`` maps ids to ``inputs_embeds`` (dynamic length up to ``max_length``); ``embeds_*``
    functions take ``inputs_embeds``. The functions share one copy of the table and the transformer.
    """
    from coreai_models.diffusion.gpu import add_function, make_converter

    converter = make_converter(include_debug_info)
    text = TextEmbeddingExportWrapper(encoder)
    for spec in text_fns:
        args = reference_inputs(spec, int(encoder.config.vocab_size))
        add_function(converter, spec.name, text, args, TEXT_INPUT_NAMES, (OUTPUT_NAME,))
    if encoder_fns:
        ids, dynamic_shapes = token_table_trace(max_length)
        add_function(
            converter,
            "main",
            TokenTableExportWrapper(encoder.embed_tokens, OUTPUT_DTYPE),
            ids,
            TOKEN_TABLE_INPUT_NAMES,
            (TOKEN_TABLE_OUTPUT_NAME,),
            dynamic_shapes=dynamic_shapes,
        )
        with without_token_table(encoder) as inner:
            embeds = InputsEmbedsEncoderWrapper(inner)
            for spec in encoder_fns:
                args = embeds_reference_inputs(spec, int(encoder.config.hidden_size))
                add_function(
                    converter, spec.name, embeds, args, ENCODER_INPUT_NAMES, (OUTPUT_NAME,)
                )
    return converter.to_coreai()


def export_image_encoder_program(
    image_encoder: nn.Module,
    image: ImageTowerFacts,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """GPU ``image_encoder``: function ``main``, pixels and positions -> soft tokens."""
    from coreai_models.diffusion.gpu import export_stateless

    return export_stateless(
        ImageEncoderExportWrapper(image_encoder),
        image_reference_inputs(image),
        IMAGE_INPUT_NAMES,
        (IMAGE_OUTPUT_NAME,),
        include_debug_info=include_debug_info,
    )


def export_gpu_programs(
    family: EmbeddingFamily,
    snapshot: Path,
    out: Path,
    *,
    with_image: bool,
    compute_precision: str,
    include_debug_info: bool,
) -> None:
    if family.key != "embeddinggemma-2":
        raise NotImplementedError(f"No GPU export for family {family.key!r}")
    from coreai_models.models.macos.embedding_gemma2 import load_embedding_gemma2_text_encoder
    from coreai_models.models.macos.embedding_gemma2_vision import EmbeddingGemma2ImageEncoder

    dtype = TORCH_DTYPES[compute_precision]
    encoder = load_embedding_gemma2_text_encoder(snapshot, dtype=dtype)
    program = export_shared_text_program(
        encoder,
        text_functions(family, "macOS"),
        encoder_functions(family, "macOS") if with_image else [],
        family.model_max_tokens,
        include_debug_info,
    )
    save_program(program, out, family, TEXT_COMPONENT)
    del program, encoder
    if not with_image:
        return
    assert family.image is not None
    image_encoder = EmbeddingGemma2ImageEncoder.from_pretrained(
        str(snapshot), dtype=dtype, max_soft_tokens=family.image.budget
    )
    save_program(
        export_image_encoder_program(image_encoder, family.image, include_debug_info),
        out,
        family,
        IMAGE_COMPONENT,
    )
