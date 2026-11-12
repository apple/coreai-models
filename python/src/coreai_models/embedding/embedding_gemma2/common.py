# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Names, facts and function plan of an EmbeddingGemma 2 bundle, and the trace helpers shared by
its macOS and iOS programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from coreai_models.embedding.contract import EmbeddingFamily
from coreai_models.embedding.export_utils import FunctionSpec, bucket_plan, function_name

TOKEN_TABLE_INPUT_NAMES = ("input_ids",)
TOKEN_TABLE_OUTPUT_NAME = "inputs_embeds"
IMAGE_INPUT_NAMES = ("pixel_values", "position_ids")
IMAGE_OUTPUT_NAME = "soft_tokens"
TOKEN_TABLE_COMPONENT = "token_table"
ENCODER_COMPONENT = "encoder"
# macOS encoder lengths are multiples of this.
ENCODER_LENGTH_MULTIPLE = 32


@dataclass(frozen=True)
class ImageTowerFacts:
    """Image-tower facts the host needs: merge ids, preprocessing and the soft-token budget.

    The exporter checks them against the snapshot ``config.json`` / ``processor_config.json``
    and raises on a mismatch. ``soft_token_dim`` is the text hidden size.
    """

    placeholder_id: int
    boi_token_id: int
    eoi_token_id: int
    patch_size: int
    pooling_kernel_size: int
    rescale_factor: float
    normalize: bool
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    max_soft_tokens: int
    soft_token_dim: int

    @property
    def max_patches(self) -> int:
        return self.max_soft_tokens * self.pooling_kernel_size**2

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2

    @property
    def max_sequence_length(self) -> int:
        """Longest image-only sequence: ``<bos> <boi> <image>*max_soft_tokens <eoi> <eos>``."""
        return self.max_soft_tokens + 4


@dataclass(frozen=True)
class EmbeddingGemma2Family(EmbeddingFamily):
    """``EmbeddingFamily`` plus the EmbeddingGemma 2 export facts.

    ``max_tokens`` is the model's token limit; ``text.truncation.max_tokens`` is the longest text
    function of the platform.
    """

    matryoshka_dimensions: tuple[int, ...] = ()
    # canonical prompt name -> prompt key in the snapshot's config_sentence_transformers.json
    prompt_names: dict[str, str] = field(default_factory=dict)
    metric: str = "cosine"
    normalized: bool = True
    renormalize_after_truncation: bool = True
    pooling_mode: str = "mean"
    include_prompt: bool = True
    document_template: str | None = None
    image_tower: ImageTowerFacts | None = None
    length_buckets: tuple[int, ...] = (128, 512, 2048)
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    max_batch_tokens: int = 16384
    ane_length_buckets: tuple[int, ...] = (128, 512, 1024)


def text_functions(family: EmbeddingGemma2Family) -> list[FunctionSpec]:
    """Functions of the macOS ``text_encoder``: ``length_buckets`` crossed with ``batch_sizes``."""
    return bucket_plan(family.length_buckets, family.batch_sizes, family.max_batch_tokens)


def encoder_functions(
    family: EmbeddingGemma2Family, platform: str, *, with_image: bool
) -> list[FunctionSpec]:
    """Batch-1 ``inputs_embeds`` functions ``embeds_len{L}_b1`` of the ``encoder`` component.

    Lengths are ``ane_length_buckets`` on iOS. On macOS the component exists only
    ``with_image`` (else ``[]``); its lengths are the longest image sequence rounded up to a
    multiple of ``ENCODER_LENGTH_MULTIPLE``, plus every longer ``length_buckets`` entry. The host
    reads the names from ``components.encoder.functions``.

    Raises:
        ValueError: if ``ane_length_buckets`` is empty or holds a length below 2 (iOS).
    """
    if platform == "iOS":
        lengths = sorted(set(family.ane_length_buckets))
        if not lengths or lengths[0] < 2:
            raise ValueError(
                f"ane_length_buckets must be >= 2 (BOS + EOS), got {family.ane_length_buckets}"
            )
    elif not with_image:
        return []
    else:
        assert family.image_tower is not None
        multiple = ENCODER_LENGTH_MULTIPLE
        first = -(-family.image_tower.max_sequence_length // multiple) * multiple
        lengths = sorted({first} | {n for n in family.length_buckets if n > first})
    return [FunctionSpec(function_name(n, 1, "embeds_"), 1, n) for n in lengths]


def output_dtype(platform: str) -> str:
    """Dtype of every component output: float16 on iOS, float32 on macOS."""
    return "float16" if platform == "iOS" else "float32"


# --- trace inputs ---


class TokenTableExportWrapper(nn.Module):
    """``input_ids [B, L] int32 -> inputs_embeds [B, L, H]`` as ``dtype`` (scaled embeddings)."""

    def __init__(self, embed_tokens: nn.Module, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids).to(self.dtype)


def image_reference_inputs(
    image: ImageTowerFacts,
    dtype: torch.dtype = torch.float32,
    position_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-encoder trace inputs: a block grid padded to the budget (-1 positions, zero pixels).

    The GPU program takes float32 pixels and int32 positions; the Neural Engine takes float16 for
    both.
    """
    from coreai_models.models.macos.embedding_gemma2_vision import pad_to_budget, patch_position_ids

    k = image.pooling_kernel_size
    # Square-ish grid of at most budget blocks (soft tokens) -> (bh*k) x (bw*k) patches, row-major.
    bw = max(1, int(image.max_soft_tokens**0.5))
    bh = max(1, image.max_soft_tokens // bw)
    generator = torch.Generator().manual_seed(image.max_soft_tokens)
    pixels = torch.rand((image.max_patches, image.patch_dim), generator=generator)
    pixels, positions = pad_to_budget(
        pixels[: bh * k * bw * k], patch_position_ids(bh * k, bw * k), image.max_patches
    )
    return pixels[None].to(dtype), positions[None].to(position_dtype)


def token_table_trace(max_length: int) -> tuple[tuple[torch.Tensor], tuple[dict[int, Any]]]:
    """Token-table trace args and dynamic shapes: batch-1 ids, length 2..``max_length``."""
    length = torch.export.Dim("length", min=2, max=max_length)
    return (torch.zeros((1, 64), dtype=torch.int32),), ({1: length},)
