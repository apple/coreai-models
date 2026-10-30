# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Helpers shared by the GPU and Neural Engine embedding exports.

Snapshot checks, the token table, trace inputs, and program assembly and saving.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import (
    TOKEN_TABLE_INPUT_NAMES,
    TOKEN_TABLE_OUTPUT_NAME,
    EmbeddingFamily,
    ImageTowerFacts,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram

logger = logging.getLogger(__name__)


TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
)


SNAPSHOT_PATTERNS = [
    "*.json",
    "*.safetensors",
    "tokenizer.model",
    "1_Pooling/*",
    "2_Normalize/*",
]


def resolve_snapshot(hf_model_id: str, model_dir: str | None = None) -> tuple[Path, str]:
    """Return ``(snapshot_dir, revision_sha)`` for a local dir or a hub download.

    The SHA is the HF cache snapshot directory name (``.../snapshots/<sha>``); a plain directory
    outside the cache yields ``""``. Raises ``FileNotFoundError`` if ``model_dir`` is not a
    directory.
    """
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model dir {path} is not a directory")
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(hf_model_id, allow_patterns=SNAPSHOT_PATTERNS)).resolve()
    sha = path.name if path.parent.name == "snapshots" else ""
    return path, sha


def read_prompts(snapshot: Path) -> tuple[dict[str, str], str | None]:
    """Read ``prompts`` / ``default_prompt_name`` from ``config_sentence_transformers.json``."""
    cfg_path = snapshot / "config_sentence_transformers.json"
    if not cfg_path.exists():
        return {}, None
    cfg = json.loads(cfg_path.read_text())
    return dict(cfg.get("prompts") or {}), cfg.get("default_prompt_name")


def verify_pooling_config(snapshot: Path, family: EmbeddingFamily) -> None:
    """Raise ``ValueError`` if the snapshot's pooling config disagrees with the family.

    Checks the pooling mode, ``include_prompt``, the embedding dimension and, for a normalized
    family, the ``2_Normalize`` module; a snapshot without ``1_Pooling/config.json`` only logs a
    warning.
    """
    pooling = snapshot / "1_Pooling" / "config.json"
    if not pooling.exists():
        logger.warning("No 1_Pooling/config.json in %s; trusting the family defaults", snapshot)
        return
    cfg = json.loads(pooling.read_text())
    mode = cfg.get("pooling_mode")
    if mode is None:
        mode = "mean" if cfg.get("pooling_mode_mean_tokens") else None
    problems = []
    if mode != family.pooling_mode:
        problems.append(f"pooling_mode {mode!r} != {family.pooling_mode!r}")
    if bool(cfg.get("include_prompt", True)) != family.include_prompt:
        problems.append(f"include_prompt {cfg.get('include_prompt')} != {family.include_prompt}")
    if int(cfg.get("embedding_dimension", family.dimension)) != family.dimension:
        problems.append(
            f"embedding_dimension {cfg.get('embedding_dimension')} != {family.dimension}"
        )
    if family.normalized and not (snapshot / "2_Normalize").exists():
        problems.append("family is normalized but the snapshot has no 2_Normalize module")
    if problems:
        raise ValueError(f"Snapshot pooling config disagrees with the family: {problems}")


def copy_tokenizer(snapshot: Path, dest: Path) -> list[str]:
    """Copy the snapshot's tokenizer files verbatim into ``dest`` and return the copied names.

    Raises ``FileNotFoundError`` if the snapshot has no ``tokenizer.json``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_FILES:
        src = snapshot / name
        if src.exists():
            shutil.copy2(src, dest / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(f"{snapshot} has no tokenizer.json")
    return copied


class TokenTableExportWrapper(nn.Module):
    """``input_ids [B, L] int32 -> inputs_embeds [B, L, H]`` as ``dtype`` (scaled embeddings)."""

    def __init__(self, embed_tokens: nn.Module, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids).to(self.dtype)


def scaled_token_table(table: torch.Tensor, scale: float, dtype: torch.dtype) -> nn.Embedding:
    """Return an ``nn.Embedding`` of ``table * scale`` (computed in float32) as ``dtype``."""
    return nn.Embedding.from_pretrained((table.float() * scale).to(dtype))


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
    bw = max(1, int(image.budget**0.5))
    bh = max(1, image.budget // bw)
    generator = torch.Generator().manual_seed(image.budget)
    pixels = torch.rand((image.max_patches, image.patch_dim), generator=generator)
    pixels, positions = pad_to_budget(
        pixels[: bh * k * bw * k], patch_position_ids(bh * k, bw * k), image.max_patches
    )
    return pixels[None].to(dtype), positions[None].to(position_dtype)


def token_table_trace(max_length: int) -> tuple[tuple[torch.Tensor], tuple[dict[int, Any]]]:
    """Token-table trace args and dynamic shapes: batch-1 ids, length 2..``max_length``."""
    length = torch.export.Dim("length", min=2, max=max_length)
    return (torch.zeros((1, 64), dtype=torch.int32),), ({1: length},)


def export_token_table_program(
    embed_tokens: nn.Module,
    max_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    dtype: torch.dtype = torch.float32,
) -> AIProgram:
    """``token_table``: function ``main``, ids -> scaled embeddings, dynamic length (batch 1)."""
    from coreai_models.diffusion.gpu import export_stateless

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


def verify_image_config(snapshot: Path, family: EmbeddingFamily) -> None:
    """Raise ``ValueError`` if the snapshot's image config disagrees with ``family.image``.

    Checks the placeholder, begin-image and end-image token ids and the text hidden size in
    ``config.json``, and the image processor settings in ``processor_config.json``. Also raises
    if the family has no image tower.
    """
    image = family.image
    if image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")
    problems = []
    config_path = snapshot / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        for key, want in (
            ("image_token_id", image.placeholder_id),
            ("boi_token_id", image.boi_token_id),
            ("eoi_token_id", image.eoi_token_id),
        ):
            if key in cfg and int(cfg[key]) != want:
                problems.append(f"config.json {key} {cfg[key]} != {want}")
        text_hidden = (cfg.get("text_config") or {}).get("hidden_size")
        if text_hidden is not None and int(text_hidden) != image.soft_token_dim:
            problems.append(f"text hidden_size {text_hidden} != {image.soft_token_dim}")
    processor_path = snapshot / "processor_config.json"
    if processor_path.exists():
        proc = json.loads(processor_path.read_text()).get("image_processor") or {}
        checks = (
            ("patch_size", image.patch_size),
            ("pooling_kernel_size", image.pooling_kernel_size),
            ("max_soft_tokens", image.budget),
            ("do_normalize", image.normalize),
            ("resample", 3),  # PIL/torchvision BICUBIC
        )
        for key, want in checks:
            if key in proc and proc[key] != want:
                problems.append(f"processor {key} {proc[key]!r} != {want!r}")
        if "rescale_factor" in proc and abs(proc["rescale_factor"] - image.rescale_factor) > 1e-12:
            problems.append(f"processor rescale_factor {proc['rescale_factor']} != 1/255")
    if problems:
        raise ValueError(f"Snapshot image config disagrees with the family: {problems}")


def save_program(program: AIProgram, out: Path, family: EmbeddingFamily, component: str) -> None:
    from coreai_models.export import metadata as export_metadata

    program.save_asset(
        out / f"{component}.aimodel",
        export_metadata.build_aimodel_metadata(family.hf_model_id, component=component),
    )
