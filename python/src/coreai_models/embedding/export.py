# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export a retrieval-embedding model to a ``kind: "embedding"`` bundle.

    uv run coreai.embedding.export embeddinggemma-2               # macOS GPU bundle: text + image
    uv run coreai.embedding.export embeddinggemma-2 --platform iOS  # Neural Engine bundle
    uv run coreai.embedding.export embeddinggemma-2 --skip-vision  # text only

The bundle contract (``metadata.json`` and the component assets) is defined in
``coreai_models.embedding.contract``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import (
    ANE_ENCODER_INPUT_NAMES,
    EMBEDDING_FAMILIES,
    ENCODER_COMPONENT,
    ENCODER_INPUT_NAMES,
    IMAGE_COMPONENT,
    IMAGE_INPUT_NAMES,
    IMAGE_OUTPUT_NAME,
    OUTPUT_NAME,
    PLATFORMS,
    TEXT_COMPONENT,
    TEXT_INPUT_NAMES,
    TOKEN_TABLE_COMPONENT,
    TOKEN_TABLE_INPUT_NAMES,
    TOKEN_TABLE_OUTPUT_NAME,
    EmbeddingFamily,
    EmbeddingPrecisionError,
    FunctionSpec,
    ImageTowerFacts,
    build_bundle_metadata,
    bundle_assets,
    bundle_name,
    check_compute_precision,
    consolidate_prompts,
    default_precision,
    embedding_block,
    encoder_functions,
    family_for_model,
    text_functions,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram

logger = logging.getLogger(__name__)

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
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _find_repo_root() -> Path | None:
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return None


def _default_output_dir() -> str:
    root = _find_repo_root()
    return str(root / "exports") if root is not None else "exports"


def resolve_snapshot(hf_model_id: str, model_dir: str | None = None) -> tuple[Path, str]:
    """Return ``(snapshot_dir, revision_sha)`` for a local dir or a hub download.

    The SHA is the HF cache snapshot directory name (``.../snapshots/<sha>``); a plain directory
    outside the cache yields ``""``.
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
    """Fail closed if the snapshot's sentence-transformers pooling disagrees with the family."""
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
    """Copy the tokenizer files verbatim (no re-serialization through a tokenizer class)."""
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


class TextEmbeddingExportWrapper(nn.Module):
    """``(input_ids, attention_mask) -> embedding`` with float32 output.

    The wrapped encoder owns pooling (masked mean incl. prompt), projection and L2 normalization.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).to(torch.float32)


class TokenTableExportWrapper(nn.Module):
    """``input_ids [B, L] int32 -> inputs_embeds [B, L, H]`` as ``dtype`` (scaled embeddings)."""

    def __init__(self, embed_tokens: nn.Module, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids).to(self.dtype)


def scaled_token_table(table: torch.Tensor, scale: float, dtype: torch.dtype) -> nn.Embedding:
    """``table * scale`` stored as ``dtype``. Scaling in float32 first keeps the cast exact."""
    return nn.Embedding.from_pretrained((table.float() * scale).to(dtype))


class InputsEmbedsEncoderWrapper(nn.Module):
    """``(inputs_embeds, attention_mask) -> embedding`` for merged (e.g. image) inputs."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        return self.encoder(
            inputs_embeds=inputs_embeds.to(dtype), attention_mask=attention_mask
        ).to(torch.float32)


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


class ImageEncoderExportWrapper(nn.Module):
    """``(pixel_values f32, position_ids int32) -> soft_tokens f32`` at the module's dtype."""

    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype
        return self.image_encoder(pixel_values.to(dtype), position_ids).to(torch.float32)


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


def image_reference_inputs(image: ImageTowerFacts) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-encoder trace inputs: a full block grid padded to the budget (int32 positions)."""
    k = image.pooling_kernel_size
    n = image.budget
    # Square-ish grid of n soft tokens (blocks) -> (bw*k) x (bh*k) patches, row-major.
    bw = max(1, int(n**0.5))
    bh = max(1, n // bw)
    pw, ph = bw * k, bh * k
    ys, xs = torch.meshgrid(torch.arange(ph), torch.arange(pw), indexing="ij")
    real = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(torch.int32)
    positions = torch.full((1, image.max_patches, 2), -1, dtype=torch.int32)
    positions[0, : real.shape[0]] = real
    generator = torch.Generator().manual_seed(image.budget)
    pixels = torch.rand((1, image.max_patches, image.patch_dim), generator=generator)
    pixels[0, real.shape[0] :] = 0.0
    return pixels, positions


def ane_text_reference_inputs(
    length: int, hidden: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neural Engine encoder trace inputs (all valid tokens): embeds, zero bias, uniform pooling."""
    generator = torch.Generator().manual_seed(length)
    embeds = torch.randn((1, length, hidden), generator=generator).to(torch.float16)
    bias = torch.zeros((1, length, 1, length), dtype=torch.float16)
    pool = torch.full((1, length), 1.0 / length, dtype=torch.float16)
    return embeds, bias, pool


def ane_image_reference_inputs(image: ImageTowerFacts) -> tuple[torch.Tensor, torch.Tensor]:
    """Neural Engine image trace inputs (float16): a block grid padded to the budget (-1 pad)."""
    k = image.pooling_kernel_size
    a = int(image.budget**0.5)
    gh, gw = k * a, k * (image.budget // a)
    generator = torch.Generator().manual_seed(image.budget)
    pixels = torch.rand((1, image.max_patches, image.patch_dim), generator=generator)
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    positions = torch.full((1, image.max_patches, 2), -1.0)
    positions[0, : gh * gw] = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1).float()
    return pixels.to(torch.float16), positions.to(torch.float16)


@dataclass
class TraceSpec:
    """One function of a multi-function program: example args, IO names, dynamic dims."""

    name: str
    args: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_shapes: tuple[dict[int, Any] | None, ...] | None = None


def _add_functions(
    converter: Any, traces: list[tuple[nn.Module, TraceSpec]], decomp: dict[Any, Any]
) -> None:
    """``torch.export`` every ``(module, trace)`` and add it to ``converter`` as one function."""
    for module, trace in traces:
        module.eval()
        start = time.time()
        with torch.no_grad():
            exported = torch.export.export(
                module, args=trace.args, dynamic_shapes=trace.dynamic_shapes
            )
        converter.add_exported_program(
            exported.run_decompositions(decomp),
            input_names=list(trace.input_names),
            output_names=list(trace.output_names),
            entrypoint_name=trace.name,
        )
        logger.info(
            "traced %s %s in %.1fs",
            trace.name,
            [tuple(a.shape) for a in trace.args],
            time.time() - start,
        )


def convert_ane_functions(
    traces: list[tuple[nn.Module, TraceSpec]],
    *,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert every ``(module, trace)`` into one multi-function Neural Engine program.

    Uses the default decompositions plus the repo's custom lowerings, as ``export/ios.py`` does.
    """
    import coreai_torch

    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    mode = coreai_torch.TorchConverter.Mode
    converter = coreai_torch.TorchConverter(mode=mode.DEBUG if include_debug_info else mode.RELEASE)
    register_custom_torch_lowering(converter)
    _add_functions(converter, traces, torch.export.default_decompositions())
    return converter.to_coreai()


def export_text_encoder_program(
    wrapper: nn.Module,
    functions: list[FunctionSpec],
    vocab_size: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """GPU ``text_encoder``: one static (batch, length) function per spec, weights shared."""
    from coreai_models.diffusion.gpu import export_multifunction

    # Static (batch, length) traces; the host pads to the next listed batch.
    return export_multifunction(
        [(spec.name, wrapper, reference_inputs(spec, vocab_size)) for spec in functions],
        TEXT_INPUT_NAMES,
        (OUTPUT_NAME,),
        include_debug_info,
    )


def export_token_table_program(
    embed_tokens: nn.Module,
    max_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    dtype: torch.dtype = torch.float32,
) -> AIProgram:
    """``token_table``: function ``main``, ids -> scaled embeddings, dynamic length (batch 1)."""
    from coreai_models.diffusion.gpu import export_stateless

    # Placeholder ids look up like any id; the host overwrites those rows with soft tokens.
    length = torch.export.Dim("length", min=2, max=max_length)
    ids = torch.zeros((1, 64), dtype=torch.int32)
    return export_stateless(
        TokenTableExportWrapper(embed_tokens, dtype),
        (ids,),
        TOKEN_TABLE_INPUT_NAMES,
        (TOKEN_TABLE_OUTPUT_NAME,),
        dynamic_shapes=({1: length},),
        include_debug_info=include_debug_info,
    )


def export_shared_text_program(
    encoder: nn.Module,
    text_fns: list[FunctionSpec],
    encoder_fns: list[FunctionSpec],
    max_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """GPU ``text_encoder`` that also holds the ``token_table`` and ``encoder`` functions.

    Text functions take ids; ``main`` maps ids to ``inputs_embeds`` (dynamic length up to
    ``max_length``); ``embeds_*`` functions take ``inputs_embeds``. The functions share one copy
    of the table and the transformer.
    """
    import coreai_torch

    vocab_size = int(encoder.config.vocab_size)
    hidden = int(encoder.config.hidden_size)
    length = torch.export.Dim("length", min=2, max=max_length)
    text = TextEmbeddingExportWrapper(encoder)
    traces = [
        (
            text,
            TraceSpec(
                spec.name, reference_inputs(spec, vocab_size), TEXT_INPUT_NAMES, (OUTPUT_NAME,)
            ),
        )
        for spec in text_fns
    ]
    traces.append(
        (
            TokenTableExportWrapper(encoder.embed_tokens),
            TraceSpec(
                "main",
                (torch.zeros((1, 64), dtype=torch.int32),),
                TOKEN_TABLE_INPUT_NAMES,
                (TOKEN_TABLE_OUTPUT_NAME,),
                ({1: length},),
            ),
        )
    )
    mode = coreai_torch.TorchConverter.Mode
    converter = coreai_torch.TorchConverter(mode=mode.DEBUG if include_debug_info else mode.RELEASE)
    decomp = coreai_torch.get_decomp_table()
    _add_functions(converter, traces, decomp)
    with without_token_table(encoder) as inner:
        embeds = InputsEmbedsEncoderWrapper(inner)
        _add_functions(
            converter,
            [
                (
                    embeds,
                    TraceSpec(
                        spec.name,
                        embeds_reference_inputs(spec, hidden),
                        ENCODER_INPUT_NAMES,
                        (OUTPUT_NAME,),
                    ),
                )
                for spec in encoder_fns
            ],
            decomp,
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
            EncoderBandWrapper(encoder, spec.length, sliding_window),
            TraceSpec(
                spec.name,
                ane_text_reference_inputs(spec.length, hidden),
                ANE_ENCODER_INPUT_NAMES,
                (OUTPUT_NAME,),
            ),
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
    trace = TraceSpec(
        "main", ane_image_reference_inputs(image), IMAGE_INPUT_NAMES, (IMAGE_OUTPUT_NAME,)
    )
    return convert_ane_functions([(image_encoder, trace)], include_debug_info=include_debug_info)


def verify_image_config(snapshot: Path, family: EmbeddingFamily) -> None:
    """Fail closed if the snapshot's merge ids / image processor disagree with the family facts."""
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


def _load_text_encoder(family: EmbeddingFamily, snapshot: Path, dtype: torch.dtype) -> nn.Module:
    if family.key == "embeddinggemma-2":
        from coreai_models.models.macos.embedding_gemma2 import load_embedding_gemma2_text_encoder

        return load_embedding_gemma2_text_encoder(snapshot, dtype=dtype)
    raise NotImplementedError(f"No text-encoder loader for family {family.key!r}")


def _load_image_encoder(
    family: EmbeddingFamily, snapshot: Path, dtype: torch.dtype, max_soft_tokens: int
) -> nn.Module:
    if family.key == "embeddinggemma-2":
        from coreai_models.models.macos.embedding_gemma2_vision import EmbeddingGemma2ImageEncoder

        return EmbeddingGemma2ImageEncoder.from_pretrained(
            str(snapshot), dtype=dtype, max_soft_tokens=max_soft_tokens
        )
    raise NotImplementedError(f"No image-encoder loader for family {family.key!r}")


def _save(program: AIProgram, out: Path, family: EmbeddingFamily, component: str) -> None:
    from coreai_models.export import metadata as export_metadata

    program.save_asset(
        out / f"{component}.aimodel",
        export_metadata.build_aimodel_metadata(family.hf_model_id, component=component),
    )


def _export_gpu_programs(
    family: EmbeddingFamily,
    snapshot: Path,
    out: Path,
    *,
    with_image: bool,
    compute_precision: str,
    include_debug_info: bool,
) -> None:
    dtype = _DTYPES[compute_precision]
    encoder = _load_text_encoder(family, snapshot, dtype)
    if with_image:
        program = export_shared_text_program(
            encoder,
            text_functions(family, "macOS"),
            encoder_functions(family, "macOS"),
            family.model_max_tokens,
            include_debug_info,
        )
    else:
        program = export_text_encoder_program(
            TextEmbeddingExportWrapper(encoder),
            text_functions(family, "macOS"),
            int(encoder.config.vocab_size),
            include_debug_info,
        )
    _save(program, out, family, TEXT_COMPONENT)
    del program, encoder
    if not with_image:
        return
    assert family.image is not None
    image_encoder = _load_image_encoder(family, snapshot, dtype, family.image.budget)
    _save(
        export_image_encoder_program(image_encoder, family.image, include_debug_info),
        out,
        family,
        IMAGE_COMPONENT,
    )


def _export_ane_programs(
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
    encoder, table = load_embedding_gemma2_text_encoder(snapshot, dtype=torch.float16)
    token_table = scaled_token_table(table, encoder.config.hidden_size**0.5, torch.float16)
    del table
    _save(
        export_token_table_program(
            token_table, family.model_max_tokens, include_debug_info, dtype=torch.float16
        ),
        out,
        family,
        TOKEN_TABLE_COMPONENT,
    )
    del token_table
    _save(
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
        _save(
            export_ane_image_encoder_program(image_encoder, family.image, include_debug_info),
            out,
            family,
            IMAGE_COMPONENT,
        )


def export_embedding_bundle(
    family: EmbeddingFamily,
    output_dir: Path,
    *,
    platform: str = "macOS",
    with_image: bool = True,
    compute_precision: str | None = None,
    overwrite: bool = False,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """Export one bundle to ``output_dir`` and return its ``metadata.json`` content.

    macOS writes a GPU bundle (``text_encoder``, plus ``image_encoder`` / ``token_table`` /
    ``encoder`` with images); iOS writes the Neural Engine bundle (``token_table``, ``encoder``,
    plus ``image_encoder``). Both carry the tokenizer and a ``metadata.json``.
    """
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {list(PLATFORMS)}, got {platform!r}")
    precision = check_compute_precision(
        family, compute_precision or default_precision(family, platform), platform=platform
    )  # before touching disk
    if with_image and family.image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        snapshot, sha = resolve_snapshot(family.hf_model_id, model_dir)
        if not sha:
            raise ValueError(
                f"Could not read the snapshot revision from {snapshot}; pass a Hugging Face "
                "cache snapshot directory (.../snapshots/<sha>), whose name is the revision the "
                "bundle records"
            )
        verify_pooling_config(snapshot, family)
        if with_image:
            verify_image_config(snapshot, family)
        snapshot_prompts, default_prompt = read_prompts(snapshot)
        prompts = consolidate_prompts(family, snapshot_prompts)
        copy_tokenizer(snapshot, out / "tokenizer")

        if platform == "iOS":
            _export_ane_programs(
                family,
                snapshot,
                out,
                with_image=with_image,
                include_debug_info=include_debug_info,
            )
        else:
            _export_gpu_programs(
                family,
                snapshot,
                out,
                with_image=with_image,
                compute_precision=precision,
                include_debug_info=include_debug_info,
            )

        block = embedding_block(
            family,
            platform=platform,
            with_image=with_image,
            revision=sha[:16],
            prompts=prompts,
            default_prompt_name=default_prompt,
            compute_precision=precision,
        )
        metadata = build_bundle_metadata(
            family,
            block,
            name=bundle_name(family, platform, with_image),
            assets=bundle_assets(platform, with_image),
            hf_revision=sha,
        )
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # a failed export leaves no partial bundle
        raise
    logger.info("wrote %s", out)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.embedding.export",
        description="Export a retrieval-embedding model to a kind=embedding bundle.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry short name (e.g. embeddinggemma-2) or HuggingFace model ID",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORMS),
        default="macOS",
        help="Target platform. macOS: GPU bundle (float32 or bfloat16). iOS: Neural Engine "
        "bundle (float16); also runs on a Mac.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Export the text tower only (skip the image encoder).",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Default: float32 on macOS, float16 on iOS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent directory for the bundle (default: <repo_root>/exports).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the export plan and exit.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _print_models() -> None:
    for family in EMBEDDING_FAMILIES.values():
        towers = "text,image" if family.image else "text"
        print(f"{family.key:24s} {family.hf_model_id:32s} {towers:12s} {','.join(PLATFORMS)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models()
        return 0
    if args.model is None:
        parser.error("a model is required (e.g. embeddinggemma-2); see --list-models")
    try:
        family = family_for_model(args.model)
    except KeyError:
        parser.error(
            f"unknown model {args.model!r}. Run `uv run coreai.embedding.export --list-models`."
        )
    platform = args.platform
    with_image = not args.skip_vision
    try:
        precision = check_compute_precision(
            family,
            args.compute_precision or default_precision(family, platform),
            platform=platform,
        )
    except EmbeddingPrecisionError as error:
        parser.error(str(error))
    out = Path(args.output_dir or _default_output_dir()) / bundle_name(family, platform, with_image)
    if args.dry_run:
        assets = bundle_assets(platform, with_image)
        towers = "text,image" if with_image else "text"
        print(f"platform:  {platform}")
        print(f"precision: {precision}")
        print(f"towers:    {towers}")
        print(f"assets:    {', '.join(dict.fromkeys(assets.values()))}")
        if platform == "macOS":
            print(f"text functions:    {[f.name for f in text_functions(family, platform)]}")
        if with_image or platform == "iOS":
            print(f"encoder functions: {[f.name for f in encoder_functions(family, platform)]}")
        print(f"output:    {out}")
        return 0
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    export_embedding_bundle(
        family,
        out,
        platform=platform,
        with_image=with_image,
        compute_precision=precision,
        overwrite=args.overwrite,
        include_debug_info=args.include_debug_info,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
