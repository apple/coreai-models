# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Export pipeline orchestration.

Ties together model loading, compression, variant-specific export,
MLIR quantization, and compilation into a single ``export_model`` call.
"""

import asyncio
import contextlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
from transformers import AutoConfig, AutoTokenizer

from coreai_models.export._constants import (
    IOS_DEFAULT_MAX_CONTEXT_LENGTH,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
)
from coreai_models.export.bundle import bundle_llm_asset
from coreai_models.export.compression import (
    get_c4,
    palettize_pytorch_model,
    quantize_pytorch_model,
)
from coreai_models.export.ios import export_ios_model
from coreai_models.export.macos import export_macos_model
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.presets import (
    DEFAULT_MACOS_COMPRESSION_PRESET,
    get_preset,
)
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.macos.cache import KVCache

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Everything needed to export a model."""

    hf_model_id: str
    variant: Literal["macOS", "iOS"] = "macOS"
    max_context_length: int | None = None
    compute_precision: str = "float16"
    compression: str = DEFAULT_MACOS_COMPRESSION_PRESET
    output_dir: str = "outputs"
    output_name: str | None = None
    num_layers: int | None = None
    overwrite: bool = False
    # Optional prebuilt coreai-opt config (KMeansPalettizerConfig or
    # QuantizerConfig) loaded from a user-provided YAML. When set, the pipeline
    # uses this directly and ignores `compression` for config resolution
    compression_config_object: Any = field(default=None, repr=False)


def _generate_output_name(config: ExportConfig) -> str:
    """Generate a filesystem-safe output name from config."""
    variant_suffix = "_dynamic" if config.variant == "macOS" else "_static"
    short_name = config.hf_model_id.split("/")[-1]
    base = re.sub(r"[^a-z0-9]+", "_", short_name.lower()).strip("_")
    # YAML-driven exports: `compression` is the YAML stem, which by convention
    # includes the model identity (e.g. `qwen3_0_6b_mixed_4bit_8bit`). Skip the
    # hf_id prefix only when the stem already starts with it, otherwise prepend
    # so generic recipes (e.g. `4bit.yaml`) don't collide across models.
    if config.compression_config_object is not None:
        stem = config.compression
        suffix = stem if stem == base or stem.startswith(f"{base}_") else f"{base}_{stem}"
        return f"{suffix}{variant_suffix}"
    suffix = f"{base}_{config.compression}" if config.compression != "none" else base
    return f"{suffix}{variant_suffix}"


def _resolve_precision(precision_str: str) -> torch.dtype:
    """Map a precision string to a torch dtype."""
    precision_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = precision_map.get(precision_str)
    if dtype is None:
        raise ValueError(
            f"Unsupported compute_precision '{precision_str}'. "
            f"Supported: {', '.join(precision_map.keys())}"
        )
    return dtype


def export_model(config_or_model_id: ExportConfig | str) -> str:
    """Export a HuggingFace model to Core AI format.

    This is the main public API. It orchestrates:
    1. Resolve model class from HuggingFace config
    2. Load model with HF weights
    3. Apply pre-export compression (torch quantization)
    4. Variant-specific export (macOS or iOS)
    5. Save as .aimodel

    Args:
        config_or_model_id: Either an ExportConfig or a HuggingFace model ID string.
            If a string, uses default settings (macOS, 4bit compression).

    Returns:
        Path to the exported .aimodel file.
    """
    if isinstance(config_or_model_id, str):
        config = ExportConfig(hf_model_id=config_or_model_id)
    else:
        config = config_or_model_id

    return asyncio.run(_async_export_model(config))


async def _async_export_model(config: ExportConfig) -> str:
    """Async implementation of export_model."""

    # ---- 1. Resolve model class ----
    hf_config = AutoConfig.from_pretrained(config.hf_model_id)
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError(
            f"Could not determine model_type from HuggingFace config for '{config.hf_model_id}'"
        )

    entry = get_model_entry(model_type)

    # Unwrap the per-modality sub-config (e.g. Gemma-3 wraps the text
    # model under `text_config`)
    if entry.hf_config_attr:
        hf_config = getattr(hf_config, entry.hf_config_attr)

    if config.variant == "iOS" and entry.ios_class is None:
        raise ValueError(f"Model '{model_type}' does not support iOS variant")

    model_class = entry.macos_class if config.variant == "macOS" else entry.ios_class

    # ---- 2. Load model ----
    target_dtype = _resolve_precision(config.compute_precision)

    # The model's native context window from its HuggingFace config. Any
    # user-provided override must not exceed it.
    native_max_ctx = getattr(hf_config, "max_position_embeddings", None)

    max_context_length = config.max_context_length
    if max_context_length is None and config.variant == "iOS":
        max_context_length = min(
            IOS_DEFAULT_MAX_CONTEXT_LENGTH, native_max_ctx or IOS_DEFAULT_MAX_CONTEXT_LENGTH
        )

    if (
        max_context_length is not None
        and native_max_ctx is not None
        and max_context_length > native_max_ctx
    ):
        raise ValueError(
            f"--max-context-length ({max_context_length}) exceeds the model's "
            f"max_position_embeddings ({native_max_ctx}). Choose a value <= {native_max_ctx}."
        )

    if max_context_length is not None:
        hf_config.max_position_embeddings = max_context_length
    if config.num_layers is not None:
        hf_config.num_hidden_layers = config.num_layers

    logger.info(f"Loading {config.hf_model_id} ({config.variant}, dtype={target_dtype})...")

    # Memory-efficient layer-by-layer loading + quantizer disk-checkpointing
    # is macOS-only for now. The iOS variant keeps the legacy full-RAM path
    # since its palettization flow has not been validated against streaming
    # weight loading.
    use_memory_efficient = config.variant == "macOS"
    temp_dir_ctx: contextlib.AbstractContextManager[str | None] = (
        tempfile.TemporaryDirectory(prefix="coreai_export_")
        if use_memory_efficient
        else contextlib.nullcontext(None)
    )

    with temp_dir_ctx as temp_dir:
        if use_memory_efficient:
            assert temp_dir is not None  # nullcontext yields None only when not memory-efficient
            layer_mmap_dir = os.path.join(temp_dir, "layers")
            os.makedirs(layer_mmap_dir, exist_ok=True)
            model = model_class.from_hf_memory_efficient(
                config.hf_model_id,
                max_context_length=max_context_length,
                target_dtype=target_dtype,
                mmap_path=layer_mmap_dir,
                num_layers=config.num_layers,
                hf_config_attr=entry.hf_config_attr,
                hf_state_dict_prefix=entry.hf_state_dict_prefix,
            )
        else:
            model = model_class.from_hf(
                config.hf_model_id,
                max_context_length=max_context_length,
                target_dtype=target_dtype,
                num_layers=config.num_layers,
            )
        model = model.eval()
        # ---- 3. Resolve compression preset ----
        if config.compression_config_object is not None:
            if isinstance(config.compression_config_object, KMeansPalettizerConfig):
                torch_palettization_config = config.compression_config_object
                torch_quantization_config = None
            else:
                torch_quantization_config = config.compression_config_object
                torch_palettization_config = None
        else:
            preset = get_preset(config.compression)
            torch_quantization_config = preset.get("torch_quantization_config")
            torch_palettization_config = preset.get("torch_palettization_config")

        assert not (
            torch_quantization_config is not None and torch_palettization_config is not None
        ), "Both a quantization and a palettization config were provided, this should never happen."

        # ---- 3a. Pre-export torch quantization (if configured) ----
        effective_max_ctx = max_context_length or getattr(
            hf_config, "max_position_embeddings", TRACE_KV_CACHE_SEQ_LEN
        )
        vocab_size = hf_config.vocab_size
        batch_size = 1
        if torch_quantization_config is not None:
            logger.info(f"Applying pre-export torch quantization (preset={config.compression})")

            input_ids = torch.randint(
                1, vocab_size, (batch_size, QUANT_TRACE_QUERY_LEN), dtype=torch.int32
            )
            position_ids = (
                torch.arange(QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32)
                .unsqueeze(0)
                .expand(batch_size, QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET)
            )

            saved_max_pos = hf_config.max_position_embeddings
            hf_config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
            k_cache, v_cache = KVCache.create_cache_tensors(hf_config, dtype=target_dtype)
            hf_config.max_position_embeddings = saved_max_pos

            quantization_inputs = (input_ids, position_ids, k_cache, v_cache)
            quantization_dynamic_shapes = {
                "input_ids": {1: torch.export.Dim("seq_ids", max=TRACE_KV_CACHE_SEQ_LEN - 2)},
                "position_ids": {
                    1: torch.export.Dim(
                        "seq_pos", min=QUANT_TRACE_QUERY_LEN, max=TRACE_KV_CACHE_SEQ_LEN - 1
                    )
                },
                "k_cache": None,
                "v_cache": None,
            }

            def get_calibration_data():  # type: ignore[no-untyped-def]
                tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
                return get_c4(tokenizer)

            quantizer_mmap_dir: str | None = None
            if use_memory_efficient:
                assert temp_dir is not None
                quantizer_mmap_dir = os.path.join(temp_dir, "quantized")
                os.makedirs(quantizer_mmap_dir, exist_ok=True)

            # Pass-through prebuilt QuantizerConfig objects.
            # copy dicts so we don't mutate the shared preset.
            quant_cfg = (
                torch_quantization_config
                if not isinstance(torch_quantization_config, dict)
                else dict(torch_quantization_config)
            )
            model = quantize_pytorch_model(
                model,
                quantization_inputs,
                quantization_dynamic_shapes,
                quant_cfg,
                calibration_data_fn=get_calibration_data,
                mmap_dir=quantizer_mmap_dir,
            )
        if torch_palettization_config is not None:
            assert config.variant == "iOS", "palettization is only supported for iOS variant."

            query_len = 8
            input_ids = torch.randint(1, vocab_size, (batch_size, query_len), dtype=torch.int32)
            position_ids = (
                torch.arange(query_len).to(torch.uint16).unsqueeze(0).expand(batch_size, query_len)
            )
            in_step = torch.zeros((1,), dtype=torch.int32)
            causal_mask = torch.zeros(1, effective_max_ctx, 1, query_len, dtype=torch.float16)
            if hasattr(hf_config, "head_dim") and isinstance(hf_config.head_dim, int):
                head_dim = hf_config.head_dim
            else:
                head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            key_cache = torch.zeros(
                hf_config.num_hidden_layers,
                1,  # batch_size
                hf_config.num_key_value_heads * head_dim,
                1,
                effective_max_ctx,
                dtype=torch.float16,
            )
            value_cache = key_cache.clone()
            palettization_inputs = (
                input_ids,
                position_ids,
                in_step,
                causal_mask,
                key_cache,
                value_cache,
            )
            model = palettize_pytorch_model(model, palettization_inputs, torch_palettization_config)

        # ---- 4. Variant-specific export ----
        if config.variant == "macOS":
            coreai_program = export_macos_model(model, hf_config, config)
        else:
            coreai_program = await export_ios_model(model, hf_config, config)

        del model

        # ---- 5. Save inside bundle directory ----
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = config.output_name or _generate_output_name(config)
        bundle_path = output_dir / output_name
        bundle_path.mkdir(parents=True, exist_ok=True)
        aimodel_path = bundle_path / f"{output_name}.aimodel"

        if aimodel_path.exists():
            if config.overwrite:
                import shutil

                shutil.rmtree(aimodel_path)
            else:
                raise FileExistsError(
                    f"{aimodel_path} already exists. Use --overwrite to replace it."
                )

        logger.info(f"Saving model to {aimodel_path}...")
        # ``AIProgram.save_asset`` is synchronous and does blocking disk I/O,
        # so offload it to a worker thread to keep the event loop responsive.
        metadata = build_aimodel_metadata(config.hf_model_id)
        await asyncio.to_thread(coreai_program.save_asset, aimodel_path, metadata)

        bundle_llm_asset(
            bundle_path=bundle_path,
            hf_model_id=config.hf_model_id,
            hf_config=hf_config,
            compression=config.compression,
            name=output_name,
        )

    logger.info(f"Export complete: {bundle_path}")
    return str(bundle_path)
