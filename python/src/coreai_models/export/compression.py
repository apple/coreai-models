# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Model compression utilities for PyTorch models.

This module provides utilities for quantizing and compressing PyTorch models
using the coreai-opt library, including calibration data preparation.
"""

import logging
from collections.abc import Callable

import torch
import torch.nn as nn

from coreai_models.export._constants import (
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
)

logger = logging.getLogger(__name__)

try:
    from coreai_opt.base_model_compressor import ExportBackend
    from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
    from coreai_opt.palettization.kmeans import (
        KMeansPalettizer,
    )
    from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

    _HAS_COREAI_OPT = True
except ImportError:
    _HAS_COREAI_OPT = False

try:
    from datasets import load_dataset
    from tqdm import tqdm

    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


def _require_coreai_opt() -> None:
    """Raise if coreai_opt is not installed."""
    if not _HAS_COREAI_OPT:
        raise ImportError(
            "coreai-opt is required for model compression. Install it with: pip install coreai-opt"
        )


def get_c4(
    tokenizer,  # type: ignore[no-untyped-def]
    max_sequence_length: int = 2048,
    num_calibration_samples: int = 16,
) -> list[torch.Tensor]:
    """
    Load calibration samples from the C4 dataset.

    Takes the first num_calibration_samples from C4 and tokenizes them.
    Samples longer than max_sequence_length are truncated.

    Args:
        tokenizer: HuggingFace tokenizer for encoding text.
        max_sequence_length: Maximum sequence length for calibration samples.
        num_calibration_samples: Number of calibration samples to load.

    Returns:
        List of tokenized samples, each of shape (1, seq_len) where
        seq_len <= max_sequence_length.
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "The 'datasets' and 'tqdm' packages are required for calibration data. "
            "Install them with: pip install datasets tqdm"
        )

    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    num_calibration_samples = min(num_calibration_samples, len(dataset))
    dataset = dataset[:num_calibration_samples]["text"]

    calibration_samples = []
    for prompt in dataset:
        tokens = tokenizer(prompt, return_tensors="pt")
        sample = tokens.input_ids[:, :max_sequence_length]
        calibration_samples.append(sample)

    return calibration_samples


def quantize_pytorch_model(
    model: nn.Module,
    inputs: tuple,
    dynamic_shapes: dict,
    quantization_config: dict,
    calibration_data_fn: Callable[[], list] | None = None,
    export_backend: object | None = None,
    mmap_dir: str | None = None,
) -> nn.Module:
    """
    Quantize a PyTorch model using PT2E quantization.

    Applies post-training quantization to a PyTorch model using the PyTorch 2 Export (PT2E)
    quantization framework. Supports weight quantization, activation quantization, and
    calibration-based quantization.

    Args:
        model: The PyTorch model to quantize.
        inputs: Example inputs for model preparation (used for torch.export).
        dynamic_shapes: Dynamic shape specifications for torch.export.
        quantization_config: Configuration dictionary matching the inner shape
            coreai-opt expects under `quantization_config`. Includes a
            `calibrate_activations` key (popped here before constructing the
            coreai-opt config).
        calibration_data_fn: Optional function that returns calibration data samples.
            Required when calibrate_activations is enabled.
        export_backend: Backend for the finalized quantized model.
            Defaults to ExportBackend.CoreAI if not specified.

    Returns:
        Quantized model ready for the specified export backend.

    Raises:
        ImportError: If coreai-opt is not installed.
        ValueError: If calibration_data_fn is not provided when calibrate_activations
            is enabled.
    """
    _require_coreai_opt()

    if export_backend is None:
        export_backend = ExportBackend.CoreAI

    run_calibration = quantization_config.pop("calibrate_activations", False)
    config = QuantizerConfig.from_dict({"quantization_config": quantization_config})

    # When doing activation quantization, run real calibration data through the
    # prepared model so the activation observers see representative ranges.
    # `inputs` follows the model forward contract: (input_ids, position_ids, k_cache, v_cache).
    if run_calibration:
        if calibration_data_fn is None:
            raise ValueError(
                "calibration_data_fn is required when activation quantization is enabled"
            )
        calibration_data = calibration_data_fn()
        device = next(model.parameters()).device

        cache_seq_len = inputs[2].shape[-2]
        # Match the dynamic-shape upper bound declared by the caller:
        #   position_ids.shape[1] <= cache_seq_len - 1   (see pipeline.py `seq_pos` Dim)
        # position_ids has length QUANT_TRACE_OFFSET + query_len, so:
        #   query_len <= cache_seq_len - QUANT_TRACE_OFFSET - 1
        max_calib_query_len = cache_seq_len - QUANT_TRACE_OFFSET - 1
        # The traced dynamic shape requires position_ids length >= QUANT_TRACE_QUERY_LEN,
        # i.e. query_len >= QUANT_TRACE_QUERY_LEN - QUANT_TRACE_OFFSET.
        min_calib_query_len = QUANT_TRACE_QUERY_LEN - QUANT_TRACE_OFFSET

        def _prep_calib_inputs(sample: torch.Tensor) -> tuple:
            sample = sample[:, :max_calib_query_len].to(device)
            position_ids = (
                torch.arange(QUANT_TRACE_OFFSET + sample.shape[1], dtype=torch.int32)
                .unsqueeze(0)
                .to(device)
            )
            zero_cache = tuple(
                torch.zeros(inp.shape, dtype=inp.dtype, device=device) for inp in inputs[2:]
            )
            return (sample, position_ids, *zero_cache)

        calibration_data = [s for s in calibration_data if s.shape[1] >= min_calib_query_len]
        if not calibration_data:
            raise ValueError(f"No calibration samples have length >= {min_calib_query_len} tokens")
        inputs = _prep_calib_inputs(calibration_data[0])

    logger.info(f"Quantization config: {config}")
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs=inputs, dynamic_shapes=dynamic_shapes)

    if run_calibration:
        if not _HAS_DATASETS:
            raise ImportError("tqdm is required for calibration progress reporting.")
        logger.info(f"Running calibration with {len(calibration_data) - 1} samples on {device}")
        with quantizer.calibration_mode(), torch.no_grad():
            for sample in tqdm(calibration_data[1:], desc="calibration"):
                prepared_model(*_prep_calib_inputs(sample))

    finalized_model = quantizer.finalize(
        prepared_model,
        backend=export_backend,
        mmap_dir=mmap_dir if quantizer._execution_mode == ExecutionMode.EAGER else None,
    )
    if isinstance(finalized_model, torch.fx.GraphModule):
        torch.ao.quantization.move_exported_model_to_eval(finalized_model)
    else:
        finalized_model.eval()

    return finalized_model


def palettize_pytorch_model(
    model: nn.Module,
    example_inputs: tuple,
    palettization_config: "dict | KMeansPalettizerConfig",
) -> nn.Module:
    """
    Palettize a PyTorch model using post-training palettization with coreai-opt.

    Args:
        model: The PyTorch model to palettize.
        example_inputs: Example inputs for model tracing (tuple matching
            model.forward() signature).
        palettization_config: Either a configuration dictionary (matching the
            inner shape coreai-opt expects under `kmeans_palettization_config`)
            or a prebuilt KMeansPalettizerConfig instance.

    Returns:
        Palettized model ready for export.

    Raises:
        ImportError: If coreai-opt is not installed.
    """
    _require_coreai_opt()

    logger.info("Palettizing model with coreai-opt")

    if isinstance(palettization_config, KMeansPalettizerConfig):
        config = palettization_config
    else:
        config = KMeansPalettizerConfig.from_dict(
            {"kmeans_palettization_config": palettization_config}
        )
    logger.info(f"Palettization config: {config}")

    palettizer = KMeansPalettizer(model, config)
    prepared_model = palettizer.prepare(example_inputs=example_inputs, num_workers=32)

    finalized_model = palettizer.finalize(prepared_model, backend=ExportBackend.CoreAI)

    logger.info("Palettization with coreai-opt complete")
    return finalized_model
