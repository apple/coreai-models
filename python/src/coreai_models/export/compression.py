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
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from coreai_models._constants import (
    MAIN_GRAPH_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
)
from coreai_models.models.base import BaseForCausalLM, TraceSpec

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


def _require_coreai_opt() -> None:
    """Raise if coreai_opt is not installed."""
    if not _HAS_COREAI_OPT:
        raise ImportError(
            "coreai-opt is required for model compression. Install it with: pip install coreai-opt"
        )


def is_compression_mode_graph(quantization_config: dict) -> bool:
    """Whether ``quantization_config`` selects coreai-opt's graph execution mode.

    Args:
        quantization_config: coreai-opt's ``quantization_config`` dict.
    """
    _require_coreai_opt()
    execution_mode = quantization_config.get("execution_mode")
    return execution_mode is not None and ExecutionMode(execution_mode) == ExecutionMode.GRAPH


def split_compression_config(
    compression_config_object: Any,
) -> "tuple[dict | None, KMeansPalettizerConfig | None]":
    """Route a prebuilt coreai-opt config to its quantization or palettization slot.

    Args:
        compression_config_object: A config loaded from a user-provided YAML.

    Returns:
        ``(quantization_config, palettization_config)``, exactly one of which is
        non-``None``.
    """
    _require_coreai_opt()
    if isinstance(compression_config_object, KMeansPalettizerConfig):
        return None, compression_config_object
    return compression_config_object, None


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
    cache_seq_len: int,
    state_indices: Sequence[int],
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
        cache_seq_len: Sequence-dim length the caches in ``inputs`` were traced at,
            used to bound the calibration query length.
        state_indices: Positions in ``inputs`` that are state and must be reset
            between calibration samples.
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
    #
    # `inputs[0]` must be input_ids and `inputs[1]` position_ids -- token-based
    # calibration cannot do anything else. Which of the rest are state comes from
    # `state_indices`; a non-state input keeps its traced tensor, which is only correct
    # if its shape does not depend on query length.
    if run_calibration:
        if calibration_data_fn is None:
            raise ValueError(
                "calibration_data_fn is required when activation quantization is enabled"
            )
        calibration_data = calibration_data_fn()
        device = next(model.parameters()).device

        reset_positions = set(state_indices)
        for pos in reset_positions:
            if pos < 2:
                raise ValueError(
                    "States cannot occupy the first two input positions. "
                    "Those must be reserved for input_ids and position_ids"
                )
            if pos >= len(inputs):
                raise IndexError(
                    f"State index out of bounds, got {pos}, while the number of inputs is "
                    f"{len(inputs)}"
                )

        # Match the caller's declared bound: position_ids.shape[1] <= cache_seq_len - 1
        # (the `seq_pos` Dim in `BaseForCausalLM.build_dynamic_shapes`).
        # position_ids has length QUANT_TRACE_OFFSET + query_len, so:
        #   query_len <= cache_seq_len - QUANT_TRACE_OFFSET - 1
        max_calib_query_len = cache_seq_len - QUANT_TRACE_OFFSET - 1
        # The traced dynamic shape requires position_ids length >= QUANT_TRACE_QUERY_LEN,
        # i.e. query_len >= QUANT_TRACE_QUERY_LEN - QUANT_TRACE_OFFSET.
        min_calib_query_len = QUANT_TRACE_QUERY_LEN - QUANT_TRACE_OFFSET

        def _prep_calib_inputs(sample: torch.Tensor) -> tuple:
            prepared = list(inputs)
            prepared[0] = sample[:, :max_calib_query_len].to(device)
            prepared[1] = (
                torch.arange(QUANT_TRACE_OFFSET + prepared[0].shape[1], dtype=torch.int32)
                .unsqueeze(0)
                .to(device)
            )
            for i in range(2, len(prepared)):
                inp = inputs[i]
                if i in reset_positions:
                    prepared[i] = torch.zeros(inp.shape, dtype=inp.dtype, device=device)
                elif isinstance(inp, torch.Tensor):
                    prepared[i] = inp.to(device)
            return tuple(prepared)

        calibration_data = [s for s in calibration_data if s.shape[1] >= min_calib_query_len]
        if not calibration_data:
            raise ValueError(f"No calibration samples have length >= {min_calib_query_len} tokens")
        inputs = _prep_calib_inputs(calibration_data[0])

    logger.info(f"Quantization config: {config}")
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs=inputs, dynamic_shapes=dynamic_shapes)

    if run_calibration:
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


def quantize_for_export(
    model: BaseForCausalLM,
    config,
    target_dtype: torch.dtype,
    quantization_config: dict,
    calibration_data_fn: Callable[[], list] | None = None,
    mmap_dir: str | None = None,
) -> nn.Module:
    """Apply pre-export torch quantization using the model's own graph contract.

    Builds the calibration trace from the export hooks rather than hardcoding a forward
    signature, so a model with extra inputs or states calibrates without the caller
    knowing about them, and activation calibration resets exactly the states.

    Args:
        model: The loaded model, in eval mode.
        config: The config the model was built from.
        target_dtype: Dtype for the trace's cache tensors.
        quantization_config: Inner coreai-opt ``quantization_config`` dict.
        calibration_data_fn: Calibration samples; required when the recipe enables
            ``calibrate_activations``.
        mmap_dir: Directory for the quantizer's disk checkpointing.
    """
    spec = TraceSpec(max_context_length=TRACE_KV_CACHE_SEQ_LEN)
    reference_inputs = model.build_reference_inputs(config, target_dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    # Same check the export path runs, so a bad contract fails identically on both.
    model.validate_export_contract(reference_inputs, dynamic_shapes)

    if MAIN_GRAPH_NAME not in reference_inputs:
        raise ValueError(
            f"{type(model).__name__} exports graphs {sorted(reference_inputs)}, not a "
            f"single {MAIN_GRAPH_NAME!r} graph. Torch quantization is only implemented "
            "for single-graph (macOS) models; iOS uses palettization."
        )
    graph_inputs = reference_inputs[MAIN_GRAPH_NAME]
    keys = list(graph_inputs)
    if quantization_config.get("calibrate_activations") and keys[:2] != [
        "input_ids",
        "position_ids",
    ]:
        raise ValueError(
            f"{type(model).__name__}: activation calibration feeds tokenized samples as "
            f"input_ids and rebuilds position_ids, so those must be the first two "
            f"parameters of forward; got {tuple(keys[:2])}."
        )

    # Which *positions* are state cannot be read off the contract: the name lists carry
    # only relative order. So this assumes the declared inputs precede the states, which
    # holds for the macOS graph -- the only graph calibration runs on. A model that
    # interleaved a non-state parameter after a cache would need them passed in.
    n_inputs = len(model.export_input_names()[MAIN_GRAPH_NAME])
    state_indices = tuple(range(n_inputs, len(keys)))

    return quantize_pytorch_model(
        model,
        model.reference_inputs_as_args(graph_inputs),
        dynamic_shapes[MAIN_GRAPH_NAME],
        quantization_config,
        calibration_data_fn=calibration_data_fn,
        mmap_dir=mmap_dir,
        cache_seq_len=spec.cache_seq_len,
        state_indices=state_indices,
    )


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
