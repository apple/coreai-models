# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Quantize a registered LLM in eager torch and save its state_dict as safetensors.

Runs the first half of ``coreai.llm.export`` — load, then coreai-opt's eager
quantization with the preset's config — and stops before torch.export. The
finalized model's ``state_dict()`` is saved as-is, so each quantized weight
appears under coreai-opt's parametrization keys::

    <module>.parametrizations.weight.original          zero-size placeholder
    <module>.parametrizations.weight.0.quantized_data  int8 codes
    <module>.parametrizations.weight.0.scale           float, one per block
    <module>.parametrizations.weight.0.zero_point      int8

The output directory is laid out like a Hugging Face snapshot — ``model.safetensors``
beside the model's ``config.json`` — so a loader that takes a snapshot reads it.

Usage::

    .venv/bin/python scripts/save_quantized_checkpoint.py Qwen/Qwen3-0.6B \\
        --output ../coreai-nn/coreai_nn_models/weights/qwen3_0_6b_4bit
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_model
from transformers import AutoConfig

from coreai_models.export.compression import quantize_for_export
from coreai_models.export.presets import get_preset
from coreai_models.model_registry import try_lookup_preset_by_hf_id
from coreai_models.models.registry import get_model_entry

logger = logging.getLogger(__name__)


def save_quantized_checkpoint(
    hf_model_id: str, output: Path, compression: str | None = None
) -> Path:
    """Quantize *hf_model_id* as its macOS export would, and save it under *output*."""
    preset = try_lookup_preset_by_hf_id(hf_model_id, variant="macOS")
    if preset is None:
        raise ValueError(f"No macOS preset is registered for {hf_model_id}")
    compression = compression or preset.compression
    quant_cfg = copy.deepcopy(get_preset(compression).get("torch_quantization_config"))
    if quant_cfg is None or quant_cfg.get("execution_mode") != "eager":
        raise ValueError(
            f"Compression '{compression}' is not an eager torch quantization"
        )
    target_dtype = getattr(torch, preset.compute_precision or "float16")

    hf_config = AutoConfig.from_pretrained(hf_model_id)
    entry = get_model_entry(hf_config.model_type)
    if entry.hf_config_attr:
        hf_config = getattr(hf_config, entry.hf_config_attr)
    if preset.max_context_length is not None:
        hf_config.max_position_embeddings = preset.max_context_length

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.safetensors"
    with tempfile.TemporaryDirectory(prefix="coreai_quantize_") as temp_dir:
        layers_dir = os.path.join(temp_dir, "layers")
        quantized_dir = os.path.join(temp_dir, "quantized")
        os.makedirs(layers_dir)
        os.makedirs(quantized_dir)

        logger.info(f"Loading {hf_model_id} at {target_dtype}...")
        model = entry.macos_class.from_hf_memory_efficient(
            hf_model_id,
            max_context_length=preset.max_context_length,
            target_dtype=target_dtype,
            mmap_path=layers_dir,
            hf_config_attr=entry.hf_config_attr,
            hf_state_dict_prefix=entry.hf_state_dict_prefix,
        )
        model.eval()

        logger.info(f"Quantizing with '{compression}'...")
        metadata = {
            "hf_model_id": hf_model_id,
            "compression": compression,
            "quantization_config": json.dumps(quant_cfg),
            "dtype": str(target_dtype),
            **{
                package: importlib.metadata.version(package)
                for package in ("coreai-models", "coreai-opt", "coreai-torch")
            },
        }
        with torch.no_grad():
            model = quantize_for_export(
                model, hf_config, target_dtype, quant_cfg, mmap_dir=quantized_dir
            )

        # Before the temporary directory goes: the quantized tensors are maps of
        # files in it. `save_model` rather than `save_file`, because the tied
        # embedding and head share one dequantize module, which `save_file` refuses.
        save_model(model, str(checkpoint), metadata=metadata)

    shutil.copy(hf_hub_download(hf_model_id, "config.json"), output / "config.json")
    logger.info(f"Saved {checkpoint} ({checkpoint.stat().st_size / 2**30:.2f} GiB)")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "hf_model_id", help="HuggingFace model ID, e.g. Qwen/Qwen3-0.6B"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--compression", help="Compression preset; defaults to the model's macOS preset"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    save_quantized_checkpoint(args.hf_model_id, args.output, args.compression)


if __name__ == "__main__":
    main()
