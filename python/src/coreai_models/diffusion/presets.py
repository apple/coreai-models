# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compression presets for diffusion model export.

Each preset is a named configuration consumed by the diffusion export pipeline.
Presets describe weight quantization applied to quantizable components (text
encoder and transformer). The VAE decoder is never quantized.

Configs are coreai-opt ``quantization_config`` dicts, applied to the PyTorch module
before torch export.

Usage::

    from coreai_models.diffusion.presets import get_preset, list_presets

    preset = get_preset("4bit-asym")
    names = list_presets()
"""

from typing import Any

DEFAULT_COMPRESSION_PRESET = "none"

# These norms end their forward with `self.weight * hidden_states`, and `torch.mul` is a
# registered eager op, so the global `weight` spec matches their rank-1 parameter. coreai-opt
# has no default axis for `torch.mul` and raises during prepare(), so these exclusions skip
# the norms instead.
#
# Entries are keyed by class, so one that a given pipeline never instantiates is a no-op.
# Norms reached through `F.layer_norm`, `F.group_norm` or `F.rms_norm` need no entry, since
# those ops stay outside the registered set.
_MODULE_TYPE_EXCLUSIONS: dict[str, Any] = {
    "diffusers.models.normalization.RMSNorm": None,
    "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": None,
    "transformers.models.umt5.modeling_umt5.UMT5LayerNorm": None,
}

# Weight-only, so the input and output specs stay None.
_WEIGHT_ONLY = {"op_input_spec": None, "op_output_spec": None}

# `symmetric_with_clipping` gives int4 the range [-7, 7]
_INT4_PER_BLOCK32 = {
    "dtype": "int4",
    "qscheme": "symmetric_with_clipping",
    "granularity": {"type": "per_block", "block_size": 32},
}
_INT4_PER_BLOCK32_ASYM = {**_INT4_PER_BLOCK32, "qscheme": "asymmetric"}

# per_channel means per output channel, which is axis 0.
_INT8_PER_CHANNEL = {
    "dtype": "int8",
    "qscheme": "symmetric_with_clipping",
    "granularity": {"type": "per_channel"},
}

PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "description": "Full precision (no quantization)",
        "config": None,
    },
    "4bit": {
        "description": "INT4 symmetric per-block (block_size=32)",
        "config": {
            "execution_mode": "eager",
            "global_config": {"op_state_spec": {"weight": _INT4_PER_BLOCK32}, **_WEIGHT_ONLY},
            "module_type_configs": _MODULE_TYPE_EXCLUSIONS,
        },
    },
    "4bit-asym": {
        "description": "INT4 asymmetric per-block (block_size=32)",
        "config": {
            "execution_mode": "eager",
            "global_config": {"op_state_spec": {"weight": _INT4_PER_BLOCK32_ASYM}, **_WEIGHT_ONLY},
            "module_type_configs": _MODULE_TYPE_EXCLUSIONS,
        },
    },
    "8bit": {
        "description": "INT8 per-channel, symmetric",
        "config": {
            "execution_mode": "eager",
            "global_config": {"op_state_spec": {"weight": _INT8_PER_CHANNEL}, **_WEIGHT_ONLY},
            "module_type_configs": _MODULE_TYPE_EXCLUSIONS,
        },
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a compression preset by name.

    Args:
        name: Preset name (e.g., ``"4bit"``, ``"none"``)

    Returns:
        Preset configuration dict

    Raises:
        KeyError: If the preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown diffusion compression preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> list[str]:
    """List all available diffusion compression preset names."""
    return sorted(PRESETS.keys())
