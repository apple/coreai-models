# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Central registry of quantization presets for LLM model export.

Each preset is a named configuration that maps directly to the fields
consumed by the export pipeline:

- ``torch_quantization_config``: PyTorch-level pre-export quantization (applied before torch export)
- ``torch_palettization_config``: PyTorch-level pre-export palettization (applied before torch
  export)

Usage::

    from coreai_models.export.presets import get_preset, list_presets

    # Get a preset by name
    preset = get_preset("4bit")

    # List all available presets
    names = list_presets()
"""

from typing import Any

# Module-type exclusions shared across torch quantization presets.
# These modules should not be quantized because they use specialized ops.
_TORCH_MODULE_EXCLUSIONS = {
    "coreai_models.primitives.macos.sdpa.SDPA": None,
    "coreai_models.primitives.macos.rope.RoPE": None,
    "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
    "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
}

# Embedding modules excluded from iOS palettization.
_IOS_PALETTIZATION_EMBEDDING_EXCLUSIONS = {
    "torch.nn.modules.sparse.Embedding": None,
    "coreai_models.primitives.ios.embedding.LoadEmbeddings": None,
}

# Per-module quantization override for MoE expert weights. The expert weight
# has shape [num_weight_sets, num_experts, output_dims, input_dims] (4D),
# which the global 2D `per_block / block_size=32 / axis=1` spec can't express.
# Block size 1 on the first three dims (no blocking) plus 32 on the last
# (input) dim mirrors how `block_size=32, axis=1` would behave on a 2D
# Linear weight. axis is `None` because block_size is itself multi-dim.
_TORCH_MOE_SWITCH_LINEAR_4BIT = {
    "coreai_models.primitives.macos.switch.SwitchLinear": {
        "module_state_spec": {
            "weight": {
                "dtype": "int4",
                "qscheme": "symmetric_with_clipping",
                "granularity": {
                    "type": "per_block",
                    "block_size": [1, 1, 1, 32],
                    "axis": None,
                },
            },
        },
        "op_input_spec": None,
        "op_output_spec": None,
    },
}

# Combined exclusions + MoE override for INT4 presets. Safe to apply on
# non-MoE models (no `SwitchLinear` instances → entry is a no-op).
_TORCH_MODULE_CONFIGS_4BIT = {
    **_TORCH_MODULE_EXCLUSIONS,
    **_TORCH_MOE_SWITCH_LINEAR_4BIT,
}

# Building blocks for the graph-mode quantization presets.

_INT4_WEIGHT_SPEC = {
    "weight": {
        "dtype": "int4",
        "qscheme": "symmetric_with_clipping",
        "granularity": {
            "type": "per_block",
            "block_size": 32,
            "axis": 1,
        },
    }
}


def _create_per_tensor_symmetric_kv_cache_config(dtype: str) -> dict[str, Any]:
    """KV-cache buffer quantization config for the fused cache-update op.

    Args:
        dtype: coreai-opt dtype name the cache buffer is stored in.
    """
    return {
        "mutable_cache_update_and_fetch": {
            "op_quantizer_config": {
                "op_input_spec": {
                    1: {
                        "dtype": dtype,
                        "qscheme": "symmetric",
                        "granularity": {"type": "per_tensor"},
                    }
                },
                "op_output_spec": None,
                "op_state_spec": None,
            },
        }
    }


_INT4_WEIGHT_ONLY_CONFIG_BASE: dict[str, Any] = {
    "global_config": {
        "op_state_spec": _INT4_WEIGHT_SPEC,
        "op_input_spec": None,
        "op_output_spec": None,
    },
    "module_type_configs": _TORCH_MODULE_CONFIGS_4BIT,
}


MACOS_PRESETS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------
    # No quantization — full precision
    # ---------------------------------------------------------------
    "none": {
        "suffix": "",
        "description": "Full precision (no quantization)",
    },
    # ---------------------------------------------------------------
    # INT4 weight-only torch quantization (default)
    # Applied to the PyTorch model before torch.export.
    # Uses coreai-opt quantization with symmetric clipping.
    # ---------------------------------------------------------------
    "4bit": {
        "torch_quantization_config": {
            "execution_mode": "eager",
            **_INT4_WEIGHT_ONLY_CONFIG_BASE,
        },
        "suffix": "4bit",
        "description": "INT4 symmetric per-block weight quantization (torch pre-export)",
    },
    # Same as the above `4bit` preset along with an INT8 per-tensor quantized KV cache.
    # Note: KV Cache quantization using ``coreai-opt`` requires graph-mode.
    "4bit_weights_8bit_kv_cache": {
        "torch_quantization_config": {
            "execution_mode": "graph",
            **_INT4_WEIGHT_ONLY_CONFIG_BASE,
            "kv_cache_quant_configs": _create_per_tensor_symmetric_kv_cache_config("int8"),
            "calibrate_activations": True,
        },
        "suffix": "4bit_weights_8bit_kv_cache",
        "description": (
            "INT4 symmetric per-block weight quantization and INT8 per-tensor KV cache (graph mode)"
        ),
    },
}

IOS_PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "suffix": "",
        "description": "Full precision (no palettization)",
    },
    "4bit_weight_palettized_group8": {
        "torch_palettization_config": {
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "n_bits": 4,
                        "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 8},
                    }
                }
            },
            "module_type_configs": _IOS_PALETTIZATION_EMBEDDING_EXCLUSIONS,
        },
        "suffix": "4bit_palettized_group8",
        "description": "INT4 weight palettization with a group size of 8 (torch pre-export)",
    },
    "4bit_weight_palettized_group32": {
        "torch_palettization_config": {
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "n_bits": 4,
                        "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 32},
                    }
                }
            },
            "module_type_configs": _IOS_PALETTIZATION_EMBEDDING_EXCLUSIONS,
        },
        "suffix": "4bit_palettized_group32",
        "description": "INT4 weight palettization with a group size of 32 (torch pre-export)",
    },
}

# Default preset when --model is used without explicit --compression
DEFAULT_MACOS_COMPRESSION_PRESET = "4bit"
DEFAULT_IOS_COMPRESSION_PRESET = "4bit_weight_palettized_group32"

# Use a set since both dicts have a "none" preset
ALL_PRESET_NAMES: list[str] = list(set(MACOS_PRESETS.keys()).union(set(IOS_PRESETS.keys())))


def get_preset(name: str) -> dict[str, Any]:
    """Get a quantization preset by name.

    Args:
        name: Preset name (e.g., ``"4bit"``, ``"none"``)

    Returns:
        Preset configuration dict

    Raises:
        KeyError: If the preset name is not found
    """
    if name not in MACOS_PRESETS and name not in IOS_PRESETS:
        available = ", ".join(ALL_PRESET_NAMES)
        raise KeyError(f"Unknown compression preset '{name}'. Available: {available}")

    if name in MACOS_PRESETS:
        return MACOS_PRESETS[name]

    return IOS_PRESETS[name]


def list_presets() -> list[str]:
    """List all available preset names."""
    return ALL_PRESET_NAMES


def list_macos_presets() -> list[str]:
    """List all available macOS preset names."""
    return sorted(MACOS_PRESETS.keys())


def list_ios_presets() -> list[str]:
    """List all available iOS preset names."""
    return sorted(IOS_PRESETS.keys())
