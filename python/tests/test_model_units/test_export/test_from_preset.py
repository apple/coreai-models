# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""``ExportConfig.from_preset`` and ``resolve_compression``."""

import dataclasses
from pathlib import Path

import pytest

from coreai_models.export.pipeline import ExportConfig
from coreai_models.llm.export import resolve_compression
from coreai_models.model_registry import LLM_PRESETS, lookup_preset


@pytest.mark.parametrize("preset", LLM_PRESETS, ids=lambda p: f"{p.short_name}-{p.variant}")
def test_every_registry_field_reaches_the_config(preset) -> None:
    config = ExportConfig.from_preset(
        preset, compute_precision=preset.compute_precision or "float16"
    )
    assert config.hf_model_id == preset.hf_id
    assert config.variant == preset.variant
    assert config.compression == preset.compression
    assert config.max_context_length == preset.max_context_length
    assert config.model_type_override == preset._model_type_override


def test_stated_values_win_over_the_preset() -> None:
    preset = lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS")
    config = ExportConfig.from_preset(
        preset, compression="none", compute_precision="float32", max_context_length=512
    )
    assert (config.compression, config.compute_precision, config.max_context_length) == (
        "none",
        "float32",
        512,
    )


def test_a_preset_without_a_precision_needs_one_stated() -> None:
    preset = dataclasses.replace(
        lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS"), compute_precision=None
    )
    with pytest.raises(ValueError, match="compute precision"):
        ExportConfig.from_preset(preset)


def test_a_missing_compression_yaml_exits_with_the_flags_message() -> None:
    with pytest.raises(SystemExit, match="--compression-config: file not found"):
        resolve_compression("macOS", compression_config=Path("does/not/exist.yaml"))
