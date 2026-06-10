# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_models.model_registry."""

from __future__ import annotations

import pytest

from coreai_models.model_registry import (
    DIFFUSION_PRESETS,
    LLM_PRESETS,
    ModelPreset,
    _preset_to_export_args,
    _preset_to_output_name,
    filter_presets,
    filter_utility_models,
    lookup_preset,
    presets_for_type,
    try_lookup_preset,
)


def test_no_duplicate_short_name_variant_pairs() -> None:
    for t in ("llm", "diffusion"):
        seen: set[tuple[str, str | None]] = set()
        for p in presets_for_type(t):
            key = (p.short_name, p.variant)
            assert key not in seen, f"Duplicate {key} in type={t}"
            seen.add(key)


def test_diffusion_presets_have_no_variant() -> None:
    for p in DIFFUSION_PRESETS:
        assert p.variant is None, f"{p.short_name} should have variant=None"


def test_filter_by_family_and_variant() -> None:
    qwen3_ne = filter_presets(LLM_PRESETS, family="qwen3", variant="iOS")
    assert qwen3_ne
    assert all(p.family == "qwen3" and p.variant == "iOS" for p in qwen3_ne)


def test_filter_excludes_experimental_by_default() -> None:
    presets = [
        ModelPreset("real", "x/r", "fam", "llm", "macOS", "4bit", "float16", 8192),
        ModelPreset(
            "hidden", "x/h", "fam", "llm", "macOS", "4bit", "float16", 8192, experimental=True
        ),
    ]
    assert [p.short_name for p in filter_presets(presets)] == ["real"]
    assert sorted(p.short_name for p in filter_presets(presets, include_experimental=True)) == [
        "hidden",
        "real",
    ]


def test_lookup_unique_preset() -> None:
    p = lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS")
    assert p.hf_id == "Qwen/Qwen3-0.6B"


def test_lookup_ambiguous_raises_when_no_variant() -> None:
    with pytest.raises(KeyError, match="Multiple variants"):
        lookup_preset("qwen3-0.6b", model_type="llm")


def test_lookup_single_variant_succeeds_without_variant() -> None:
    """A model with only one variant doesn't need --platform to disambiguate."""
    # gpt-oss-20b is registered for macOS only.
    p = lookup_preset("gpt-oss-20b", model_type="llm")
    assert p.variant == "macOS"
    assert p.hf_id == "openai/gpt-oss-20b"


def test_lookup_missing_raises() -> None:
    with pytest.raises(KeyError, match="No preset"):
        lookup_preset("nonexistent-model", model_type="llm")


def test_export_args_macos_omits_variant_flag() -> None:
    p = lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS")
    args = _preset_to_export_args(p)
    assert "--platform" not in args
    assert args[0] == "Qwen/Qwen3-0.6B"


def test_export_args_iOS_emits_variant_flag() -> None:
    p = lookup_preset("qwen3-0.6b", model_type="llm", variant="iOS")
    args = _preset_to_export_args(p)
    assert args[args.index("--platform") + 1] == "iOS"


def test_export_args_diffusion_no_variant_no_context() -> None:
    p = lookup_preset("flux2-klein-4b", model_type="diffusion")
    args = _preset_to_export_args(p)
    assert "--platform" not in args
    assert "--max-context-length" not in args


def test_output_name_llm_macos_matches_pipeline_format() -> None:
    p = lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS")
    assert _preset_to_output_name(p) == "qwen3_0_6b_4bit_dynamic"


def test_output_name_llm_iOS_uses_yaml_stem_when_compression_config_set() -> None:
    p = lookup_preset("qwen3-0.6b", model_type="llm", variant="iOS")
    assert _preset_to_output_name(p) == "qwen3_0_6b_mixed_4bit_8bit_static"


def test_output_name_llm_iOS_prepends_hf_tail_when_yaml_stem_lacks_it() -> None:
    """A generic recipe stem (no model prefix) must still get the hf tail
    prepended so exports of different models with the same recipe don't
    collide on disk."""
    p = ModelPreset(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B",
        "qwen3",
        "llm",
        "iOS",
        "none",
        "float16",
        4096,
        compression_config="generic_4bit.yaml",
    )
    assert _preset_to_output_name(p) == "qwen3_0_6b_generic_4bit_static"


def test_output_name_diffusion_is_hf_tail() -> None:
    p = lookup_preset("flux2-klein-4b", model_type="diffusion")
    assert _preset_to_output_name(p) == "FLUX.2-klein-4B"


# --- try_lookup_preset ---


def test_try_lookup_returns_none_for_unknown() -> None:
    assert try_lookup_preset("nonexistent-model", model_type="llm") is None


def test_try_lookup_resolves_with_type() -> None:
    p = try_lookup_preset("qwen3-0.6b", model_type="llm", variant="macOS")
    assert p is not None
    assert p.hf_id == "Qwen/Qwen3-0.6B"


def test_try_lookup_resolves_diffusion() -> None:
    p = try_lookup_preset("flux2-klein-4b", model_type="diffusion")
    assert p is not None
    assert p.hf_id == "black-forest-labs/FLUX.2-klein-4B"


def test_try_lookup_without_type_finds_across_tables() -> None:
    p = try_lookup_preset("flux2-klein-4b")
    assert p is not None
    assert p.type == "diffusion"


def test_try_lookup_prefers_macos_when_ambiguous() -> None:
    p = try_lookup_preset("qwen3-0.6b", model_type="llm")
    assert p is not None
    assert p.variant == "macOS"


def test_try_lookup_with_explicit_variant() -> None:
    """Passing an explicit variant resolves the correct preset."""
    result = try_lookup_preset("qwen3-0.6b", model_type="llm", variant="iOS")
    assert result is not None
    assert result.variant == "iOS"


def test_utility_platform_filter_excludes_macos_only() -> None:
    ios_models = filter_utility_models(platform="iOS")
    names = [u.short_name for u in ios_models]
    assert "depth-anything-3-small" not in names
    assert "clip-vit-b32" in names
