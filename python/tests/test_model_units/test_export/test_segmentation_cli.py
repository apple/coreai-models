# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``coreai.segmentation.export`` CLI.

Flag plumbing only — nothing here downloads weights or runs an export, so
these are safe on any machine.
"""

from __future__ import annotations

import argparse
import logging

import pytest

from coreai_models.segmentation.export import (
    _SUPPORTED,
    _resolve_hf_model_id,
    _resolve_image_size,
    _warn_unused_flags,
    build_parser,
)
from coreai_models.segmentation.pipeline import FullExportConfig, SegmentationExportConfig


def _parse(*argv: str) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Build a fresh parser and parse ``argv``, returning both."""
    parser = build_parser()
    return parser, parser.parse_args(list(argv))


# --- --model ---------------------------------------------------------


def test_every_supported_spelling_parses_and_resolves() -> None:
    """``--model`` takes its ``choices`` from ``_SUPPORTED``, which is what lets
    ``_resolve_hf_model_id`` be a bare dict lookup. Pin that coupling: every
    key must parse, and every value must be a canonical HF id."""
    for spelling in _SUPPORTED:
        _, args = _parse("--model", spelling)
        assert _resolve_hf_model_id(args.model) == _SUPPORTED[spelling]


def test_model_defaults_to_sam3() -> None:
    _, args = _parse()
    assert _resolve_hf_model_id(args.model) == "facebook/sam3"


def test_model_accepts_registry_short_name() -> None:
    _, args = _parse("--model", "sam3")
    assert _resolve_hf_model_id(args.model) == "facebook/sam3"


def test_model_accepts_hf_id() -> None:
    _, args = _parse("--model", "facebook/sam3")
    assert _resolve_hf_model_id(args.model) == "facebook/sam3"


def test_model_rejects_unknown_value() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "facebook/sam2"])


def test_bare_positional_is_rejected() -> None:
    """``--model`` replaced a positional, so a bare value is no longer valid."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sam3"])


# --- mode + image size -----------------------------------------------


def test_lite_is_the_default_mode() -> None:
    _, args = _parse()
    assert args.full is False


def test_image_size_defaults_per_mode() -> None:
    _, lite = _parse()
    _, full = _parse("--full")
    assert _resolve_image_size(lite) == SegmentationExportConfig.image_size
    assert _resolve_image_size(full) == FullExportConfig.image_size


def test_explicit_image_size_overrides_mode_default() -> None:
    _, args = _parse("--full", "--image-size", "512")
    assert _resolve_image_size(args) == 512


# --- _warn_unused_flags ----------------------------------------------


def test_warn_unused_flags_does_not_reparse_argv() -> None:
    """Regression: this recovered defaults via ``parse_args([args.model])``,
    which argparse rejected as a stray positional once ``model`` became
    ``--model`` — killing the process before the export started."""
    _, args = _parse("--full", "--dtype", "float16")
    _warn_unused_flags(args)


def test_warns_on_dtype_equal_to_default_in_lite_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: comparing against the default missed ``--dtype float32``,
    since float32 *is* the default — the flag was silently ignored with no
    warning. Mode-specific flags now default to None so "passed" is detectable
    regardless of the value."""
    _, args = _parse("--dtype", "float32")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert "--dtype" in caplog.text


def test_warns_on_lite_only_flag_equal_to_default_in_full_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same class of bug as the --dtype case: 32 is the resolved default."""
    _, args = _parse("--full", "--max-text-seq-len", "32")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert "--max-text-seq-len" in caplog.text


def test_warns_on_lite_only_flags_in_full_mode(caplog: pytest.LogCaptureFixture) -> None:
    _, args = _parse("--full", "--n-bits", "4", "--max-text-seq-len", "64")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert "--n-bits" in caplog.text
    assert "--max-text-seq-len" in caplog.text


def test_warns_on_dtype_in_lite_mode(caplog: pytest.LogCaptureFixture) -> None:
    _, args = _parse("--dtype", "float16")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert "--dtype" in caplog.text


def test_no_warning_for_lite_flags_in_lite_mode(caplog: pytest.LogCaptureFixture) -> None:
    _, args = _parse("--n-bits", "4", "--group-size", "16")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert caplog.text == ""


def test_no_warning_for_dtype_in_full_mode(caplog: pytest.LogCaptureFixture) -> None:
    _, args = _parse("--full", "--dtype", "float16")
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert caplog.text == ""


def test_no_warning_when_nothing_mode_specific_is_passed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    for argv in ((), ("--full",)):
        _, args = _parse(*argv)
        with caplog.at_level(logging.WARNING):
            _warn_unused_flags(args)
        assert caplog.text == "", f"unexpected warning for {argv}"


# --- --include-debug-info -----------------------------------------------------


def test_include_debug_info_defaults_to_off() -> None:
    """Exports default to the converter's RELEASE mode: minimum debug information."""
    _, args = _parse()
    assert args.include_debug_info is False


def test_include_debug_info_can_be_enabled() -> None:
    _, args = _parse("--include-debug-info")
    assert args.include_debug_info is True


def test_include_debug_info_reaches_both_config_dataclasses() -> None:
    """The flag is mode-agnostic, so both branches of ``main`` must carry it."""
    for config_cls in (SegmentationExportConfig, FullExportConfig):
        assert config_cls().include_debug_info is False
        assert config_cls(include_debug_info=True).include_debug_info is True


@pytest.mark.parametrize("argv", [("--include-debug-info",), ("--full", "--include-debug-info")])
def test_include_debug_info_is_not_treated_as_mode_specific(
    argv: tuple[str, ...], caplog: pytest.LogCaptureFixture
) -> None:
    """Regression guard: --include-debug-info applies to both modes, so it must never be
    added to the mode-specific warn lists in ``_warn_unused_flags``."""
    _, args = _parse(*argv)
    with caplog.at_level(logging.WARNING):
        _warn_unused_flags(args)
    assert caplog.text == ""
