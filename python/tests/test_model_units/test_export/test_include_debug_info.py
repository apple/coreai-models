# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``--include-debug-info`` flag across every export surface.

Exports default to the converter's ``RELEASE`` mode so shipped ``.aimodel``
assets embed minimum debug information; ``--include-debug-info`` opts into ``DEBUG``
mode and its full debug information.

Flag plumbing only — nothing here downloads weights or runs an export.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.diffusion.pipeline import DiffusionExportConfig
from coreai_models.export.pipeline import ExportConfig
from coreai_models.llm.export import build_parser
from coreai_models.segmentation.export import build_parser as build_segmentation_parser


def _repo_root() -> Path:
    """Walk up to the workspace root (where ``pyproject.toml`` + ``python/`` live)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    raise RuntimeError("workspace root not found")


# --- the shared default ----------------------------------------------


def test_the_default_is_release_mode() -> None:
    """Pins the repo-wide default. Flipping this changes every shipped asset."""
    assert DEFAULT_INCLUDE_DEBUG_INFO is False


@pytest.mark.parametrize("config_cls", [ExportConfig, DiffusionExportConfig])
def test_config_dataclasses_default_to_the_shared_constant(config_cls) -> None:
    config = config_cls(hf_model_id="org/model")
    assert config.include_debug_info is DEFAULT_INCLUDE_DEBUG_INFO


# --- coreai.llm.export ------------------------------------------------


def test_llm_cli_defaults_to_off() -> None:
    args = build_parser().parse_args(["qwen3-0.6b"])
    assert args.include_debug_info is False


def test_llm_cli_accepts_the_flag() -> None:
    args = build_parser().parse_args(["qwen3-0.6b", "--include-debug-info"])
    assert args.include_debug_info is True


def test_llm_cli_flag_is_independent_of_verbose() -> None:
    """``-v`` is console log level only; it must not change the asset contents."""
    args = build_parser().parse_args(["qwen3-0.6b", "-v"])
    assert args.verbose is True
    assert args.include_debug_info is False


def test_llm_cli_resolves_the_flag_onto_the_export_config() -> None:
    """Covers the wiring in ``_resolve_export_config``, not just the parser."""
    from coreai_models.llm.export import _resolve_export_config

    for argv, expected in ((["qwen3-0.6b"], False), (["qwen3-0.6b", "--include-debug-info"], True)):
        config = _resolve_export_config(build_parser().parse_args(argv))
        assert config.include_debug_info is expected


# --- standalone models/*/export.py recipes ----------------------------

# ``models/sam3/export.py`` and ``models/sam3_video/export.py`` delegate to the
# segmentation CLI, so they inherit the flag rather than declaring one.
_DELEGATING_RECIPES = {"sam3", "sam3_video"}


def _standalone_recipes() -> list[Path]:
    recipes = sorted(
        p
        for p in _repo_root().glob("models/*/export.py")
        if p.parent.name not in _DELEGATING_RECIPES
    )
    assert recipes, "no standalone recipes found — glob or layout changed"
    return recipes


def test_every_standalone_recipe_is_discovered() -> None:
    # Guards the glob itself: a silently-empty list would make the checks below vacuous.
    assert len(_standalone_recipes()) >= 12


@pytest.mark.parametrize("recipe_name", sorted(_DELEGATING_RECIPES))
def test_delegating_recipe_actually_routes_through_the_shared_cli(recipe_name: str) -> None:
    """Keep the exemption above honest.

    A recipe is excused from declaring ``--include-debug-info`` only because it
    hands argv to the segmentation CLI, which declares it. If one ever stopped
    delegating, the exemption would silently become a hole.
    """
    source = (_repo_root() / "models" / recipe_name / "export.py").read_text()
    assert "coreai_models.segmentation.export" in source, (
        f"{recipe_name}: exempted as delegating, but does not call the segmentation CLI"
    )
    assert "--include-debug-info" in build_segmentation_parser().format_help()


@pytest.mark.parametrize("recipe", _standalone_recipes(), ids=lambda p: p.parent.name)
def test_standalone_recipe_defaults_to_release_and_offers_include_debug_info(recipe: Path) -> None:
    """A ``.aimodel`` must not differ based on which script produced it.

    Source-text scan rather than an import: these are PEP 723 scripts whose
    inline dependencies are not installed in the test environment.
    """
    source = recipe.read_text()
    assert "--include-debug-info" in source, (
        f"{recipe.parent.name}: missing the --include-debug-info flag"
    )
    assert "Mode.RELEASE" in source, f"{recipe.parent.name}: does not default to RELEASE mode"
    assert "TorchConverter()" not in source, (
        f"{recipe.parent.name}: constructs TorchConverter() with no mode, "
        "which silently inherits the library's DEBUG default"
    )
