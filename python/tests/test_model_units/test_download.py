# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_models._download.

These cover the backend-resolution priority matrix, error paths, the
optional-dependency guard, and the pass-through id contract. They are
pure-logic: no network, no installed modelscope/huggingface_hub required
for the ImportError case.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from coreai_models._download import (
    download_snapshot,
    resolve_backend,
    resolve_model_path,
)

# ---------------------------------------------------------------------------
# resolve_backend — priority matrix
# ---------------------------------------------------------------------------


def test_default_is_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set and no explicit arg, default is huggingface."""
    monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)
    assert resolve_backend() == "huggingface"


def test_download_backend_modelscope_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")
    assert resolve_backend() == "modelscope"


def test_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit backend arg has the highest priority."""
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")
    assert resolve_backend("huggingface") == "huggingface"
    # And the reverse
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "huggingface")
    assert resolve_backend("modelscope") == "modelscope"


# ---------------------------------------------------------------------------
# resolve_backend — error paths
# ---------------------------------------------------------------------------


def test_invalid_explicit_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown download backend"):
        resolve_backend("garbage")  # type: ignore[arg-type]


def test_invalid_env_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "garbage")
    with pytest.raises(ValueError, match="is invalid"):
        resolve_backend()


# ---------------------------------------------------------------------------
# _download_modelscope — optional dependency guard
# ---------------------------------------------------------------------------


def test_modelscope_backend_without_package_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the modelscope package is absent, the modelscope backend gives a
    helpful ImportError instead of a bare ModuleNotFoundError.

    We simulate absence by making ``importlib.import_module('modelscope')``
    raise ImportError, which is what the real environment looks like when
    the extra isn't installed.
    """
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "modelscope":
            raise ImportError("simulated: modelscope not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Remove any cached modelscope so the fake import path is taken.
    for mod in list(sys.modules):
        if mod == "modelscope" or mod.startswith("modelscope."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError, match="modelscope"):
        download_snapshot("AI-ModelScope/whisper-large-v3", allow_patterns=["config.json"])


# ---------------------------------------------------------------------------
# resolve_model_path — delegation
# ---------------------------------------------------------------------------


def test_resolve_model_path_delegates_to_download_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_model_path is a thin wrapper over download_snapshot; it should
    forward its kwargs (allow_patterns, cache_dir, backend) verbatim."""

    captured: dict = {}

    def fake_download(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return "/fake/path"

    monkeypatch.setattr("coreai_models._download.download_snapshot", fake_download)
    result = resolve_model_path(
        "Qwen/Qwen3-0.6B",
        allow_patterns=["config.json"],
        cache_dir="/tmp/cache",
        backend="modelscope",
    )
    assert result == "/fake/path"
    assert captured["model_id"] == "Qwen/Qwen3-0.6B"
    assert captured["kwargs"]["allow_patterns"] == ["config.json"]
    assert captured["kwargs"]["cache_dir"] == "/tmp/cache"
    assert captured["kwargs"]["backend"] == "modelscope"


# ---------------------------------------------------------------------------
# Pass-through id contract
# ---------------------------------------------------------------------------


def _install_fake_modelscope(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """Install a fake modelscope module capturing snapshot_download calls."""

    def fake_ms_snapshot(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        return "/fake/ms/path"

    fake_mod = type(sys)("modelscope")
    fake_mod.snapshot_download = fake_ms_snapshot
    monkeypatch.setitem(sys.modules, "modelscope", fake_mod)


def test_modelscope_backend_passes_id_through_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With backend=modelscope, the id reaches modelscope's snapshot_download
    unchanged."""
    captured: dict = {}
    _install_fake_modelscope(monkeypatch, captured)
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")

    download_snapshot("AI-ModelScope/whisper-large-v3", allow_patterns=["config.json"])

    assert captured["model_id"] == "AI-ModelScope/whisper-large-v3"


def test_modelscope_style_id_does_not_auto_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ModelScope-style id with the default (huggingface) backend goes to
    the HuggingFace Hub; the backend is selected only by arg or env var."""
    monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)

    def fail_ms(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("modelscope must not be called without an explicit backend selection")

    fake_ms_mod = type(sys)("modelscope")
    fake_ms_mod.snapshot_download = fail_ms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", fake_ms_mod)

    captured: dict = {}

    def fake_hf(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        return "/fake/hf/path"

    fake_hf_mod = type(sys)("huggingface_hub")
    fake_hf_mod.snapshot_download = fake_hf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_mod)

    p = download_snapshot("AI-ModelScope/whisper-large-v3")

    assert captured["model_id"] == "AI-ModelScope/whisper-large-v3"
    assert p == "/fake/hf/path"


def test_modelscope_style_id_does_not_auto_route_even_against_explicit_hf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit backend='huggingface' is honored literally even when the id
    looks like a ModelScope id."""
    monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)

    def fail_ms(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("modelscope must not be called; backend was explicitly huggingface")

    fake_ms_mod = type(sys)("modelscope")
    fake_ms_mod.snapshot_download = fail_ms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", fake_ms_mod)

    captured: dict = {}

    def fake_hf(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        return "/fake/hf/path"

    fake_hf_mod = type(sys)("huggingface_hub")
    fake_hf_mod.snapshot_download = fake_hf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_mod)

    p = download_snapshot("AI-ModelScope/whisper-large-v3", backend="huggingface")

    assert captured["model_id"] == "AI-ModelScope/whisper-large-v3"
    assert p == "/fake/hf/path"


def test_hf_id_with_modelscope_backend_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HF-style id with the ModelScope backend is likewise passed to
    ModelScope unchanged (same-name repos download fine)."""
    captured: dict = {}
    _install_fake_modelscope(monkeypatch, captured)
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")

    download_snapshot("Qwen/Qwen3-0.6B")

    assert captured["model_id"] == "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# models/_download_shim.py — standalone-script download shim
#
# The shim must stay in sync with coreai_models._download: same env-var
# precedence and same invalid-value behavior.
# These tests pin that so a drift is caught immediately.
# ---------------------------------------------------------------------------

_SHIM_PATH = str(Path(__file__).resolve().parents[3] / "models" / "_download_shim.py")


def _import_shim():
    """Import the shim as an isolated module (it's not inside the package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_shim_under_test", _SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_shim_default_is_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _import_shim()
    monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)
    assert shim._resolve_backend() == "huggingface"


def test_shim_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shim must raise on an invalid COREAI_DOWNLOAD_BACKEND value — same
    as the main module. This pins the shared behavior."""
    shim = _import_shim()
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "garbage")
    with pytest.raises(ValueError, match="is invalid"):
        shim._resolve_backend()


def test_shim_backend_selection_matches_main_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim resolves the backend exactly like the main module across the
    env matrix (default / modelscope / explicit-value validation)."""
    shim = _import_shim()
    for env_value, expected in ((None, "huggingface"), ("modelscope", "modelscope")):
        if env_value is None:
            monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)
        else:
            monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", env_value)
        assert shim._resolve_backend() == expected, (
            f"shim and main module disagree on backend for env={env_value!r}"
        )


def test_shim_modelscope_backend_passes_id_through_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With COREAI_DOWNLOAD_BACKEND=modelscope the shim passes the id to
    modelscope unchanged, mirroring the main module."""
    shim = _import_shim()
    monkeypatch.setenv("COREAI_DOWNLOAD_BACKEND", "modelscope")

    captured: dict = {}

    def fake_ms_snapshot(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        return "/fake/ms/path"

    fake_mod = type(sys)("modelscope")
    fake_mod.snapshot_download = fake_ms_snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", fake_mod)

    p = shim.resolve_model_path("AI-ModelScope/whisper-large-v3", allow_patterns=["config.json"])

    assert captured["model_id"] == "AI-ModelScope/whisper-large-v3"
    assert p == "/fake/ms/path"


def test_shim_modelscope_style_id_does_not_auto_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ModelScope-style id with the default backend goes to the HF hub in
    the shim too; the backend is selected only via the env var."""
    shim = _import_shim()
    monkeypatch.delenv("COREAI_DOWNLOAD_BACKEND", raising=False)

    def fail_ms(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("shim must not call modelscope without COREAI_DOWNLOAD_BACKEND")

    fake_ms_mod = type(sys)("modelscope")
    fake_ms_mod.snapshot_download = fail_ms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", fake_ms_mod)

    captured: dict = {}

    def fake_hf(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        return "/fake/hf/path"

    fake_hf_mod = type(sys)("huggingface_hub")
    fake_hf_mod.snapshot_download = fake_hf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_mod)

    p = shim.resolve_model_path("AI-ModelScope/whisper-large-v3")

    assert captured["model_id"] == "AI-ModelScope/whisper-large-v3"
    assert p == "/fake/hf/path"
