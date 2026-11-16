# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the hub backend abstraction and ModelScope download support."""

from __future__ import annotations

import pytest

from coreai_models.hub import (
    BackendName,
    HuggingFaceBackend,
    ModelScopeBackend,
    get_backend,
    resolve_download_id,
)

# ---------------------------------------------------------------------------
# Backend factory and singleton behavior
# ---------------------------------------------------------------------------


def test_get_backend() -> None:
    """Factory returns the right backend, rejects unknown names, and shares instances."""
    assert isinstance(get_backend(), HuggingFaceBackend)
    assert isinstance(get_backend("modelscope"), ModelScopeBackend)
    assert get_backend() is get_backend("huggingface")  # singleton
    with pytest.raises(ValueError, match="Unknown download backend"):
        get_backend("unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HuggingFace pass-through (the no-op guarantee)
# ---------------------------------------------------------------------------


def test_hf_backend_is_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF backend forwards the exact original huggingface_hub call."""
    import huggingface_hub

    calls: list[dict] = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: calls.append({"repo_id": repo_id, **kw}) or "/fake",
    )

    backend = HuggingFaceBackend()
    result = backend.snapshot("Qwen/Qwen3-0.6B", allow_patterns=["*.json"])

    assert result == "/fake"
    assert calls == [{"repo_id": "Qwen/Qwen3-0.6B", "allow_patterns": ["*.json"]}]


# ---------------------------------------------------------------------------
# ModelScope backend (lazy import + id routing)
# ---------------------------------------------------------------------------


def test_modelscope_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """MS backend routes the id to modelscope.snapshot_download; missing package raises."""
    import sys

    calls: list[tuple[str, list[str] | None]] = []

    class _FakeMS:
        def snapshot_download(self, model_id: str, *, allow_patterns=None):
            calls.append((model_id, allow_patterns))
            return "/tmp/fake"

    fake_module = type(sys)("modelscope")
    fake_module.snapshot_download = _FakeMS().snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_module)

    backend = ModelScopeBackend()
    backend.snapshot("Qwen/Qwen3-0.6B", allow_patterns=["*.json"])
    assert calls == [("Qwen/Qwen3-0.6B", ["*.json"])]

    # Missing package → helpful ImportError
    monkeypatch.setitem(sys.modules, "modelscope", None)
    with pytest.raises(ImportError, match="pip install"):
        ModelScopeBackend().snapshot("any/id")


# ---------------------------------------------------------------------------
# Default backend (no-op guarantee)
# ---------------------------------------------------------------------------


def test_export_config_default_backend_is_huggingface() -> None:
    """ExportConfig defaults to huggingface — no backend change without opt-in."""
    from coreai_models.export.pipeline import ExportConfig

    assert ExportConfig(hf_model_id="Qwen/Qwen3-0.6B").backend == "huggingface"


# ---------------------------------------------------------------------------
# Preset modelscope_id data
# ---------------------------------------------------------------------------


def test_preset_modelscope_id() -> None:
    """Only gpt-oss-20b carries an override; same-name presets are None."""
    from coreai_models.hub import resolve_download_id
    from coreai_models.model_registry import presets_for_type

    presets = presets_for_type("llm")
    gpt = next(p for p in presets if p.short_name == "gpt-oss-20b")
    assert gpt.hf_id == "openai/gpt-oss-20b"
    assert gpt.modelscope_id == "openai-mirror/gpt-oss-20b"

    same_name = next(p for p in presets if p.short_name == "qwen3-0.6b" and p.variant == "macOS")
    assert same_name.modelscope_id is None
    # Display id (used by --list-models) derives from the same resolve path:
    assert resolve_download_id(same_name.hf_id, "modelscope") == same_name.hf_id
    assert resolve_download_id(gpt.hf_id, "modelscope") == "openai-mirror/gpt-oss-20b"


# ---------------------------------------------------------------------------
# resolve_download_id (id translation at the download layer)
# ---------------------------------------------------------------------------


def test_resolve_download_id() -> None:
    """HF id passes through; gpt-oss overrides to openai-mirror namespace."""
    assert resolve_download_id("Qwen/Qwen3-0.6B", "modelscope") == "Qwen/Qwen3-0.6B"
    assert resolve_download_id("unknown/model", "modelscope") == "unknown/model"
    assert resolve_download_id("openai/gpt-oss-20b", "modelscope") == "openai-mirror/gpt-oss-20b"
    assert resolve_download_id("openai/gpt-oss-20b", "huggingface") == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# resolve_local (single entry point for id → path resolution)
# ---------------------------------------------------------------------------


def test_resolve_local(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """HF returns the id unchanged; a local dir passes through; MS pre-downloads."""

    from coreai_models import hub

    # HF identity
    assert hub.resolve_local("Qwen/Qwen3-0.6B", "huggingface", ["*.json"]) == "Qwen/Qwen3-0.6B"

    # Pre-downloaded local dir passes through unchanged
    assert hub.resolve_local(str(tmp_path), "modelscope", ["*.json"]) == str(tmp_path)

    # MS pre-downloads and returns the local path (id translated first)
    calls: list[tuple[str, list[str] | None]] = []

    class _FakeMS:
        name: BackendName = "modelscope"

        def snapshot(self, model_id: str, *, allow_patterns=None):
            calls.append((model_id, allow_patterns))
            return "/tmp/fake"

    monkeypatch.setitem(hub._BACKENDS, "modelscope", _FakeMS())
    result = hub.resolve_local("openai/gpt-oss-20b", "modelscope", ["*.json"])
    assert result == "/tmp/fake"
    # Id was translated before reaching the backend:
    assert calls == [("openai-mirror/gpt-oss-20b", ["*.json"])]


# ---------------------------------------------------------------------------
# Model-class override forwarding (drafter regression)
# ---------------------------------------------------------------------------


def test_drafter_memory_efficient_forwards_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drafter's from_hf_memory_efficient must forward backend to from_hf."""
    from coreai_models.models.macos.muse_glimmer_drafter_ring import (
        MuseGlimmerDrafterForCausalLM,
    )

    received: dict = {}

    def _fake_from_hf(cls, model_id, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(MuseGlimmerDrafterForCausalLM, "from_hf", classmethod(_fake_from_hf))
    MuseGlimmerDrafterForCausalLM.from_hf_memory_efficient(
        "meta-models/Muse-Glimmer-30B-assistant",
        backend="modelscope",
    )
    assert received.get("backend") == "modelscope"
