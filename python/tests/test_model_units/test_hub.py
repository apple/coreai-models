# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_models.hub — the model download backend abstraction.

Pure-logic tests: no network, no real modelscope/huggingface_hub install
required (the ImportError case simulates the package's absence).
"""

from __future__ import annotations

import builtins
import sys

import pytest

from coreai_models.hub import (
    HuggingFaceBackend,
    ModelScopeBackend,
    get_backend,
)

# ---------------------------------------------------------------------------
# get_backend — factory
# ---------------------------------------------------------------------------


def test_get_backend_default_is_huggingface() -> None:
    assert get_backend().name == "huggingface"
    assert isinstance(get_backend(), HuggingFaceBackend)


def test_get_backend_modelscope() -> None:
    backend = get_backend("modelscope")
    assert backend.name == "modelscope"
    assert isinstance(backend, ModelScopeBackend)


def test_get_backend_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown download backend"):
        get_backend("garbage")  # type: ignore[arg-type]


def test_get_backend_caches_instances() -> None:
    """Backends are stateless and shared: the factory hands out the same
    top-level singleton instances (no per-call construction, no lazy
    machinery — just the module-level ``_BACKENDS`` dict)."""
    assert get_backend("huggingface") is get_backend("huggingface")
    assert get_backend("modelscope") is get_backend("modelscope")


# ---------------------------------------------------------------------------
# HuggingFaceBackend — A1 no-op guarantee: pure pass-through
# ---------------------------------------------------------------------------


def test_hf_backend_is_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF backend must call huggingface_hub.snapshot_download with the
    exact same arguments the code passed before this abstraction existed."""
    captured: dict = {}

    def fake_snapshot_download(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return "/fake/hf/path"

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = HuggingFaceBackend().snapshot(
        "Qwen/Qwen3-0.6B", allow_patterns=["*.safetensors", "config.json"]
    )

    assert captured["model_id"] == "Qwen/Qwen3-0.6B"
    assert captured["kwargs"] == {"allow_patterns": ["*.safetensors", "config.json"]}
    assert result == "/fake/hf/path"


def test_hf_backend_default_allow_patterns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No allow_patterns → None must reach snapshot_download verbatim
    (matching the pre-abstraction call shape)."""
    captured: dict = {}

    def fake_snapshot_download(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return "/fake/hf/path"

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    HuggingFaceBackend().snapshot("Qwen/Qwen3-0.6B")
    assert captured["kwargs"] == {"allow_patterns": None}


# ---------------------------------------------------------------------------
# ModelScopeBackend — lazy import + pass-through
# ---------------------------------------------------------------------------


def test_modelscope_backend_uses_given_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model id and allow_patterns reach modelscope's snapshot_download
    verbatim — no id translation happens in the backend."""

    captured: dict = {}

    def fake_ms_snapshot(model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return "/fake/ms/path"

    fake_mod = type(sys)("modelscope")
    fake_mod.snapshot_download = fake_ms_snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", fake_mod)

    result = ModelScopeBackend().snapshot("Qwen/Qwen3-0.6B", allow_patterns=["config.json"])

    assert captured["model_id"] == "Qwen/Qwen3-0.6B"
    assert captured["kwargs"] == {"allow_patterns": ["config.json"]}
    assert result == "/fake/ms/path"


def test_modelscope_backend_without_package_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the modelscope package is absent, selecting its backend gives a
    helpful ImportError instead of a bare ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "modelscope":
            raise ImportError("simulated: modelscope not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in list(sys.modules):
        if mod == "modelscope" or mod.startswith("modelscope."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError, match="modelscope"):
        ModelScopeBackend().snapshot("Qwen/Qwen3-0.6B")


# ---------------------------------------------------------------------------
# Protocol shape — both backends are interchangeable
# ---------------------------------------------------------------------------


def test_backends_share_protocol_shape() -> None:
    """Both implementations expose name + snapshot(callable with the same
    signature), so callers can treat them interchangeably."""
    for backend in (HuggingFaceBackend(), ModelScopeBackend()):
        assert callable(backend.snapshot)
        assert backend.name in ("huggingface", "modelscope")


# ---------------------------------------------------------------------------
# Pipeline-level wiring — backend field, preset override map, list filtering
# (pure logic; no network)
# ---------------------------------------------------------------------------


def test_export_config_default_backend_is_huggingface() -> None:
    """An ExportConfig built without an explicit backend keeps the default,
    which is the no-op guarantee for the whole pipeline."""
    from coreai_models.export.pipeline import ExportConfig

    assert ExportConfig(hf_model_id="Qwen/Qwen3-0.6B").backend == "huggingface"


def test_preset_modelscope_id_override() -> None:
    """The registry override map: a preset with a modelscope_id resolves to it
    under the ModelScope backend and stays on the HF id otherwise. Presets
    without an entry (the same-name majority) keep the HF id on both backends.
    """
    from coreai_models.model_registry import try_lookup_preset

    preset = try_lookup_preset("gpt-oss-20b", model_type="llm")
    assert preset is not None
    assert preset.modelscope_id == "openai-mirror/gpt-oss-20b"
    assert preset.hf_id == "openai/gpt-oss-20b"

    same_name = try_lookup_preset("qwen3-0.6b", model_type="llm")
    assert same_name is not None
    assert same_name.modelscope_id is None


def test_presets_display_id_swaps_by_backend() -> None:
    """--list-models support: presets_for_type returns the full preset list
    (same-name models are assumed available on ModelScope); the backend only
    affects the display id, which callers derive via ``preset.modelscope_id
    or preset.hf_id``."""
    from coreai_models.model_registry import presets_for_type

    presets = presets_for_type("llm")

    gpt = next(p for p in presets if p.short_name == "gpt-oss-20b")
    assert gpt.hf_id == "openai/gpt-oss-20b"
    # Display id under --backend modelscope:
    assert (gpt.modelscope_id or gpt.hf_id) == "openai-mirror/gpt-oss-20b"

    same_name = next(p for p in presets if p.short_name == "qwen3-0.6b" and p.variant == "macOS")
    assert same_name.modelscope_id is None
    # Same display id under either backend:
    assert (same_name.modelscope_id or same_name.hf_id) == same_name.hf_id


# ---------------------------------------------------------------------------
# Model-class override forwarding — backend must survive every from_hf* hop
# ---------------------------------------------------------------------------


def test_drafter_memory_efficient_forwards_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drafter's from_hf_memory_efficient override must forward ``backend``
    to ``from_hf`` instead of swallowing it via ``**kwargs``.

    Regression test: the override used to accept ``**kwargs`` without
    forwarding, so exporting the drafter as the main model
    (``coreai.llm.export muse-glimmer-30b-drafter --backend modelscope``,
    a macOS export — all macOS exports take the memory-efficient path)
    silently downloaded weights from HuggingFace despite the flag.
    """
    from coreai_models.models.macos.muse_glimmer_drafter_ring import (
        MuseGlimmerDrafterForCausalLM,
    )

    captured: dict = {}

    def fake_from_hf(cls_, model_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        MuseGlimmerDrafterForCausalLM, "from_hf", classmethod(fake_from_hf)
    )

    # Call with the exact kwarg shape the export pipeline passes: backend plus
    # the registry hooks the override intentionally swallows.
    MuseGlimmerDrafterForCausalLM.from_hf_memory_efficient(
        "meta-models/Muse-Glimmer-30B-assistant",
        backend="modelscope",
        hf_config_attr=None,
        hf_state_dict_prefix="",
    )
    assert captured["model_id"] == "meta-models/Muse-Glimmer-30B-assistant"
    assert captured["kwargs"].get("backend") == "modelscope"
