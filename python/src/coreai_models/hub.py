# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model download backends.

A :class:`HubBackend` knows how to fetch a model snapshot from one hub to a
local directory. The HuggingFace backend is the default and is a pure
pass-through of ``huggingface_hub.snapshot_download`` — selecting it (or not
selecting a backend at all) keeps the exact original download behavior.
The ModelScope backend is opt-in (``--backend modelscope`` on the export
CLIs) and downloads from modelscope.cn; it lazily imports the ``modelscope``
package, so environments that never opt in need not install it.
"""

from __future__ import annotations

from typing import Literal, Protocol

BackendName = Literal["huggingface", "modelscope"]

# Glob patterns for the config/tokenizer snapshot fetched when resolving a
# model id for ``from_pretrained`` under the ModelScope backend. Shared by
# the export pipeline (AutoConfig resolution, calibration tokenizer, bundle
# tokenizer) so every site fetches the same file set.
SNAPSHOT_PATTERNS = [
    "*.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "*.model",
    "*.txt",
    "*.jinja",
]

# Patterns for a single comprehensive pre-download (config + tokenizer + weights).
# Used by the export pipeline to fetch everything in one hub call.
FULL_PATTERNS = sorted(
    set(
        SNAPSHOT_PATTERNS
        + [
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "generation_config.json",
        ]
    )
)


class HubBackend(Protocol):
    """Interface every download backend implements."""

    name: BackendName

    def snapshot(self, model_id: str, *, allow_patterns: list[str] | None = None) -> str:
        """Download (or resolve from the hub's local cache) a full repo snapshot.

        Args:
            model_id: Model identifier in the selected hub's namespace.
            allow_patterns: Glob patterns restricting which files to fetch.

        Returns:
            Local filesystem path to the snapshot directory.
        """
        ...


class HuggingFaceBackend(HubBackend):
    """Default backend: pure pass-through to ``huggingface_hub``.

    The call made here is identical to the one the code made before this
    abstraction existed, so routing through it is a no-op for the default
    path.
    """

    name: BackendName = "huggingface"

    def snapshot(self, model_id: str, *, allow_patterns: list[str] | None = None) -> str:
        import huggingface_hub

        return huggingface_hub.snapshot_download(model_id, allow_patterns=allow_patterns)


class ModelScopeBackend(HubBackend):
    """Opt-in backend for modelscope.cn.

    ``modelscope`` is a runtime-optional dependency: it is imported here only
    when this backend is actually selected, never at module import time.
    """

    name: BackendName = "modelscope"

    def snapshot(self, model_id: str, *, allow_patterns: list[str] | None = None) -> str:
        try:
            from modelscope import snapshot_download as _ms_snapshot
        except ImportError as exc:
            raise ImportError(
                "The ModelScope download backend requires the 'modelscope' package "
                "(>=1.37). Install it with: pip install 'modelscope>=1.37'"
            ) from exc
        # ModelScope's snapshot_download accepts ``model_id`` as the first
        # positional argument and returns a local path string, mirroring
        # huggingface_hub.snapshot_download closely.
        return _ms_snapshot(model_id, allow_patterns=allow_patterns)


# Backends are stateless, so a single top-level mapping suffices.
_BACKENDS: dict[str, HubBackend] = {
    "huggingface": HuggingFaceBackend(),
    "modelscope": ModelScopeBackend(),
}


def get_backend(name: BackendName = "huggingface") -> HubBackend:
    """Return the backend instance for ``name``.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown download backend {name!r}. Expected 'huggingface' or 'modelscope'."
        )
    return _BACKENDS[name]


def resolve_download_id(hf_model_id: str, backend: BackendName = "huggingface") -> str:
    """Return the model id the selected backend should use for downloading.

    HuggingFace uses the id unchanged. ModelScope consults the registry's
    per-preset ``modelscope_id`` override; repos that exist under the same
    org/name on both hubs pass through unchanged. This is the only place
    ids are translated, so every other consumer (metadata lookup, bundle
    writing, tokenizer references) keeps the original HuggingFace id.
    """
    if backend == "huggingface":
        return hf_model_id
    from coreai_models.model_registry import try_lookup_preset_by_hf_id

    preset = try_lookup_preset_by_hf_id(hf_model_id, model_type="llm")
    if preset is not None and preset.modelscope_id:
        return preset.modelscope_id
    return hf_model_id


def resolve_local(
    hf_model_id_or_path: str,
    backend: BackendName = "huggingface",
    allow_patterns: list[str] | None = None,
) -> str:
    """Return an id or local path that ``from_pretrained`` can consume.

    A pre-downloaded local directory is returned unchanged. HuggingFace
    (default) returns the hub id unchanged — ``from_pretrained`` keeps its
    own download path (the no-op guarantee). ModelScope resolves the
    download id (translating the HuggingFace id via the registry override
    when needed), pre-downloads the snapshot, and returns the local path.
    """
    from pathlib import Path

    if Path(hf_model_id_or_path).is_dir():
        return hf_model_id_or_path
    if backend == "huggingface":
        return hf_model_id_or_path
    ms_id = resolve_download_id(hf_model_id_or_path, backend)
    return get_backend("modelscope").snapshot(ms_id, allow_patterns=allow_patterns)
