# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unified model-download abstraction.

This module is the single entry point for fetching model snapshots. It can
route to either the HuggingFace Hub (default) or the ModelScope hub, so the
rest of the codebase never imports ``huggingface_hub`` or ``modelscope``
directly for downloads.

Backend selection (in priority order):

1. Explicit ``backend`` argument to :func:`download_snapshot` /
   :func:`resolve_model_path`.
2. ``COREAI_DOWNLOAD_BACKEND`` environment variable (``huggingface`` |
   ``modelscope``).
3. Default: HuggingFace.

Model ids belong to the selected hub's namespace: use a HuggingFace id
(``org/name``) with the default backend, or a ModelScope id (e.g.
``AI-ModelScope/whisper-large-v3`` — some models live under a different
namespace on ModelScope, such as ``AI-ModelScope/*``, ``openai-mirror/*``,
``nv-community/*``) with the ModelScope backend.

``modelscope`` is an *optional* dependency — it is only imported when the
ModelScope backend is actually selected, so existing users who never opt in
pay no cost and need not install it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

DownloadBackend = Literal["huggingface", "modelscope"]


def resolve_backend(backend: DownloadBackend | None = None) -> DownloadBackend:
    """Return the effective download backend.

    Precedence: explicit arg > COREAI_DOWNLOAD_BACKEND > huggingface.
    """
    if backend is not None:
        if backend not in ("huggingface", "modelscope"):
            raise ValueError(
                f"Unknown download backend {backend!r}. Expected 'huggingface' or 'modelscope'."
            )
        return backend

    env = os.environ.get("COREAI_DOWNLOAD_BACKEND", "").strip().lower()
    if env:
        if env not in ("huggingface", "modelscope"):
            raise ValueError(
                f"COREAI_DOWNLOAD_BACKEND={env!r} is invalid. "
                "Expected 'huggingface' or 'modelscope'."
            )
        return env  # type: ignore[return-value]

    return "huggingface"


def _download_huggingface(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    import huggingface_hub

    kwargs: dict = {
        "allow_patterns": allow_patterns,
        "local_files_only": local_files_only,
    }
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return huggingface_hub.snapshot_download(model_id, **kwargs)


def _download_modelscope(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    try:
        from modelscope import snapshot_download as _ms_snapshot
    except ImportError as exc:
        raise ImportError(
            "The ModelScope download backend requires the 'modelscope' package. "
            "Install it with: pip install modelscope  (or: uv pip install modelscope)"
        ) from exc

    # The caller supplies a ModelScope id (e.g. ``AI-ModelScope/whisper-large-v3``)
    # when the ModelScope backend is selected.
    kwargs: dict = {
        "allow_patterns": allow_patterns,
        "local_files_only": local_files_only,
    }
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    # ModelScope's snapshot_download accepts ``model_id`` (or ``repo_id``) as
    # the first positional argument and returns a local path string, mirroring
    # huggingface_hub.snapshot_download closely.
    return _ms_snapshot(model_id, **kwargs)


def download_snapshot(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    backend: DownloadBackend | None = None,
) -> str:
    """Download (or resolve from cache) a full model snapshot and return its
    local path string.

    Args:
        model_id: Model identifier in the *selected hub's* namespace —
            HuggingFace ids (``org/name``) for the default backend,
            ModelScope ids (e.g. ``AI-ModelScope/<name>``) for the
            ModelScope backend.
        allow_patterns: Glob patterns restricting which files to fetch.
        local_files_only: When True, only use the local cache; do not hit the network.
        token: Optional auth token (HF token or ModelScope SDK token).
        cache_dir: Optional cache root override.
        backend: Force a backend; otherwise resolved via :func:`resolve_backend`.

    Returns:
        Local filesystem path to the snapshot directory.
    """
    resolved = resolve_backend(backend)
    fn = _download_modelscope if resolved == "modelscope" else _download_huggingface
    return fn(
        model_id,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        token=token,
        cache_dir=cache_dir,
    )


def resolve_model_path(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    cache_dir: str | Path | None = None,
    backend: DownloadBackend | None = None,
) -> str:
    """Resolve a model id to a local snapshot path suitable for passing to
    ``transformers``/``diffusers`` ``from_pretrained(local_path)``.

    This is the bridge that lets ModelScope-backed downloads flow through the
    existing ``AutoConfig.from_pretrained(path)`` / ``AutoTokenizer.from_pretrained(path)``
    call sites: we pre-download the snapshot (only the files the caller needs,
    via ``allow_patterns``) and hand back the local directory.

    Pass the returned path where a HuggingFace model id currently goes.
    Callers that also need ``local_files_only`` or ``token`` should call
    :func:`download_snapshot` directly.
    """
    return download_snapshot(
        model_id,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        backend=backend,
    )
