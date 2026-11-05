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
                "(>=1.37). Install it with: uv run --extra modelscope "
                '(or uv pip install "modelscope>=1.37")'
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
