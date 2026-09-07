# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Zero-dependency download shim for standalone ``models/*/export.py`` scripts.

Standalone export scripts run via PEP 723 inline dependencies whose
transformers constraints conflict with the ``coreai-models`` package (most
pin ``transformers==4.x``; the rest pin 5.x ranges that exclude the
package's exact ``transformers==5.12.1``). Adding ``coreai-models`` to
those inline dependencies would fail resolution, so they cannot import
``coreai_models._download``. This shim is a self-contained subset that
replicates the same backend-selection logic *without* importing
``coreai_models``.

Model ids belong to the selected hub's namespace: use a HuggingFace id
(``org/name``) with the default backend, or a ModelScope id (e.g.
``AI-ModelScope/whisper-large-v3``) together with
``COREAI_DOWNLOAD_BACKEND=modelscope``.

Usage (from any ``models/<name>/export.py``):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _download_shim import resolve_model_path

    local_path = resolve_model_path(model_name)
    model = transformers.XXX.from_pretrained(local_path)

Keep this module in sync with ``python/src/coreai_models/_download.py`` —
specifically the env-var precedence and the invalid-value ``ValueError``
on ``COREAI_DOWNLOAD_BACKEND`` (the sync tests in
``python/tests/test_model_units/test_download.py`` pin both behaviors).

This is a *minimal subset* of the main module: ``resolve_model_path`` here
does not support ``local_files_only``, ``token``, or ``backend`` kwargs.
Since shim users by definition cannot install ``coreai-models`` (see above),
use the environment instead: ``COREAI_DOWNLOAD_BACKEND`` selects the backend
(this shim reads it), and the hubs' own variables (e.g. ``HF_HUB_OFFLINE=1``)
cover offline behaviour. A script that needs more than that has outgrown
the standalone pattern and should move to the ``coreai.*.export`` pipelines
or, like ``models/sam3``, depend on ``coreai-models`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_backend() -> str:
    """Resolve the download backend from environment variables.

    Precedence: COREAI_DOWNLOAD_BACKEND > huggingface.

    Kept in sync with ``coreai_models._download.resolve_backend`` — including
    raising ``ValueError`` on an invalid ``COREAI_DOWNLOAD_BACKEND`` value so
    that standalone scripts and the package-level pipeline fail the same way
    on a typo.
    """
    env = os.environ.get("COREAI_DOWNLOAD_BACKEND", "").strip().lower()
    if env:
        if env not in ("huggingface", "modelscope"):
            raise ValueError(
                f"COREAI_DOWNLOAD_BACKEND={env!r} is invalid. "
                "Expected 'huggingface' or 'modelscope'."
            )
        return env
    return "huggingface"


def resolve_model_path(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Pre-download a model snapshot and return its local path.

    Routes to HuggingFace (default) or ModelScope based on the
    ``COREAI_DOWNLOAD_BACKEND`` env var. The returned path is suitable for
    ``transformers.from_pretrained(path)``.
    """
    if _resolve_backend() == "modelscope":
        try:
            from modelscope import snapshot_download as _ms_snapshot
        except ImportError as exc:
            raise ImportError(
                "The ModelScope download backend requires the 'modelscope' package. "
                "This script runs in its own PEP 723 environment, which does not "
                "include it. Re-run with:  uv run --with modelscope <this-script.py> "
                "(plus COREAI_DOWNLOAD_BACKEND=modelscope)"
            ) from exc
        kwargs: dict = {"allow_patterns": allow_patterns}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        return _ms_snapshot(model_id, **kwargs)

    import huggingface_hub

    kwargs = {"allow_patterns": allow_patterns}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return huggingface_hub.snapshot_download(model_id, **kwargs)
