# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def embeddinggemma2_snapshot() -> Path:
    """Local google/embeddinggemma-2 snapshot; skips when it is not cached."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download("google/embeddinggemma-2", local_files_only=True))
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"google/embeddinggemma-2 is not in the local Hugging Face cache ({error})")
