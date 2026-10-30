# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Attribution metadata in the Parakeet TDT export recipe (``models/parakeet/export.py``).

Nothing here downloads weights or runs an export.
"""

import importlib.util
from pathlib import Path

import pytest

# The export script is co-located under models/, loaded by path.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "parakeet_export", _REPO_ROOT / "models/parakeet/export.py"
)
export_parakeet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_parakeet)


def test_v3_metadata_credits_v3() -> None:
    metadata = export_parakeet._build_aimodel_metadata("encoder", "nvidia/parakeet-tdt-0.6b-v3")
    assert metadata.author == "M. Sekoyan et al."
    assert metadata.license == "CC-BY-4.0"
    assert metadata.model_description.startswith("Parakeet-TDT v3 ASR (encoder subgraph).")
    assert metadata.model_description.endswith("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3")


def test_v2_metadata_credits_nvidias_original() -> None:
    """The weights come from a converted repo, but the asset credits NVIDIA's model."""
    metadata = export_parakeet._build_aimodel_metadata("joint", "rahulrachuri/parakeet-tdt-0.6b-v2")
    assert metadata.author == "NVIDIA"
    assert metadata.license == "CC-BY-4.0"
    assert metadata.model_description.startswith("Parakeet-TDT v2 ASR (joint subgraph).")
    assert metadata.model_description.endswith("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2")


def test_unsupported_model_has_no_metadata() -> None:
    with pytest.raises(KeyError):
        export_parakeet._build_aimodel_metadata("encoder", "nvidia/parakeet-tdt-0.6b-v1")
