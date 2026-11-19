# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Parakeet export recipe (``models/parakeet/export.py``).

The recipe is not part of the installed package, so it is loaded by path, after its sibling
``nemo_checkpoint.py``, which it imports by that name.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("yaml")

from tests._runner_infra._deps import _HAS_COREAI, _MSG_COREAI_NOT_FOUND  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND)

_PARAKEET = Path(__file__).resolve().parents[4] / "models/parakeet"


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PARAKEET / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if _HAS_COREAI:
    _load("nemo_checkpoint", "nemo_checkpoint.py")
    # Not registered as `export`: the Gemma 4 tests already claim that name.
    export_parakeet = _load("parakeet_export", "export.py")


def test_v3_metadata_credits_v3() -> None:
    metadata = export_parakeet._build_aimodel_metadata("encoder", "nvidia/parakeet-tdt-0.6b-v3")
    assert metadata.author == "M. Sekoyan et al."
    assert metadata.license == "CC-BY-4.0"
    assert metadata.model_description.startswith("Parakeet-TDT v3 ASR (encoder subgraph).")
    assert metadata.model_description.endswith("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3")


def test_v2_metadata_credits_nvidia() -> None:
    """v2's weights are converted from NeMo at export time, but the asset credits NVIDIA's repo."""
    metadata = export_parakeet._build_aimodel_metadata("joint", "nvidia/parakeet-tdt-0.6b-v2")
    assert metadata.author == "NVIDIA"
    assert metadata.license == "CC-BY-4.0"
    assert metadata.model_description.startswith("Parakeet-TDT v2 ASR (joint subgraph).")
    assert metadata.model_description.endswith("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2")


def test_unsupported_model_has_no_metadata() -> None:
    with pytest.raises(KeyError):
        export_parakeet._build_aimodel_metadata("encoder", "nvidia/parakeet-tdt-0.6b-v1")
