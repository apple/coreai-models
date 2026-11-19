# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""LLM bundle metadata_version contract: 0.3 (with max_batch_size) for batched bundles,
0.2 for everything else."""

import json
from pathlib import Path
from types import SimpleNamespace

from coreai_models._constants import METADATA_VERSION, METADATA_VERSION_BATCHED
from coreai_models.export.bundle import _write_metadata


def test_llm_writer_defaults_to_0_2(tmp_path: Path) -> None:
    """Omitted max_batch_size => single-sequence => 0.2 schema."""
    cfg = SimpleNamespace(vocab_size=256, max_position_embeddings=4096)
    _write_metadata(tmp_path, "org/model", cfg, "4bit", "model")
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["metadata_version"] == "0.2"


def test_llm_writer_omits_max_batch_size_when_single_sequence(tmp_path: Path) -> None:
    """A 0.2 bundle does not carry max_batch_size, keeping its output byte-identical."""
    cfg = SimpleNamespace(vocab_size=256, max_position_embeddings=4096)
    _write_metadata(tmp_path, "org/model", cfg, "4bit", "model")
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert "max_batch_size" not in meta["language"]


def test_llm_writer_emits_0_3_when_batched(tmp_path: Path) -> None:
    """A dynamic-batch export (max_batch_size > 1) stamps the 0.3 schema."""
    cfg = SimpleNamespace(vocab_size=256, max_position_embeddings=4096)
    _write_metadata(tmp_path, "org/model", cfg, "4bit", "model", max_batch_size=8)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["metadata_version"] == "0.3"


def test_llm_writer_stamps_max_batch_size_when_batched(tmp_path: Path) -> None:
    """A batched export records its max batch so the runtime can reject larger requests."""
    cfg = SimpleNamespace(vocab_size=256, max_position_embeddings=4096)
    _write_metadata(tmp_path, "org/model", cfg, "4bit", "model", max_batch_size=8)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["language"]["max_batch_size"] == 8


def test_version_constants() -> None:
    """The default schema is 0.2; batched LLM bundles use 0.3."""
    assert METADATA_VERSION == "0.2"
    assert METADATA_VERSION_BATCHED == "0.3"


def test_other_bundle_kinds_stay_on_0_2() -> None:
    """Conditional versioning leaves diffusion (and the other kinds) on 0.2."""
    from coreai_models.diffusion.pipeline import METADATA_VERSION as DIFFUSION_VERSION

    assert DIFFUSION_VERSION == "0.2"
