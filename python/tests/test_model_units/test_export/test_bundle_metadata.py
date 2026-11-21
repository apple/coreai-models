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


def _fake_pipe(neural_engine: bool) -> tuple[object, str]:
    """A pipe for the type the exporter pairs with ``neural_engine``: only Sana Sprint has a
    Neural Engine build."""
    if not neural_engine:
        pipe = SimpleNamespace(
            vae=SimpleNamespace(config=SimpleNamespace(block_out_channels=[1, 2, 3, 4])),
            transformer=SimpleNamespace(config=SimpleNamespace()),
        )
        return pipe, "flux2"

    from diffusers import SanaSprintPipeline

    class FakeSanaPipe:
        # `_build_sana_sprint_config` reads the real pipeline's call defaults.
        __call__ = SanaSprintPipeline.__call__
        vae = SimpleNamespace(
            config=SimpleNamespace(encoder_block_out_channels=[1, 2, 3, 4, 5, 6], scaling_factor=1)
        )
        transformer = SimpleNamespace(config=SimpleNamespace(sample_size=32))
        tokenizer = SimpleNamespace(encode=lambda text: [0] * 8)

    return FakeSanaPipe(), "sana_sprint"


def _write_diffusion_metadata(out: Path, assets: list[str], *, neural_engine: bool) -> dict:
    """Write a diffusion manifest for empty ``<name>.aimodel`` assets."""
    from coreai_models.diffusion.pipeline import _write_metadata_json

    exported = {}
    for name in assets:
        (out / f"{name}.aimodel").mkdir(exist_ok=True)
        exported[name] = str(out / f"{name}.aimodel")
    pipe, pipeline_type = _fake_pipe(neural_engine)
    _write_metadata_json(
        pipe, "m", pipeline_type, out, "none", exported, neural_engine=neural_engine
    )
    return json.loads((out / "metadata.json").read_text())["diffusion"]


def test_diffusion_lists_the_compute_unit_of_every_asset(tmp_path: Path) -> None:
    gpu = _write_diffusion_metadata(tmp_path, ["transformer", "text_encoder"], neural_engine=False)
    assert gpu["compute_units"] == {"transformer": ["gpu"], "text_encoder": ["gpu"]}

    ne = tmp_path / "ne"
    ne.mkdir()
    units = _write_diffusion_metadata(ne, ["transformer", "vae_decoder"], neural_engine=True)
    assert units["type"] == "sana-sprint"
    assert units["compute_units"] == {
        "transformer": ["neural_engine"],
        "vae_decoder": ["neural_engine"],
    }


def test_partial_diffusion_export_stamps_every_merged_asset_with_its_unit(tmp_path: Path) -> None:
    """The merged asset list takes this run's unit, as the whole directory shares one: the
    Neural Engine build writes to its own ``<model>-ios/`` directory."""
    _write_diffusion_metadata(tmp_path, ["transformer", "vae_decoder"], neural_engine=True)
    units = _write_diffusion_metadata(tmp_path, ["transformer"], neural_engine=True)
    assert units["compute_units"] == {
        "transformer": ["neural_engine"],
        "vae_decoder": ["neural_engine"],
    }
