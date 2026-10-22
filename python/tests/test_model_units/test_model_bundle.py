# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_models.model_bundle."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from coreai_models._constants import METADATA_VERSION, METADATA_VERSION_BATCHED
from coreai_models.model_bundle import (
    BundleError,
    BundleValidationError,
    ModelBundle,
)


def _llm_metadata(name: str = "qwen3_0_6b_4bit_dynamic") -> dict[str, Any]:
    return {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": "Qwen/Qwen3-0.6B",
            "vocab_size": 151936,
            "max_context_length": 40960,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": "Qwen/Qwen3-0.6B"},
        "user_data": {"git_sha": "abc1234"},
    }


def _write_bundle(root: Path, metadata: dict[str, Any]) -> Path:
    bundle = root / metadata.get("name", "unnamed")
    bundle.mkdir(parents=True)
    for file in metadata.get("assets", {}).values():
        (bundle / file).mkdir()
        (bundle / file / "main.mlirb").write_bytes(b"\x00")
    (bundle / "tokenizer").mkdir()
    (bundle / "tokenizer" / "tokenizer.json").write_text("{}")
    (bundle / "metadata.json").write_text(json.dumps(metadata))
    return bundle


def _codes(issues: list, severity: str) -> list[str]:
    return [i.code for i in issues if i.severity == severity]


# --- load -------------------------------------------------------------------------------


def test_load_reads_the_common_fields(tmp_path: Path) -> None:
    bundle = ModelBundle.load(_write_bundle(tmp_path, _llm_metadata()))
    assert bundle.metadata_version == METADATA_VERSION
    assert bundle.kind == "llm"
    assert bundle.name == "qwen3_0_6b_4bit_dynamic"
    assert bundle.assets == {"main": "qwen3_0_6b_4bit_dynamic.aimodel"}
    assert bundle.auxiliary_assets == {}
    assert bundle.user_data == {"git_sha": "abc1234"}
    assert bundle.metadata["language"]["vocab_size"] == 151936


def test_asset_path(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _llm_metadata())
    bundle = ModelBundle.load(path)
    assert bundle.asset_path("main") == path / "qwen3_0_6b_4bit_dynamic.aimodel"
    with pytest.raises(BundleError, match="no asset with role 'vision'"):
        bundle.asset_path("vision")


def test_load_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="no such directory"):
        ModelBundle.load(tmp_path / "absent")


def test_load_reports_a_directory_without_metadata(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(BundleError, match="has no metadata.json"):
        ModelBundle.load(tmp_path / "empty")


def test_load_reports_invalid_json_with_its_location(tmp_path: Path) -> None:
    bundle = tmp_path / "broken"
    bundle.mkdir()
    (bundle / "metadata.json").write_text('{"name": "x",\n "kind": }')
    with pytest.raises(BundleError, match=r"not valid JSON \(line 2"):
        ModelBundle.load(bundle)


def test_load_needs_a_json_object(tmp_path: Path) -> None:
    bundle = tmp_path / "list"
    bundle.mkdir()
    (bundle / "metadata.json").write_text("[]")
    with pytest.raises(BundleError, match="JSON object"):
        ModelBundle.load(bundle)


# --- validate ---------------------------------------------------------------------------


def _validate(tmp_path: Path, metadata: dict[str, Any]) -> list:
    return ModelBundle.load(_write_bundle(tmp_path, metadata)).validate()


def test_a_well_formed_llm_bundle_validates_clean(tmp_path: Path) -> None:
    assert _validate(tmp_path, _llm_metadata()) == []


def test_the_batched_metadata_version_is_known(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    metadata["metadata_version"] = METADATA_VERSION_BATCHED
    assert _validate(tmp_path, metadata) == []


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("name", "missing_name"),
        ("kind", "missing_kind"),
        ("metadata_version", "missing_metadata_version"),
        ("language", "missing_language_block"),
    ],
)
def test_missing_required_fields_are_errors(tmp_path: Path, field: str, code: str) -> None:
    metadata = _llm_metadata()
    del metadata[field]
    assert code in _codes(_validate(tmp_path, metadata), "error")


def test_no_assets_is_an_error(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    metadata["assets"] = {}
    assert "no_assets" in _codes(_validate(tmp_path, metadata), "error")


def test_listed_assets_must_exist(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _llm_metadata())
    shutil.rmtree(path / "qwen3_0_6b_4bit_dynamic.aimodel")
    issues = ModelBundle.load(path).validate()
    assert "missing_asset" in _codes(issues, "error")
    assert any("assets.main" in i.message for i in issues)


def test_listed_assets_must_be_model_folders(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    path = _write_bundle(tmp_path, metadata)
    (path / "weights.bin").write_bytes(b"\x00")
    metadata["assets"]["main"] = "weights.bin"
    (path / "metadata.json").write_text(json.dumps(metadata))
    assert "not_a_model_asset" in _codes(ModelBundle.load(path).validate(), "error")


def test_listed_auxiliary_assets_must_exist(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    metadata["auxiliary_assets"] = {"per_layer_embeddings": "ple.bin"}
    assert "missing_auxiliary_asset" in _codes(_validate(tmp_path, metadata), "error")


def test_embedded_tokenizer_needs_the_tokenizer_folder(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _llm_metadata())
    shutil.rmtree(path / "tokenizer")
    assert "missing_tokenizer" in _codes(ModelBundle.load(path).validate(), "error")


def test_vlm_needs_the_vision_asset_and_block(tmp_path: Path) -> None:
    metadata = _llm_metadata("vlm")
    metadata["kind"] = "vlm"
    errors = _codes(_validate(tmp_path, metadata), "error")
    assert {"missing_vision_asset", "missing_vision_block"} <= set(errors)


@pytest.mark.parametrize("kind", ["diffusion", "video-diffusion"])
def test_diffusion_kinds_need_the_diffusion_block(tmp_path: Path, kind: str) -> None:
    metadata = {
        "metadata_version": METADATA_VERSION,
        "kind": kind,
        "name": "d",
        "assets": {"transformer": "Transformer.aimodel"},
    }
    assert "missing_diffusion_block" in _codes(_validate(tmp_path, metadata), "error")


def test_incomplete_language_block_is_a_warning(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    del metadata["language"]["vocab_size"]
    issues = _validate(tmp_path, metadata)
    assert _codes(issues, "error") == []
    assert "missing_language_field" in _codes(issues, "warning")


def test_unknown_kind_and_newer_version_are_warnings(tmp_path: Path) -> None:
    metadata = {
        "metadata_version": "0.9",
        "kind": "decision-tree",
        "name": "trees",
        "assets": {"main": "trees.aimodel"},
    }
    issues = _validate(tmp_path, metadata)
    assert _codes(issues, "error") == []
    assert {"unknown_kind", "unknown_metadata_version"} <= set(_codes(issues, "warning"))


def test_unlisted_model_folder_is_a_warning(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _llm_metadata())
    (path / "leftover.aimodel").mkdir()
    issues = ModelBundle.load(path).validate()
    assert "orphaned_asset" in _codes(issues, "warning")
    assert _codes(issues, "error") == []


# --- check ------------------------------------------------------------------------------


def test_check_raises_with_every_error(tmp_path: Path) -> None:
    metadata = _llm_metadata()
    del metadata["kind"]
    metadata["assets"] = {}
    bundle = ModelBundle.load(_write_bundle(tmp_path, metadata))
    with pytest.raises(BundleValidationError) as raised:
        bundle.check()
    assert {"missing_kind", "no_assets"} <= {i.code for i in raised.value.issues}
    assert "error [missing_kind]" in str(raised.value)


def test_check_returns_warnings_for_a_usable_bundle(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _llm_metadata())
    (path / "leftover.aimodel").mkdir()
    warnings = ModelBundle.load(path).check()
    assert [w.code for w in warnings] == ["orphaned_asset"]
