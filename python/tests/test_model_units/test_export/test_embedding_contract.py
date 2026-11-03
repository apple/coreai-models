# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the generic embedding bundle code: contract, export helpers, orchestrator and CLI.

The orchestrator and CLI tests use a synthetic text-only exporter (a tiny ``nn.Module``). Its
programs also run on the Core AI runtime (CPU and GPU) when it is available.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import tomllib
import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.embedding import export as embedding_export
from coreai_models.embedding.contract import (
    AudioPreprocessing,
    EmbeddingFamily,
    EmbeddingPrecisionError,
    ImagePreprocessing,
    build_embedding_bundle_metadata,
    bundle_name,
    function_entry,
    media_block,
    resolve_compute_precision,
    text_block,
    validate_block,
)
from coreai_models.embedding.export_utils import (
    EmbeddingExporter,
    export_program,
    parse_spec,
    resolve_snapshot,
    save_program,
    save_tokenizer,
    trace_inputs,
)
from tests.test_model_units.test_export._embedding_host import (
    REVISION,
    Bundle,
    asset_signatures,
    runtime_options,
    runtime_type,
    save_tiny_tokenizer,
    snapshot_dir,
)

WORDS = ["<unk>", "a", "photo", "of", "the", "cat", "dog", "sound"]
TOY = EmbeddingFamily(
    key="toy-embed",
    hf_model_id="acme/toy-embed",
    dimension=8,
    bos_token_id=8,
    eos_token_id=9,
    pad_token_id=10,
    max_tokens=6,
    precisions={"macOS": ("float32", "float16"), "iOS": ("float16",)},
)


class ToyTextTower(nn.Module):
    """Masked mean of token embeddings, L2-normalized."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(11, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask[..., None].float()
        pooled = (self.embed(input_ids) * mask).sum(1) / mask.sum(1)
        return F.normalize(pooled, dim=-1)


def toy_block(revision: str) -> dict[str, Any]:
    component = {
        "asset": "text_encoder",
        "inputs": {"input_ids": "int32[B,L]", "attention_mask": "int32[B,L]"},
        "outputs": {"embedding": "float32[B,8]"},
        "compute_units": ["gpu"],
        "functions": [function_entry("text_encoder", (1, 4), TOY.max_tokens)],
    }
    return {
        "space": {"id": TOY.hf_model_id, "revision": revision},
        "dimension": 8,
        "normalized": True,
        "metric": "cosine",
        "logit_scale": 1.0,
        "compute_precision": "float32",
        "components": {"text_encoder": component},
        "towers": [{"modality": "text", "pipeline": ["text_encoder"]}],
        "text": text_block(TOY),
    }


def toy_export_programs(snapshot, out, *, revision, include_debug_info, **_):
    block = toy_block(revision)
    spec = block["components"]["text_encoder"]
    program = export_program([(ToyTextTower(), spec)], include_debug_info)
    save_program(program, out, TOY, "text_encoder")
    return {"text_encoder": "text_encoder.aimodel"}, block


TOY_EXPORTER = EmbeddingExporter(TOY, lambda snapshot: None, toy_export_programs)


@pytest.fixture
def toy_snapshot(tmp_path) -> Path:
    snapshot = snapshot_dir(tmp_path)
    save_tiny_tokenizer(snapshot, WORDS, TOY.bos_token_id, TOY.eos_token_id, TOY.pad_token_id)
    return snapshot


# --- contract (torch-free) ---


def test_block_builders():
    assert function_entry("f", 1) == {"function": "f", "batch": 1}
    assert function_entry("f", (1, 64), 77) == {
        "function": "f",
        "batch": {"min": 1, "max": 64},
        "length": 77,
    }
    assert text_block(TOY) == {
        "embedded_tokenizer": True,
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token_id": 8,
        "eos_token_id": 9,
        "pad_token_id": 10,
        "padding_side": "right",
        "truncation": {"max_tokens": 6, "keep_eos": True},
    }
    image = ImagePreprocessing("center_crop", 32, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    audio = AudioPreprocessing(16000, 400, 400, 160, 8, 11, 0.0, 8000.0)
    family = dataclasses.replace(TOY, image=image)
    assert family.modalities == ("text", "image")
    assert media_block(family, "image") == {
        "image_strategy": "center_crop",
        "image_size": 32,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
    }
    assert list(media_block(dataclasses.replace(TOY, audio=audio), "audio")) == [
        "sample_rate",
        "n_fft",
        "win_length",
        "hop_length",
        "n_mel_bins",
        "n_frames",
        "f_min",
        "f_max",
    ]
    with pytest.raises(ValueError, match="no audio tower"):
        media_block(family, "audio")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b.pop("logit_scale"), "missing 'logit_scale'"),
        (lambda b: b["components"]["text_encoder"].update(asset="other"), "not in assets"),
        (lambda b: b["components"]["text_encoder"].update(functions={}), "non-empty array"),
        (
            lambda b: b["components"]["text_encoder"]["functions"].append(
                function_entry("text_encoder", 1, 6)
            ),
            "duplicate",
        ),
        (lambda b: b["towers"].append({"modality": "text", "pipeline": ["x"]}), "one text"),
        (lambda b: b["towers"][0].update(pipeline=["missing"]), "unknown components"),
        (
            lambda b: b["towers"].append({"modality": "video", "pipeline": ["text_encoder"]}),
            "unknown modality",
        ),
        (
            lambda b: b["towers"].append({"modality": "image", "pipeline": ["text_encoder"]}),
            "image block",
        ),
        (lambda b: b.update(audio={}), "audio block"),
    ],
)
def test_validate_block_rejects(mutate, message):
    block = copy.deepcopy(toy_block(REVISION))
    mutate(block)
    with pytest.raises(ValueError, match=message):
        validate_block(block, {"text_encoder": "text_encoder.aimodel"})


def test_precision_follows_the_family_table():
    assert resolve_compute_precision(TOY, None, "macOS") == "float32"
    assert resolve_compute_precision(TOY, None, "iOS") == "float16"
    assert resolve_compute_precision(TOY, "float16", "macOS") == "float16"
    with pytest.raises(EmbeddingPrecisionError, match="float32 or float16, not bfloat16"):
        resolve_compute_precision(TOY, "bfloat16", "macOS")
    with pytest.raises(EmbeddingPrecisionError, match="no visionOS export"):
        resolve_compute_precision(TOY, None, "visionOS")


def test_bundle_names_and_envelope():
    assert bundle_name(TOY, "macOS", False, False) == "toy-embed"
    assert bundle_name(TOY, "iOS", True, False) == "toy-embed-ios-text"
    assert bundle_name(TOY, "macOS", False, True) == "toy-embed-fused"
    meta = build_embedding_bundle_metadata(
        TOY, {}, name="toy-embed", assets={}, hf_revision=REVISION
    )
    assert meta["metadata_version"] == "0.2" and meta["kind"] == "embedding"
    assert meta["source"] == {
        "model_definition": "torch",
        "hf_model_id": TOY.hf_model_id,
        "hf_revision": REVISION,
    }
    assert meta["compression"] is None and meta["compilation"]["targets"] == []


# --- export helpers ---


def test_trace_inputs_follow_the_specs():
    inputs = {
        "pixel_values": "float32[N,3,4,4]",
        "input_ids": "int32[B,L]",
        "attention_mask": "int32[B,L]",
        "is_longer": "bool[N,1]",
    }
    args, shapes = trace_inputs(inputs, function_entry("main", (1, 64), 7))
    assert [tuple(a.shape) for a in args] == [(3, 3, 4, 4), (2, 7), (2, 7), (3, 1)]
    assert [a.dtype for a in args] == [torch.float32, torch.int32, torch.int32, torch.bool]
    assert torch.all(args[2] == 1) and torch.all(args[1] == 0)
    batch, other = shapes[1][0], shapes[0][0]
    assert shapes[2][0] is batch and shapes[3][0] is other and batch is not other
    assert (batch.min, batch.max) == (1, 64) and list(shapes[1]) == [0]
    args, shapes = trace_inputs({"x": "float32[B,2]"}, function_entry("f", 5))
    assert args[0].shape == (5, 2) and shapes == ({},)
    assert parse_spec("float32[B,512]") == ("float32", ["B", "512"])
    with pytest.raises(ValueError, match="bad tensor spec"):
        parse_spec("float32(B)")


def test_resolve_snapshot_reads_the_revision(tmp_path, toy_snapshot):
    assert resolve_snapshot(TOY.hf_model_id, str(toy_snapshot)) == (toy_snapshot, REVISION)
    assert resolve_snapshot(TOY.hf_model_id, str(tmp_path))[1] == ""
    with pytest.raises(FileNotFoundError):
        resolve_snapshot(TOY.hf_model_id, str(tmp_path / "missing"))


def test_save_tokenizer_turns_padding_off_and_checks_ids(tmp_path, toy_snapshot):
    save_tokenizer(toy_snapshot, tmp_path / "tok", TOY)
    spec = json.loads((tmp_path / "tok" / "tokenizer.json").read_text())
    assert spec["padding"] is None and spec["truncation"] is None
    with pytest.raises(ValueError, match="disagrees with the family"):
        save_tokenizer(toy_snapshot, tmp_path / "bad", dataclasses.replace(TOY, pad_token_id=9))


def test_bpe_merges_are_written_as_pairs(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        special_tokens=["<unk>", "<s>", "</s>"], end_of_word_suffix="</w>", vocab_size=60
    )
    tokenizer.train_from_iterator(["a photo of a cat", "a photo of the dog"] * 4, trainer)
    tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1))
    snapshot = tmp_path / "snap"
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", pad_token="</s>"
    ).save_pretrained(snapshot)
    family = dataclasses.replace(TOY, bos_token_id=1, eos_token_id=2, pad_token_id=2)
    save_tokenizer(snapshot, tmp_path / "tok", family)
    merges = json.loads((tmp_path / "tok" / "tokenizer.json").read_text())["model"]["merges"]
    assert merges and all(isinstance(m, list) and len(m) == 2 for m in merges)


# --- orchestrator (synthetic exporter) ---


def test_export_writes_the_bundle(tmp_path, toy_snapshot):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "stale.txt").write_text("replaced")
    meta = embedding_export.export_embedding_bundle(
        TOY_EXPORTER, out, overwrite=True, model_dir=str(toy_snapshot)
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "metadata.json",
        "text_encoder.aimodel",
        "tokenizer",
    ]
    assert json.loads((out / "metadata.json").read_text()) == meta
    assert meta["name"] == "toy-embed" and meta["source"]["hf_revision"] == REVISION
    spec = meta["embedding"]["components"]["text_encoder"]
    inputs, outputs = asset_signatures(out / "text_encoder.aimodel")["text_encoder"]
    assert inputs == [(k, runtime_type(v, TOY.max_tokens)) for k, v in spec["inputs"].items()]
    assert outputs == [(k, runtime_type(v, None)) for k, v in spec["outputs"].items()]


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_toy_bundle_runs_from_its_metadata(compute_unit, tmp_path, toy_snapshot):
    """The reference host tokenizes (with truncation keeping eos), pads, batches above
    ``batch.max`` and matches the PyTorch tower.
    """
    out = tmp_path / "bundle"
    embedding_export.export_embedding_bundle(TOY_EXPORTER, out, model_dir=str(toy_snapshot))
    bundle = Bundle(out, runtime_options(compute_unit))
    texts = ["a cat", "the dog of the cat photo", "a", "sound", "cat dog"]
    ids, mask = bundle.tokenize(texts)
    assert ids[1].tolist() == [8, 4, 6, 3, 4, 9] and mask[1].tolist() == [1] * 6
    assert ids[0].tolist() == [8, 1, 5, 9, 10, 10] and mask[0].tolist() == [1, 1, 1, 1, 0, 0]
    got = bundle.embed_text(texts)
    with torch.no_grad():
        want = ToyTextTower()(ids, mask)
    assert got.shape == (5, 8)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_failed_or_refused_exports_leave_no_bundle(tmp_path, toy_snapshot):
    def broken(snapshot, out, **kwargs):
        assets, block = toy_export_programs(snapshot, out, **kwargs)
        block["towers"] = []
        return assets, block

    export = embedding_export.export_embedding_bundle
    out = tmp_path / "bundle"
    with pytest.raises(EmbeddingPrecisionError):
        export(TOY_EXPORTER, out, compute_precision="bfloat16")
    with pytest.raises(ValueError, match="no fused export"):
        export(TOY_EXPORTER, out, fused=True)
    with pytest.raises(ValueError, match="one text tower"):
        export(
            dataclasses.replace(TOY_EXPORTER, export_programs=broken),
            out,
            model_dir=str(toy_snapshot),
        )
    with pytest.raises(ValueError, match="snapshot revision"):
        export(TOY_EXPORTER, out, model_dir=str(tmp_path))
    assert not out.exists()
    out.mkdir()
    (out / "stale.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export(TOY_EXPORTER, out, model_dir=str(toy_snapshot))
    assert (out / "stale.txt").read_text() == "keep"


# --- CLI ---


@pytest.fixture
def seen(monkeypatch) -> dict[str, Any]:
    """Registers the toy exporter and records the CLI's ``export_embedding_bundle`` call."""
    monkeypatch.setitem(embedding_export.EXPORTERS, TOY.key, TOY_EXPORTER)
    calls: dict[str, Any] = {}

    def fake(exporter, out, **kwargs):
        calls.update(exporter=exporter, out=Path(out), **kwargs)
        return {}

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fake)
    return calls


@pytest.mark.parametrize(
    ("argv", "name", "text_only"),
    [
        ([], "toy-embed", False),
        (["--text-only"], "toy-embed-text", True),
        (["--skip-vision"], "toy-embed-text", True),
        (["--platform", "iOS"], "toy-embed-ios", False),
    ],
)
def test_cli_passes_the_options(argv, name, text_only, tmp_path, seen):
    assert embedding_export.main(["acme/toy-embed", *argv, "--output-dir", str(tmp_path)]) == 0
    assert seen["exporter"] is TOY_EXPORTER and seen["out"] == tmp_path / name
    assert seen["text_only"] is text_only and seen["fused"] is False
    assert seen["compute_precision"] == ("float16" if "iOS" in argv else "float32")


def test_cli_default_output_is_under_exports(seen):
    assert embedding_export.main(["toy-embed"]) == 0
    assert seen["out"].parent.name == "exports"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["acme/none"], "--list-models"),
        ([], "a model is required"),
        (["--dry-run"], "a model is required"),
        (["toy-embed", "--compute-precision", "bfloat16"], "not bfloat16"),
        (["toy-embed", "--fused"], "no fused export"),
    ],
)
def test_cli_errors_exit_2_before_exporting(argv, message, seen, capsys):
    with pytest.raises(SystemExit) as info:
        embedding_export.main(argv)
    assert info.value.code == 2 and message in capsys.readouterr().err
    assert not seen


def test_cli_dry_run_and_list_models(seen, capsys):
    assert embedding_export.main(["toy-embed", "--platform", "iOS", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "iOS" in out and "float16" in out and "toy-embed-ios" in out
    assert not seen
    assert embedding_export.main(["--list-models"]) == 0
    assert "acme/toy-embed" in capsys.readouterr().out


def test_console_script_is_registered():
    root = Path(__file__).resolve().parents[4]
    scripts = tomllib.loads((root / "python" / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["coreai.embedding.export"] == "coreai_models.embedding.export:main"
