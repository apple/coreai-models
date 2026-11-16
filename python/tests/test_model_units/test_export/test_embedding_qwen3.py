# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the Qwen3-Embedding exporter (``coreai_models.embedding.qwen3_embedding``).

Metadata, snapshot and CLI tests need no hardware or weights. The other tests save a tiny random
Qwen3 checkpoint in a fake snapshot and compare the encoder and the exported bundle with
Hugging Face's ``Qwen3Model`` (last token, L2-normalized); the bundle also runs on the Core AI
runtime (CPU and GPU) when it is available.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from coreai_models.embedding import export as embedding_export
from coreai_models.embedding import qwen3_embedding as q3
from coreai_models.embedding.contract import CheckpointFacts, validate_block
from coreai_models.embedding.qwen3_embedding import (
    QWEN3_EMBEDDING_0_6B,
    embedding_block,
    read_facts,
    text_functions,
)
from tests.test_model_units.test_export._embedding_host import (
    REVISION,
    Bundle,
    asset_signatures,
    run_function,
    runtime_options,
    save_tiny_tokenizer,
)

FAMILY = QWEN3_EMBEDDING_0_6B
# The real checkpoint adds no bos; <|endoftext|> ends every text and is the pad token.
FACTS = CheckpointFacts(
    dimension=1024, bos_token_id=None, eos_token_id=151643, pad_token_id=151643, max_tokens=32768
)
PROMPTS = {
    "query": "Instruct: Given a web search query, retrieve relevant passages that answer the "
    "query\nQuery:",
    "document": "",
}
MODULES = [
    {"path": "", "type": "sentence_transformers.models.Transformer"},
    {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]
POOLING = {
    "word_embedding_dimension": 32,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": True,
    "include_prompt": True,
}
EOS = 2
# Like the real tokenizer, the tiny one names <|im_end|> as its eos_token but appends EOS.
IM_END = 9
WORDS = ["<unk>", "a", "photo", "of", "the", "cat", "dog", "sound"]
TINY_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 16,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "bos_token_id": EOS,
    "eos_token_id": EOS,
}
# Functions len16_b1 and len16_b2.
TINY = dataclasses.replace(
    FAMILY, length_buckets=(16,), batch_sizes=(2,), max_tokens=64, matryoshka_dimensions=(32, 16)
)
# Functions len512_b1 and len512_b8, on a checkpoint with 512 positions.
TINY_512 = dataclasses.replace(TINY, length_buckets=(512,), batch_sizes=(8,), max_tokens=512)
TINY_512_CONFIG = {**TINY_CONFIG, "max_position_embeddings": 512}


def _snapshot(
    root: Path,
    *,
    pooling=POOLING,
    modules=MODULES,
    weights: bool = False,
    config: dict = TINY_CONFIG,
) -> Path:
    """A fake HF snapshot of a tiny Qwen3-Embedding: tokenizer, sentence-transformers files and
    ``config.json``; with ``weights``, a random ``Qwen3Model`` checkpoint (keys without a prefix,
    as in the real one).
    """
    snap = root / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / REVISION
    (snap / "1_Pooling").mkdir(parents=True)
    files = {
        "modules.json": modules,
        "1_Pooling/config.json": pooling,
        "config_sentence_transformers.json": {"prompts": PROMPTS, "default_prompt_name": None},
    }
    for name, content in files.items():
        (snap / name).write_text(json.dumps(content))
    save_tiny_tokenizer(snap, WORDS, None, EOS, EOS, named_eos=IM_END)
    if weights:
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(0)
        model = Qwen3Model(Qwen3Config(**config))
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(std=0.5)
        model.save_pretrained(snap)
    else:
        (snap / "config.json").write_text(json.dumps({"model_type": "qwen3", **config}))
    return snap


def _ragged_inputs(
    lengths: tuple[int, ...] = (9, 16), width: int = 16
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Right-padded rows of ``lengths`` tokens in ``width`` columns, eos last (by default two rows
    of 9 and 16 tokens for the length-16 functions).
    """
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(3, TINY_CONFIG["vocab_size"], (len(lengths), width), generator=generator)
    mask = torch.zeros((len(lengths), width), dtype=torch.int32)
    for row, n in enumerate(lengths):
        ids[row, n - 1] = EOS
        ids[row, n:] = EOS
        mask[row, :n] = 1
    return ids.to(torch.int32), mask, list(lengths)


def _hugging_face_embeddings(snapshot: Path, ids: torch.Tensor, lengths: list[int]):
    """The model card's reference: each row unpadded, last token, L2-normalized."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(snapshot, dtype=torch.float32).eval()
    with torch.no_grad():
        rows = [
            model(input_ids=ids[i : i + 1, :n].long()).last_hidden_state[0, -1]
            for i, n in enumerate(lengths)
        ]
    return F.normalize(torch.stack(rows), dim=-1)


# --- Function plan and metadata ---


def test_function_plan():
    assert [f.name for f in text_functions(FAMILY)] == [
        "len128_b1",
        "len128_b8",
        "len128_b32",
        "len512_b1",
        "len512_b8",
        "len512_b32",
        "len2048_b1",
        "len2048_b8",
    ]


def test_embedding_block_describes_the_text_tower():
    block = embedding_block(FAMILY, FACTS, revision=REVISION, prompts=PROMPTS)
    validate_block(block, {"text_encoder": "text_encoder.aimodel"})
    assert block["space"] == {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": REVISION}
    assert block["compute_precision"] == "float32"
    assert (block["dimension"], block["normalized"], block["metric"]) == (1024, True, "cosine")
    assert block["matryoshka_dimensions"] == [1024, 512, 256, 128, 64, 32]
    assert block["towers"] == [{"modality": "text", "pipeline": ["text_encoder"]}]
    (component,) = block["components"].values()
    assert component["inputs"] == {"input_ids": "int32[B,L]", "attention_mask": "int32[B,L]"}
    assert component["outputs"] == {"embedding": "float32[B,1024]"}
    assert component["compute_units"] == ["gpu"]
    assert block["text"] == {
        "embedded_tokenizer": True,
        "add_bos_token": False,
        "add_eos_token": True,
        "bos_token_id": None,
        "eos_token_id": 151643,
        "pad_token_id": 151643,
        "padding_side": "right",
        "truncation": {"max_tokens": 2048, "keep_eos": True},
        "model_max_tokens": 32768,
        "prompts": PROMPTS,
        "default_prompt_name": None,
    }
    for precision in ("float16", "bfloat16"):
        block = embedding_block(
            FAMILY, FACTS, revision=REVISION, prompts=PROMPTS, compute_precision=precision
        )
        assert block["compute_precision"] == precision
        assert block["components"]["text_encoder"]["outputs"] == {"embedding": "float32[B,1024]"}


@pytest.mark.parametrize(
    ("family", "overrides", "message"),
    [
        (FAMILY, {"revision": ""}, "revision"),
        (FAMILY, {"prompts": {"query": PROMPTS["query"]}}, r"missing \['document'\]"),
        (FAMILY, {"default_prompt_name": "nope"}, "default_prompt_name 'nope'"),
        (dataclasses.replace(FAMILY, matryoshka_dimensions=(512,)), {}, "native dimension"),
        (dataclasses.replace(FAMILY, length_buckets=(65536,)), {}, "exceeds max_tokens"),
    ],
)
def test_embedding_block_rejects(family, overrides, message):
    kwargs = {"revision": REVISION, "prompts": PROMPTS, **overrides}
    with pytest.raises(ValueError, match=message):
        embedding_block(family, FACTS, **kwargs)


# --- Snapshot checks ---


def test_read_facts_from_the_snapshot(tmp_path):
    """eos is the token the tokenizer appends (config.json ``eos_token_id``), not its
    ``eos_token``; a family without ``max_tokens`` is refused.
    """
    from transformers import AutoTokenizer

    snapshot = _snapshot(tmp_path)
    facts = read_facts(TINY, snapshot)
    assert facts == CheckpointFacts(
        dimension=32, bos_token_id=None, eos_token_id=EOS, pad_token_id=EOS, max_tokens=64
    )
    assert AutoTokenizer.from_pretrained(snapshot).eos_token_id == IM_END
    with pytest.raises(ValueError, match="sets no max_tokens"):
        read_facts(dataclasses.replace(TINY, max_tokens=None), snapshot)


@pytest.mark.parametrize(
    ("pooling", "modules", "message"),
    [
        ({**POOLING, "pooling_mode_mean_tokens": True}, MODULES, "pooling modes"),
        ({**POOLING, "include_prompt": False}, MODULES, "include_prompt"),
        ({**POOLING, "word_embedding_dimension": 16}, MODULES, "word_embedding_dimension"),
        (POOLING, MODULES[:2], "modules"),
    ],
)
def test_read_facts_rejects_a_snapshot_unlike_the_family(tmp_path, pooling, modules, message):
    with pytest.raises(ValueError, match=message):
        read_facts(TINY, _snapshot(tmp_path, pooling=pooling, modules=modules))


# --- Encoder and bundle (tiny random checkpoint) ---


def test_encoder_matches_hugging_face(tmp_path):
    """Right padding changes nothing: the encoder on a ragged batch equals Hugging Face on each
    row alone. A row without real tokens stays finite.
    """
    from coreai_models.models.macos.qwen3_embedding import load_qwen3_embedding_encoder

    snapshot = _snapshot(tmp_path, weights=True)
    ids, mask, lengths = _ragged_inputs()
    encoder = load_qwen3_embedding_encoder(snapshot)
    with torch.no_grad():
        got = encoder(ids, mask)
        empty = encoder(ids[:1], torch.zeros_like(mask[:1]))
    torch.testing.assert_close(got, _hugging_face_embeddings(snapshot, ids, lengths))
    assert torch.isfinite(empty).all()


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory) -> tuple[Path, Path]:
    """``(snapshot, bundle)``: ``TINY`` exported by the exporter from a tiny random checkpoint."""
    root = tmp_path_factory.mktemp("qwen3")
    snapshot = _snapshot(root / "hf", weights=True)
    out = root / "bundle"
    embedding_export.export_embedding_bundle(q3.exporter(TINY), out, model_dir=str(snapshot))
    return snapshot, out


def _runtime_dtypes(signature: list) -> list[tuple[str, str]]:
    """``[(name, dtype)]`` of an ``asset_signatures`` input or output list."""
    return [(name, kind.split("(")[1].split(",")[0].lower()) for name, kind in signature]


def test_tiny_bundle_files_match_metadata(tiny_bundle):
    """The bundle holds ``text_encoder``, the tokenizer and ``metadata.json``; the asset holds
    exactly the listed functions with the metadata's names and dtypes, and the tokenizer adds no
    bos and ends with eos.
    """
    from tokenizers import Tokenizer

    _, out = tiny_bundle
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["name"] == "qwen3-embedding-0.6b"
    assert sorted(p.name for p in out.iterdir()) == [
        "metadata.json",
        "text_encoder.aimodel",
        "tokenizer",
    ]
    block = meta["embedding"]
    assert block["space"]["revision"] == meta["source"]["hf_revision"] == REVISION
    assert (block["text"]["add_bos_token"], block["text"]["bos_token_id"]) == (False, None)
    spec = block["components"]["text_encoder"]
    signatures = asset_signatures(out / "text_encoder.aimodel")
    assert sorted(signatures) == ["len16_b1", "len16_b2"]
    for inputs, outputs in signatures.values():
        assert _runtime_dtypes(inputs) == [("input_ids", "int32"), ("attention_mask", "int32")]
        assert _runtime_dtypes(outputs) == [("embedding", "float32")]
    assert [f["function"] for f in spec["functions"]] == ["len16_b1", "len16_b2"]
    tokenizer = Tokenizer.from_file(str(out / "tokenizer" / "tokenizer.json"))
    assert tokenizer.padding is None and tokenizer.truncation is None
    words = ["a", "photo", "of", "the", "cat"]
    ids = tokenizer.encode(" ".join(words)).ids
    assert ids == [tokenizer.token_to_id(word) for word in words] + [EOS]


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_bundle_matches_hugging_face(tiny_bundle, compute_unit):
    """``len16_b2`` on a ragged right-padded batch and ``len16_b1`` on each row, run by the Core
    AI runtime, match Hugging Face within atol = rtol = 1e-4.
    """
    options = runtime_options(compute_unit)
    snapshot, out = tiny_bundle
    ids, mask, lengths = _ragged_inputs()
    want = _hugging_face_embeddings(snapshot, ids, lengths)
    asset = out / "text_encoder.aimodel"
    inputs = {"input_ids": ids, "attention_mask": mask}
    (got,) = run_function(asset, "len16_b2", inputs, options).values()
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    for row in range(2):
        one = {name: value[row : row + 1] for name, value in inputs.items()}
        (got,) = run_function(asset, "len16_b1", one, options).values()
        torch.testing.assert_close(got[0], want[row], atol=1e-4, rtol=1e-4)


def test_reference_host_tokenizes_the_bundle(tiny_bundle, tmp_path):
    """The reference host (R6) sends the tokenizer's ids with no bos and eos last, right-padded to
    16, whether the bundle lists both functions or one; a longer text keeps eos last.
    """
    from tokenizers import Tokenizer

    _, out = tiny_bundle
    one = tmp_path / "one-function"
    shutil.copytree(out, one)
    meta = json.loads((one / "metadata.json").read_text())
    spec = meta["embedding"]["components"]["text_encoder"]
    spec["functions"] = spec["functions"][:1]
    (one / "metadata.json").write_text(json.dumps(meta))
    tokenizer = Tokenizer.from_file(str(out / "tokenizer" / "tokenizer.json"))
    short = [tokenizer.token_to_id(word) for word in ["a", "photo", "of", "the", "cat"]] + [EOS]
    long = [tokenizer.token_to_id(word) for word in ["cat", "dog"] * 10][:15] + [EOS]
    for path in (out, one):
        ids, mask = Bundle(path, None).tokenize(["a photo of the cat", " ".join(["cat dog"] * 10)])
        assert ids.tolist() == [short + [EOS] * (16 - len(short)), long]
        assert mask.tolist() == [[1] * len(short) + [0] * (16 - len(short)), [1] * 16]


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_reference_host_embeds_like_hugging_face(tiny_bundle, compute_unit):
    """The reference host picks ``len16_b2`` for three texts (two calls, the second padded with a
    copy of its row, R4) and ``len16_b1`` for one; both match Hugging Face within 1e-4.
    """
    snapshot, out = tiny_bundle
    bundle = Bundle(out, runtime_options(compute_unit))
    texts = ["a photo of the cat", "the dog", "cat dog sound of a photo the cat dog"]
    ids, mask = bundle.tokenize(texts)
    want = _hugging_face_embeddings(snapshot, ids, mask.sum(dim=1).tolist())
    assert bundle.function("text_encoder", len(texts), ids.shape[1])["function"] == "len16_b2"
    assert bundle.function("text_encoder", 1, ids.shape[1])["function"] == "len16_b1"
    torch.testing.assert_close(bundle.embed_text(texts), want, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(bundle.embed_text(texts[2:]), want[2:], atol=1e-4, rtol=1e-4)


@pytest.fixture(scope="module")
def tiny_bundle_512(tmp_path_factory) -> tuple[Path, Path]:
    """``(snapshot, bundle)``: ``TINY_512`` exported from a tiny random checkpoint."""
    root = tmp_path_factory.mktemp("qwen3-512")
    snapshot = _snapshot(root / "hf", weights=True, config=TINY_512_CONFIG)
    out = root / "bundle"
    embedding_export.export_embedding_bundle(q3.exporter(TINY_512), out, model_dir=str(snapshot))
    return snapshot, out


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_len512_b8_matches_hugging_face(tiny_bundle_512, compute_unit):
    """``len512_b8`` on eight right-padded rows of 1 to 512 tokens, run by the Core AI runtime,
    matches Hugging Face on each row within atol = rtol = 1e-4.
    """
    options = runtime_options(compute_unit)
    snapshot, out = tiny_bundle_512
    ids, mask, lengths = _ragged_inputs((1, 2, 31, 128, 200, 384, 511, 512), 512)
    inputs = {"input_ids": ids, "attention_mask": mask}
    (got,) = run_function(out / "text_encoder.aimodel", "len512_b8", inputs, options).values()
    want = _hugging_face_embeddings(snapshot, ids, lengths)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


# --- CLI ---


@pytest.fixture
def no_export(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("export must not start")

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fail)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--platform", "iOS"], "has no iOS export"),
        (["--fused"], "no fused export"),
    ],
)
def test_cli_refuses_before_exporting(argv, message, no_export, capsys):
    with pytest.raises(SystemExit) as info:
        embedding_export.main(["qwen3-embedding-0.6b", *argv])
    assert info.value.code == 2 and message in capsys.readouterr().err


def test_cli_dry_run_and_list_models(no_export, capsys):
    argv = ["Qwen/Qwen3-Embedding-0.6B", "--compute-precision", "float16", "--dry-run"]
    assert embedding_export.main(argv) == 0
    out = capsys.readouterr().out
    assert "float16" in out and "towers:    text" in out
    assert out.rstrip().endswith("qwen3-embedding-0.6b")
    assert embedding_export.main(["--list-models"]) == 0
    rows = [r for r in capsys.readouterr().out.splitlines() if r.startswith("qwen3-embedding")]
    assert len(rows) == 1 and " text " in rows[0]
    assert rows[0].endswith("macOS:float32,float16,bfloat16")
