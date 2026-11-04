# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the EmbeddingGemma 2 embedding exporter (``coreai_models.embedding.embedding_gemma2``).

Metadata, precision and CLI tests need no hardware or weights. The conversion tests use tiny random
EmbeddingGemma 2 configs; the programs also run on the Core AI runtime (CPU and GPU) when it is
available.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path

import pytest
import torch

from coreai_models.embedding import embedding_gemma2 as eg2
from coreai_models.embedding import export as embedding_export
from coreai_models.embedding.contract import (
    CheckpointFacts,
    EmbeddingPrecisionError,
    build_embedding_bundle_metadata,
    bundle_name,
    resolve_compute_precision,
    validate_block,
)
from coreai_models.embedding.embedding_gemma2 import (
    EMBEDDINGGEMMA2,
    EMBEDDINGGEMMA2_EXPORTER,
    bundle_assets,
    common,
    consolidate_prompts,
    embedding_block,
    ios,
    macos,
)
from coreai_models.embedding.embedding_gemma2.common import (
    FunctionSpec,
    bucket_plan,
    encoder_functions,
    text_functions,
)
from tests.test_model_units.test_export._embedding_host import (
    asset_signatures,
    save_tiny_tokenizer,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    TINY_TEXT_CONFIG,
    TINY_VISION_CONFIG,
    host_bias_and_pool,
)

EG2 = EMBEDDINGGEMMA2
IMAGE = EG2.image_tower
# config.json text_config token ids of the snapshot
TEXT_IDS = {"bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0}
EG2_FACTS = CheckpointFacts(
    dimension=768, bos_token_id=2, eos_token_id=1, pad_token_id=0, max_tokens=8192
)
POOLING = {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "sentence_similarity": "task: sentence similarity | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "question_answering": "task: question answering | query: ",
    "fact_checking": "task: fact checking | query: ",
    "code_retrieval": "task: code retrieval | query: ",
}
# The snapshot's 20-entry prompt table (several keys share one prefix).
SNAPSHOT_PROMPTS = {
    **dict.fromkeys(
        ["query", "Retrieval-query", "SearchQuery", "Retrieval", "Reranking", "BitextMining"],
        PROMPTS["query"],
    ),
    **dict.fromkeys(["document", "Document", "Retrieval-document"], PROMPTS["document"]),
    **dict.fromkeys(
        ["STS", "SentenceSimilarity", "PairClassification", "Summarization"],
        PROMPTS["sentence_similarity"],
    ),
    **dict.fromkeys(["Classification", "MultilabelClassification"], PROMPTS["classification"]),
    "Clustering": PROMPTS["clustering"],
    "QuestionAnswering": PROMPTS["question_answering"],
    "FactChecking": PROMPTS["fact_checking"],
    **dict.fromkeys(["CodeRetrieval", "InstructionRetrieval"], PROMPTS["code_retrieval"]),
}

# Keys the embedding block must not contain at any depth.
REMOVED_KEYS = {
    "pooling",
    "attention",
    "granularity",
    "budgets",
    "max_sequence_length",
    "prompt_applies_to",
    "side",
    "strategy",
    "wrap",
    "soft_tokens_scaled",
    "floor",
    "norm",
    "dynamic_batch",
    "dynamic_length",
    "hidden_size",
    "note",
    "max_patches",
    "soft_token_dim",
    "convert_rgb",
    "exif_transpose",
    "resize",
    "resize_policy",
    "patch_pixel_order",
    "position_order",
    "padding_position",
    "prompt",
    "truncation_side",
    # names replaced by the contract's
    "precision",
    "forbidden",
    "output",
    "output_dtype",
    "mask_encoding",
    "tokenizer",
    "special_tokens",
    "batch_sizes",
    "length_buckets",
    "max_batch_tokens",
    "merge",
    "placeholder_ids",
    "token_ids",
    "preprocessing",
    "rescale_factor",
    "normalize",
}
WORDS = ["<unk>", "a", "photo", "of", "the", "cat"]


def _block(platform="macOS", with_image=False, **overrides):
    kwargs = dict(revision=REVISION, prompts=PROMPTS)
    kwargs.update(overrides)
    return embedding_block(EG2, EG2_FACTS, platform=platform, with_image=with_image, **kwargs)


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def _dtype(spec: str) -> str:
    return spec.split("[")[0]


# --- Precision ---


def test_precisions_per_platform():
    assert EG2.precisions == {"macOS": ("float32", "bfloat16"), "iOS": ("float16",)}
    assert resolve_compute_precision(EG2, None, "iOS") == "float16"
    assert resolve_compute_precision(EG2, None, "macOS") == "float32"
    assert resolve_compute_precision(EG2, "bfloat16", "macOS") == "bfloat16"
    with pytest.raises(EmbeddingPrecisionError, match="float32 or bfloat16, not float16"):
        resolve_compute_precision(EG2, "float16", "macOS")
    for precision in ("float32", "bfloat16"):
        with pytest.raises(EmbeddingPrecisionError, match="computes in float16"):
            resolve_compute_precision(EG2, precision, "iOS")


# --- Prompts ---


def test_consolidate_prompts_maps_the_snapshot_table_to_eight_names():
    assert len(SNAPSHOT_PROMPTS) == 20
    prompts, default = consolidate_prompts(EG2, SNAPSHOT_PROMPTS)
    assert prompts == PROMPTS and default is None
    assert list(prompts) == list(EG2.prompt_names)


def test_consolidate_prompts_maps_the_default_prompt_name():
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "STS") == (PROMPTS, "sentence_similarity")
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "query")[1] == "query"
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "SearchQuery")[1] == "query"  # by prefix
    with pytest.raises(ValueError, match="default_prompt_name 'Translation'"):
        consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "Translation")


def test_consolidate_prompts_rejects_unknown_and_missing_keys():
    with pytest.raises(ValueError, match="canonical"):
        consolidate_prompts(EG2, {**SNAPSHOT_PROMPTS, "Translation": "task: translate | query: "})
    missing = {k: v for k, v in SNAPSHOT_PROMPTS.items() if k != "FactChecking"}
    with pytest.raises(ValueError, match="FactChecking"):
        consolidate_prompts(EG2, missing)


# --- Bucket plan ---


def test_bucket_plan_respects_token_cap_and_keeps_batch_one():
    plan = bucket_plan([2048, 128, 512], [32, 8], 16384)
    shapes = [(f.length, f.batch) for f in plan]
    assert shapes == [
        (128, 1),
        (128, 8),
        (128, 32),
        (512, 1),
        (512, 8),
        (512, 32),
        (2048, 1),
        (2048, 8),
    ]
    assert all(f.batch == 1 or f.batch * f.length <= 16384 for f in plan)
    assert plan[0].name == "len128_b1"
    assert len({f.name for f in plan}) == len(plan)


def test_bucket_plan_rejects_bad_lengths():
    with pytest.raises(ValueError):
        bucket_plan([1], [1], 1024)


def test_function_plans_per_platform():
    assert [f.length for f in encoder_functions(EG2, "macOS", with_image=True)] == [288, 512, 2048]
    assert encoder_functions(EG2, "macOS", with_image=False) == []
    for with_image in (False, True):
        ios = encoder_functions(EG2, "iOS", with_image=with_image)
        assert [(f.length, f.batch) for f in ios] == [(128, 1), (512, 1), (1024, 1)]
    with pytest.raises(ValueError, match="ane_length_buckets"):
        encoder_functions(
            dataclasses.replace(EG2, ane_length_buckets=(1, 128)), "iOS", with_image=False
        )
    assert IMAGE.max_sequence_length == 284  # <bos> <boi> 280 x <image> <eoi> <eos>


# --- embedding_block schema ---


@pytest.mark.parametrize(
    ("platform", "with_image", "components", "pipelines", "units"),
    [
        ("macOS", False, ["text_encoder"], [["text_encoder"]], {"text_encoder": ["gpu"]}),
        (
            "macOS",
            True,
            ["text_encoder", "image_encoder", "token_table", "encoder"],
            [["text_encoder"], ["image_encoder", "token_table", "encoder"]],
            {
                "text_encoder": ["gpu"],
                "image_encoder": ["gpu"],
                "token_table": ["gpu"],
                "encoder": ["gpu"],
            },
        ),
        (
            "iOS",
            False,
            ["token_table", "encoder"],
            [["token_table", "encoder"]],
            {"token_table": ["gpu"], "encoder": ["neural_engine"]},
        ),
        (
            "iOS",
            True,
            ["token_table", "encoder", "image_encoder"],
            [["token_table", "encoder"], ["image_encoder", "token_table", "encoder"]],
            {
                "token_table": ["gpu"],
                "encoder": ["neural_engine"],
                "image_encoder": ["neural_engine"],
            },
        ),
    ],
)
def test_embedding_block_platform_matrix(platform, with_image, components, pipelines, units):
    block = _block(platform, with_image)
    assets = bundle_assets(platform, with_image)
    validate_block(block, assets)
    assert list(block["components"]) == components
    assert [t["pipeline"] for t in block["towers"]] == pipelines
    assert all(list(t) == ["modality", "pipeline"] for t in block["towers"])
    assert {k: v["compute_units"] for k, v in block["components"].items()} == units
    assert ("image" in block) == with_image
    assert {spec["asset"] for spec in block["components"].values()} == set(assets)
    assert len(set(assets.values())) == len(assets)
    want_dtype = "float16" if platform == "iOS" else "float32"
    outputs = [s for c in block["components"].values() for s in c["outputs"].values()]
    assert {_dtype(s) for s in outputs} == {want_dtype}
    assert not REMOVED_KEYS & set(_keys(block)), REMOVED_KEYS & set(_keys(block))
    meta = build_embedding_bundle_metadata(
        EG2,
        block,
        name=bundle_name(EG2, platform, not with_image, False),
        assets=assets,
        hf_revision=REVISION,
    )
    assert meta["metadata_version"] == "0.2" and meta["kind"] == "embedding"
    assert {c for t in meta["embedding"]["towers"] for c in t["pipeline"]} == set(components)
    assert meta["source"]["hf_model_id"] == "google/embeddinggemma-2"


def test_embedding_block_ios_details():
    block = _block("iOS", True)
    enc = block["components"]["encoder"]
    assert enc["inputs"] == {
        "inputs_embeds": "float16[B,L,512]",
        "bias": "float16[B,L,1,L]",
        "pool_weights": "float16[B,L]",
    }
    assert enc["outputs"] == {"embedding": "float16[B,768]"}
    assert enc["bias"] == {
        "layout": "[1, key, 1, query]",
        "masked": -30000.0,
        "unmasked": 0.0,
        "sliding_window": "in_graph",
    }
    assert enc["pool_weights"] == "attention_mask / attention_mask.sum(-1)"
    assert [f["length"] for f in enc["functions"]] == [128, 512, 1024]
    assert block["components"]["image_encoder"]["inputs"] == {
        "pixel_values": "float16[B,2520,768]",
        "position_ids": "float16[B,2520,2]",
    }
    assert block["compute_precision"] == "float16"
    assert block["text"]["truncation"]["max_tokens"] == 1024
    assert block["space"] == _block("macOS")["space"]
    assert bundle_assets("iOS", True) == {
        "token_table": "token_table.aimodel",
        "encoder": "encoder.aimodel",
        "image_encoder": "image_encoder.aimodel",
    }
    assert {k: v["asset"] for k, v in block["components"].items()} == {
        "token_table": "token_table",
        "encoder": "encoder",
        "image_encoder": "image_encoder",
    }


def test_embedding_block_macos_image_details():
    block = _block("macOS", True)
    assert list(block) == [
        "space",
        "dimension",
        "matryoshka_dimensions",
        "renormalize_after_truncation",
        "normalized",
        "metric",
        "logit_scale",
        "compute_precision",
        "components",
        "towers",
        "text",
        "image",
    ]
    assert block["space"] == {"id": "google/embeddinggemma-2", "revision": REVISION}
    assert block["matryoshka_dimensions"] == [768, 512, 256, 128]
    assert block["logit_scale"] is None and block["compute_precision"] == "float32"
    text_encoder = block["components"]["text_encoder"]
    assert text_encoder["outputs"] == {"embedding": "float32[B,768]"}
    assert {"function": "len2048_b8", "batch": 8, "length": 2048} in text_encoder["functions"]
    assert block["text"] == {
        "embedded_tokenizer": True,
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "padding_side": "right",
        "truncation": {"max_tokens": 2048, "keep_eos": True},
        "model_max_tokens": 8192,
        "prompts": PROMPTS,
        "default_prompt_name": None,
        "document_template": "title: {title} | text: {text}",
    }
    assert bundle_assets("macOS", True) == {
        "text_encoder": "text_encoder.aimodel",
        "image_encoder": "image_encoder.aimodel",
    }
    assert {k: v["asset"] for k, v in block["components"].items()} == {
        "text_encoder": "text_encoder",
        "image_encoder": "image_encoder",
        "token_table": "text_encoder",
        "encoder": "text_encoder",
    }
    names = [f.name for f in text_functions(EG2) + encoder_functions(EG2, "macOS", with_image=True)]
    assert len(names) == len(set(names)) and "main" not in names
    enc = block["components"]["encoder"]
    assert enc["inputs"] == {"inputs_embeds": "float32[B,L,512]", "attention_mask": "int32[B,L]"}
    assert [f["length"] for f in enc["functions"]] == [288, 512, 2048]
    assert block["components"]["token_table"]["functions"] == [
        {"function": "main", "batch": 1, "length": {"min": 2, "max": 8192}}
    ]
    assert block["components"]["image_encoder"]["outputs"] == {"soft_tokens": "float32[B,280,512]"}
    assert block["image"] == {
        "image_strategy": "aspect_ratio_preserving",
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "max_soft_tokens": 280,
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
    }


def test_embedding_block_validation():
    with pytest.raises(ValueError, match="revision"):
        _block(revision="")
    with pytest.raises(ValueError, match="prompts"):
        _block(prompts={})
    with pytest.raises(ValueError, match="default_prompt_name"):
        _block(default_prompt_name="missing")
    with pytest.raises(ValueError, match="platform"):
        _block("watchOS")
    no_image = dataclasses.replace(EG2, image_tower=None)
    with pytest.raises(ValueError, match="no image tower"):
        embedding_block(
            no_image, EG2_FACTS, platform="macOS", with_image=True, revision="x", prompts=PROMPTS
        )


def test_embedding_block_records_the_compute_precision():
    assert _block()["compute_precision"] == "float32"
    bf16 = _block(with_image=True, compute_precision="bfloat16")
    assert bf16["compute_precision"] == "bfloat16"
    outputs = [s for c in bf16["components"].values() for s in c["outputs"].values()]
    assert {_dtype(s) for s in outputs} == {"float32"}
    assert _block("iOS")["compute_precision"] == "float16"
    with pytest.raises(EmbeddingPrecisionError, match="not float16"):
        _block(compute_precision="float16")


@pytest.fixture
def stub_snapshot(tmp_path, monkeypatch):
    """Snapshot helpers stubbed so ``export_embedding_bundle`` runs without weights."""
    monkeypatch.setattr(
        embedding_export, "resolve_snapshot", lambda *a: (tmp_path, "0123456789abcdef0123")
    )
    monkeypatch.setattr(embedding_export, "save_tokenizer", lambda *a: None)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": TEXT_IDS}))
    monkeypatch.setattr(eg2, "read_pooling_config", lambda *a: 768)
    monkeypatch.setattr(eg2, "read_prompts", lambda *a: (SNAPSHOT_PROMPTS, None))
    monkeypatch.setattr(eg2, "fetch_weights", lambda *a: None)


def test_export_writes_the_requested_precision_to_metadata(tmp_path, monkeypatch, stub_snapshot):
    seen = {}
    monkeypatch.setattr(eg2, "export_macos_programs", lambda *a, **kw: seen.update(kw))
    meta = embedding_export.export_embedding_bundle(
        EMBEDDINGGEMMA2_EXPORTER, tmp_path / "out", text_only=True, compute_precision="bfloat16"
    )
    assert seen["compute_precision"] == "bfloat16" and seen["with_image"] is False
    assert meta["name"] == "embeddinggemma-2-text"
    written = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert written == meta
    assert written["embedding"]["compute_precision"] == "bfloat16"
    assert written["embedding"]["space"]["revision"] == "0123456789abcdef0123"
    assert written["source"]["hf_revision"] == "0123456789abcdef0123"
    assert written["embedding"]["text"]["prompts"] == PROMPTS
    assert written["embedding"]["text"]["default_prompt_name"] is None


def test_export_maps_the_default_prompt_before_converting(tmp_path, monkeypatch, stub_snapshot):
    seen = []
    monkeypatch.setattr(eg2, "export_macos_programs", lambda *a, **kw: seen.append(kw))
    monkeypatch.setattr(eg2, "read_prompts", lambda *a: (SNAPSHOT_PROMPTS, "STS"))
    out = tmp_path / "out"
    meta = embedding_export.export_embedding_bundle(EMBEDDINGGEMMA2_EXPORTER, out, text_only=True)
    assert meta["embedding"]["text"]["default_prompt_name"] == "sentence_similarity"
    monkeypatch.setattr(eg2, "read_prompts", lambda *a: (SNAPSHOT_PROMPTS, "Nope"))
    with pytest.raises(ValueError, match="default_prompt_name 'Nope'"):
        embedding_export.export_embedding_bundle(
            EMBEDDINGGEMMA2_EXPORTER, tmp_path / "bad", text_only=True
        )
    assert len(seen) == 1 and not (tmp_path / "bad").exists()


def test_exporter_is_registered():
    assert embedding_export.EXPORTERS["embeddinggemma-2"] is EMBEDDINGGEMMA2_EXPORTER
    assert (
        embedding_export.exporter_for_model("Google/EmbeddingGemma-2") is EMBEDDINGGEMMA2_EXPORTER
    )
    assert EMBEDDINGGEMMA2_EXPORTER.family is EG2 and not EMBEDDINGGEMMA2_EXPORTER.fused
    assert EG2.modalities == ("text", "image")
    with pytest.raises(KeyError, match="embeddinggemma-2"):
        embedding_export.exporter_for_model("google/embeddinggemma-300m")


# --- Snapshot helpers ---


def _fake_snapshot(root: Path, pooling: dict) -> Path:
    snap = root / "models--google--embeddinggemma-2" / "snapshots" / ("a" * 40)
    (snap / "1_Pooling").mkdir(parents=True)
    (snap / "2_Normalize").mkdir()
    (snap / "1_Pooling" / "config.json").write_text(json.dumps(pooling))
    (snap / "config.json").write_text(json.dumps({"text_config": TEXT_IDS}))
    (snap / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": SNAPSHOT_PROMPTS, "default_prompt_name": None})
    )
    return snap


def test_snapshot_helpers(tmp_path):
    snap = _fake_snapshot(tmp_path, POOLING)
    assert eg2.read_prompts(snap) == (SNAPSHOT_PROMPTS, None)
    assert eg2.read_pooling_config(snap, EG2) == 768
    assert EMBEDDINGGEMMA2_EXPORTER.read_facts(snap) == EG2_FACTS


def test_read_pooling_config_rejects_include_prompt_false(tmp_path):
    snap = _fake_snapshot(tmp_path, {**POOLING, "include_prompt": False})
    with pytest.raises(ValueError, match="include_prompt"):
        eg2.read_pooling_config(snap, EG2)
    with pytest.raises(ValueError, match="include_prompt"):
        EMBEDDINGGEMMA2_EXPORTER.read_facts(snap)


def _image_snapshot(root: Path, **config_overrides) -> Path:
    snap = root / "snap"
    snap.mkdir(parents=True)
    cfg = {
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "text_config": {"hidden_size": 512},
    }
    cfg.update(config_overrides)
    (snap / "config.json").write_text(json.dumps(cfg))
    proc = {
        "image_processor": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "do_normalize": False,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
        }
    }
    (snap / "processor_config.json").write_text(json.dumps(proc))
    return snap


def test_verify_image_config(tmp_path):
    eg2.verify_image_config(_image_snapshot(tmp_path / "ok"), EG2)
    with pytest.raises(ValueError, match="boi_token_id"):
        eg2.verify_image_config(_image_snapshot(tmp_path / "bad", boi_token_id=1), EG2)


def test_read_facts_takes_the_snapshot_values(tmp_path):
    snap = _fake_snapshot(tmp_path, {**POOLING, "embedding_dimension": 512})
    (snap / "config.json").write_text(
        json.dumps({"text_config": {**TEXT_IDS, "eos_token_id": 106}})
    )
    facts = EMBEDDINGGEMMA2_EXPORTER.read_facts(snap)
    assert (facts.dimension, facts.eos_token_id, facts.max_tokens) == (512, 106, 8192)
    (snap / "1_Pooling" / "config.json").unlink()
    with pytest.raises(ValueError, match="no 1_Pooling"):
        EMBEDDINGGEMMA2_EXPORTER.read_facts(snap)


def test_fetch_weights_downloads_only_a_missing_checkpoint(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: calls.append(kw))
    snap = tmp_path / "hub" / "models--google--embeddinggemma-2" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    eg2.fetch_weights(EG2, snap)
    assert calls == [
        {"revision": "a" * 40, "cache_dir": tmp_path / "hub", "allow_patterns": ["*.safetensors"]}
    ]
    (snap / "model.safetensors").write_bytes(b"")
    eg2.fetch_weights(EG2, snap)
    assert len(calls) == 1


def test_model_token_ids_match_the_family():
    from coreai_models.models.macos import embedding_gemma2_vision as vision

    model_ids = (
        vision.EG2_BOS_TOKEN_ID,
        vision.EG2_EOS_TOKEN_ID,
        vision.EG2_PAD_TOKEN_ID,
        vision.EG2_IMAGE_TOKEN_ID,
        vision.EG2_BOI_TOKEN_ID,
        vision.EG2_EOI_TOKEN_ID,
    )
    family_ids = (
        EG2_FACTS.bos_token_id,
        EG2_FACTS.eos_token_id,
        EG2_FACTS.pad_token_id,
        IMAGE.placeholder_id,
        IMAGE.boi_token_id,
        IMAGE.eoi_token_id,
    )
    assert model_ids == family_ids


# --- Conversion (tiny random config) ---


def test_tiny_multifunction_conversion():
    wrapper = macos.TextEmbeddingExportWrapper(_tiny_text_encoder())
    plan = bucket_plan([16], [1, 4], 1024)
    ids, mask = macos.reference_inputs(plan[1], 128)
    assert ids.dtype == torch.int32 and mask.dtype == torch.int32 and ids.shape == (4, 16)
    out = wrapper(ids, mask)
    assert out.dtype == torch.float32 and out.shape == (4, 48)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(4))
    assert macos.export_text_encoder_program(wrapper.encoder, plan, [], 64) is not None


def _tiny_text_encoder():
    from coreai_models.models.macos.embedding_gemma2 import (
        EmbeddingGemma2TextConfig,
        EmbeddingGemma2TextEncoder,
    )

    config = EmbeddingGemma2TextConfig.from_dict(TINY_TEXT_CONFIG)
    torch.manual_seed(0)
    return EmbeddingGemma2TextEncoder(config).eval()


def _tiny_image_encoder():
    from coreai_models.models.macos.embedding_gemma2_vision import EmbeddingGemma2ImageEncoder

    torch.manual_seed(1)
    return EmbeddingGemma2ImageEncoder(TINY_VISION_CONFIG).eval()


TINY_IMAGE = dataclasses.replace(IMAGE, patch_size=4, max_soft_tokens=12, soft_token_dim=32)


def test_inputs_embeds_path_equals_ids_path():
    """``encoder(token_table(ids))`` matches ``text_encoder(ids)`` within atol 1e-6, and
    ``without_token_table`` restores ``embed_tokens`` afterwards.
    """
    text = _tiny_text_encoder()
    ids = torch.randint(3, 128, (1, 16), dtype=torch.int32)
    mask = torch.ones(1, 16, dtype=torch.int32)
    mask[0, 11:] = 0
    with torch.no_grad():
        direct = macos.TextEmbeddingExportWrapper(text)(ids, mask)
        embeds = common.TokenTableExportWrapper(text.embed_tokens)(ids)
        with macos.without_token_table(text) as encoder:
            merged = macos.InputsEmbedsEncoderWrapper(encoder)(embeds, mask)
    assert embeds.dtype == torch.float32 and embeds.shape == (1, 16, 64)
    torch.testing.assert_close(merged, direct, atol=1e-6, rtol=0)
    assert isinstance(text.embed_tokens, torch.nn.Embedding)  # restored


def test_float16_scaled_token_table_equals_the_float32_table_cast():
    """Lookups in the float16 ``scaled_token_table`` equal the float32 ``embed_tokens`` lookups
    cast to float16.
    """
    text = _tiny_text_encoder()
    ids = torch.randint(3, 128, (1, 16), dtype=torch.int32)
    table = ios.scaled_token_table(
        text.embed_tokens.weight, text.config.hidden_size**0.5, torch.float16
    )
    with torch.no_grad():
        want = common.TokenTableExportWrapper(text.embed_tokens)(ids).half()
        got = common.TokenTableExportWrapper(table, torch.float16)(ids)
    assert got.dtype == torch.float16
    assert torch.equal(got, want)


def test_tiny_float16_token_table_conversion():
    text = _tiny_text_encoder()
    table = ios.scaled_token_table(
        text.embed_tokens.weight, text.config.hidden_size**0.5, torch.float16
    )
    program = ios.export_token_table_program(table, 64, dtype=torch.float16)
    assert program is not None


def test_image_reference_inputs_are_a_valid_padded_grid():
    pixels, positions = common.image_reference_inputs(TINY_IMAGE)
    assert pixels.shape == (1, 108, 48) and pixels.dtype == torch.float32
    assert positions.shape == (1, 108, 2) and positions.dtype == torch.int32
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0
    assert torch.all(pixels[0, ~real] == 0)
    soft = macos.ImageEncoderExportWrapper(_tiny_image_encoder())(pixels, positions)
    n = int(real.sum()) // 9
    assert soft.shape == (1, 12, 32) and soft.dtype == torch.float32
    assert torch.all(soft[0, n:] == 0) and torch.all(soft[0, :n].abs().sum(-1) > 0)


def _function_names(program) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        asset = program.save_asset(Path(tmp) / "program.aimodel")
        return sorted(asset.summary(include_statistics=False).function_names)


@pytest.fixture(scope="module")
def tiny_gpu_programs():
    """Component name -> (float32 PyTorch module, its converted program) for the tiny macOS bundle.

    ``text_encoder`` holds ``len16_b2`` (longer than the sliding window of 3), ``main`` and
    ``embeds_len32_b1``.
    """
    text = _tiny_text_encoder()
    image = _tiny_image_encoder()
    program = macos.export_text_encoder_program(
        text, [FunctionSpec("len16_b2", 2, 16)], [FunctionSpec("embeds_len32_b1", 1, 32)], 64
    )
    return {
        "text_encoder": (text, program),
        "image_encoder": (image, macos.export_image_encoder_program(image, TINY_IMAGE)),
    }


def test_tiny_gpu_programs_have_their_functions(tiny_gpu_programs):
    text_encoder = tiny_gpu_programs["text_encoder"][1]
    assert _function_names(text_encoder) == ["embeds_len32_b1", "len16_b2", "main"]
    assert _function_names(tiny_gpu_programs["image_encoder"][1]) == ["main"]


def test_ane_reference_inputs_shapes():
    embeds, bias, pool = ios.encoder_reference_inputs(128, 512)
    assert embeds.shape == (1, 128, 512) and bias.shape == (1, 128, 1, 128)
    assert pool.shape == (1, 128) and {t.dtype for t in (embeds, bias, pool)} == {torch.float16}
    pixels, positions = common.image_reference_inputs(IMAGE, torch.float16, torch.float16)
    assert pixels.shape == (1, 2520, 768) and positions.shape == (1, 2520, 2)
    assert pixels.dtype == positions.dtype == torch.float16
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0 and torch.all(positions[0, ~real] == -1)


# --- Neural Engine conversion (tiny random config) ---

# Encoder lengths 4 (unbanded) and 16 (banded: > sliding_window + 1); 16 also fits the image
# sequence (max_soft_tokens 12 + 4).
TINY_ANE_FAMILY = dataclasses.replace(
    EG2, image_tower=TINY_IMAGE, ane_length_buckets=(4, 16), max_tokens=64
)
TINY_FACTS = dataclasses.replace(EG2_FACTS, max_tokens=64)


@pytest.fixture(scope="module")
def tiny_ane_programs():
    """Component name -> (float16 PyTorch module, its converted program) for the tiny iOS bundle."""
    from coreai_models.models.ios import embedding_gemma2_vision as ios_vision
    from coreai_models.models.ios.embedding_gemma2 import text_encoder_from_state_dict

    text = _tiny_text_encoder()
    encoder, table = text_encoder_from_state_dict(
        text.config, dict(text.state_dict()), torch.float16
    )
    encoder.eval()
    token_table = ios.scaled_token_table(table, text.config.hidden_size**0.5, torch.float16)
    torch.manual_seed(1)
    image_encoder = ios_vision.EmbeddingGemma2ImageEncoder(TINY_VISION_CONFIG).half().eval()
    functions = encoder_functions(TINY_ANE_FAMILY, "iOS", with_image=True)
    window = int(text.config.sliding_window)
    return {
        "encoder": (
            encoder,
            ios.export_ane_encoder_program(encoder, functions, window),
        ),
        "token_table": (
            token_table,
            ios.export_token_table_program(token_table, 64, dtype=torch.float16),
        ),
        "image_encoder": (
            image_encoder,
            ios.export_ane_image_encoder_program(image_encoder, TINY_IMAGE),
        ),
    }


def _runtime_options(compute_unit: str):
    """Core AI runtime options for ``"cpu"`` or ``"gpu"``; skips when the runtime cannot run."""
    import coreai.runtime as rt

    if not rt.SpecializationOptions.is_supported():
        pytest.skip("the Core AI runtime cannot run programs here")
    if compute_unit == "cpu":
        return rt.SpecializationOptions.cpu_only()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())


async def _run_function(program, name: str, inputs: dict, options) -> dict:
    """Outputs of function ``name`` of ``program`` on ``inputs``, run by the Core AI runtime."""
    import coreai.runtime as rt

    with tempfile.TemporaryDirectory() as tmp:
        asset = program.save_asset(Path(tmp) / "program.aimodel")
        async with asset.executable(options) as model:
            outputs = await model.load_function(name)(
                {key: rt.NDArray(data=value.contiguous()) for key, value in inputs.items()}
            )
            return {key: torch.from_numpy(value.numpy().copy()) for key, value in outputs.items()}


def test_tiny_ane_programs_have_the_contract_functions(tiny_ane_programs):
    block = embedding_block(
        TINY_ANE_FAMILY,
        TINY_FACTS,
        platform="iOS",
        with_image=True,
        revision=REVISION,
        prompts=PROMPTS,
    )
    encoder = block["components"]["encoder"]
    assert [f["function"] for f in encoder["functions"]] == ["embeds_len4_b1", "embeds_len16_b1"]
    for name, spec in block["components"].items():
        functions = [f["function"] for f in spec["functions"]]
        assert _function_names(tiny_ane_programs[name][1]) == sorted(functions), name


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_ane_programs_match_pytorch(tiny_ane_programs, compute_unit):
    """Each converted program, run by the Core AI runtime, outputs the metadata's output dtype and
    matches its float16 module: the token table exactly, the encoders within atol = rtol = 1e-2.
    """
    from coreai_models.models.macos.embedding_gemma2_vision import pad_to_budget, patch_position_ids

    options = _runtime_options(compute_unit)
    dtypes = {
        name: getattr(torch, _dtype(*spec["outputs"].values()))
        for name, spec in _block("iOS", True)["components"].items()
    }
    encoder, program = tiny_ane_programs["encoder"]
    for spec in encoder_functions(TINY_ANE_FAMILY, "iOS", with_image=True):
        n = spec.length
        embeds, _, _ = ios.encoder_reference_inputs(n, 64)
        mask = (torch.arange(n) < n - 1).to(torch.int32)[None]
        bias, pool = host_bias_and_pool(mask, torch.float16)
        inputs = {"inputs_embeds": embeds, "bias": bias, "pool_weights": pool}
        with torch.no_grad():
            wrapper = ios.EncoderBandWrapper(encoder, n, encoder.config.sliding_window)
            want = wrapper(*inputs.values())
        got = asyncio.run(_run_function(program, spec.name, inputs, options))["embedding"]
        assert got.dtype == dtypes["encoder"] and got.shape == want.shape == (1, 48)
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)

    table, program = tiny_ane_programs["token_table"]
    ids = torch.randint(3, 128, (1, 9), dtype=torch.int32)
    got = asyncio.run(_run_function(program, "main", {"input_ids": ids}, options))
    assert got["inputs_embeds"].dtype == dtypes["token_table"]
    assert torch.equal(
        got["inputs_embeds"], common.TokenTableExportWrapper(table, torch.float16)(ids)
    )

    image_encoder, program = tiny_ane_programs["image_encoder"]
    pixels = torch.rand((54, TINY_IMAGE.patch_dim), generator=torch.Generator().manual_seed(2))
    pixels, positions = pad_to_budget(pixels, patch_position_ids(6, 9), TINY_IMAGE.max_patches)
    inputs = {"pixel_values": pixels[None].half(), "position_ids": positions[None].half()}
    with torch.no_grad():
        want = image_encoder(*inputs.values())
    got = asyncio.run(_run_function(program, "main", inputs, options))["soft_tokens"]
    assert got.dtype == dtypes["image_encoder"] and got.shape == want.shape == (1, 12, 32)
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    assert torch.all(got[0, 6:] == 0)


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_gpu_programs_match_pytorch(tiny_gpu_programs, compute_unit):
    """Each macOS function, run by the Core AI runtime, matches its float32 PyTorch wrapper within
    atol = rtol = 1e-4: ids on a ragged int32 mask longer than the sliding window (fully masked
    padded query rows), the token table, ``inputs_embeds`` and the image encoder.
    """
    from coreai_models.models.macos.embedding_gemma2_vision import pad_to_budget, patch_position_ids

    options = _runtime_options(compute_unit)
    text, program = tiny_gpu_programs["text_encoder"]
    ids, _ = macos.reference_inputs(FunctionSpec("len16_b2", 2, 16), 128)
    mask = torch.ones((2, 16), dtype=torch.int32)
    mask[1, 7:] = 0
    with torch.no_grad():
        want = macos.TextEmbeddingExportWrapper(text)(ids, mask)
    inputs = {"input_ids": ids, "attention_mask": mask}
    got = asyncio.run(_run_function(program, "len16_b2", inputs, options))["embedding"]
    assert got.dtype == torch.float32 and got.shape == (2, 48)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    row = ids[1:, :9]
    with torch.no_grad():
        want = common.TokenTableExportWrapper(text.embed_tokens)(row)
    got = asyncio.run(_run_function(program, "main", {"input_ids": row}, options))
    torch.testing.assert_close(got["inputs_embeds"], want, atol=1e-4, rtol=1e-4)

    embeds = torch.randn((1, 32, 64), generator=torch.Generator().manual_seed(3))
    mask = (torch.arange(32) < 20).to(torch.int32)[None]
    with torch.no_grad(), macos.without_token_table(text) as encoder:
        want = macos.InputsEmbedsEncoderWrapper(encoder)(embeds, mask)
    inputs = {"inputs_embeds": embeds, "attention_mask": mask}
    got = asyncio.run(_run_function(program, "embeds_len32_b1", inputs, options))["embedding"]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    image_encoder, program = tiny_gpu_programs["image_encoder"]
    pixels = torch.rand((54, TINY_IMAGE.patch_dim), generator=torch.Generator().manual_seed(2))
    pixels, positions = pad_to_budget(pixels, patch_position_ids(6, 9), TINY_IMAGE.max_patches)
    inputs = {"pixel_values": pixels[None], "position_ids": positions[None].to(torch.int32)}
    with torch.no_grad():
        want = macos.ImageEncoderExportWrapper(image_encoder)(*inputs.values())
    got = asyncio.run(_run_function(program, "main", inputs, options))["soft_tokens"]
    assert got.dtype == torch.float32 and got.shape == want.shape == (1, 12, 32)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    assert torch.all(got[0, 6:] == 0)


# --- Bundle orchestration (tiny random config) ---

TINY_BUNDLE_FAMILY = dataclasses.replace(TINY_ANE_FAMILY, length_buckets=(16,), batch_sizes=(1,))


def _bundle_snapshot(root: Path) -> Path:
    """A fake HF snapshot whose configs and tokenizer agree with ``TINY_BUNDLE_FAMILY``."""
    snap = _fake_snapshot(root, POOLING)
    text_config = {"hidden_size": 32, **TEXT_IDS}
    image = _image_snapshot(root / "image", text_config=text_config)
    (snap / "config.json").write_text((image / "config.json").read_text())
    processor = json.loads((image / "processor_config.json").read_text())
    processor["image_processor"].update(patch_size=4, max_soft_tokens=12)
    (snap / "processor_config.json").write_text(json.dumps(processor))
    facts = EG2_FACTS
    save_tiny_tokenizer(snap, WORDS, facts.bos_token_id, facts.eos_token_id, facts.pad_token_id)
    return snap


@pytest.fixture
def tiny_loaders(monkeypatch):
    """The snapshot loaders the exporters import lazily, patched to return the tiny modules."""
    from coreai_models.models.ios import embedding_gemma2 as ios_text
    from coreai_models.models.ios import embedding_gemma2_vision as ios_vision
    from coreai_models.models.macos import embedding_gemma2 as macos_text
    from coreai_models.models.macos import embedding_gemma2_vision as macos_vision

    def text_and_table(snapshot, dtype):
        text = _tiny_text_encoder()
        return ios_text.text_encoder_from_state_dict(text.config, dict(text.state_dict()), dtype)

    def ios_image(snapshot, dtype, max_soft_tokens=None):
        torch.manual_seed(1)
        return ios_vision.EmbeddingGemma2ImageEncoder(TINY_VISION_CONFIG).to(dtype).eval()

    monkeypatch.setattr(
        macos_text,
        "load_embedding_gemma2_text_encoder",
        lambda snapshot, dtype: _tiny_text_encoder().to(dtype),
    )
    monkeypatch.setattr(
        macos_vision.EmbeddingGemma2ImageEncoder,
        "from_pretrained",
        lambda snapshot, dtype, max_soft_tokens=None: _tiny_image_encoder().to(dtype),
    )
    monkeypatch.setattr(ios_text, "load_embedding_gemma2_text_encoder_and_table", text_and_table)
    monkeypatch.setattr(ios_vision, "load_embedding_gemma2_image_encoder", ios_image)
    monkeypatch.setattr(eg2, "fetch_weights", lambda *a: None)


@pytest.mark.parametrize("platform", ["macOS", "iOS"])
def test_bundle_files_match_metadata(platform, tmp_path, tiny_loaders):
    """The files on disk are the metadata's assets, the tokenizer and ``metadata.json``; every
    component's functions are in its asset with the metadata's input and output names and dtypes.
    """
    snapshot = str(_bundle_snapshot(tmp_path / "hf"))
    out = tmp_path / "bundle"
    exporter = eg2.exporter(TINY_BUNDLE_FAMILY)
    meta = embedding_export.export_embedding_bundle(
        exporter, out, platform=platform, model_dir=snapshot
    )
    files = [*meta["assets"].values(), "metadata.json", "tokenizer"]
    assert sorted(p.name for p in out.iterdir()) == sorted(files)
    assert json.loads((out / "metadata.json").read_text()) == meta
    for name, spec in meta["embedding"]["components"].items():
        signatures = asset_signatures(out / meta["assets"][spec["asset"]])
        for entry in spec["functions"]:
            inputs, outputs = signatures[entry["function"]]
            for want, got in ((spec["inputs"], inputs), (spec["outputs"], outputs)):
                got_dtypes = [(n, t.split("(")[1].split(",")[0].lower()) for n, t in got]
                assert got_dtypes == [(n, _dtype(s)) for n, s in want.items()], name


# --- CLI ---


@pytest.fixture
def no_export(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("export must not start")

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fail)


def test_cli_precision_conflicts_exit_before_loading(tmp_path, no_export, capsys):
    out = str(tmp_path)
    with pytest.raises(SystemExit) as macos:
        embedding_export.main(
            ["embeddinggemma-2", "--compute-precision", "float16", "--output-dir", out]
        )
    assert macos.value.code == 2 and "not float16" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ios:
        embedding_export.main(
            ["embeddinggemma-2", "--platform", "iOS", "--compute-precision", "float32"]
        )
    assert ios.value.code == 2 and "computes in float16" in capsys.readouterr().err
    with pytest.raises(SystemExit) as fused:
        embedding_export.main(["embeddinggemma-2", "--fused"])
    assert fused.value.code == 2 and "no fused export" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        ([], "embeddinggemma-2"),
        (["--text-only"], "embeddinggemma-2-text"),
        (["--platform", "iOS"], "embeddinggemma-2-ios"),
        (["--platform", "iOS", "--skip-vision"], "embeddinggemma-2-ios-text"),
    ],
)
def test_cli_default_output_names(argv, name, tmp_path, monkeypatch):
    seen = {}

    def fake(exporter, out, **kwargs):
        seen.update(exporter=exporter, out=Path(out), **kwargs)
        return {}

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fake)
    rc = embedding_export.main(["embeddinggemma-2", *argv, "--output-dir", str(tmp_path)])
    assert rc == 0
    assert seen["out"] == tmp_path / name and seen["exporter"] is EMBEDDINGGEMMA2_EXPORTER


def test_cli_dry_run_never_exports(no_export, capsys):
    assert embedding_export.main(["embeddinggemma-2", "--platform", "iOS", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "iOS" in out and "float16" in out and "text,image" in out
    assert "embeddinggemma-2-ios" in out


def test_cli_list_models(capsys):
    assert embedding_export.main(["--list-models"]) == 0
    rows = [r for r in capsys.readouterr().out.splitlines() if r.startswith("embeddinggemma-2 ")]
    assert len(rows) == 1 and "text,image" in rows[0]
    assert rows[0].endswith("macOS:float32,bfloat16 iOS:float16")
