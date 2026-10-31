# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``kind: "embedding"`` bundle contract and exporter (``coreai_models.embedding``).

Metadata, precision and CLI tests need no hardware or weights. The conversion tests use tiny random
EmbeddingGemma 2 configs; the Neural Engine programs also run on the Core AI runtime (CPU and GPU)
when it is available.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path

import pytest
import tomllib
import torch

from coreai_models.embedding import export as embedding_export
from coreai_models.embedding import export_ios, export_macos, export_utils
from coreai_models.embedding.contract import (
    EMBEDDING_FAMILIES,
    EmbeddingPrecisionError,
    FunctionSpec,
    bucket_plan,
    build_bundle_metadata,
    bundle_assets,
    bundle_name,
    consolidate_prompts,
    embedding_block,
    encoder_functions,
    family_for_model,
    text_functions,
    validate_compute_precision,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    TINY_TEXT_CONFIG,
    TINY_VISION_CONFIG,
)

EG2 = family_for_model("embeddinggemma-2")
IMAGE = EG2.image
REVISION = "914f7f89142e33e7"
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

# Keys of earlier contract drafts that must not come back.
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
}


def _block(platform="macOS", with_image=False, **overrides):
    kwargs = dict(revision=REVISION, prompts=PROMPTS)
    kwargs.update(overrides)
    return embedding_block(EG2, platform=platform, with_image=with_image, **kwargs)


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


# --- Precision guard ---


def test_ios_requires_float16():
    assert validate_compute_precision(EG2, "float16", platform="iOS") == "float16"
    for precision in ("float32", "bfloat16"):
        with pytest.raises(EmbeddingPrecisionError, match="float16"):
            validate_compute_precision(EG2, precision, platform="iOS")


def test_macos_refuses_float16_and_allows_bfloat16():
    with pytest.raises(EmbeddingPrecisionError, match="forbids float16"):
        validate_compute_precision(EG2, "float16")
    assert validate_compute_precision(EG2, "bfloat16") == "bfloat16"
    assert validate_compute_precision(EG2, "float32") == "float32"


def test_unknown_precision_is_refused():
    with pytest.raises(EmbeddingPrecisionError, match="Unknown compute precision"):
        validate_compute_precision(EG2, "int8")


def test_export_bundle_refuses_float16_before_touching_out(tmp_path):
    out = tmp_path / "bundle"
    with pytest.raises(EmbeddingPrecisionError):
        embedding_export.export_embedding_bundle(EG2, out, compute_precision="float16")
    assert not out.exists()


# --- Prompts ---


def test_consolidate_prompts_maps_the_snapshot_table_to_eight_names():
    assert len(SNAPSHOT_PROMPTS) == 20
    prompts = consolidate_prompts(EG2, SNAPSHOT_PROMPTS)
    assert prompts == PROMPTS
    assert list(prompts) == list(EG2.prompt_names)


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
    assert [f.length for f in encoder_functions(EG2, "macOS")] == [288, 512, 2048]
    assert [f.length for f in encoder_functions(EG2, "iOS")] == [128, 512, 1024]
    assert [(f.length, f.batch) for f in text_functions(EG2, "iOS")] == [
        (128, 1),
        (512, 1),
        (1024, 1),
    ]
    assert IMAGE.sequence_length() == 284  # <bos> <boi> 280 x <image> <eoi> <eos>


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
    assert list(block["components"]) == components
    assert [t["pipeline"] for t in block["towers"]] == pipelines
    assert {k: v["compute_units"] for k, v in block["components"].items()} == units
    assert ("merge" in block) == with_image and ("preprocessing" in block) == with_image
    assets = bundle_assets(platform, with_image)
    assert {spec["asset"] for spec in block["components"].values()} == set(assets)
    assert len(set(assets.values())) == len(assets)
    want_dtype = "float16" if platform == "iOS" else "float32"
    assert {spec["output_dtype"] for spec in block["components"].values()} == {want_dtype}
    assert not REMOVED_KEYS & set(_keys(block)), REMOVED_KEYS & set(_keys(block))
    meta = build_bundle_metadata(
        EG2,
        block,
        name=bundle_name(EG2, platform, with_image),
        assets=assets,
        hf_revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    )
    assert meta["metadata_version"] == "0.2" and meta["kind"] == "embedding"
    assert {c for t in meta["embedding"]["towers"] for c in t["pipeline"]} == set(components)
    assert meta["source"]["hf_model_id"] == "google/embeddinggemma-2"


def test_embedding_block_ios_details():
    block = _block("iOS", True)
    enc = block["components"]["encoder"]
    assert enc["inputs"] == ["inputs_embeds", "bias", "pool_weights"]
    assert enc["mask_encoding"] == "additive_bias"
    assert [f["length"] for f in enc["functions"]] == [128, 512, 1024]
    assert block["precision"] == {
        "compute": "float16",
        "forbidden": [],
        "compute_units": ["neural_engine", "gpu"],
    }
    text_tower, image_tower = block["towers"]
    assert "max_batch_tokens" not in text_tower
    assert text_tower["length_buckets"] == [128, 512, 1024]
    assert image_tower["length_buckets"] == [512, 1024]
    assert image_tower["batch_sizes"] == [1]
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
        "precision",
        "components",
        "towers",
        "text",
        "merge",
        "preprocessing",
    ]
    assert block["space"] == {"id": "google/embeddinggemma-2", "revision": REVISION}
    assert block["matryoshka_dimensions"] == [768, 512, 256, 128]
    assert block["precision"] == {
        "compute": "float32",
        "forbidden": ["float16"],
        "compute_units": ["gpu"],
    }
    text_encoder = block["components"]["text_encoder"]
    assert text_encoder["output"] == "embedding"
    assert {"function": "len2048_b8", "batch": 8, "length": 2048} in text_encoder["functions"]
    text_tower = block["towers"][0]
    assert text_tower["batch_sizes"] == [1, 8, 32]
    assert text_tower["length_buckets"] == [128, 512, 2048]
    assert text_tower["max_batch_tokens"] == 16384
    text = block["text"]
    assert text["truncation"] == {"keep_eos": True, "max_tokens": 2048}
    assert text["special_tokens"] == {
        "bos": True,
        "eos": True,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    assert text["prompts"] == PROMPTS and text["default_prompt_name"] is None
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
    names = [f.name for f in text_functions(EG2, "macOS") + encoder_functions(EG2, "macOS")]
    assert len(names) == len(set(names)) and "main" not in names
    enc = block["components"]["encoder"]
    assert enc["inputs"] == ["inputs_embeds", "attention_mask"]
    assert enc["mask_encoding"] == "attention_mask"
    assert block["towers"][1]["length_buckets"] == [288, 512, 2048]
    assert block["merge"] == {
        "placeholder_ids": {"image": 258880},
        "token_ids": {"boi": 255999, "eoi": 258882},
    }
    pre = block["preprocessing"]["image"]
    assert pre["patch_size"] == 16 and pre["pooling_kernel_size"] == 3
    assert pre["max_soft_tokens"] == 280
    assert pre["rescale_factor"] == pytest.approx(1 / 255)
    assert pre["normalize"] is False


def test_embedding_block_validation():
    with pytest.raises(ValueError, match="revision"):
        _block(revision="")
    with pytest.raises(ValueError, match="prompts"):
        _block(prompts={})
    with pytest.raises(ValueError, match="default_prompt_name"):
        _block(default_prompt_name="missing")
    with pytest.raises(ValueError, match="platform"):
        _block("watchOS")
    no_image = dataclasses.replace(EG2, image=None)
    with pytest.raises(ValueError, match="no image tower"):
        embedding_block(no_image, platform="macOS", with_image=True, revision="x", prompts=PROMPTS)


def test_embedding_block_records_the_compute_precision():
    assert _block()["precision"]["compute"] == "float32"
    bf16 = _block(with_image=True, compute_precision="bfloat16")
    assert bf16["precision"]["compute"] == "bfloat16"
    assert {spec["output_dtype"] for spec in bf16["components"].values()} == {"float32"}
    assert _block("iOS")["precision"]["compute"] == "float16"
    with pytest.raises(EmbeddingPrecisionError, match="forbids float16"):
        _block(compute_precision="float16")


@pytest.fixture
def stub_snapshot(tmp_path, monkeypatch):
    """Snapshot helpers stubbed so ``export_embedding_bundle`` runs without weights."""
    monkeypatch.setattr(
        embedding_export, "resolve_snapshot", lambda *a: (tmp_path, "0123456789abcdef0123")
    )
    monkeypatch.setattr(embedding_export, "verify_pooling_config", lambda *a: None)
    monkeypatch.setattr(embedding_export, "read_prompts", lambda *a: (PROMPTS, None))
    monkeypatch.setattr(embedding_export, "consolidate_prompts", lambda family, prompts: prompts)
    monkeypatch.setattr(embedding_export, "copy_tokenizer", lambda *a: ["tokenizer.json"])


def test_export_writes_the_requested_precision_to_metadata(tmp_path, monkeypatch, stub_snapshot):
    seen = {}
    monkeypatch.setattr(embedding_export, "export_gpu_programs", lambda *a, **kw: seen.update(kw))
    meta = embedding_export.export_embedding_bundle(
        EG2, tmp_path / "out", with_image=False, compute_precision="bfloat16"
    )
    assert seen["compute_precision"] == "bfloat16"
    assert meta["embedding"]["precision"]["compute"] == "bfloat16"
    written = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert written["embedding"]["precision"]["compute"] == "bfloat16"


def test_failed_export_removes_the_bundle_dir(tmp_path, monkeypatch, stub_snapshot):
    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(embedding_export, "export_gpu_programs", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        embedding_export.export_embedding_bundle(EG2, tmp_path / "out", with_image=False)
    assert not (tmp_path / "out").exists()


def test_forbidden_precision_message_lists_the_allowed_ones():
    with pytest.raises(EmbeddingPrecisionError, match=r"Use float32 \(default\) or bfloat16\."):
        validate_compute_precision(EG2, "float16")
    no_floor = dataclasses.replace(EG2, precision_floor=None)
    with pytest.raises(EmbeddingPrecisionError, match=r"Use float32 \(default\)\.$"):
        validate_compute_precision(no_floor, "float16")


def test_family_registry():
    assert "embeddinggemma-2" in EMBEDDING_FAMILIES
    assert family_for_model("Google/EmbeddingGemma-2") is EG2
    with pytest.raises(KeyError, match="embeddinggemma-2"):
        family_for_model("google/embeddinggemma-300m")


# --- Snapshot helpers ---


def _fake_snapshot(root: Path, pooling: dict) -> Path:
    snap = root / "models--google--embeddinggemma-2" / "snapshots" / ("a" * 40)
    (snap / "1_Pooling").mkdir(parents=True)
    (snap / "2_Normalize").mkdir()
    (snap / "1_Pooling" / "config.json").write_text(json.dumps(pooling))
    (snap / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": SNAPSHOT_PROMPTS, "default_prompt_name": None})
    )
    (snap / "tokenizer.json").write_text("{}")
    (snap / "tokenizer_config.json").write_text("{}")
    return snap


def test_snapshot_helpers(tmp_path):
    snap = _fake_snapshot(
        tmp_path, {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}
    )
    path, sha = export_utils.resolve_snapshot("google/embeddinggemma-2", str(snap))
    assert sha == "a" * 40 and path == snap.resolve()
    assert export_utils.read_prompts(snap) == (SNAPSHOT_PROMPTS, None)
    export_utils.verify_pooling_config(snap, EG2)
    copied = export_utils.copy_tokenizer(snap, tmp_path / "bundle" / "tokenizer")
    assert copied == ["tokenizer.json", "tokenizer_config.json"]


def test_verify_pooling_config_rejects_include_prompt_false(tmp_path):
    snap = _fake_snapshot(
        tmp_path, {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": False}
    )
    with pytest.raises(ValueError, match="include_prompt"):
        export_utils.verify_pooling_config(snap, EG2)


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
    export_utils.verify_image_config(_image_snapshot(tmp_path / "ok"), EG2)
    with pytest.raises(ValueError, match="boi_token_id"):
        export_utils.verify_image_config(_image_snapshot(tmp_path / "bad", boi_token_id=1), EG2)


# --- Conversion (tiny random config) ---


def test_tiny_multifunction_conversion():
    wrapper = export_macos.TextEmbeddingExportWrapper(_tiny_text_encoder())
    plan = bucket_plan([16], [1, 4], 1024)
    ids, mask = export_macos.reference_inputs(plan[1], 128)
    assert ids.dtype == torch.int32 and mask.dtype == torch.int32 and ids.shape == (4, 16)
    out = wrapper(ids, mask)
    assert out.dtype == torch.float32 and out.shape == (4, 48)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(4))
    assert export_macos.export_shared_text_program(wrapper.encoder, plan, [], 64) is not None


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


TINY_IMAGE = dataclasses.replace(IMAGE, patch_size=4, budget=12, soft_token_dim=32)


def test_inputs_embeds_path_equals_ids_path():
    """``encoder(token_table(ids))`` matches ``text_encoder(ids)`` within atol 1e-6, and
    ``without_token_table`` restores ``embed_tokens`` afterwards.
    """
    text = _tiny_text_encoder()
    ids = torch.randint(3, 128, (1, 16), dtype=torch.int32)
    mask = torch.ones(1, 16, dtype=torch.int32)
    mask[0, 11:] = 0
    with torch.no_grad():
        direct = export_macos.TextEmbeddingExportWrapper(text)(ids, mask)
        embeds = export_utils.TokenTableExportWrapper(text.embed_tokens)(ids)
        with export_macos.without_token_table(text) as encoder:
            merged = export_macos.InputsEmbedsEncoderWrapper(encoder)(embeds, mask)
    assert embeds.dtype == torch.float32 and embeds.shape == (1, 16, 64)
    torch.testing.assert_close(merged, direct, atol=1e-6, rtol=0)
    assert isinstance(text.embed_tokens, torch.nn.Embedding)  # restored


def test_float16_scaled_token_table_equals_the_float32_table_cast():
    """Lookups in the float16 ``scaled_token_table`` equal the float32 ``embed_tokens`` lookups
    cast to float16.
    """
    text = _tiny_text_encoder()
    ids = torch.randint(3, 128, (1, 16), dtype=torch.int32)
    table = export_utils.scaled_token_table(
        text.embed_tokens.weight, text.config.hidden_size**0.5, torch.float16
    )
    with torch.no_grad():
        want = export_utils.TokenTableExportWrapper(text.embed_tokens)(ids).half()
        got = export_utils.TokenTableExportWrapper(table, torch.float16)(ids)
    assert got.dtype == torch.float16
    assert torch.equal(got, want)


def test_tiny_float16_token_table_conversion():
    text = _tiny_text_encoder()
    table = export_utils.scaled_token_table(
        text.embed_tokens.weight, text.config.hidden_size**0.5, torch.float16
    )
    program = export_utils.export_token_table_program(table, 64, dtype=torch.float16)
    assert program is not None


def test_image_reference_inputs_are_a_valid_padded_grid():
    pixels, positions = export_utils.image_reference_inputs(TINY_IMAGE)
    assert pixels.shape == (1, 108, 48) and pixels.dtype == torch.float32
    assert positions.shape == (1, 108, 2) and positions.dtype == torch.int32
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0
    assert torch.all(pixels[0, ~real] == 0)
    soft = export_macos.ImageEncoderExportWrapper(_tiny_image_encoder())(pixels, positions)
    n = int(real.sum()) // 9
    assert soft.shape == (1, 12, 32) and soft.dtype == torch.float32
    assert torch.all(soft[0, n:] == 0) and torch.all(soft[0, :n].abs().sum(-1) > 0)


def _function_names(program) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        asset = program.save_asset(Path(tmp) / "program.aimodel")
        return sorted(asset.summary(include_statistics=False).function_names)


def test_tiny_shared_text_program_conversion():
    program = export_macos.export_shared_text_program(
        _tiny_text_encoder(),
        [FunctionSpec("len16_b1", 1, 16), FunctionSpec("len32_b1", 1, 32)],
        [FunctionSpec("embeds_len32_b1", 1, 32)],
        max_length=64,
    )
    assert _function_names(program) == ["embeds_len32_b1", "len16_b1", "len32_b1", "main"]
    image = export_macos.export_image_encoder_program(_tiny_image_encoder(), TINY_IMAGE)
    assert image is not None


def test_ane_reference_inputs_shapes():
    embeds, bias, pool = export_ios.ane_text_reference_inputs(128, 512)
    assert embeds.shape == (1, 128, 512) and bias.shape == (1, 128, 1, 128)
    assert pool.shape == (1, 128) and {t.dtype for t in (embeds, bias, pool)} == {torch.float16}
    pixels, positions = export_utils.image_reference_inputs(IMAGE, torch.float16, torch.float16)
    assert pixels.shape == (1, 2520, 768) and positions.shape == (1, 2520, 2)
    assert pixels.dtype == positions.dtype == torch.float16
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0 and torch.all(positions[0, ~real] == -1)


# --- Neural Engine conversion (tiny random config) ---

# Encoder lengths 4 (unbanded) and 16 (banded: > sliding_window + 1); 16 also fits the image
# sequence (budget 12 + 4).
TINY_ANE_FAMILY = dataclasses.replace(
    EG2, image=TINY_IMAGE, ane_length_buckets=(4, 16), model_max_tokens=64
)


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
    token_table = export_utils.scaled_token_table(
        table, text.config.hidden_size**0.5, torch.float16
    )
    torch.manual_seed(1)
    image_encoder = ios_vision.EmbeddingGemma2ImageEncoder(TINY_VISION_CONFIG).half().eval()
    functions = encoder_functions(TINY_ANE_FAMILY, "iOS")
    window = int(text.config.sliding_window)
    return {
        "encoder": (
            encoder,
            export_ios.export_ane_encoder_program(encoder, functions, window),
        ),
        "token_table": (
            token_table,
            export_utils.export_token_table_program(token_table, 64, dtype=torch.float16),
        ),
        "image_encoder": (
            image_encoder,
            export_ios.export_ane_image_encoder_program(image_encoder, TINY_IMAGE),
        ),
    }


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
        TINY_ANE_FAMILY, platform="iOS", with_image=True, revision=REVISION, prompts=PROMPTS
    )
    encoder = block["components"]["encoder"]
    assert [f["function"] for f in encoder["functions"]] == ["embeds_len4_b1", "embeds_len16_b1"]
    for name, spec in block["components"].items():
        if "functions" in spec:
            functions = [f["function"] for f in spec["functions"]]
        else:
            functions = [spec["function"]]
        assert _function_names(tiny_ane_programs[name][1]) == sorted(functions), name


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_ane_programs_match_pytorch(tiny_ane_programs, compute_unit):
    """Each converted program, run by the Core AI runtime, outputs the metadata's ``output_dtype``
    and matches its float16 module: the token table exactly, the encoders within atol = rtol = 1e-2.
    """
    import coreai.runtime as rt

    from coreai_models.models.ios.embedding_gemma2 import MASK_NEG
    from coreai_models.models.macos.embedding_gemma2_vision import pad_to_budget, patch_position_ids

    if not rt.SpecializationOptions.is_supported():
        pytest.skip("the Core AI runtime cannot run programs here")
    options = (
        rt.SpecializationOptions.cpu_only()
        if compute_unit == "cpu"
        else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    )

    dtypes = {
        name: getattr(torch, spec["output_dtype"])
        for name, spec in _block("iOS", True)["components"].items()
    }
    encoder, program = tiny_ane_programs["encoder"]
    for spec in encoder_functions(TINY_ANE_FAMILY, "iOS"):
        n = spec.length
        valid = torch.arange(n) < n - 1
        embeds, _, _ = export_ios.ane_text_reference_inputs(n, 64)
        bias = torch.where(valid.reshape(1, n, 1, 1), 0.0, MASK_NEG).expand(1, n, 1, n)
        pool = (valid.float() / valid.sum()).reshape(1, n)
        inputs = {"inputs_embeds": embeds, "bias": bias.half(), "pool_weights": pool.half()}
        with torch.no_grad():
            wrapper = export_ios.EncoderBandWrapper(encoder, n, encoder.config.sliding_window)
            want = wrapper(*inputs.values())
        got = asyncio.run(_run_function(program, spec.name, inputs, options))["embedding"]
        assert got.dtype == dtypes["encoder"] and got.shape == want.shape == (1, 48)
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)

    table, program = tiny_ane_programs["token_table"]
    ids = torch.randint(3, 128, (1, 9), dtype=torch.int32)
    got = asyncio.run(_run_function(program, "main", {"input_ids": ids}, options))
    assert got["inputs_embeds"].dtype == dtypes["token_table"]
    assert torch.equal(
        got["inputs_embeds"], export_utils.TokenTableExportWrapper(table, torch.float16)(ids)
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
    assert macos.value.code == 2 and "forbids float16" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ios:
        embedding_export.main(
            ["embeddinggemma-2", "--platform", "iOS", "--compute-precision", "float32"]
        )
    assert ios.value.code == 2 and "float16" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_cli_unknown_model_exits_2(no_export, capsys):
    with pytest.raises(SystemExit) as info:
        embedding_export.main(["acme/none"])
    assert info.value.code == 2
    assert "--list-models" in capsys.readouterr().err


def test_cli_missing_model_says_model_required(no_export, capsys):
    for argv in ([], ["--dry-run"]):
        with pytest.raises(SystemExit) as missing:
            embedding_export.main(argv)
        assert missing.value.code == 2
        assert "a model is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        ([], "embeddinggemma-2"),
        (["--skip-vision"], "embeddinggemma-2-text"),
        (["--platform", "iOS"], "embeddinggemma-2-ios"),
        (["--platform", "iOS", "--skip-vision"], "embeddinggemma-2-ios-text"),
    ],
)
def test_cli_default_output_names(argv, name, tmp_path, monkeypatch):
    seen = {}

    def fake(family, out, **kwargs):
        seen["out"] = Path(out)
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fake)
    rc = embedding_export.main(["embeddinggemma-2", *argv, "--output-dir", str(tmp_path)])
    assert rc == 0
    assert seen["out"] == tmp_path / name


def test_cli_default_output_is_under_exports(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        embedding_export, "export_embedding_bundle", lambda f, out, **kw: seen.update(out=out)
    )
    assert embedding_export.main(["embeddinggemma-2"]) == 0
    assert Path(seen["out"]).parent.name == "exports"


def test_cli_dry_run_never_exports(no_export, capsys):
    assert embedding_export.main(["embeddinggemma-2", "--platform", "iOS", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "iOS" in out and "float16" in out and "encoder.aimodel" in out
    assert "embeds_len128_b1" in out and "embeddinggemma-2-ios" in out


def test_cli_list_models(capsys):
    assert embedding_export.main(["--list-models"]) == 0
    assert "embeddinggemma-2" in capsys.readouterr().out


def test_console_script_is_registered():
    root = Path(__file__).resolve().parents[4]
    scripts = tomllib.loads((root / "python" / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["coreai.embedding.export"] == "coreai_models.embedding.export:main"
