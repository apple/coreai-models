# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``kind: "embedding"`` bundle contract and exporter (``coreai_models.embedding``).

Metadata, precision and CLI tests need no hardware or weights. The conversion tests use tiny random
EmbeddingGemma 2 configs and skip when ``coreai_torch`` is unavailable.
"""

from __future__ import annotations

import dataclasses
import json
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
    check_compute_precision,
    consolidate_prompts,
    embedding_block,
    encoder_functions,
    family_for_model,
    text_functions,
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
    assert check_compute_precision(EG2, "float16", platform="iOS") == "float16"
    for precision in ("float32", "bfloat16"):
        with pytest.raises(EmbeddingPrecisionError, match="float16"):
            check_compute_precision(EG2, precision, platform="iOS")


def test_macos_refuses_float16_and_allows_bfloat16():
    with pytest.raises(EmbeddingPrecisionError, match="forbids float16"):
        check_compute_precision(EG2, "float16")
    assert check_compute_precision(EG2, "bfloat16") == "bfloat16"
    assert check_compute_precision(EG2, "float32") == "float32"


def test_unknown_precision_is_refused():
    with pytest.raises(EmbeddingPrecisionError, match="Unknown compute precision"):
        check_compute_precision(EG2, "int8")


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


def test_consolidate_prompts_fails_closed():
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


def test_embedding_block_macos_text():
    block = _block()
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
    ]
    assert block["space"] == {"id": "google/embeddinggemma-2", "revision": REVISION}
    assert block["matryoshka_dimensions"] == [768, 512, 256, 128]
    assert block["precision"] == {
        "compute": "float32",
        "forbidden": ["float16"],
        "compute_units": ["gpu"],
    }
    assert set(block["components"]) == {"text_encoder"}
    component = block["components"]["text_encoder"]
    assert component["output"] == "embedding" and component["compute_units"] == ["gpu"]
    assert {"function": "len2048_b8", "batch": 8, "length": 2048} in component["functions"]
    (tower,) = block["towers"]
    assert tower["pipeline"] == ["text_encoder"]
    assert tower["batch_sizes"] == [1, 8, 32]
    assert tower["length_buckets"] == [128, 512, 2048]
    assert tower["max_batch_tokens"] == 16384
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
    json.dumps(block)


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
    assert set(bundle_assets(platform, with_image)) == set(components)
    assert not REMOVED_KEYS & set(_keys(block)), REMOVED_KEYS & set(_keys(block))
    assert all("batch" not in tower and "output" in tower for tower in block["towers"])
    assert "output" not in block["precision"]
    json.dumps(block)


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


def test_embedding_block_macos_image_details():
    block = _block("macOS", True)
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
    assert _block(compute_precision="bfloat16")["precision"]["compute"] == "bfloat16"
    assert _block("iOS")["precision"]["compute"] == "float16"
    with pytest.raises(EmbeddingPrecisionError, match="forbids float16"):
        _block(compute_precision="float16")


def test_export_writes_the_requested_precision_to_metadata(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        embedding_export, "resolve_snapshot", lambda *a: (tmp_path, "0123456789abcdef0123")
    )
    monkeypatch.setattr(embedding_export, "verify_pooling_config", lambda *a: None)
    monkeypatch.setattr(embedding_export, "read_prompts", lambda *a: (PROMPTS, None))
    monkeypatch.setattr(embedding_export, "consolidate_prompts", lambda family, prompts: prompts)
    monkeypatch.setattr(embedding_export, "copy_tokenizer", lambda *a: ["tokenizer.json"])
    monkeypatch.setattr(embedding_export, "export_gpu_programs", lambda *a, **kw: seen.update(kw))
    meta = embedding_export.export_embedding_bundle(
        EG2, tmp_path / "out", with_image=False, compute_precision="bfloat16"
    )
    assert seen["compute_precision"] == "bfloat16"
    assert meta["embedding"]["precision"]["compute"] == "bfloat16"
    written = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert written["embedding"]["precision"]["compute"] == "bfloat16"


def test_failed_export_removes_the_bundle_dir(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        embedding_export, "resolve_snapshot", lambda *a: (tmp_path, "0123456789abcdef0123")
    )
    monkeypatch.setattr(embedding_export, "verify_pooling_config", lambda *a: None)
    monkeypatch.setattr(embedding_export, "read_prompts", lambda *a: (PROMPTS, None))
    monkeypatch.setattr(embedding_export, "consolidate_prompts", lambda family, prompts: prompts)
    monkeypatch.setattr(embedding_export, "copy_tokenizer", lambda *a: ["tokenizer.json"])
    monkeypatch.setattr(embedding_export, "export_gpu_programs", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        embedding_export.export_embedding_bundle(EG2, tmp_path / "out", with_image=False)
    assert not (tmp_path / "out").exists()


def test_forbidden_precision_message_lists_the_allowed_ones():
    with pytest.raises(EmbeddingPrecisionError, match=r"Use float32 \(default\) or bfloat16\."):
        check_compute_precision(EG2, "float16")
    no_floor = dataclasses.replace(EG2, precision_floor=None)
    with pytest.raises(EmbeddingPrecisionError, match=r"Use float32 \(default\)\.$"):
        check_compute_precision(no_floor, "float16")


def test_bundle_metadata_envelope():
    meta = build_bundle_metadata(
        EG2,
        _block("iOS", True),
        name=bundle_name(EG2, "iOS", True),
        assets=bundle_assets("iOS", True),
        hf_revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    )
    assert meta["metadata_version"] == "0.2" and meta["kind"] == "embedding"
    assert meta["name"] == "embeddinggemma-2-ios"
    pipelines = {c for t in meta["embedding"]["towers"] for c in t["pipeline"]}
    assert pipelines == set(meta["assets"])
    assert meta["source"]["hf_model_id"] == "google/embeddinggemma-2"


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


def test_verify_pooling_config_fails_closed(tmp_path):
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

TINY_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "sliding_window": 3,
    "layer_types": ["sliding_attention", "full_attention"],
    "per_layer_config": {"01": {"head_dim": 32, "num_key_value_heads": 1}},
    "rope_parameters": {
        "full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
    },
    "hidden_size_per_layer_input": 8,
    "embedding_dim": 24,
}


def test_tiny_multifunction_conversion():
    pytest.importorskip("coreai_torch")
    wrapper = export_macos.TextEmbeddingExportWrapper(_tiny_text_encoder())
    plan = bucket_plan([16], [1, 4], 1024)
    ids, mask = export_macos.reference_inputs(plan[1], 64)
    assert ids.dtype == torch.int32 and mask.dtype == torch.int32 and ids.shape == (4, 16)
    out = wrapper(ids, mask)
    assert out.dtype == torch.float32 and out.shape == (4, 24)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(4))
    assert export_macos.export_text_encoder_program(wrapper, plan, 64) is not None


def _tiny_text_encoder():
    from coreai_models.models.macos.embedding_gemma2 import (
        EmbeddingGemma2TextConfig,
        EmbeddingGemma2TextEncoder,
    )

    config = EmbeddingGemma2TextConfig.from_dict(TINY_CONFIG)
    torch.manual_seed(0)
    return EmbeddingGemma2TextEncoder(config).eval()


def _tiny_image_encoder():
    from coreai_models.models.macos.embedding_gemma2_vision import (
        EmbeddingGemma2ImageEncoder,
        EmbeddingGemma2VisionConfig,
    )

    config = EmbeddingGemma2VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        patch_size=4,
        pooling_kernel_size=3,
        position_embedding_size=64,
        text_hidden_size=32,
        max_soft_tokens=12,
    )
    torch.manual_seed(1)
    return EmbeddingGemma2ImageEncoder(config).eval()


TINY_IMAGE = dataclasses.replace(IMAGE, patch_size=4, budget=12, soft_token_dim=32)


def test_inputs_embeds_path_equals_ids_path():
    """token_table(ids) -> encoder(inputs_embeds) == text_encoder(ids) (the merge is lossless)."""
    text = _tiny_text_encoder()
    ids = torch.randint(3, 64, (1, 16), dtype=torch.int32)
    mask = torch.ones(1, 16, dtype=torch.int32)
    mask[0, 11:] = 0
    with torch.no_grad():
        direct = export_macos.TextEmbeddingExportWrapper(text)(ids, mask)
        embeds = export_utils.TokenTableExportWrapper(text.embed_tokens)(ids)
        with export_macos.without_token_table(text) as encoder:
            merged = export_macos.InputsEmbedsEncoderWrapper(encoder)(embeds, mask)
    assert embeds.dtype == torch.float32 and embeds.shape == (1, 16, 32)
    torch.testing.assert_close(merged, direct, atol=1e-6, rtol=0)
    assert isinstance(text.embed_tokens, torch.nn.Embedding)  # restored


def test_float16_scaled_token_table_equals_the_float32_table_cast():
    """The float16 token table gives the encoder the same inputs as the float32 one."""
    text = _tiny_text_encoder()
    ids = torch.randint(3, 64, (1, 16), dtype=torch.int32)
    table = export_utils.scaled_token_table(
        text.embed_tokens.weight, text.config.hidden_size**0.5, torch.float16
    )
    with torch.no_grad():
        want = export_utils.TokenTableExportWrapper(text.embed_tokens)(ids).half()
        got = export_utils.TokenTableExportWrapper(table, torch.float16)(ids)
    assert got.dtype == torch.float16
    assert torch.equal(got, want)


def test_tiny_float16_token_table_conversion():
    pytest.importorskip("coreai_torch")
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


def test_tiny_shared_text_program_conversion():
    pytest.importorskip("coreai_torch")
    program = export_macos.export_shared_text_program(
        _tiny_text_encoder(),
        [FunctionSpec("len16_b1", 1, 16), FunctionSpec("len32_b1", 1, 32)],
        [FunctionSpec("embeds_len32_b1", 1, 32)],
        max_length=64,
    )
    assert program is not None
    image = export_macos.export_image_encoder_program(_tiny_image_encoder(), TINY_IMAGE)
    assert image is not None


def test_macos_image_bundle_shares_the_text_encoder_asset():
    assets = bundle_assets("macOS", True)
    assert assets["token_table"] == assets["encoder"] == assets["text_encoder"]
    assert assets["image_encoder"] == "image_encoder.aimodel"
    names = [f.name for f in text_functions(EG2, "macOS") + encoder_functions(EG2, "macOS")]
    assert len(names) == len(set(names)) and "main" not in names


def test_ane_reference_inputs_shapes():
    embeds, bias, pool = export_ios.ane_text_reference_inputs(128, 512)
    assert embeds.shape == (1, 128, 512) and bias.shape == (1, 128, 1, 128)
    assert pool.shape == (1, 128) and {t.dtype for t in (embeds, bias, pool)} == {torch.float16}
    pixels, positions = export_utils.image_reference_inputs(IMAGE, torch.float16, torch.float16)
    assert pixels.shape == (1, 2520, 768) and positions.shape == (1, 2520, 2)
    assert pixels.dtype == positions.dtype == torch.float16
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0 and torch.all(positions[0, ~real] == -1)


def test_ane_band_wrapper_matches_explicit_sliding_bias():
    """The in-graph band equals an explicit sliding-window bias built on the host."""
    from coreai_models.models.ios.embedding_gemma2 import MASK_NEG, EmbeddingGemma2TextEncoder
    from coreai_models.models.macos.embedding_gemma2 import EmbeddingGemma2TextConfig

    config = EmbeddingGemma2TextConfig.from_dict(TINY_CONFIG)
    torch.manual_seed(0)
    model = EmbeddingGemma2TextEncoder(config).eval()
    length, valid = 16, 12
    pos = torch.arange(length)
    key_ok = (pos < valid)[:, None]  # (key, query)
    bias = torch.where(key_ok, 0.0, MASK_NEG).expand(length, length)
    band = (pos[:, None] - pos[None, :]).abs() <= 3
    sliding = torch.where(key_ok & band, 0.0, MASK_NEG)
    bias, sliding = (t.reshape(1, length, 1, length) for t in (bias, sliding))
    embeds = torch.randn(1, length, 32)
    pool = (pos < valid).float().reshape(1, length) / valid
    wrapper = export_ios.EncoderBandWrapper(model, length, 3)
    with torch.no_grad():
        want = model(embeds, bias, pool, sliding)
        got = wrapper(embeds, bias, pool)
    torch.testing.assert_close(got, want)
    assert not export_ios.EncoderBandWrapper(model, 4, 3).banded


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
