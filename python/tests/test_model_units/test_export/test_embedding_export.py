# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the EmbeddingGemma 2 embedding exporter (``coreai_models.embedding.embedding_gemma2``).

``test_embedding_contract.py`` covers the shared contract, orchestrator and CLI. Metadata,
snapshot and CLI tests here need no hardware or weights. The bundle tests export tiny random
EmbeddingGemma 2 modules from a fake snapshot; the bundles also run on the Core AI runtime (CPU and
GPU) when it is available.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from coreai_models.embedding import embedding_gemma2 as eg2
from coreai_models.embedding import export as embedding_export
from coreai_models.embedding.contract import CheckpointFacts, validate_block
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
    run_function,
    runtime_options,
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
COMPONENT_KEYS = ["asset", "inputs", "outputs", "compute_units", "functions"]
WORDS = ["<unk>", "a", "photo", "of", "the", "cat"]

TINY_IMAGE = dataclasses.replace(IMAGE, patch_size=4, max_soft_tokens=12, soft_token_dim=32)
# Text functions len16_b1 and len16_b2 (longer than the sliding window of 3), macOS encoder
# embeds_len32_b1, iOS encoder lengths 4 (unbanded) and 16 (banded; fits the image sequence).
TINY = dataclasses.replace(
    EG2,
    image_tower=TINY_IMAGE,
    length_buckets=(16,),
    batch_sizes=(2,),
    ane_length_buckets=(4, 16),
    max_tokens=64,
)


def _block(platform="macOS", with_image=False, family=EG2, **overrides):
    kwargs = dict(revision=REVISION, prompts=PROMPTS)
    kwargs.update(overrides)
    return embedding_block(family, EG2_FACTS, platform=platform, with_image=with_image, **kwargs)


def _dtype(spec: str) -> str:
    return spec.split("[")[0]


def _snapshot(root: Path, image=IMAGE, pooling=POOLING, default_prompt_name=None) -> Path:
    """A fake HF snapshot: pooling, normalize, config, processor and prompt files for ``image``."""
    snap = root / "models--google--embeddinggemma-2" / "snapshots" / ("a" * 40)
    (snap / "1_Pooling").mkdir(parents=True)
    (snap / "2_Normalize").mkdir()
    config = {
        "image_token_id": image.placeholder_id,
        "boi_token_id": image.boi_token_id,
        "eoi_token_id": image.eoi_token_id,
        "text_config": {**TEXT_IDS, "hidden_size": image.soft_token_dim},
    }
    processor = {
        "patch_size": image.patch_size,
        "pooling_kernel_size": image.pooling_kernel_size,
        "max_soft_tokens": image.max_soft_tokens,
        "do_normalize": image.normalize,
        "resample": 3,
        "rescale_factor": image.rescale_factor,
    }
    files = {
        "1_Pooling/config.json": pooling,
        "config.json": config,
        "processor_config.json": {"image_processor": processor},
        "config_sentence_transformers.json": {
            "prompts": SNAPSHOT_PROMPTS,
            "default_prompt_name": default_prompt_name,
        },
    }
    for name, content in files.items():
        (snap / name).write_text(json.dumps(content))
    return snap


# --- Prompts and function plan ---


def test_consolidate_prompts_maps_the_snapshot_table():
    assert len(SNAPSHOT_PROMPTS) == 20
    prompts, default = consolidate_prompts(EG2, SNAPSHOT_PROMPTS)
    assert prompts == PROMPTS and default is None
    assert list(prompts) == list(EG2.prompt_names)
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "STS") == (PROMPTS, "sentence_similarity")
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "query")[1] == "query"
    assert consolidate_prompts(EG2, SNAPSHOT_PROMPTS, "SearchQuery")[1] == "query"  # by prefix


@pytest.mark.parametrize(
    ("snapshot_prompts", "default", "message"),
    [
        ({**SNAPSHOT_PROMPTS, "Translation": "task: translate | query: "}, None, "canonical"),
        ({k: v for k, v in SNAPSHOT_PROMPTS.items() if k != "FactChecking"}, None, "FactChecking"),
        (SNAPSHOT_PROMPTS, "Translation", "default_prompt_name 'Translation'"),
    ],
    ids=["unknown-prefix", "missing-key", "unknown-default"],
)
def test_consolidate_prompts_rejects(snapshot_prompts, default, message):
    with pytest.raises(ValueError, match=message):
        consolidate_prompts(EG2, snapshot_prompts, default)


def _names(functions: list[FunctionSpec]) -> str:
    return " ".join(f.name for f in functions)


def test_function_plans():
    """Batch 1 is kept for every length; batch 32 at length 2048 is above the 16384-token cap."""
    plan = bucket_plan([2048, 128, 512], [32, 8], 16384)
    assert _names(plan) == (
        "len128_b1 len128_b8 len128_b32 len512_b1 len512_b8 len512_b32 len2048_b1 len2048_b8"
    )
    assert text_functions(EG2) == plan
    assert encoder_functions(EG2, "macOS", with_image=False) == []
    macos_plan = encoder_functions(EG2, "macOS", with_image=True)
    assert _names(macos_plan) == "embeds_len288_b1 embeds_len512_b1 embeds_len2048_b1"
    for with_image in (False, True):
        ios_plan = encoder_functions(EG2, "iOS", with_image=with_image)
        assert _names(ios_plan) == "embeds_len128_b1 embeds_len512_b1 embeds_len1024_b1"
    assert IMAGE.max_sequence_length == 284  # <bos> <boi> 280 x <image> <eoi> <eos>
    with pytest.raises(ValueError, match="length_buckets"):
        bucket_plan([1], [1], 1024)
    with pytest.raises(ValueError, match="ane_length_buckets"):
        encoder_functions(
            dataclasses.replace(EG2, ane_length_buckets=(1, 128)), "iOS", with_image=False
        )


# --- embedding_block ---


@pytest.mark.parametrize(
    ("platform", "with_image", "components", "pipelines"),
    [
        ("macOS", False, {"text_encoder": ("text_encoder", "gpu")}, [["text_encoder"]]),
        (
            "macOS",
            True,
            {
                "text_encoder": ("text_encoder", "gpu"),
                "image_encoder": ("image_encoder", "gpu"),
                "token_table": ("text_encoder", "gpu"),
                "encoder": ("text_encoder", "gpu"),
            },
            [["text_encoder"], ["image_encoder", "token_table", "encoder"]],
        ),
        (
            "iOS",
            False,
            {"token_table": ("token_table", "gpu"), "encoder": ("encoder", "neural_engine")},
            [["token_table", "encoder"]],
        ),
        (
            "iOS",
            True,
            {
                "token_table": ("token_table", "gpu"),
                "encoder": ("encoder", "neural_engine"),
                "image_encoder": ("image_encoder", "neural_engine"),
            },
            [["token_table", "encoder"], ["image_encoder", "token_table", "encoder"]],
        ),
    ],
    ids=["macOS-text", "macOS-image", "iOS-text", "iOS-image"],
)
def test_embedding_block_platform_matrix(platform, with_image, components, pipelines):
    """Components in order with their asset and compute unit, one ``.aimodel`` per asset, the
    pipelines and the platform's output dtype.
    """
    block = _block(platform, with_image)
    assets = bundle_assets(platform, with_image)
    validate_block(block, assets)
    got = {name: (c["asset"], *c["compute_units"]) for name, c in block["components"].items()}
    assert list(got.items()) == list(components.items())
    assert assets == {a: f"{a}.aimodel" for a in dict.fromkeys(a for a, _ in components.values())}
    assert block["towers"] == [
        {"modality": modality, "pipeline": pipeline}
        for modality, pipeline in zip(["text", "image"], pipelines, strict=False)
    ]
    assert ("image" in block) == with_image
    outputs = [s for c in block["components"].values() for s in c["outputs"].values()]
    assert {_dtype(s) for s in outputs} == {"float16" if platform == "iOS" else "float32"}


def test_embedding_block_ios_details():
    block = _block("iOS", True)
    assert list(block) == list(_block("macOS", True))
    assert [list(c) for c in block["components"].values()] == [
        COMPONENT_KEYS,
        [*COMPONENT_KEYS, "bias", "pool_weights"],
        COMPONENT_KEYS,
    ]
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
    assert all(list(c) == COMPONENT_KEYS for c in block["components"].values())
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision": ""}, "revision"),
        ({"prompts": {}}, "prompts"),
        ({"default_prompt_name": "missing"}, "default_prompt_name"),
        ({"platform": "watchOS"}, "platform"),
        ({"compute_precision": "float16"}, "not float16"),
        (
            {"family": dataclasses.replace(EG2, image_tower=None), "with_image": True},
            "no image tower",
        ),
    ],
)
def test_embedding_block_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        _block(**overrides)


# --- Snapshot ---


def test_read_facts_and_prompts_from_the_snapshot(tmp_path):
    snap = _snapshot(tmp_path / "published")
    assert EMBEDDINGGEMMA2_EXPORTER.read_facts(snap) == EG2_FACTS
    assert eg2.read_prompts(snap) == (SNAPSHOT_PROMPTS, None)
    other = _snapshot(tmp_path / "other", pooling={**POOLING, "embedding_dimension": 512})
    config = json.loads((other / "config.json").read_text())
    config["text_config"]["eos_token_id"] = 106
    (other / "config.json").write_text(json.dumps(config))
    facts = EMBEDDINGGEMMA2_EXPORTER.read_facts(other)
    assert (facts.dimension, facts.eos_token_id, facts.max_tokens) == (512, 106, 8192)
    (other / "1_Pooling" / "config.json").unlink()
    with pytest.raises(ValueError, match="no 1_Pooling"):
        EMBEDDINGGEMMA2_EXPORTER.read_facts(other)


@pytest.mark.parametrize(
    ("pooling", "message"),
    [
        ({**POOLING, "include_prompt": False}, "include_prompt"),
        ({**POOLING, "pooling_mode": "cls"}, "pooling_mode"),
    ],
)
def test_read_facts_rejects_a_pooling_config_unlike_the_family(tmp_path, pooling, message):
    with pytest.raises(ValueError, match=message):
        EMBEDDINGGEMMA2_EXPORTER.read_facts(_snapshot(tmp_path, pooling=pooling))


def test_verify_image_config(tmp_path):
    eg2.verify_image_config(_snapshot(tmp_path / "ok"), EG2)
    bad = _snapshot(tmp_path / "bad", dataclasses.replace(IMAGE, boi_token_id=1))
    with pytest.raises(ValueError, match="boi_token_id"):
        eg2.verify_image_config(bad, EG2)


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
    from coreai_models.models.macos import embedding_gemma2_vision as v

    text_ids = (v.EG2_BOS_TOKEN_ID, v.EG2_EOS_TOKEN_ID, v.EG2_PAD_TOKEN_ID)
    assert text_ids == (EG2_FACTS.bos_token_id, EG2_FACTS.eos_token_id, EG2_FACTS.pad_token_id)
    image_ids = (v.EG2_IMAGE_TOKEN_ID, v.EG2_BOI_TOKEN_ID, v.EG2_EOI_TOKEN_ID)
    assert image_ids == (IMAGE.placeholder_id, IMAGE.boi_token_id, IMAGE.eoi_token_id)


# --- Export (no conversion) ---


@pytest.fixture
def macos_calls(monkeypatch) -> list[dict]:
    """Keyword arguments of each ``export_macos_programs`` call; nothing is fetched or converted."""
    calls: list[dict] = []
    monkeypatch.setattr(eg2, "fetch_weights", lambda *a: None)
    monkeypatch.setattr(eg2, "export_macos_programs", lambda *a, **kw: calls.append(kw))
    return calls


def _tokenized_snapshot(root: Path, **kwargs) -> Path:
    snap = _snapshot(root, **kwargs)
    facts = EG2_FACTS
    save_tiny_tokenizer(snap, WORDS, facts.bos_token_id, facts.eos_token_id, facts.pad_token_id)
    return snap


def test_export_writes_the_metadata(tmp_path, macos_calls):
    snap = _tokenized_snapshot(tmp_path / "hf", default_prompt_name="STS")
    meta = embedding_export.export_embedding_bundle(
        EMBEDDINGGEMMA2_EXPORTER,
        tmp_path / "out",
        text_only=True,
        compute_precision="bfloat16",
        model_dir=str(snap),
    )
    assert [(c["with_image"], c["compute_precision"]) for c in macos_calls] == [(False, "bfloat16")]
    assert meta["name"] == "embeddinggemma-2-text"
    block = meta["embedding"]
    assert block["compute_precision"] == "bfloat16"
    outputs = [s for c in block["components"].values() for s in c["outputs"].values()]
    assert {_dtype(s) for s in outputs} == {"float32"}
    assert block["space"]["revision"] == meta["source"]["hf_revision"] == "a" * 40
    assert block["text"]["prompts"] == PROMPTS
    assert block["text"]["default_prompt_name"] == "sentence_similarity"


def test_export_checks_the_default_prompt_before_converting(tmp_path, macos_calls):
    snap = _tokenized_snapshot(tmp_path / "hf", default_prompt_name="Nope")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="default_prompt_name 'Nope'"):
        embedding_export.export_embedding_bundle(
            EMBEDDINGGEMMA2_EXPORTER, out, text_only=True, model_dir=str(snap)
        )
    assert macos_calls == [] and not out.exists()


# --- Program wrappers and trace inputs (tiny random config) ---


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


def _tiny_ios_text_encoder_and_table(dtype):
    from coreai_models.models.ios.embedding_gemma2 import text_encoder_from_state_dict

    text = _tiny_text_encoder()
    return text_encoder_from_state_dict(text.config, dict(text.state_dict()), dtype)


def _tiny_ios_image_encoder(dtype):
    from coreai_models.models.ios.embedding_gemma2_vision import EmbeddingGemma2ImageEncoder

    torch.manual_seed(1)
    return EmbeddingGemma2ImageEncoder(TINY_VISION_CONFIG).to(dtype).eval()


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
    assert isinstance(text.embed_tokens, torch.nn.Embedding)


def test_float16_scaled_token_table_equals_the_float32_table_cast():
    text = _tiny_text_encoder()
    ids = torch.randint(3, 128, (1, 16), dtype=torch.int32)
    scale = text.config.hidden_size**0.5
    table = ios.scaled_token_table(text.embed_tokens.weight, scale, torch.float16)
    with torch.no_grad():
        want = common.TokenTableExportWrapper(text.embed_tokens)(ids).half()
        got = common.TokenTableExportWrapper(table, torch.float16)(ids)
    assert got.dtype == torch.float16
    assert torch.equal(got, want)


@pytest.mark.parametrize(
    "dtypes", [(torch.float32, torch.int32), (torch.float16, torch.float16)], ids=["gpu", "ane"]
)
def test_image_reference_inputs_are_a_padded_block_grid(dtypes):
    pixels, positions = common.image_reference_inputs(IMAGE, *dtypes)
    assert (pixels.shape, positions.shape) == ((1, 2520, 768), (1, 2520, 2))
    assert (pixels.dtype, positions.dtype) == dtypes
    real = positions[0, :, 0] >= 0
    assert int(real.sum()) % 9 == 0
    assert torch.all(pixels[0, ~real] == 0) and torch.all(positions[0, ~real] == -1)


def test_ane_encoder_reference_inputs():
    embeds, bias, pool = ios.encoder_reference_inputs(128, 512)
    assert (embeds.shape, bias.shape, pool.shape) == ((1, 128, 512), (1, 128, 1, 128), (1, 128))
    assert {t.dtype for t in (embeds, bias, pool)} == {torch.float16}


# --- Tiny bundles ---


@pytest.fixture(scope="module")
def tiny_bundles(tmp_path_factory) -> dict[tuple[str, bool], Path]:
    """``(platform, text_only)`` -> bundle directory of ``TINY``, exported from a fake snapshot by
    the exporter with the snapshot loaders patched to return the tiny modules.
    """
    from coreai_models.models.ios import embedding_gemma2 as ios_text
    from coreai_models.models.ios import embedding_gemma2_vision as ios_vision
    from coreai_models.models.macos import embedding_gemma2 as macos_text
    from coreai_models.models.macos import embedding_gemma2_vision as macos_vision

    root = tmp_path_factory.mktemp("tiny")
    snapshot = str(_tokenized_snapshot(root / "hf", image=TINY_IMAGE))
    bundles = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            macos_text,
            "load_embedding_gemma2_text_encoder",
            lambda snapshot, dtype: _tiny_text_encoder().to(dtype),
        )
        patch.setattr(
            macos_vision.EmbeddingGemma2ImageEncoder,
            "from_pretrained",
            lambda snapshot, dtype, max_soft_tokens: _tiny_image_encoder().to(dtype),
        )
        patch.setattr(
            ios_text,
            "load_embedding_gemma2_text_encoder_and_table",
            lambda snapshot, dtype: _tiny_ios_text_encoder_and_table(dtype),
        )
        patch.setattr(
            ios_vision,
            "load_embedding_gemma2_image_encoder",
            lambda snapshot, dtype, max_soft_tokens: _tiny_ios_image_encoder(dtype),
        )
        patch.setattr(eg2, "fetch_weights", lambda *a: None)
        for platform in ("macOS", "iOS"):
            for text_only in (False, True):
                out = root / f"{platform}-{text_only}"
                embedding_export.export_embedding_bundle(
                    eg2.exporter(TINY),
                    out,
                    platform=platform,
                    text_only=text_only,
                    model_dir=snapshot,
                )
                bundles[platform, text_only] = out
    return bundles


def _runtime_dtypes(signature: list) -> list[tuple[str, str]]:
    """``[(name, dtype)]`` of an ``asset_signatures`` input or output list."""
    return [(name, kind.split("(")[1].split(",")[0].lower()) for name, kind in signature]


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("platform", ["macOS", "iOS"])
def test_tiny_bundle_files_match_metadata(tiny_bundles, platform, text_only):
    """The bundle holds its assets, the tokenizer and ``metadata.json``. Each asset holds exactly
    the functions its components list, with the metadata's input and output names and dtypes.
    """
    out = tiny_bundles[platform, text_only]
    meta = json.loads((out / "metadata.json").read_text())
    files = [*meta["assets"].values(), "metadata.json", "tokenizer"]
    assert sorted(p.name for p in out.iterdir()) == sorted(files)
    components = meta["embedding"]["components"].values()
    for asset, file in meta["assets"].items():
        signatures = asset_signatures(out / file)
        listed = [
            (entry["function"], spec)
            for spec in components
            if spec["asset"] == asset
            for entry in spec["functions"]
        ]
        assert sorted(signatures) == sorted(name for name, _ in listed)
        for name, spec in listed:
            inputs, outputs = signatures[name]
            assert _runtime_dtypes(inputs) == [(n, _dtype(s)) for n, s in spec["inputs"].items()]
            assert _runtime_dtypes(outputs) == [(n, _dtype(s)) for n, s in spec["outputs"].items()]


def _image_inputs(dtype, position_dtype) -> dict[str, torch.Tensor]:
    """A 6x9-patch image (6 soft tokens) padded to ``TINY_IMAGE``'s budget."""
    from coreai_models.models.macos.embedding_gemma2_vision import pad_to_budget, patch_position_ids

    pixels = torch.rand((54, TINY_IMAGE.patch_dim), generator=torch.Generator().manual_seed(2))
    pixels, positions = pad_to_budget(pixels, patch_position_ids(6, 9), TINY_IMAGE.max_patches)
    return {
        "pixel_values": pixels[None].to(dtype),
        "position_ids": positions[None].to(position_dtype),
    }


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_macos_bundle_matches_pytorch(tiny_bundles, compute_unit):
    """Each macOS function, run by the Core AI runtime, matches its float32 PyTorch wrapper within
    atol = rtol = 1e-4: ids on a ragged mask longer than the sliding window (fully masked padded
    query rows), the token table, ``inputs_embeds`` and the image encoder (zero padding tokens).
    """
    options = runtime_options(compute_unit)
    out = tiny_bundles["macOS", False]
    text = _tiny_text_encoder()
    ids, _ = macos.reference_inputs(FunctionSpec("len16_b2", 2, 16), 128)
    mask = torch.ones((2, 16), dtype=torch.int32)
    mask[1, 7:] = 0
    embeds = torch.randn((1, 32, 64), generator=torch.Generator().manual_seed(3))
    embeds_mask = (torch.arange(32) < 20).to(torch.int32)[None]
    with torch.no_grad():
        cases = [
            (
                "len16_b2",
                {"input_ids": ids, "attention_mask": mask},
                macos.TextEmbeddingExportWrapper(text)(ids, mask),
            ),
            (
                "main",
                {"input_ids": ids[1:, :9]},
                common.TokenTableExportWrapper(text.embed_tokens)(ids[1:, :9]),
            ),
        ]
        with macos.without_token_table(text) as encoder:
            want = macos.InputsEmbedsEncoderWrapper(encoder)(embeds, embeds_mask)
        cases.append(
            ("embeds_len32_b1", {"inputs_embeds": embeds, "attention_mask": embeds_mask}, want)
        )
    for function, inputs, want in cases:
        (got,) = run_function(out / "text_encoder.aimodel", function, inputs, options).values()
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    image = _image_inputs(torch.float32, torch.int32)
    with torch.no_grad():
        want = macos.ImageEncoderExportWrapper(_tiny_image_encoder())(*image.values())
    got = run_function(out / "image_encoder.aimodel", "main", image, options)["soft_tokens"]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    assert torch.all(got[0, 6:] == 0)


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_tiny_ios_bundle_matches_pytorch(tiny_bundles, compute_unit):
    """Each iOS function, run by the Core AI runtime, matches its float16 PyTorch module: the token
    table exactly, the encoders within atol = rtol = 1e-2 (zero padding tokens).
    """
    options = runtime_options(compute_unit)
    out = tiny_bundles["iOS", False]
    encoder, table = _tiny_ios_text_encoder_and_table(torch.float16)
    for spec in encoder_functions(TINY, "iOS", with_image=True):
        n = spec.length
        embeds, _, _ = ios.encoder_reference_inputs(n, 64)
        bias, pool = host_bias_and_pool(
            (torch.arange(n) < n - 1).to(torch.int32)[None], torch.float16
        )
        inputs = {"inputs_embeds": embeds, "bias": bias, "pool_weights": pool}
        with torch.no_grad():
            want = ios.EncoderBandWrapper(encoder, n, encoder.config.sliding_window)(
                *inputs.values()
            )
        got = run_function(out / "encoder.aimodel", spec.name, inputs, options)["embedding"]
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)

    token_table = ios.scaled_token_table(table, encoder.config.hidden_size**0.5, torch.float16)
    ids = torch.randint(3, 128, (1, 9), dtype=torch.int32)
    want = common.TokenTableExportWrapper(token_table, torch.float16)(ids)
    got = run_function(out / "token_table.aimodel", "main", {"input_ids": ids}, options)
    torch.testing.assert_close(got["inputs_embeds"], want, atol=0, rtol=0)

    image = _image_inputs(torch.float16, torch.float16)
    with torch.no_grad():
        want = _tiny_ios_image_encoder(torch.float16)(*image.values())
    got = run_function(out / "image_encoder.aimodel", "main", image, options)["soft_tokens"]
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    assert torch.all(got[0, 6:] == 0)


# --- CLI ---


@pytest.fixture
def no_export(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("export must not start")

    monkeypatch.setattr(embedding_export, "export_embedding_bundle", fail)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--compute-precision", "float16"], "not float16"),
        (["--platform", "iOS", "--compute-precision", "float32"], "computes in float16"),
        (["--fused"], "no fused export"),
    ],
)
def test_cli_refuses_before_exporting(argv, message, no_export, capsys):
    with pytest.raises(SystemExit) as info:
        embedding_export.main(["embeddinggemma-2", *argv])
    assert info.value.code == 2 and message in capsys.readouterr().err


def test_cli_dry_run_and_list_models(no_export, capsys):
    assert embedding_export.main(["embeddinggemma-2", "--platform", "iOS", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "iOS" in out and "float16" in out and "text,image" in out
    assert "embeddinggemma-2-ios" in out
    assert embedding_export.main(["--list-models"]) == 0
    rows = [r for r in capsys.readouterr().out.splitlines() if r.startswith("embeddinggemma-2 ")]
    assert len(rows) == 1 and "text,image" in rows[0]
    assert rows[0].endswith("macOS:float32,bfloat16 iOS:float16")
