# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the CLIP and CLAP embedding exporters (``coreai_models.embedding.clip_clap``).

On tiny random configs:

- the towers and the fused ``main`` match Hugging Face, in PyTorch and on the Core AI runtime
  (CPU and GPU)
- the bundle layout and the fused asset
- each allowed precision
- the reference host, reading only the bundle

The real-weight tests (``slow``) run when the snapshots are in the Hugging Face cache.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import transformers

from coreai_models.embedding import clip_clap
from coreai_models.embedding.contract import (
    AudioPreprocessing,
    CheckpointFacts,
    ImagePreprocessing,
    validate_block,
)
from coreai_models.embedding.export import EXPORTERS, export_embedding_bundle
from tests.test_model_units.test_export._embedding_host import (
    Bundle,
    asset_signatures,
    runtime_options,
    runtime_type,
    save_tiny_tokenizer,
    snapshot_dir,
)

CLIP_PREPROCESSOR = {
    "crop_size": 224,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "resample": 3,
    "size": 224,
}
CLAP_PREPROCESSOR = {
    "chunk_length_s": 10,
    "feature_extractor_type": "ClapFeatureExtractor",
    "feature_size": 64,
    "fft_window_size": 1024,
    "frequency_max": 14000,
    "frequency_min": 50,
    "hop_length": 480,
    "max_length_s": 10,
    "n_fft": 1024,
    "nb_frequency_bins": 513,
    "nb_max_frames": 1000,
    "nb_max_samples": 480000,
    "padding": "repeatpad",
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": False,
    "sampling_rate": 48000,
    "top_db": None,
    "truncation": "rand_trunc",
}
TEXTS = ["a photo of a cat", "the sound of a dog", "a", " ".join(["photo of the cat"] * 30)]
DIM = 24
# bos, eos, pad of the tiny tokenizers
TINY_TOKEN_IDS = {"clip": (62, 63, 63), "clap": (0, 2, 1)}


def _tiny_config(model: str):
    if model == "clip":
        return transformers.CLIPConfig(
            # eos_token_id 2 selects the argmax pooling the checkpoint uses.
            text_config=dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=37,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=77,
                eos_token_id=2,
            ),
            vision_config=dict(
                image_size=224,
                patch_size=32,
                hidden_size=32,
                intermediate_size=37,
                num_hidden_layers=2,
                num_attention_heads=4,
            ),
            projection_dim=DIM,
        )
    return transformers.ClapConfig(
        text_config=dict(
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            max_position_embeddings=80,
        ),
        audio_config=dict(
            depths=[1, 1],
            num_attention_heads=[2, 4],
            patch_embeds_hidden_size=16,
            hidden_size=32,
            window_size=8,
            enable_fusion=False,
        ),
        projection_dim=DIM,
    )


@dataclasses.dataclass
class Tiny:
    spec: clip_clap.DualEncoder
    snapshot: Path
    model: Any
    bundles: dict[str, Path]

    @property
    def exporter(self):
        return clip_clap.exporter(self.spec)

    @property
    def facts(self) -> CheckpointFacts:
        return clip_clap.read_facts(self.spec, self.snapshot)


def _save_tiny_snapshot(model: str, root: Path) -> Path:
    snapshot = snapshot_dir(root)
    torch.manual_seed(0)
    hf = (transformers.CLIPModel if model == "clip" else transformers.ClapModel)(
        _tiny_config(model)
    )
    hf.save_pretrained(snapshot)
    words = ["<unk>", "a", "photo", "of", "the", "cat", "sound", "dog"]
    save_tiny_tokenizer(snapshot, words, *TINY_TOKEN_IDS[model], model_max_length=77)
    proc = CLIP_PREPROCESSOR if model == "clip" else CLAP_PREPROCESSOR
    (snapshot / "preprocessor_config.json").write_text(json.dumps(proc))
    return snapshot


@pytest.fixture(scope="module", params=["clip", "clap"])
def tiny(request, tmp_path_factory) -> Tiny:
    """A tiny snapshot of the model and its float32 bundles: ``plain``, ``fused``."""
    root = tmp_path_factory.mktemp(request.param)
    spec = clip_clap.CLIP if request.param == "clip" else clip_clap.CLAP
    snapshot = _save_tiny_snapshot(request.param, root)
    model = clip_clap.load_model(spec, snapshot)
    exporter = clip_clap.exporter(spec)
    bundles = {}
    for name, fused in (("plain", False), ("fused", True)):
        bundles[name] = root / name
        export_embedding_bundle(exporter, bundles[name], fused=fused, model_dir=str(snapshot))
    return Tiny(spec, snapshot, model, bundles)


def _text_inputs(tiny: Tiny, n: int) -> dict[str, torch.Tensor]:
    """``n`` rows of ids at length 77 with ragged masks; pads with ``pad_token_id``."""
    facts = tiny.facts
    generator = torch.Generator().manual_seed(n)
    ids = torch.full((n, facts.max_tokens), facts.pad_token_id, dtype=torch.int32)
    mask = torch.zeros_like(ids)
    words = torch.randint(3, 10, (n, facts.max_tokens), generator=generator, dtype=torch.int32)
    for row in range(n):
        size = 3 + (row * 7) % (facts.max_tokens - 2)
        ids[row, :size] = words[row, :size]
        ids[row, 0], ids[row, size - 1] = facts.bos_token_id, facts.eos_token_id
        mask[row, :size] = 1
    return {"input_ids": ids, "attention_mask": mask}


def _media_inputs(tiny: Tiny, n: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + n)
    block = json.loads((tiny.bundles["plain"] / "metadata.json").read_text())["embedding"]
    inputs = {}
    for name, spec in block["components"][tiny.spec.media_component]["inputs"].items():
        shape = [int(d) for d in spec[:-1].split("[")[1].split(",")[1:]]
        inputs[name] = torch.randn([n, *shape], generator=generator)
    return inputs


def _similarity_inputs(tiny: Tiny, b: int, n: int) -> dict[str, torch.Tensor]:
    inputs = {**_text_inputs(tiny, b), **_media_inputs(tiny, n)}
    if "is_longer" in tiny.spec.similarity_inputs:
        inputs["is_longer"] = torch.zeros((n, 1), dtype=torch.bool)
    return {name: inputs[name] for name in tiny.spec.similarity_inputs}


def _hf_features(tiny: Tiny, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    model = tiny.model
    with torch.no_grad():
        if "input_ids" in inputs:
            out = model.get_text_features(**inputs)
        elif "pixel_values" in inputs:
            out = model.get_image_features(**inputs)
        else:
            out = model.get_audio_features(**inputs)
    return F.normalize(out.pooler_output, dim=-1)


def _hf_similarity(tiny: Tiny, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    with torch.no_grad():
        out = tiny.model(**inputs)
    return [getattr(out, name) for name in tiny.spec.similarity_outputs]


def test_towers_match_hugging_face(tiny):
    """The wrappers equal HF's normalized features and ``forward`` (pad == eos for CLIP)."""
    text = _text_inputs(tiny, 3)
    with torch.no_grad():
        got = clip_clap.TextTower(tiny.model)(*text.values())
        torch.testing.assert_close(got, _hf_features(tiny, text), atol=1e-6, rtol=0)
        media_inputs = _media_inputs(tiny, 2)
        got = tiny.spec.media_tower(tiny.model)(*media_inputs.values())
        torch.testing.assert_close(got, _hf_features(tiny, media_inputs), atol=1e-6, rtol=0)
        inputs = _similarity_inputs(tiny, 3, 2)
        got = tiny.spec.similarity(tiny.model)(*inputs.values())
    for g, w in zip(got, _hf_similarity(tiny, inputs), strict=True):
        torch.testing.assert_close(g, w, atol=1e-6, rtol=0)


def test_bundle_layout_matches_metadata(tiny):
    """Files are the assets, ``tokenizer/`` and ``metadata.json``; each component's function is
    in its asset with the metadata's input and output names, dtypes and shapes.
    """
    media = tiny.spec.media_component
    expected = {"plain": ["text_encoder", media], "fused": ["model"]}
    for name, path in tiny.bundles.items():
        meta = json.loads((path / "metadata.json").read_text())
        block = meta["embedding"]
        validate_block(block, meta["assets"])
        assert sorted(meta["assets"]) == sorted(expected[name])
        files = [*meta["assets"].values(), "metadata.json", "tokenizer"]
        assert sorted(p.name for p in path.iterdir()) == sorted(files)
        assert sorted(p.name for p in (path / "tokenizer").iterdir()) == [
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        components = ["text_encoder", media] + ["similarity"] * (name == "fused")
        assert list(block["components"]) == components
        for spec in block["components"].values():
            signatures = asset_signatures(path / meta["assets"][spec["asset"]])
            (entry,) = spec["functions"]
            assert entry["batch"] == {"min": 1, "max": 64}
            length = entry.get("length")
            inputs, outputs = signatures[entry["function"]]
            assert inputs == [(k, runtime_type(v, length)) for k, v in spec["inputs"].items()]
            assert outputs == [(k, runtime_type(v, None)) for k, v in spec["outputs"].items()]
        if name == "fused":
            assert sorted(signatures) == sorted(["text_encoder", media, "main"])
        assert block["towers"] == [
            {"modality": "text", "pipeline": ["text_encoder"]},
            {"modality": tiny.spec.modality, "pipeline": [media]},
        ]
    scale = tiny.model.logit_scale if tiny.spec.modality == "image" else tiny.model.logit_scale_a
    assert block["logit_scale"] == pytest.approx(float(scale.exp()))


@pytest.mark.parametrize("compute_unit", ["cpu", "gpu"])
def test_runtime_matches_pytorch(tiny, compute_unit):
    """Every function at batch 1, 3 and 64 matches PyTorch within 1e-4; the fused asset's tower
    functions equal the plain bundle's within 1e-6.
    """
    plain = Bundle(tiny.bundles["plain"], runtime_options(compute_unit))
    fused = Bundle(tiny.bundles["fused"], plain.options)
    media = tiny.spec.media_component
    for n in (1, 3, 64):
        for component, inputs in (
            ("text_encoder", _text_inputs(tiny, n)),
            (media, _media_inputs(tiny, n)),
        ):
            got = plain.run(component, inputs)["embedding"]
            assert got.shape == (n, DIM) and got.dtype == torch.float32
            torch.testing.assert_close(got, _hf_features(tiny, inputs), atol=1e-4, rtol=1e-4)
            same = fused.run(component, inputs)["embedding"]
            torch.testing.assert_close(same, got, atol=1e-6, rtol=0)
    for b, n in ((1, 1), (3, 2), (64, 5)):
        inputs = _similarity_inputs(tiny, b, n)
        outputs = fused.run("similarity", inputs)
        want = dict(zip(tiny.spec.similarity_outputs, _hf_similarity(tiny, inputs), strict=True))
        assert sorted(outputs) == sorted(want)
        for name, value in want.items():
            assert outputs[name].shape == value.shape
            torch.testing.assert_close(outputs[name], value, atol=1e-4, rtol=1e-4)


def test_reference_host_text_matches_hugging_face(tiny):
    """From ``metadata.json`` and ``tokenizer/`` alone: ids equal HF's tokenizer with truncation
    to 77 (eos kept), and the vectors equal HF's features.
    """
    bundle = Bundle(tiny.bundles["plain"], runtime_options("gpu"))
    ids, mask = bundle.tokenize(TEXTS)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny.snapshot)
    want = tokenizer(TEXTS, padding="max_length", max_length=77, truncation=True)
    assert ids.tolist() == want["input_ids"] and mask.tolist() == want["attention_mask"]
    assert ids[3, -1] == tiny.facts.eos_token_id and mask[3].all()
    got = bundle.embed_text(TEXTS)
    hf = _hf_features(tiny, {"input_ids": ids, "attention_mask": mask})
    torch.testing.assert_close(got, hf, atol=1e-4, rtol=1e-4)


def _images():
    from PIL import Image

    generator = np.random.default_rng(0)
    sizes = [(480, 320), (224, 224), (300, 600)]
    return [
        Image.fromarray(generator.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in sizes
    ]


def _waveforms() -> list[np.ndarray]:
    generator = np.random.default_rng(0)
    return [generator.standard_normal(n).astype(np.float32) * 0.1 for n in (7 * 48000, 12 * 48000)]


def _check_media_host(bundle: Bundle, tiny_model, snapshot: Path, modality: str, atol: float):
    """The host's preprocessing equals HF's processor; its vectors equal HF's features."""
    if modality == "image":
        processor = transformers.CLIPImageProcessorPil.from_pretrained(snapshot)
        images = _images()
        pixels = bundle.preprocess_images(images)
        want = processor(images=images, return_tensors="pt")["pixel_values"]
        torch.testing.assert_close(pixels, want, atol=1e-5, rtol=0)
        got = bundle.embed_images(images)
        with torch.no_grad():
            hf = tiny_model.get_image_features(pixel_values=want).pooler_output
    else:
        extractor = transformers.ClapFeatureExtractor.from_pretrained(snapshot)
        waves = _waveforms()
        features = bundle.audio_features(waves)
        heads = [w[:480000] for w in waves]
        want = extractor(heads, sampling_rate=48000, return_tensors="pt")["input_features"]
        torch.testing.assert_close(features, want, atol=2e-3, rtol=0)
        got = bundle.embed_audio(waves)
        with torch.no_grad():
            hf = tiny_model.get_audio_features(input_features=want).pooler_output
    hf = F.normalize(hf, dim=-1)
    assert F.cosine_similarity(got, hf).min() >= 1 - atol
    return got


def test_reference_host_media_matches_hugging_face(tiny):
    bundle = Bundle(tiny.bundles["plain"], runtime_options("gpu"))
    _check_media_host(bundle, tiny.model, tiny.snapshot, tiny.spec.modality, 1e-6)


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_reduced_precisions(tiny, precision, tmp_path):
    """Allowed precisions give finite vectors close to float32 HF on the GPU; CLAP refuses
    bfloat16.
    """
    if precision not in tiny.spec.family.precisions["macOS"]:
        with pytest.raises(ValueError, match=f"not {precision}"):
            export_embedding_bundle(tiny.exporter, tmp_path / "b", compute_precision=precision)
        return
    out = tmp_path / precision
    meta = export_embedding_bundle(
        tiny.exporter, out, compute_precision=precision, model_dir=str(tiny.snapshot)
    )
    assert meta["embedding"]["compute_precision"] == precision
    bundle = Bundle(out, runtime_options("gpu"))
    floor = {"float16": 0.999, "bfloat16": 0.99}[precision]
    for component, inputs in (
        ("text_encoder", _text_inputs(tiny, 3)),
        (tiny.spec.media_component, _media_inputs(tiny, 3)),
    ):
        got = bundle.run(component, inputs)["embedding"]
        assert got.dtype == torch.float32 and torch.isfinite(got).all()
        assert F.cosine_similarity(got, _hf_features(tiny, inputs)).min() >= floor


UNSUPPORTED = {
    "image": [("resample", 2), ("crop_size", 200), ("do_normalize", False)],
    "audio": [("truncation", "fusion"), ("padding", "pad"), ("top_db", 80)],
}


def test_read_facts_refuses_unsupported_preprocessing(tiny, tmp_path):
    """The tiny snapshots preprocess like the real ones; settings outside R7 / R8 are refused."""
    facts = tiny.exporter.read_facts(tiny.snapshot)
    real = REAL_FACTS["clip" if tiny.spec.modality == "image" else "clap"]
    assert (facts.dimension, facts.max_tokens) == (DIM, 77)
    assert (facts.image, facts.audio) == (real.image, real.audio)
    for key, value in UNSUPPORTED[tiny.spec.modality]:
        bad = tmp_path / key
        shutil.copytree(tiny.snapshot, bad, ignore=shutil.ignore_patterns("*.safetensors"))
        proc = json.loads((bad / "preprocessor_config.json").read_text())
        (bad / "preprocessor_config.json").write_text(json.dumps({**proc, key: value}))
        with pytest.raises(ValueError, match=f"cannot express: .*{key}"):
            tiny.exporter.read_facts(bad)


def test_exporters_are_registered():
    assert EXPORTERS["clip-vit-b32"] is clip_clap.CLIP_EXPORTER
    assert EXPORTERS["clap-htsat"] is clip_clap.CLAP_EXPORTER


# --- real weights (local only) ---

REAL = {
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "clap": ("laion/clap-htsat-unfused", "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"),
}
REAL_FACTS = {
    "clip": CheckpointFacts(
        512,
        49406,
        49407,
        49407,
        77,
        image=ImagePreprocessing(
            "center_crop",
            224,
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ),
    "clap": CheckpointFacts(
        512, 0, 2, 1, 77, audio=AudioPreprocessing(48000, 1024, 1024, 480, 64, 1001, 50, 14000)
    ),
}
LABELS = {
    "clip": ["a photo of a cat", "a photo of a dog", "a diagram"],
    "clap": ["the sound of a dog", "the sound of rain", "a piano"],
}


def _real_snapshot(model: str) -> Path:
    hf_id, revision = REAL[model]
    root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = root / f"models--{hf_id.replace('/', '--')}" / "snapshots" / revision
    # A snapshot cached by another loader can hold the weights without the processor config.
    needed = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    if not all((snapshot / name).exists() for name in needed):
        pytest.skip(f"{hf_id} is not in the HF cache")
    return snapshot


@pytest.mark.parametrize("model", ["clip", "clap"])
def test_real_facts(model):
    spec = clip_clap.CLIP if model == "clip" else clip_clap.CLAP
    assert clip_clap.read_facts(spec, _real_snapshot(model)) == REAL_FACTS[model]


@pytest.mark.slow
@pytest.mark.parametrize("model", ["clip", "clap"])
def test_real_weights_match_hugging_face(model, tmp_path):
    """Fused float32 bundle from the cached snapshot: host vectors vs HF features (cos >= 0.9999),
    host logits ``logit_scale * cos`` and the fused ``main`` vs HF's logits; float16 / bfloat16
    bundles stay finite with cos >= 0.999 on both towers.
    """
    snapshot = _real_snapshot(model)
    spec = clip_clap.CLIP if model == "clip" else clip_clap.CLAP
    exporter = clip_clap.exporter(spec)
    out = tmp_path / "fused"
    export_embedding_bundle(exporter, out, fused=True, model_dir=str(snapshot))
    bundle = Bundle(out, runtime_options("gpu"))
    hf: Any = clip_clap.load_model(spec, snapshot)
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)
    # The shape Swift CLIPTokenizer decodes: model.vocab [String: Int32], model.merges [[String]].
    bpe = json.loads((out / "tokenizer" / "tokenizer.json").read_text())["model"]
    assert all(isinstance(i, int) for i in bpe["vocab"].values())
    assert all(isinstance(m, list) and len(m) == 2 for m in bpe["merges"])

    texts = LABELS[model] + [TEXTS[3]]
    text = bundle.embed_text(texts)
    enc = tokenizer(
        texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        want = F.normalize(hf.get_text_features(**enc).pooler_output, dim=-1)
    assert F.cosine_similarity(text, want).min() >= 0.9999
    media = _check_media_host(bundle, hf, snapshot, spec.modality, 1e-4)

    scale = bundle.block["logit_scale"]
    ids, mask = bundle.tokenize(texts)
    inputs = {"input_ids": ids, "attention_mask": mask}
    if model == "clip":
        processor = transformers.CLIPImageProcessorPil.from_pretrained(snapshot)
        inputs["pixel_values"] = processor(images=_images(), return_tensors="pt")["pixel_values"]
        per_media = "logits_per_image"
    else:
        extractor = transformers.ClapFeatureExtractor.from_pretrained(snapshot)
        heads = [w[:480000] for w in _waveforms()]
        features = extractor(heads, sampling_rate=48000, return_tensors="pt")
        inputs.update(input_features=features["input_features"], is_longer=features["is_longer"])
        per_media = "logits_per_audio"
    with torch.no_grad():
        out_hf = hf(**inputs)
    torch.testing.assert_close(
        scale * media @ text.T, getattr(out_hf, per_media), atol=2e-2, rtol=0
    )
    main = bundle.run("similarity", {k: inputs[k] for k in spec.similarity_inputs})
    for name, got in main.items():
        torch.testing.assert_close(got, getattr(out_hf, name), atol=2e-2, rtol=1e-3)

    for precision in spec.family.precisions["macOS"][1:]:
        reduced = tmp_path / precision
        export_embedding_bundle(
            exporter, reduced, compute_precision=precision, model_dir=str(snapshot)
        )
        low = Bundle(reduced, bundle.options)
        got = low.embed_text(texts)
        assert torch.isfinite(got).all() and F.cosine_similarity(got, want).min() >= 0.999
        got = low.embed_images(_images()) if model == "clip" else low.embed_audio(_waveforms())
        assert torch.isfinite(got).all() and F.cosine_similarity(got, media).min() >= 0.999
