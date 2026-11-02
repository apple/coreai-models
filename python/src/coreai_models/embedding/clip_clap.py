# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLIP (text and image) and CLAP (text and audio) embedding exporters.

Each tower is one function that maps a batch of 1 to 64 inputs to L2-normalized embeddings:

- text: ``input_ids``, ``attention_mask`` -> ``embedding``
- CLIP image: ``pixel_values`` -> ``embedding``
- CLAP audio: log-mel ``input_features`` -> ``embedding``

``logit_scale`` goes in the metadata. ``--fused`` puts both towers and ``main`` in one asset,
``model.aimodel``. ``main`` is the Hugging Face ``forward``: logits and embeddings for a batch of
texts and a batch of images or clips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.embedding.contract import (
    AUDIO_COMPONENT,
    IMAGE_COMPONENT,
    OUTPUT_NAME,
    SIMILARITY_COMPONENT,
    TEXT_COMPONENT,
    TEXT_INPUT_NAMES,
    AudioPreprocessing,
    EmbeddingFamily,
    ImagePreprocessing,
    function_entry,
    media_block,
    text_block,
)
from coreai_models.embedding.export_utils import (
    TORCH_DTYPES,
    EmbeddingExporter,
    export_program,
    save_program,
)

BATCH = (1, 64)
FUSED_ASSET = "model"
FUSED_FUNCTION = "main"


def _features(output: Any) -> torch.Tensor:
    return F.normalize(output.pooler_output.float(), dim=-1)


class _Wrapper(nn.Module):
    model: Any

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def dtype(self) -> torch.dtype:
        dtype: torch.dtype = next(self.model.parameters()).dtype
        return dtype


class TextTower(_Wrapper):
    """``(input_ids, attention_mask) -> embedding``: projected, L2-normalized, float32."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return _features(
            self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )


class ImageTower(_Wrapper):
    """``pixel_values -> embedding``: projected, L2-normalized, float32."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _features(self.model.get_image_features(pixel_values=pixel_values.to(self.dtype)))


class AudioTower(_Wrapper):
    """``input_features -> embedding``: projected, L2-normalized, float32."""

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        features = input_features.to(self.dtype)
        return _features(self.model.get_audio_features(input_features=features))


class ClipSimilarity(_Wrapper):
    """``CLIPModel.forward``: logits per image and text, then image and text embeddings."""

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        out = self.model(
            pixel_values=pixel_values.to(self.dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        outputs = (out.logits_per_image, out.logits_per_text, out.image_embeds, out.text_embeds)
        return tuple(x.float() for x in outputs)


class ClapSimilarity(_Wrapper):
    """``ClapModel.forward``: logits per audio and text, then text and audio embeddings."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        is_longer: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features.to(self.dtype),
            is_longer=is_longer,
        )
        outputs = (out.logits_per_audio, out.logits_per_text, out.text_embeds, out.audio_embeds)
        return tuple(x.float() for x in outputs)


@dataclass(frozen=True)
class DualEncoder:
    """A Hugging Face text + media model exported as two towers.

    ``similarity_inputs`` / ``similarity_outputs`` are the fused ``main`` signature, in the order
    of ``similarity``'s arguments and results.
    """

    family: EmbeddingFamily
    model_class: str
    logit_scale: str
    media_component: str
    media_tower: type[nn.Module]
    media_inputs: dict[str, str]
    similarity: type[nn.Module]
    similarity_inputs: dict[str, str]
    similarity_outputs: dict[str, str]

    @property
    def modality(self) -> str:
        return self.family.modalities[1]


CLIP = DualEncoder(
    family=EmbeddingFamily(
        key="clip-vit-b32",
        hf_model_id="openai/clip-vit-base-patch32",
        dimension=512,
        # The tokenizer wraps text as <|startoftext|> ... <|endoftext|> and pads with <|endoftext|>.
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=49407,
        max_tokens=77,
        precisions={p: ("float32", "float16", "bfloat16") for p in ("macOS", "iOS")},
        image=ImagePreprocessing(
            image_strategy="center_crop",
            image_size=224,
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ),
    model_class="CLIPModel",
    logit_scale="logit_scale",
    media_component=IMAGE_COMPONENT,
    media_tower=ImageTower,
    media_inputs={"pixel_values": "float32[B,3,224,224]"},
    similarity=ClipSimilarity,
    similarity_inputs={
        "pixel_values": "float32[N,3,224,224]",
        "input_ids": "int32[B,L]",
        "attention_mask": "int32[B,L]",
    },
    similarity_outputs={
        "logits_per_image": "float32[N,B]",
        "logits_per_text": "float32[B,N]",
        "image_embeds": "float32[N,512]",
        "text_embeds": "float32[B,512]",
    },
)

CLAP = DualEncoder(
    family=EmbeddingFamily(
        key="clap-htsat",
        hf_model_id="laion/clap-htsat-unfused",
        dimension=512,
        # The tokenizer wraps text as <s> ... </s> and pads with <pad>.
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        # The training length; the tokenizer accepts up to 512.
        max_tokens=77,
        precisions={p: ("float32", "float16") for p in ("macOS", "iOS")},
        # From preprocessor_config.json: 10 s clips.
        audio=AudioPreprocessing(
            sample_rate=48000,
            n_fft=1024,
            win_length=1024,
            hop_length=480,
            n_mel_bins=64,
            n_frames=1001,
            f_min=50,
            f_max=14000,
        ),
    ),
    model_class="ClapModel",
    # The scale of HF logits_per_audio. HF logits_per_text uses logit_scale_t, still at its
    # initial 1/0.07.
    logit_scale="logit_scale_a",
    media_component=AUDIO_COMPONENT,
    media_tower=AudioTower,
    media_inputs={"input_features": "float32[B,1,1001,64]"},
    similarity=ClapSimilarity,
    similarity_inputs={
        "input_ids": "int32[B,L]",
        "attention_mask": "int32[B,L]",
        "input_features": "float32[N,1,1001,64]",
        "is_longer": "bool[N,1]",
    },
    similarity_outputs={
        "logits_per_audio": "float32[N,B]",
        "logits_per_text": "float32[B,N]",
        "text_embeds": "float32[B,512]",
        "audio_embeds": "float32[N,512]",
    },
)


def _component(
    asset: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    function: str,
    length: int | None = None,
) -> dict[str, Any]:
    return {
        "asset": asset,
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": ["gpu"],
        "functions": [function_entry(function, BATCH, length)],
    }


def dual_encoder_block(
    spec: DualEncoder,
    *,
    revision: str,
    compute_precision: str,
    logit_scale: float,
    text_only: bool,
    fused: bool,
) -> dict[str, Any]:
    """Build the ``embedding`` block. Component ``c`` is function ``c`` in asset ``c``, or in
    asset ``model`` when ``fused``.
    """
    family = spec.family
    embedding = {OUTPUT_NAME: f"float32[B,{family.dimension}]"}
    text_inputs = {name: "int32[B,L]" for name in TEXT_INPUT_NAMES}

    def asset(component: str) -> str:
        return FUSED_ASSET if fused else component

    components = {
        TEXT_COMPONENT: _component(
            asset(TEXT_COMPONENT), text_inputs, embedding, TEXT_COMPONENT, family.max_tokens
        )
    }
    towers = [{"modality": "text", "pipeline": [TEXT_COMPONENT]}]
    if not text_only:
        media = spec.media_component
        components[media] = _component(asset(media), spec.media_inputs, embedding, media)
        towers.append({"modality": spec.modality, "pipeline": [media]})
    if fused:
        components[SIMILARITY_COMPONENT] = _component(
            FUSED_ASSET,
            spec.similarity_inputs,
            spec.similarity_outputs,
            FUSED_FUNCTION,
            family.max_tokens,
        )
    block: dict[str, Any] = {
        "space": {"id": family.hf_model_id, "revision": revision},
        "dimension": family.dimension,
        "normalized": True,
        "metric": "cosine",
        "logit_scale": logit_scale,
        "compute_precision": compute_precision,
        "components": components,
        "towers": towers,
        "text": text_block(family),
    }
    if not text_only:
        block[spec.modality] = media_block(family, spec.modality)
    return block


def load_model(spec: DualEncoder, snapshot: Path) -> nn.Module:
    """The float32 Hugging Face model of ``spec`` from ``snapshot``, in eval mode."""
    import transformers

    model: nn.Module = getattr(transformers, spec.model_class).from_pretrained(snapshot).eval()
    return model


def export_programs(
    spec: DualEncoder,
    snapshot: Path,
    out: Path,
    *,
    platform: str,
    text_only: bool,
    fused: bool,
    compute_precision: str,
    include_debug_info: bool,
    revision: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Write one asset per component (one ``model`` asset when ``fused``) to ``out``.

    Both platforms get the same GPU programs. Returns ``(assets, embedding block)``.
    """
    model = load_model(spec, snapshot)
    logit_scale = getattr(model, spec.logit_scale).detach().exp().item()
    model.to(TORCH_DTYPES[compute_precision])
    block = dual_encoder_block(
        spec,
        revision=revision,
        compute_precision=compute_precision,
        logit_scale=logit_scale,
        text_only=text_only,
        fused=fused,
    )
    wrappers: dict[str, type[nn.Module]] = {
        TEXT_COMPONENT: TextTower,
        spec.media_component: spec.media_tower,
        SIMILARITY_COMPONENT: spec.similarity,
    }
    by_asset: dict[str, list[tuple[nn.Module, dict[str, Any]]]] = {}
    for name, component in block["components"].items():
        by_asset.setdefault(component["asset"], []).append((wrappers[name](model), component))
    for asset, functions in by_asset.items():
        save_program(export_program(functions, include_debug_info), out, spec.family, asset)
    return {asset: f"{asset}.aimodel" for asset in by_asset}, block


def _mismatches(source: str, got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    return [f"{source} {k} {got.get(k)!r} != {v!r}" for k, v in want.items() if got.get(k) != v]


def check_clip_snapshot(spec: DualEncoder, snapshot: Path) -> None:
    """Raise ``ValueError`` if the snapshot's image preprocessing or projection disagree."""
    image = spec.family.image
    assert image is not None
    proc = json.loads((snapshot / "preprocessor_config.json").read_text())
    size = proc.get("size")
    crop = proc.get("crop_size")
    got = {
        "size": size.get("shortest_edge") if isinstance(size, dict) else size,
        "crop_size": crop.get("height") if isinstance(crop, dict) else crop,
        "do_center_crop": proc.get("do_center_crop"),
        "resample": proc.get("resample"),
        "image_mean": proc.get("image_mean"),
        "image_std": proc.get("image_std"),
    }
    want = {
        "size": image.image_size,
        "crop_size": image.image_size,
        "do_center_crop": True,
        "resample": 3,  # PIL BICUBIC
        "image_mean": list(image.image_mean),
        "image_std": list(image.image_std),
    }
    config = json.loads((snapshot / "config.json").read_text())
    problems = _mismatches("preprocessor_config.json", got, want) + _mismatches(
        "config.json", config, {"projection_dim": spec.family.dimension}
    )
    if problems:
        raise ValueError(f"Snapshot disagrees with the CLIP family: {problems}")


def check_clap_snapshot(spec: DualEncoder, snapshot: Path) -> None:
    """Raise ``ValueError`` if the snapshot's audio front end, fusion or projection disagree."""
    audio = spec.family.audio
    assert audio is not None
    proc = json.loads((snapshot / "preprocessor_config.json").read_text())
    want = {
        "sampling_rate": audio.sample_rate,
        "n_fft": audio.n_fft,
        "fft_window_size": audio.win_length,
        "hop_length": audio.hop_length,
        "feature_size": audio.n_mel_bins,
        "frequency_min": audio.f_min,
        "frequency_max": audio.f_max,
        "nb_max_samples": (audio.n_frames - 1) * audio.hop_length,
        "padding": "repeatpad",
    }
    config = json.loads((snapshot / "config.json").read_text())
    problems = _mismatches("preprocessor_config.json", proc, want)
    problems += _mismatches("config.json", config, {"projection_dim": spec.family.dimension})
    problems += _mismatches(
        "config.json audio_config", config.get("audio_config", {}), {"enable_fusion": False}
    )
    if problems:
        raise ValueError(f"Snapshot disagrees with the CLAP family: {problems}")


def exporter(spec: DualEncoder) -> EmbeddingExporter:
    """The ``EmbeddingExporter`` for ``spec``."""
    check = check_clip_snapshot if spec.modality == "image" else check_clap_snapshot
    return EmbeddingExporter(
        family=spec.family,
        check_snapshot=partial(check, spec),
        export_programs=partial(export_programs, spec),
        fused=True,
    )


CLIP_EXPORTER = exporter(CLIP)
CLAP_EXPORTER = exporter(CLAP)
