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
    CheckpointFacts,
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
BICUBIC = 3  # PIL.Image.Resampling.BICUBIC


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
    of ``similarity``'s arguments and results. Shapes in these and ``media_inputs`` take ``{dim}``
    (the embedding dimension), ``{size}`` (the image size), ``{frames}`` and ``{mels}`` (the audio
    frames and mel bins) from the checkpoint facts.
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
        precisions={p: ("float32", "float16", "bfloat16") for p in ("macOS", "iOS")},
        modalities=("text", "image"),
    ),
    model_class="CLIPModel",
    logit_scale="logit_scale",
    media_component=IMAGE_COMPONENT,
    media_tower=ImageTower,
    media_inputs={"pixel_values": "float32[B,3,{size},{size}]"},
    similarity=ClipSimilarity,
    similarity_inputs={
        "pixel_values": "float32[N,3,{size},{size}]",
        "input_ids": "int32[B,L]",
        "attention_mask": "int32[B,L]",
    },
    similarity_outputs={
        "logits_per_image": "float32[N,B]",
        "logits_per_text": "float32[B,N]",
        "image_embeds": "float32[N,{dim}]",
        "text_embeds": "float32[B,{dim}]",
    },
)

CLAP = DualEncoder(
    family=EmbeddingFamily(
        key="clap-htsat",
        hf_model_id="laion/clap-htsat-unfused",
        precisions={p: ("float32", "float16") for p in ("macOS", "iOS")},
        modalities=("text", "audio"),
        # The training length; the tokenizer accepts up to 512.
        max_tokens=77,
    ),
    model_class="ClapModel",
    # The scale of HF logits_per_audio. HF logits_per_text uses logit_scale_t, still at its
    # initial 1/0.07.
    logit_scale="logit_scale_a",
    media_component=AUDIO_COMPONENT,
    media_tower=AudioTower,
    media_inputs={"input_features": "float32[B,1,{frames},{mels}]"},
    similarity=ClapSimilarity,
    similarity_inputs={
        "input_ids": "int32[B,L]",
        "attention_mask": "int32[B,L]",
        "input_features": "float32[N,1,{frames},{mels}]",
        "is_longer": "bool[N,1]",
    },
    similarity_outputs={
        "logits_per_audio": "float32[N,B]",
        "logits_per_text": "float32[B,N]",
        "text_embeds": "float32[B,{dim}]",
        "audio_embeds": "float32[N,{dim}]",
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


def _shapes(specs: dict[str, str], facts: CheckpointFacts) -> dict[str, str]:
    sizes = {"dim": facts.dimension}
    if facts.image is not None:
        sizes["size"] = facts.image.image_size
    if facts.audio is not None:
        sizes.update(frames=facts.audio.n_frames, mels=facts.audio.n_mel_bins)
    return {name: spec.format(**sizes) for name, spec in specs.items()}


def dual_encoder_block(
    spec: DualEncoder,
    facts: CheckpointFacts,
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
    embedding = {OUTPUT_NAME: f"float32[B,{facts.dimension}]"}
    text_inputs = {name: "int32[B,L]" for name in TEXT_INPUT_NAMES}

    def asset(component: str) -> str:
        return FUSED_ASSET if fused else component

    components = {
        TEXT_COMPONENT: _component(
            asset(TEXT_COMPONENT), text_inputs, embedding, TEXT_COMPONENT, facts.max_tokens
        )
    }
    towers = [{"modality": "text", "pipeline": [TEXT_COMPONENT]}]
    if not text_only:
        media = spec.media_component
        media_inputs = _shapes(spec.media_inputs, facts)
        components[media] = _component(asset(media), media_inputs, embedding, media)
        towers.append({"modality": spec.modality, "pipeline": [media]})
    if fused:
        components[SIMILARITY_COMPONENT] = _component(
            FUSED_ASSET,
            _shapes(spec.similarity_inputs, facts),
            _shapes(spec.similarity_outputs, facts),
            FUSED_FUNCTION,
            facts.max_tokens,
        )
    block: dict[str, Any] = {
        "space": {"id": spec.family.hf_model_id, "revision": revision},
        "dimension": facts.dimension,
        "normalized": True,
        "metric": "cosine",
        "logit_scale": logit_scale,
        "compute_precision": compute_precision,
        "components": components,
        "towers": towers,
        "text": text_block(facts),
    }
    if not text_only:
        block[spec.modality] = media_block(facts, spec.modality)
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
    facts: CheckpointFacts,
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
        facts,
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


def _refuse(source: str, got: dict[str, Any], want: dict[str, Any]) -> None:
    problems = [f"{k} {got[k]!r} (supported: {v!r})" for k, v in want.items() if got[k] != v]
    if problems:
        raise ValueError(f"{source} has settings the embedding contract cannot express: {problems}")


def _image_preprocessing(processor: Any) -> ImagePreprocessing:
    size = processor.size.get("shortest_edge")
    crop = processor.crop_size
    got = {
        "do_convert_rgb": processor.do_convert_rgb,
        "do_resize": processor.do_resize,
        "resample": processor.resample,
        "do_center_crop": processor.do_center_crop,
        "crop_size": (crop.get("height"), crop.get("width")),
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
    }
    want = {
        "do_convert_rgb": True,
        "do_resize": True,
        "resample": BICUBIC,
        "do_center_crop": True,
        "crop_size": (size, size),
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
    }
    _refuse("the image processor", got, want)
    return ImagePreprocessing(
        image_strategy="center_crop",
        image_size=size,
        image_mean=tuple(processor.image_mean),
        image_std=tuple(processor.image_std),
    )


def _audio_preprocessing(extractor: Any, enable_fusion: bool) -> AudioPreprocessing:
    hop, samples = extractor.hop_length, extractor.nb_max_samples
    got = {
        "padding": extractor.padding,
        "truncation": extractor.truncation,
        "top_db": extractor.top_db,
        "nb_max_samples % hop_length": samples % hop,
        "audio_config.enable_fusion": enable_fusion,
    }
    want = {
        "padding": "repeatpad",
        "truncation": "rand_trunc",
        "top_db": None,
        "nb_max_samples % hop_length": 0,
        "audio_config.enable_fusion": False,
    }
    _refuse("the feature extractor", got, want)
    return AudioPreprocessing(
        sample_rate=extractor.sampling_rate,
        n_fft=extractor.fft_window_size,
        win_length=extractor.fft_window_size,
        hop_length=hop,
        n_mel_bins=extractor.feature_size,
        n_frames=samples // hop + 1,
        f_min=extractor.frequency_min,
        f_max=extractor.frequency_max,
    )


def read_facts(spec: DualEncoder, snapshot: Path) -> CheckpointFacts:
    """The snapshot's facts, from ``transformers.AutoProcessor`` and ``AutoConfig``.

    ``max_tokens`` is the family's, else the tokenizer's ``model_max_length``.

    Raises:
        ValueError: if the image processor does anything but convert to RGB, resize the shortest
            edge bicubic, crop the center square, scale by 1/255 and normalize (R7), or the audio
            front end is not the R8 one: ``repeatpad`` padding, ``rand_trunc`` truncation, no
            ``top_db``, a clip of whole hops and an unfused model.
    """
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(snapshot)
    config = AutoConfig.from_pretrained(snapshot)
    tokenizer = processor.tokenizer
    image = audio = None
    if spec.modality == "image":
        image = _image_preprocessing(processor.image_processor)
    else:
        audio = _audio_preprocessing(processor.feature_extractor, config.audio_config.enable_fusion)
    return CheckpointFacts(
        dimension=config.projection_dim,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_tokens=spec.family.max_tokens or tokenizer.model_max_length,
        image=image,
        audio=audio,
    )


def exporter(spec: DualEncoder) -> EmbeddingExporter:
    """The ``EmbeddingExporter`` for ``spec``."""
    return EmbeddingExporter(
        family=spec.family,
        read_facts=partial(read_facts, spec),
        export_programs=partial(export_programs, spec),
        fused=True,
    )


CLIP_EXPORTER = exporter(CLIP)
CLAP_EXPORTER = exporter(CLAP)
