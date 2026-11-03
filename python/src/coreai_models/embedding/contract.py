# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The ``kind: "embedding"`` bundle: what ``metadata.json`` holds and how a host uses it.

A bundle directory holds ``metadata.json``, the ``.aimodel`` files listed in ``assets`` and the
tokenizer in ``tokenizer/``. A host embeds an input by running the tower for its modality.
Vectors from the towers of one bundle are comparable.

R1. Top-level keys: ``metadata_version``, ``kind`` (``"embedding"``), ``name``, ``assets``
    (asset name -> file), ``embedding`` (below), ``source``, ``compression``, ``compilation``.

R2. ``embedding`` keys:
    - ``space``: ``{id, revision}``. Vectors are comparable only within one space.
    - ``dimension``, ``normalized``, ``metric``, ``logit_scale``, ``compute_precision``
    - ``components``, ``towers``, ``text``
    - ``image`` / ``audio``: present exactly when a tower has that modality.

R3. ``components.<name>``: a program the host can call.
    - ``asset``: a key of ``assets``.
    - ``inputs``, ``outputs``: name -> ``"dtype[shape]"``. Always float32, int32 or bool,
      whatever the ``compute_precision``.
    - ``compute_units``
    - ``functions``: an array of ``{function, batch[, length]}``, where ``batch`` and ``length``
      are an int or ``{min, max}``. Function names are unique within an asset.

    Shape axes: ``B`` is the batch, ``L`` the text length, ``N`` a second batch axis with the
    same range as ``B`` but independent of it. Other dimensions are fixed sizes.

R4. ``towers``: ``[{modality, pipeline}]``, with ``modality`` one of ``text``, ``image``,
    ``audio``.
    - The host runs the ``pipeline`` components in order. The last one's ``embedding`` output
      is the vector.
    - Inputs: ``input_ids`` and ``attention_mask`` for text, ``pixel_values`` for images,
      ``input_features`` for audio.
    - Batches smaller than ``batch.min`` are padded; batches larger than ``batch.max`` are split.

R5. A reader refuses a bundle with a tower, component key or enum value it does not know.

R6. ``text``: how to tokenize.
    - ``embedded_tokenizer``: true when the tokenizer files are in ``tokenizer/``.
    - ``add_bos_token``, ``add_eos_token``: the tokenizer adds these tokens; the host does not.
    - ``bos_token_id``, ``eos_token_id``, ``pad_token_id`` (may equal ``eos_token_id``)
    - ``padding_side``, ``truncation {max_tokens, keep_eos}``

    ``tokenizer.json`` has padding and truncation off. The host truncates to ``max_tokens``
    with ``eos`` kept last, pads to the function's ``length`` with ``pad_token_id`` and sends the
    tokenizer's ``attention_mask``.

R7. ``image``: preprocessing, with Swift ``VisionConfig`` names: ``image_strategy``,
    ``image_size``, ``image_mean``, ``image_std``.
    - Every strategy: convert to RGB, resize bicubic, scale bytes by 1/255, then
      ``(x - mean) / std``.
    - ``center_crop``: resize the shortest edge to ``image_size``, then crop the center square.

R8. ``audio``: the log-mel front end, with snake_case Swift ``MelConfig`` names (provisional
    until a Swift reader exists): ``sample_rate``, ``n_fft``, ``win_length``, ``hop_length``,
    ``n_mel_bins``, ``n_frames``, ``f_min``, ``f_max``.
    - Clip: mono at ``sample_rate``, ``(n_frames - 1) * hop_length`` samples. A shorter clip is
      repeated whole as many times as fits, then zero-padded; a longer one keeps its head.
    - Spectrogram: periodic Hann window, centered frames with reflect padding, power spectrum,
      Slaney-scale mel filters with Slaney area normalization,
      ``10 * log10(max(power, 1e-10))``, no pre-emphasis.

R9. ``logit_scale``: ``logits = logit_scale * cosine``.

R10. ``components.similarity`` (fused bundles only, in no tower): both towers and the logits in
    one call, with the inputs and outputs of the model's Hugging Face ``forward``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from coreai_models._constants import METADATA_VERSION

BUNDLE_KIND = "embedding"
OUTPUT_NAME = "embedding"
TEXT_INPUT_NAMES = ("input_ids", "attention_mask")
TEXT_COMPONENT = "text_encoder"
IMAGE_COMPONENT = "image_encoder"
AUDIO_COMPONENT = "audio_encoder"
SIMILARITY_COMPONENT = "similarity"
SUPPORTED_COMPUTE_PRECISIONS = ("float32", "bfloat16", "float16")
PLATFORMS = ("macOS", "iOS")
MODALITIES = ("text", "image", "audio")


class EmbeddingPrecisionError(ValueError):
    """Raised when an export asks for a compute precision the model does not support on the
    platform."""


@dataclass(frozen=True)
class ImagePreprocessing:
    """The ``image`` block (R7)."""

    image_strategy: str
    image_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]


@dataclass(frozen=True)
class AudioPreprocessing:
    """The ``audio`` block (R8)."""

    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mel_bins: int
    n_frames: int
    f_min: float
    f_max: float


@dataclass(frozen=True)
class EmbeddingFamily:
    """An exportable embedding model: its Hugging Face id, token ids, precisions and
    preprocessing."""

    key: str
    hf_model_id: str
    dimension: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    max_tokens: int
    # platform -> allowed compute precisions, default first
    precisions: dict[str, tuple[str, ...]]
    image: ImagePreprocessing | None = None
    audio: AudioPreprocessing | None = None

    @property
    def modalities(self) -> tuple[str, ...]:
        media = [m for m, facts in (("image", self.image), ("audio", self.audio)) if facts]
        return ("text", *media)


def resolve_compute_precision(family: EmbeddingFamily, requested: str | None, platform: str) -> str:
    """Return ``requested``, or the family's default on ``platform`` if it is None.

    Raises:
        EmbeddingPrecisionError: if the family has no export for the platform or the precision
            is unsupported on it.
    """
    allowed = family.precisions.get(platform)
    if not allowed:
        raise EmbeddingPrecisionError(f"{family.key} has no {platform} export")
    precision = requested or allowed[0]
    if precision not in allowed:
        raise EmbeddingPrecisionError(
            f"{family.key} on {platform} computes in {' or '.join(allowed)}, not {precision}"
        )
    return precision


def function_entry(
    name: str, batch: int | tuple[int, int], length: int | tuple[int, int] | None = None
) -> dict[str, Any]:
    """Return a ``functions`` entry (R3). A ``(min, max)`` pair is a dynamic range."""

    def axis(value: int | tuple[int, int]) -> int | dict[str, int]:
        return value if isinstance(value, int) else {"min": value[0], "max": value[1]}

    entry: dict[str, Any] = {"function": name, "batch": axis(batch)}
    if length is not None:
        entry["length"] = axis(length)
    return entry


def text_block(family: EmbeddingFamily) -> dict[str, Any]:
    """The ``text`` block (R6)."""
    return {
        "embedded_tokenizer": True,
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token_id": family.bos_token_id,
        "eos_token_id": family.eos_token_id,
        "pad_token_id": family.pad_token_id,
        "padding_side": "right",
        "truncation": {"max_tokens": family.max_tokens, "keep_eos": True},
    }


def media_block(family: EmbeddingFamily, modality: str) -> dict[str, Any]:
    """The ``image`` or ``audio`` block (R7, R8)."""
    facts = family.image if modality == "image" else family.audio
    if facts is None:
        raise ValueError(f"{family.key} has no {modality} tower")
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(facts).items()}


_BLOCK_KEYS = (
    "space",
    "dimension",
    "normalized",
    "metric",
    "logit_scale",
    "compute_precision",
    "components",
    "towers",
    "text",
)


def validate_block(block: dict[str, Any], assets: dict[str, str]) -> None:
    """Raise ``ValueError`` if ``block`` breaks R2-R4 or names an asset not in ``assets``."""
    problems = [f"missing {key!r}" for key in _BLOCK_KEYS if key not in block]
    components = block.get("components", {})
    seen: set[tuple[str, str]] = set()
    for name, spec in components.items():
        if spec.get("asset") not in assets:
            problems.append(f"{name}: asset {spec.get('asset')!r} not in assets")
        functions = spec.get("functions")
        if not isinstance(functions, list) or not functions:
            problems.append(f"{name}: functions must be a non-empty array")
            continue
        for entry in functions:
            key = (spec.get("asset"), entry.get("function"))
            if key in seen or "batch" not in entry:
                problems.append(f"{name}: duplicate or incomplete function {entry}")
            seen.add(key)
    towers = block.get("towers", [])
    modalities = [tower.get("modality") for tower in towers]
    if modalities.count("text") != 1 or len(set(modalities)) != len(modalities):
        problems.append(f"towers must hold one text tower and distinct modalities: {modalities}")
    for tower in towers:
        if tower.get("modality") not in MODALITIES:
            problems.append(f"unknown modality {tower.get('modality')!r}")
        missing = [c for c in tower.get("pipeline") or [None] if c not in components]
        if missing:
            problems.append(f"{tower.get('modality')} pipeline names unknown components {missing}")
    for modality in ("image", "audio"):
        if (modality in block) != (modality in modalities):
            problems.append(f"the {modality} block must be present exactly with its tower")
    if problems:
        raise ValueError(f"embedding block breaks the contract: {problems}")


def bundle_name(family: EmbeddingFamily, platform: str, text_only: bool, fused: bool) -> str:
    """The bundle directory name: the family key plus ``-ios``, ``-text`` and ``-fused``."""
    return (
        family.key
        + ("-ios" if platform == "iOS" else "")
        + ("-text" if text_only else "")
        + ("-fused" if fused else "")
    )


def build_embedding_bundle_metadata(
    family: EmbeddingFamily,
    block: dict[str, Any],
    *,
    name: str,
    assets: dict[str, str],
    hf_revision: str | None,
) -> dict[str, Any]:
    """Wrap an ``embedding`` block in the ``metadata.json`` envelope (R1)."""
    return {
        "metadata_version": METADATA_VERSION,
        "kind": BUNDLE_KIND,
        "name": name,
        "assets": assets,
        "embedding": block,
        "source": {
            "model_definition": "torch",
            "hf_model_id": family.hf_model_id,
            "hf_revision": hf_revision,
        },
        "compression": None,
        "compilation": {"date": datetime.now().astimezone().isoformat(), "targets": []},
    }
