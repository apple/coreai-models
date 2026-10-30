# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Builds the ``embedding`` block and ``metadata.json`` for ``kind: "embedding"`` bundles.

Also holds the per-family export facts the exporter and the Swift runtime both read.
EmbeddingGemma 2 overflows in float16 on the GPU, so macOS bundles compute in float32 or
bfloat16, while Neural Engine (iOS) bundles use float16 with scale-invariant norms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from coreai_models._constants import METADATA_VERSION

BUNDLE_KIND = "embedding"
OUTPUT_NAME = "embedding"
TEXT_INPUT_NAMES = ("input_ids", "attention_mask")
TOKEN_TABLE_INPUT_NAMES = ("input_ids",)
TOKEN_TABLE_OUTPUT_NAME = "inputs_embeds"
ENCODER_INPUT_NAMES = ("inputs_embeds", "attention_mask")
ANE_ENCODER_INPUT_NAMES = ("inputs_embeds", "bias", "pool_weights")
IMAGE_INPUT_NAMES = ("pixel_values", "position_ids")
IMAGE_OUTPUT_NAME = "soft_tokens"
TEXT_COMPONENT = "text_encoder"
TOKEN_TABLE_COMPONENT = "token_table"
ENCODER_COMPONENT = "encoder"
IMAGE_COMPONENT = "image_encoder"
SUPPORTED_COMPUTE_PRECISIONS = ("float32", "bfloat16", "float16")
PLATFORMS = ("macOS", "iOS")


class EmbeddingPrecisionError(ValueError):
    """Raised when an export asks for a compute precision the platform or family forbids."""


@dataclass(frozen=True)
class ImageTowerFacts:
    """Image-tower facts the host needs: merge ids, preprocessing and the soft-token budget.

    The exporter checks them against the snapshot ``config.json`` / ``processor_config.json``
    and raises on a mismatch.
    """

    placeholder_id: int
    boi_token_id: int
    eoi_token_id: int
    patch_size: int
    pooling_kernel_size: int
    rescale_factor: float
    normalize: bool
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    budget: int
    soft_token_dim: int

    @property
    def max_patches(self) -> int:
        return self.budget * self.pooling_kernel_size**2

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2

    def sequence_length(self) -> int:
        """Longest image-only sequence: ``<bos> <boi> <image>*budget <eoi> <eos>``."""
        return self.budget + 4


@dataclass(frozen=True)
class EmbeddingFamily:
    """Facts about one retrieval-embedding family that the bundle must carry.

    Values come from the HF snapshot files and model card; the space *revision* is the snapshot
    SHA of the weights actually exported.
    """

    key: str
    hf_model_id: str
    dimension: int
    matryoshka_dimensions: tuple[int, ...]
    # canonical prompt name -> prompt key in the snapshot's config_sentence_transformers.json
    prompt_names: dict[str, str]
    metric: str = "cosine"
    normalized: bool = True
    renormalize_after_truncation: bool = True
    pooling_mode: str = "mean"
    include_prompt: bool = True
    # Precision of the GPU encoder, which runs the reference math. The Neural Engine encoder
    # rescales its norms to compute in float16.
    default_compute_precision: str = "float32"
    precision_floor: str | None = None
    forbidden_precisions: tuple[str, ...] = ()
    model_max_tokens: int = 512
    special_tokens: dict[str, bool] = field(default_factory=dict)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    padding_side: str = "right"
    truncation_keep_eos: bool = True
    document_template: str | None = None
    image: ImageTowerFacts | None = None
    length_buckets: tuple[int, ...] = (128, 512, 2048)
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    max_batch_tokens: int = 16384
    ane_length_buckets: tuple[int, ...] = (128, 512, 1024)


EMBEDDINGGEMMA2 = EmbeddingFamily(
    key="embeddinggemma-2",
    hf_model_id="google/embeddinggemma-2",
    dimension=768,
    # Model card only (no HF JSON lists them); quality is near-lossless down to 256.
    matryoshka_dimensions=(768, 512, 256, 128),
    prompt_names={
        "query": "query",
        "document": "document",
        "sentence_similarity": "STS",
        "classification": "Classification",
        "clustering": "Clustering",
        "question_answering": "QuestionAnswering",
        "fact_checking": "FactChecking",
        "code_retrieval": "CodeRetrieval",
    },
    precision_floor="bfloat16",
    forbidden_precisions=("float16",),
    model_max_tokens=8192,
    # tokenizer.json post-processor is TemplateProcessing "<bos> A <eos>".
    special_tokens={"bos": True, "eos": True},
    bos_token_id=2,
    eos_token_id=1,
    pad_token_id=0,
    document_template="title: {title} | text: {text}",
    # config.json image/boi/eoi token ids; processor_config.json image_processor.
    image=ImageTowerFacts(
        placeholder_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        patch_size=16,
        pooling_kernel_size=3,
        rescale_factor=1.0 / 255.0,
        normalize=False,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        budget=280,
        soft_token_dim=512,
    ),
)

EMBEDDING_FAMILIES: dict[str, EmbeddingFamily] = {EMBEDDINGGEMMA2.key: EMBEDDINGGEMMA2}


def family_for_model(name: str) -> EmbeddingFamily:
    """Return the family whose key or HF id matches ``name`` (case-insensitive)."""
    wanted = name.lower()
    for family in EMBEDDING_FAMILIES.values():
        if wanted in (family.key.lower(), family.hf_model_id.lower()):
            return family
    raise KeyError(f"No embedding family {name!r}; known: {sorted(EMBEDDING_FAMILIES)}")


def default_precision(family: EmbeddingFamily, platform: str) -> str:
    return "float16" if platform == "iOS" else family.default_compute_precision


def validate_compute_precision(
    family: EmbeddingFamily, precision: str, *, platform: str = "macOS"
) -> str:
    """Validate ``precision`` for ``family`` on ``platform`` and return it.

    Raises:
        EmbeddingPrecisionError: if the precision is unknown, not float16 on iOS, or forbidden
            for the family on macOS.
    """
    if precision not in SUPPORTED_COMPUTE_PRECISIONS:
        raise EmbeddingPrecisionError(
            f"Unknown compute precision {precision!r}; expected one of "
            f"{list(SUPPORTED_COMPUTE_PRECISIONS)}"
        )
    if platform == "iOS":
        if precision != "float16":
            raise EmbeddingPrecisionError(
                f"iOS (Neural Engine) bundles compute in float16; got {precision}."
            )
        return precision
    if precision in family.forbidden_precisions:
        floor = family.precision_floor
        alternatives = "float32 (default)" + (
            f" or {floor}" if floor and floor != "float32" else ""
        )
        raise EmbeddingPrecisionError(
            f"{family.hf_model_id} forbids {precision} compute on macOS (it overflows: NaN or "
            f"silently degraded vectors). Use {alternatives}."
        )
    return precision


def consolidate_prompts(
    family: EmbeddingFamily, snapshot_prompts: dict[str, str]
) -> dict[str, str]:
    """Collapse the snapshot's prompt table to canonical name -> prefix, in family order.

    Fails closed: a mapped snapshot key that is missing, or a snapshot prefix that no canonical
    prompt carries (a new upstream task), raises instead of being dropped.
    """
    missing = [key for key in family.prompt_names.values() if key not in snapshot_prompts]
    if missing:
        raise ValueError(f"snapshot prompts are missing {missing}")
    canonical = {name: snapshot_prompts[key] for name, key in family.prompt_names.items()}
    unknown = sorted(
        key for key, prefix in snapshot_prompts.items() if prefix not in canonical.values()
    )
    if unknown:
        raise ValueError(f"snapshot prompts {unknown} use prefixes outside the canonical set")
    return canonical


@dataclass(frozen=True)
class FunctionSpec:
    """One static function; text inputs are ``int32[batch, length]``."""

    name: str
    batch: int
    length: int

    def as_dict(self) -> dict[str, Any]:
        return {"function": self.name, "batch": self.batch, "length": self.length}


def function_name(length: int, batch: int, prefix: str = "") -> str:
    return f"{prefix}len{length}_b{batch}"


def bucket_plan(
    length_buckets: list[int] | tuple[int, ...],
    batch_sizes: list[int] | tuple[int, ...],
    max_batch_tokens: int,
) -> list[FunctionSpec]:
    """Cross length buckets with batch sizes, dropping pairs above ``max_batch_tokens``.

    Batch 1 is always kept for every length (the unbatched path). Attention cost is O(B * L^2), so
    the token cap is what keeps the large buckets small.
    """
    lengths = sorted(set(int(x) for x in length_buckets))
    batches = sorted(set(int(x) for x in batch_sizes) | {1})
    if not lengths or lengths[0] < 2:
        raise ValueError(f"length_buckets must be >= 2 (BOS + EOS), got {length_buckets}")
    if batches[0] < 1:
        raise ValueError(f"batch_sizes must be >= 1, got {batch_sizes}")
    return [
        FunctionSpec(function_name(length, batch), batch, length)
        for length in lengths
        for batch in batches
        if batch == 1 or batch * length <= max_batch_tokens
    ]


def text_functions(family: EmbeddingFamily, platform: str) -> list[FunctionSpec]:
    """Text-tower functions: batched buckets on macOS, batch 1 per Neural Engine length on iOS."""
    if platform == "iOS":
        return bucket_plan(family.ane_length_buckets, (1,), family.max_batch_tokens)
    return bucket_plan(family.length_buckets, family.batch_sizes, family.max_batch_tokens)


def encoder_functions(family: EmbeddingFamily, platform: str) -> list[FunctionSpec]:
    """Batch-1 ``inputs_embeds`` functions of the ``encoder`` component.

    Named ``embeds_len{L}_b1`` on both platforms; the host reads the names from
    ``components.encoder.functions``.
    """
    if platform == "iOS":
        lengths = sorted(set(family.ane_length_buckets))
    else:
        assert family.image is not None
        first = -(-family.image.sequence_length() // 32) * 32
        lengths = sorted({first} | {n for n in family.length_buckets if n > first})
    return [FunctionSpec(function_name(n, 1, "embeds_"), 1, n) for n in lengths]


def bundle_name(family: EmbeddingFamily, platform: str, with_image: bool) -> str:
    return family.key + ("-ios" if platform == "iOS" else "") + ("" if with_image else "-text")


IMAGE_PIPELINE = (IMAGE_COMPONENT, TOKEN_TABLE_COMPONENT, ENCODER_COMPONENT)


def text_pipeline(platform: str) -> tuple[str, ...]:
    """Components the text tower runs, in order."""
    return (TOKEN_TABLE_COMPONENT, ENCODER_COMPONENT) if platform == "iOS" else (TEXT_COMPONENT,)


def bundle_components(platform: str, with_image: bool) -> tuple[str, ...]:
    """Components of the bundle's pipelines, text first."""
    return tuple(dict.fromkeys(text_pipeline(platform) + (IMAGE_PIPELINE if with_image else ())))


def component_asset(platform: str, component: str) -> str:
    """Name of the asset holding ``component``.

    On macOS, ``token_table`` and ``encoder`` are functions of the ``text_encoder`` asset.
    """
    shared = platform != "iOS" and component in (TOKEN_TABLE_COMPONENT, ENCODER_COMPONENT)
    return TEXT_COMPONENT if shared else component


def bundle_assets(platform: str, with_image: bool) -> dict[str, str]:
    """Asset name -> ``.aimodel`` file, one entry per file; components name theirs in ``asset``."""
    names = (component_asset(platform, c) for c in bundle_components(platform, with_image))
    return {name: f"{name}.aimodel" for name in dict.fromkeys(names)}


def output_dtype(platform: str) -> str:
    """Dtype of every component output: float16 on iOS, float32 on macOS."""
    return "float16" if platform == "iOS" else "float32"


def embedding_block(
    family: EmbeddingFamily,
    *,
    platform: str,
    with_image: bool,
    revision: str,
    prompts: dict[str, str],
    default_prompt_name: str | None = None,
    compute_precision: str | None = None,
) -> dict[str, Any]:
    """Build the ``embedding`` block of ``metadata.json``.

    Text runs ``text_encoder`` on macOS and ``token_table -> encoder`` on iOS. With
    ``with_image``, images run ``image_encoder -> token_table -> encoder``, and the host writes
    the soft tokens into the ``token_table`` output at the placeholder rows. ``token_table`` runs
    on the GPU; on iOS the encoders run on the Neural Engine. Each component names its
    ``bundle_assets`` entry (``asset``), its functions and its ``output_dtype``.
    ``compute_precision`` defaults to the platform's precision.

    Raises:
        ValueError: on an unknown platform, a missing revision or prompts, an unknown
            ``default_prompt_name``, or family facts that do not fit the function plan.
        EmbeddingPrecisionError: if ``compute_precision`` is not allowed on ``platform``.
    """
    ios = platform == "iOS"
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {list(PLATFORMS)}, got {platform!r}")
    precision = validate_compute_precision(
        family, compute_precision or default_precision(family, platform), platform=platform
    )
    if not revision:
        raise ValueError("revision (snapshot SHA) is required to stamp the embedding space")
    if not prompts:
        raise ValueError("prompts must not be empty")
    if default_prompt_name is not None and default_prompt_name not in prompts:
        raise ValueError(f"default_prompt_name {default_prompt_name!r} not in prompts")
    if family.matryoshka_dimensions and family.matryoshka_dimensions[0] != family.dimension:
        raise ValueError("matryoshka_dimensions must start with the native dimension")
    image = family.image
    if with_image and image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")

    text_fns = text_functions(family, platform)
    text_lengths = sorted({f.length for f in text_fns})
    if text_lengths[-1] > family.model_max_tokens:
        raise ValueError(
            f"length bucket {text_lengths[-1]} exceeds model_max_tokens {family.model_max_tokens}"
        )
    enc_fns: list[FunctionSpec] = []
    if with_image or ios:
        enc_fns = encoder_functions(family, platform)
    functions_json = [f.as_dict() for f in enc_fns]
    out_dtype = output_dtype(platform)

    specs: dict[str, Any] = {
        TEXT_COMPONENT: {
            "functions": [f.as_dict() for f in text_fns],
            "inputs": list(TEXT_INPUT_NAMES),
            "output": OUTPUT_NAME,
            "output_dtype": out_dtype,
            "compute_units": ["gpu"],
        },
        IMAGE_COMPONENT: {
            "function": "main",
            "inputs": list(IMAGE_INPUT_NAMES),
            "output": IMAGE_OUTPUT_NAME,
            "output_dtype": out_dtype,
            "compute_units": ["neural_engine"] if ios else ["gpu"],
        },
        TOKEN_TABLE_COMPONENT: {
            "function": "main",
            "inputs": list(TOKEN_TABLE_INPUT_NAMES),
            "output": TOKEN_TABLE_OUTPUT_NAME,
            "output_dtype": out_dtype,
            "compute_units": ["gpu"],
        },
        ENCODER_COMPONENT: {
            "functions": functions_json,
            "inputs": list(ANE_ENCODER_INPUT_NAMES if ios else ENCODER_INPUT_NAMES),
            "output": OUTPUT_NAME,
            "output_dtype": out_dtype,
            "mask_encoding": "additive_bias" if ios else "attention_mask",
            "compute_units": ["neural_engine"] if ios else ["gpu"],
        },
    }
    components = {
        name: {"asset": component_asset(platform, name), **specs[name]}
        for name in bundle_components(platform, with_image)
    }

    text_tower: dict[str, Any] = {
        "modality": "text",
        "pipeline": list(text_pipeline(platform)),
        "inputs": {"input_ids": "int32[B,L]", "attention_mask": "int32[B,L]"},
        "output": OUTPUT_NAME,
        "batch_sizes": sorted({f.batch for f in text_fns}),
        "length_buckets": text_lengths,
    }
    if not ios:
        text_tower["max_batch_tokens"] = family.max_batch_tokens
    towers = [text_tower]
    if with_image:
        assert image is not None
        image_lengths = sorted(f.length for f in enc_fns if f.length >= image.sequence_length())
        if not image_lengths:
            raise ValueError(
                f"no encoder function fits the longest image sequence ({image.sequence_length()})"
            )
        towers.append(
            {
                "modality": "image",
                "pipeline": list(IMAGE_PIPELINE),
                "output": OUTPUT_NAME,
                "batch_sizes": [1],
                "length_buckets": image_lengths,
            }
        )

    special_tokens: dict[str, Any] = dict(family.special_tokens)
    for key, value in (
        ("bos_token_id", family.bos_token_id),
        ("eos_token_id", family.eos_token_id),
        ("pad_token_id", family.pad_token_id),
    ):
        if value is not None:
            special_tokens[key] = value

    block: dict[str, Any] = {
        "space": {"id": family.hf_model_id, "revision": revision},
        "dimension": family.dimension,
        "matryoshka_dimensions": list(family.matryoshka_dimensions),
        "renormalize_after_truncation": family.renormalize_after_truncation,
        "normalized": family.normalized,
        "metric": family.metric,
        "precision": {
            "compute": precision,
            # The Neural Engine graphs compute in float16.
            "forbidden": [] if ios else list(family.forbidden_precisions),
            "compute_units": ["neural_engine", "gpu"] if ios else ["gpu"],
        },
        "components": components,
        "towers": towers,
        "text": {
            "tokenizer": "tokenizer",
            "padding_side": family.padding_side,
            "special_tokens": special_tokens,
            "truncation": {"keep_eos": family.truncation_keep_eos, "max_tokens": text_lengths[-1]},
            "model_max_tokens": family.model_max_tokens,
            "prompts": dict(prompts),
            "default_prompt_name": default_prompt_name,
            "document_template": family.document_template,
        },
    }
    if with_image:
        assert image is not None
        block["merge"] = {
            "placeholder_ids": {"image": image.placeholder_id},
            "token_ids": {"boi": image.boi_token_id, "eoi": image.eoi_token_id},
        }
        block["preprocessing"] = {
            "image": {
                "patch_size": image.patch_size,
                "rescale_factor": image.rescale_factor,
                "normalize": image.normalize,
                "image_mean": list(image.image_mean),
                "image_std": list(image.image_std),
                "max_soft_tokens": image.budget,
                "pooling_kernel_size": image.pooling_kernel_size,
            }
        }
    return block


def build_bundle_metadata(
    family: EmbeddingFamily,
    block: dict[str, Any],
    *,
    name: str,
    assets: dict[str, str],
    hf_revision: str | None,
) -> dict[str, Any]:
    """Wrap an :func:`embedding_block` in the 0.2 ``metadata.json`` envelope."""
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
