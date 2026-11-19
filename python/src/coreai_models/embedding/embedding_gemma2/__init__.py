# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""EmbeddingGemma 2 exporter: family facts, snapshot checks and the ``embedding`` block.

macOS bundles run on the GPU and compute in float32 or bfloat16; float16 overflows on the GPU.
iOS bundles run the encoders on the Neural Engine and compute in float16 with scale-invariant
norms. ``macos`` and ``ios`` build the programs.

Keys beyond the contract: ``matryoshka_dimensions`` and ``renormalize_after_truncation`` in the
block; ``model_max_tokens``, ``prompts``, ``default_prompt_name`` and ``document_template`` in
``text``; ``patch_size``, ``pooling_kernel_size``, ``max_soft_tokens``, ``image_token_id``,
``boi_token_id`` and ``eoi_token_id`` in ``image``; ``bias`` and ``pool_weights`` in the iOS
``encoder`` component.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from coreai_models._constants import MASK_NEG
from coreai_models.embedding.contract import (
    IMAGE_COMPONENT,
    OUTPUT_NAME,
    PLATFORMS,
    TEXT_COMPONENT,
    TEXT_INPUT_NAMES,
    CheckpointFacts,
    function_entry,
    resolve_compute_precision,
    text_block,
)
from coreai_models.embedding.embedding_gemma2.common import (
    ENCODER_COMPONENT,
    IMAGE_OUTPUT_NAME,
    TOKEN_TABLE_COMPONENT,
    TOKEN_TABLE_OUTPUT_NAME,
    EmbeddingGemma2Family,
    ImageTowerFacts,
    encoder_functions,
    output_dtype,
    text_functions,
)
from coreai_models.embedding.embedding_gemma2.ios import export_ios_programs
from coreai_models.embedding.embedding_gemma2.macos import export_macos_programs
from coreai_models.embedding.export_utils import EmbeddingExporter, fetch_weights, read_prompts

IMAGE_STRATEGY = "aspect_ratio_preserving"


EMBEDDINGGEMMA2 = EmbeddingGemma2Family(
    key="embeddinggemma-2",
    hf_model_id="google/embeddinggemma-2",
    precisions={"macOS": ("float32", "bfloat16"), "iOS": ("float16",)},
    modalities=("text", "image"),
    max_tokens=8192,
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
    document_template="title: {title} | text: {text}",
    # config.json image/boi/eoi token ids; processor_config.json image_processor.
    image_tower=ImageTowerFacts(
        placeholder_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        patch_size=16,
        pooling_kernel_size=3,
        rescale_factor=1.0 / 255.0,
        normalize=False,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        max_soft_tokens=280,
        soft_token_dim=512,
    ),
)


def consolidate_prompts(
    family: EmbeddingGemma2Family, snapshot_prompts: dict[str, str], default: str | None = None
) -> tuple[dict[str, str], str | None]:
    """Collapse the snapshot's prompt table to canonical name -> prefix, in family order.

    ``default`` is the snapshot's ``default_prompt_name``. It maps to the canonical name that
    ``family.prompt_names`` gives that key, else to the one canonical name with the same prefix.
    Returns ``(prompts, default_prompt_name)``.

    Raises:
        ValueError: if a snapshot key mapped in ``family.prompt_names`` is missing, a snapshot
            prompt uses a prefix that no canonical prompt carries, or ``default`` is not a
            snapshot prompt or matches more than one canonical name.
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
    if default is None:
        return canonical, None
    if default not in snapshot_prompts:
        raise ValueError(f"default_prompt_name {default!r} is not a snapshot prompt")
    by_key = {key: name for name, key in family.prompt_names.items()}
    names = (
        [by_key[default]]
        if default in by_key
        else [name for name, prefix in canonical.items() if prefix == snapshot_prompts[default]]
    )
    if len(names) != 1:
        raise ValueError(f"default_prompt_name {default!r} matches {names}")
    return canonical, names[0]


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


def embedding_block(
    family: EmbeddingGemma2Family,
    facts: CheckpointFacts,
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
    ``with_image``, images run ``image_encoder -> token_table -> encoder``: the host resizes the
    image (``aspect_ratio_preserving``: the largest size with sides divisible by ``patch_size *
    pooling_kernel_size`` and at most ``max_soft_tokens * pooling_kernel_size**2`` patches), feeds
    the ids ``<bos> <boi> <image>*n <eoi> <eos>`` to ``token_table`` and writes the ``n`` soft
    tokens into its output at the ``image_token_id`` rows. ``token_table`` runs on the GPU; on iOS
    the encoders run on the Neural Engine. Each component names its ``bundle_assets`` entry
    (``asset``). ``compute_precision`` defaults to the platform's precision.

    Raises:
        ValueError: on an unknown platform, a missing revision or prompts, an unknown
            ``default_prompt_name``, ``with_image`` on a family without an image tower, or family
            facts that do not fit the function plan.
        EmbeddingPrecisionError: if ``compute_precision`` is not allowed on ``platform``.
    """
    ios = platform == "iOS"
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {list(PLATFORMS)}, got {platform!r}")
    precision = resolve_compute_precision(family, compute_precision, platform)
    if not revision:
        raise ValueError("revision (snapshot SHA) is required to stamp the embedding space")
    if not prompts:
        raise ValueError("prompts must not be empty")
    if default_prompt_name is not None and default_prompt_name not in prompts:
        raise ValueError(f"default_prompt_name {default_prompt_name!r} not in prompts")
    if family.matryoshka_dimensions and family.matryoshka_dimensions[0] != facts.dimension:
        raise ValueError("matryoshka_dimensions must start with the native dimension")
    image = family.image_tower
    if with_image and image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")

    text_fns = [] if ios else text_functions(family)
    enc_fns = encoder_functions(family, platform, with_image=with_image)
    # iOS text runs token_table -> encoder, so the text tower has the encoder's lengths.
    tower_fns = enc_fns if ios else text_fns
    max_length = max(f.length for f in tower_fns)
    if max_length > facts.max_tokens:
        raise ValueError(f"length bucket {max_length} exceeds max_tokens {facts.max_tokens}")
    if with_image:
        assert image is not None
        if all(f.length < image.max_sequence_length for f in enc_fns):
            raise ValueError(
                f"no encoder function fits the longest image sequence ({image.max_sequence_length})"
            )

    dtype = output_dtype(platform)
    units = ["neural_engine"] if ios else ["gpu"]
    embedding = {OUTPUT_NAME: f"{dtype}[B,{facts.dimension}]"}
    specs: dict[str, dict[str, Any]] = {
        TEXT_COMPONENT: {
            "inputs": {name: "int32[B,L]" for name in TEXT_INPUT_NAMES},
            "outputs": embedding,
            "compute_units": ["gpu"],
            "functions": [f.as_dict() for f in text_fns],
        }
    }
    if image is not None:
        hidden = image.soft_token_dim
        positions = "float16" if ios else "int32"
        specs[IMAGE_COMPONENT] = {
            "inputs": {
                "pixel_values": f"{dtype}[B,{image.max_patches},{image.patch_dim}]",
                "position_ids": f"{positions}[B,{image.max_patches},2]",
            },
            "outputs": {IMAGE_OUTPUT_NAME: f"{dtype}[B,{image.max_soft_tokens},{hidden}]"},
            "compute_units": units,
            "functions": [function_entry("main", 1)],
        }
        specs[TOKEN_TABLE_COMPONENT] = {
            "inputs": {"input_ids": "int32[B,L]"},
            "outputs": {TOKEN_TABLE_OUTPUT_NAME: f"{dtype}[B,L,{hidden}]"},
            "compute_units": ["gpu"],
            "functions": [function_entry("main", 1, (2, facts.max_tokens))],
        }
        if ios:
            inputs = {
                "inputs_embeds": f"{dtype}[B,L,{hidden}]",
                "bias": f"{dtype}[B,L,1,L]",
                "pool_weights": f"{dtype}[B,L]",
            }
        else:
            inputs = {"inputs_embeds": f"{dtype}[B,L,{hidden}]", "attention_mask": "int32[B,L]"}
        specs[ENCODER_COMPONENT] = {
            "inputs": inputs,
            "outputs": embedding,
            "compute_units": units,
            "functions": [f.as_dict() for f in enc_fns],
        }
        if ios:
            # bias[0, k, 0, q] is 0.0 if key k is valid, else MASK_NEG; the graph adds the band.
            specs[ENCODER_COMPONENT]["bias"] = {
                "layout": "[1, key, 1, query]",
                "masked": MASK_NEG,
                "unmasked": 0.0,
                "sliding_window": "in_graph",
            }
            specs[ENCODER_COMPONENT]["pool_weights"] = "attention_mask / attention_mask.sum(-1)"
    components = {
        name: {"asset": component_asset(platform, name), **specs[name]}
        for name in bundle_components(platform, with_image)
    }
    towers = [{"modality": "text", "pipeline": list(text_pipeline(platform))}]
    if with_image:
        towers.append({"modality": "image", "pipeline": list(IMAGE_PIPELINE)})

    text = text_block(facts)
    text["truncation"]["max_tokens"] = max_length
    text.update(
        model_max_tokens=facts.max_tokens,
        prompts=dict(prompts),
        default_prompt_name=default_prompt_name,
        document_template=family.document_template,
    )
    block: dict[str, Any] = {
        "space": {"id": family.hf_model_id, "revision": revision},
        "dimension": facts.dimension,
        "matryoshka_dimensions": list(family.matryoshka_dimensions),
        "renormalize_after_truncation": family.renormalize_after_truncation,
        "normalized": family.normalized,
        "metric": family.metric,
        "logit_scale": None,
        "compute_precision": precision,
        "components": components,
        "towers": towers,
        "text": text,
    }
    if with_image:
        assert image is not None
        block["image"] = {
            "image_strategy": IMAGE_STRATEGY,
            "image_mean": list(image.image_mean),
            "image_std": list(image.image_std),
            "patch_size": image.patch_size,
            "pooling_kernel_size": image.pooling_kernel_size,
            "max_soft_tokens": image.max_soft_tokens,
            "image_token_id": image.placeholder_id,
            "boi_token_id": image.boi_token_id,
            "eoi_token_id": image.eoi_token_id,
        }
    return block


# --- snapshot checks ---


def read_pooling_config(snapshot: Path, family: EmbeddingGemma2Family) -> int:
    """Return the embedding dimension in the snapshot's ``1_Pooling/config.json``.

    Raises:
        ValueError: if the file is missing, or the pooling mode, ``include_prompt`` or, for a
            normalized family, the ``2_Normalize`` module disagree with the family.
    """
    pooling = snapshot / "1_Pooling" / "config.json"
    if not pooling.exists():
        raise ValueError(f"{snapshot} has no 1_Pooling/config.json")
    cfg = json.loads(pooling.read_text())
    mode = cfg.get("pooling_mode")
    if mode is None:
        mode = "mean" if cfg.get("pooling_mode_mean_tokens") else None
    problems = []
    if mode != family.pooling_mode:
        problems.append(f"pooling_mode {mode!r} != {family.pooling_mode!r}")
    if bool(cfg.get("include_prompt", True)) != family.include_prompt:
        problems.append(f"include_prompt {cfg.get('include_prompt')} != {family.include_prompt}")
    if family.normalized and not (snapshot / "2_Normalize").exists():
        problems.append("family is normalized but the snapshot has no 2_Normalize module")
    if problems:
        raise ValueError(f"Snapshot pooling config disagrees with the family: {problems}")
    return int(cfg["embedding_dimension"])


def verify_image_config(snapshot: Path, family: EmbeddingGemma2Family) -> None:
    """Raise ``ValueError`` if the snapshot's image config disagrees with ``family.image_tower``.

    Checks the placeholder, begin-image and end-image token ids and the text hidden size in
    ``config.json``, and the image processor settings in ``processor_config.json``. Also raises
    if the family has no image tower.
    """
    image = family.image_tower
    if image is None:
        raise ValueError(f"{family.hf_model_id} has no image tower")
    problems = []
    config_path = snapshot / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        for key, want in (
            ("image_token_id", image.placeholder_id),
            ("boi_token_id", image.boi_token_id),
            ("eoi_token_id", image.eoi_token_id),
        ):
            if key in cfg and int(cfg[key]) != want:
                problems.append(f"config.json {key} {cfg[key]} != {want}")
        text_hidden = (cfg.get("text_config") or {}).get("hidden_size")
        if text_hidden is not None and int(text_hidden) != image.soft_token_dim:
            problems.append(f"text hidden_size {text_hidden} != {image.soft_token_dim}")
    processor_path = snapshot / "processor_config.json"
    if processor_path.exists():
        proc = json.loads(processor_path.read_text()).get("image_processor") or {}
        checks = (
            ("patch_size", image.patch_size),
            ("pooling_kernel_size", image.pooling_kernel_size),
            ("max_soft_tokens", image.max_soft_tokens),
            ("do_normalize", image.normalize),
            ("resample", 3),  # PIL/torchvision BICUBIC
        )
        for key, want in checks:
            if key in proc and proc[key] != want:
                problems.append(f"processor {key} {proc[key]!r} != {want!r}")
        if "rescale_factor" in proc and abs(proc["rescale_factor"] - image.rescale_factor) > 1e-12:
            problems.append(
                f"processor rescale_factor {proc['rescale_factor']} != {image.rescale_factor}"
            )
    if problems:
        raise ValueError(f"Snapshot image config disagrees with the family: {problems}")


# --- exporter ---


def read_facts(family: EmbeddingGemma2Family, snapshot: Path) -> CheckpointFacts:
    """The snapshot's facts: the dimension from ``1_Pooling/config.json``, the token ids from
    ``config.json`` ``text_config`` and the family's ``max_tokens``.

    Raises:
        ValueError: if the pooling config is missing or disagrees with the family
            (``read_pooling_config``).
    """
    dimension = read_pooling_config(snapshot, family)
    text = json.loads((snapshot / "config.json").read_text())["text_config"]
    assert family.max_tokens is not None
    return CheckpointFacts(
        dimension=dimension,
        bos_token_id=int(text["bos_token_id"]),
        eos_token_id=int(text["eos_token_id"]),
        pad_token_id=int(text["pad_token_id"]),
        max_tokens=family.max_tokens,
    )


def export_programs(
    family: EmbeddingGemma2Family,
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
    """Write the bundle's assets to ``out`` and return ``(assets, embedding block)``.

    macOS writes a GPU bundle: ``text_encoder``; unless ``text_only``, also ``image_encoder``,
    and ``text_encoder`` then holds the ``token_table`` and ``encoder`` functions. iOS writes a
    Neural Engine bundle: ``token_table`` and ``encoder``; unless ``text_only``, also
    ``image_encoder``. The image config and prompts are checked before converting.

    Raises:
        ValueError: if the snapshot's image config or prompts disagree with the family.
    """
    with_image = not text_only
    if with_image:
        verify_image_config(snapshot, family)
    prompts, default_prompt = consolidate_prompts(family, *read_prompts(snapshot))
    block = embedding_block(
        family,
        facts,
        platform=platform,
        with_image=with_image,
        revision=revision,
        prompts=prompts,
        default_prompt_name=default_prompt,
        compute_precision=compute_precision,
    )
    fetch_weights(family, snapshot)
    if platform == "iOS":
        export_ios_programs(
            family,
            snapshot,
            out,
            facts,
            with_image=with_image,
            include_debug_info=include_debug_info,
        )
    else:
        export_macos_programs(
            family,
            snapshot,
            out,
            facts,
            with_image=with_image,
            compute_precision=compute_precision,
            include_debug_info=include_debug_info,
        )
    return bundle_assets(platform, with_image), block


def exporter(family: EmbeddingGemma2Family) -> EmbeddingExporter:
    return EmbeddingExporter(
        family=family,
        read_facts=partial(read_facts, family),
        export_programs=partial(export_programs, family),
    )


EMBEDDINGGEMMA2_EXPORTER = exporter(EMBEDDINGGEMMA2)
