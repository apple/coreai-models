# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3-Embedding exporter: family facts, snapshot checks and the ``embedding`` block.

macOS bundles run the text tower on the GPU and compute in float32, float16 or bfloat16. The
tower is the ``text_encoder`` asset with ``len{L}_b{B}`` functions; the graph takes the last real
token and L2-normalizes it. The tokenizer adds no bos and ends every text with
``<|endoftext|>``, which is also the pad token.

Keys beyond the contract: ``matryoshka_dimensions`` and ``renormalize_after_truncation`` in the
block; ``model_max_tokens``, ``prompts`` and ``default_prompt_name`` in ``text``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from coreai_models.embedding.contract import (
    OUTPUT_NAME,
    TEXT_COMPONENT,
    TEXT_INPUT_NAMES,
    CheckpointFacts,
    EmbeddingFamily,
    resolve_compute_precision,
    text_block,
)
from coreai_models.embedding.embedding_gemma2 import fetch_weights, read_prompts
from coreai_models.embedding.embedding_gemma2.common import FunctionSpec, bucket_plan
from coreai_models.embedding.export_utils import (
    TORCH_DTYPES,
    EmbeddingExporter,
    export_program,
    save_program,
)

# The sentence-transformers modules the checkpoint must list, in order (modules.json).
ST_MODULES = (
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Normalize",
)


@dataclass(frozen=True)
class Qwen3EmbeddingFamily(EmbeddingFamily):
    """``EmbeddingFamily`` plus the Qwen3-Embedding export facts.

    ``max_tokens`` is the model's token limit; ``text.truncation.max_tokens`` is the longest text
    function.
    """

    matryoshka_dimensions: tuple[int, ...] = ()
    length_buckets: tuple[int, ...] = (128, 512, 2048)
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    max_batch_tokens: int = 16384


QWEN3_EMBEDDING_0_6B = Qwen3EmbeddingFamily(
    key="qwen3-embedding-0.6b",
    hf_model_id="Qwen/Qwen3-Embedding-0.6B",
    precisions={"macOS": ("float32", "float16", "bfloat16")},
    modalities=("text",),
    # config.json max_position_embeddings
    max_tokens=32768,
    # The model card gives the range 32 to 1024.
    matryoshka_dimensions=(1024, 512, 256, 128, 64, 32),
)


def text_functions(family: Qwen3EmbeddingFamily) -> list[FunctionSpec]:
    """Functions of ``text_encoder``: ``length_buckets`` crossed with ``batch_sizes``."""
    return bucket_plan(family.length_buckets, family.batch_sizes, family.max_batch_tokens)


def embedding_block(
    family: Qwen3EmbeddingFamily,
    facts: CheckpointFacts,
    *,
    revision: str,
    prompts: dict[str, str],
    default_prompt_name: str | None = None,
    compute_precision: str | None = None,
) -> dict[str, Any]:
    """Build the ``embedding`` block of ``metadata.json`` for a macOS bundle.

    The text tower is ``text_encoder``. ``compute_precision`` defaults to the family's.

    Raises:
        ValueError: on a missing revision, prompts without ``query`` and ``document``, an unknown
            ``default_prompt_name``, or family facts that do not fit the function plan.
        EmbeddingPrecisionError: if ``compute_precision`` is not allowed on macOS.
    """
    precision = resolve_compute_precision(family, compute_precision, "macOS")
    if not revision:
        raise ValueError("revision (snapshot SHA) is required to stamp the embedding space")
    missing = [name for name in ("query", "document") if name not in prompts]
    if missing:
        raise ValueError(f"prompts are missing {missing}")
    if default_prompt_name is not None and default_prompt_name not in prompts:
        raise ValueError(f"default_prompt_name {default_prompt_name!r} not in prompts")
    if family.matryoshka_dimensions and family.matryoshka_dimensions[0] != facts.dimension:
        raise ValueError("matryoshka_dimensions must start with the native dimension")
    functions = text_functions(family)
    max_length = max(f.length for f in functions)
    if max_length > facts.max_tokens:
        raise ValueError(f"length bucket {max_length} exceeds max_tokens {facts.max_tokens}")

    text = text_block(facts)
    text["truncation"]["max_tokens"] = max_length
    text.update(
        model_max_tokens=facts.max_tokens,
        prompts=dict(prompts),
        default_prompt_name=default_prompt_name,
    )
    component = {
        "asset": TEXT_COMPONENT,
        "inputs": {name: "int32[B,L]" for name in TEXT_INPUT_NAMES},
        "outputs": {OUTPUT_NAME: f"float32[B,{facts.dimension}]"},
        "compute_units": ["gpu"],
        "functions": [f.as_dict() for f in functions],
    }
    return {
        "space": {"id": family.hf_model_id, "revision": revision},
        "dimension": facts.dimension,
        "matryoshka_dimensions": list(family.matryoshka_dimensions),
        "renormalize_after_truncation": True,
        "normalized": True,
        "metric": "cosine",
        "logit_scale": None,
        "compute_precision": precision,
        "components": {TEXT_COMPONENT: component},
        "towers": [{"modality": "text", "pipeline": [TEXT_COMPONENT]}],
        "text": text,
    }


# --- snapshot checks ---


def read_pooling_dimension(snapshot: Path) -> int:
    """Return the embedding dimension of the snapshot's last-token pooling.

    Raises:
        ValueError: if ``modules.json`` is not Transformer -> Pooling -> Normalize, or
            ``1_Pooling/config.json`` pools anything but the last token, excludes the prompt or
            disagrees with ``config.json`` ``hidden_size``.
    """
    modules = json.loads((snapshot / "modules.json").read_text())
    problems = []
    types = tuple(module.get("type") for module in modules)
    if types != ST_MODULES:
        problems.append(f"modules {list(types)} != {list(ST_MODULES)}")
    pooling = json.loads((snapshot / "1_Pooling" / "config.json").read_text())
    modes = sorted(key for key, on in pooling.items() if key.startswith("pooling_mode_") and on)
    if modes != ["pooling_mode_lasttoken"]:
        problems.append(f"pooling modes {modes} != ['pooling_mode_lasttoken']")
    if not pooling.get("include_prompt", True):
        problems.append("include_prompt is false")
    dimension = int(pooling["word_embedding_dimension"])
    hidden = json.loads((snapshot / "config.json").read_text())["hidden_size"]
    if dimension != hidden:
        problems.append(f"word_embedding_dimension {dimension} != hidden_size {hidden}")
    if problems:
        raise ValueError(f"Snapshot pooling disagrees with the family: {problems}")
    return dimension


# --- exporter ---


def read_facts(family: Qwen3EmbeddingFamily, snapshot: Path) -> CheckpointFacts:
    """The snapshot's facts: the dimension from the pooling config, eos from ``config.json``,
    bos (none) and pad from the tokenizer, and the family's ``max_tokens``.

    Raises:
        ValueError: if the pooling config disagrees with the family (``read_pooling_dimension``).
    """
    from transformers import AutoTokenizer

    dimension = read_pooling_dimension(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    assert family.max_tokens is not None
    return CheckpointFacts(
        dimension=dimension,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=int(config["eos_token_id"]),
        pad_token_id=tokenizer.pad_token_id,
        max_tokens=family.max_tokens,
    )


def export_programs(
    family: Qwen3EmbeddingFamily,
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
    """Write the ``text_encoder`` asset to ``out`` and return ``(assets, embedding block)``.

    The model has only a text tower, so ``text_only`` changes nothing. The prompts are checked
    before converting.

    Raises:
        ValueError: if the snapshot's prompts disagree with the family.
    """
    from coreai_models.models.macos.qwen3_embedding import load_qwen3_embedding_encoder

    prompts, default_prompt = read_prompts(snapshot)
    block = embedding_block(
        family,
        facts,
        revision=revision,
        prompts=prompts,
        default_prompt_name=default_prompt,
        compute_precision=compute_precision,
    )
    fetch_weights(family, snapshot)
    encoder = load_qwen3_embedding_encoder(snapshot, dtype=TORCH_DTYPES[compute_precision])
    spec = block["components"][TEXT_COMPONENT]
    functions = [(encoder, {**spec, "functions": [entry]}) for entry in spec["functions"]]
    save_program(export_program(functions, include_debug_info), out, family, TEXT_COMPONENT)
    return {TEXT_COMPONENT: f"{TEXT_COMPONENT}.aimodel"}, block


def exporter(family: Qwen3EmbeddingFamily) -> EmbeddingExporter:
    return EmbeddingExporter(
        family=family,
        read_facts=partial(read_facts, family),
        export_programs=partial(export_programs, family),
    )


QWEN3_EMBEDDING_EXPORTER = exporter(QWEN3_EMBEDDING_0_6B)
