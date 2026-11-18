#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Standalone export recipe for Qwen3.5 2B / 4B (text backbone), iOS.

Qwen3.5 is a hybrid decoder — three Gated-DeltaNet layers per full-attention layer — so it
carries two pieces of mutable state beyond the KV cache and does not fit the generic
``coreai.llm.export`` pipeline. This recipe is the only iOS path for Qwen3.5; see
``models/qwen3_5/README.md``.

Runs in the workspace environment, whose ``transformers`` carries
``transformers.models.qwen3_5``:

    cd models/qwen3_5
    uv run export.py --model Qwen/Qwen3.5-2B --max-context-length 32768

The export is a ladder of static programs, one per (context bucket, family), named
``extend_{ctx}_{q}`` / ``prompt_opt_{ctx}_{q}``. Each family has one query length; see
``DECODE_QUERY_LEN`` / ``PREFILL_QUERY_LEN`` in ``models/ios/qwen3_5.py``.

The graph contract, beyond the shared iOS one:

* ``conv_state`` — the DeltaNet short-conv ring, rank 4 and context-independent, addressed by
  two extra runtime inputs (``conv_step``, ``conv_seed_selector``).
* ``recurrent_state`` — the delta rule's matrix-valued state, rank 4 and context-independent.
* ``ssm_update_flag`` — 1.0 only where a slot holds a real, not-yet-processed token. A block's
  start is aligned down, so a call re-sends already-processed tokens; the KV cache is
  position-addressed and idempotent under that, but a recurrent state is not.
* the KV cache is sized by the full-attention layers.
* every cache constraint is keyed to its own bucket's ``ctx``, so the runner can size each
  bucket's cache separately.
* Compression is k-means palettization driven by a coreai-opt YAML, one per checkpoint
  (``IOS_COMPRESSION_CONFIGS``).

Developer overrides (comma-separated flags): ``--dev-extend-qlens`` / ``--dev-prompt-qlens``
(per-family query lengths), ``--dev-ladder-only`` (restrict the context buckets). See
``DevOverrides``.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from coreai.authoring import AIProgram
from coreai.authoring.types import AllocationType, HardwareConstraints
from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
from coreai_torch import TorchConverter
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    DEFAULT_INCLUDE_DEBUG_INFO,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    POSITION_IDS_INPUT_NAME,
    PROMPT_OPT_FUNCTION_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
)
from coreai_models.export.bundle import bundle_llm_asset
from coreai_models.export.compression import palettize_pytorch_model
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.models.ios.qwen3_5 import (
    CONV_SEED_SELECTOR_INPUT_NAME,
    CONV_STATE_INPUT_NAME,
    CONV_STEP_INPUT_NAME,
    DECODE_QUERY_LEN,
    PREFILL_QUERY_LEN,
    RECURRENT_STATE_INPUT_NAME,
    SSM_UPDATE_FLAG_INPUT_NAME,
    Qwen3_5ForCausalLMForiOS,
)
from coreai_models.primitives.ios.sdpa import BlockedSDPA
from coreai_models.primitives.ios.ssm_cache import MAX_QUERY_LEN

KV_CACHE_INTERLEAVE_FACTOR = BaseForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR

logger = logging.getLogger("qwen3_5.export")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Checkpoints whose exports have been accuracy- and performance-verified, each with the
# palettization recipe it ships with. The 2B and 4B take different group sizes; see the yaml
# headers. Other checkpoints are accepted with a warning and need an explicit
# --compression-config: they may trace and export, but nothing about the result is verified.
IOS_COMPRESSION_CONFIGS = {
    "Qwen/Qwen3.5-2B": Path(__file__).resolve().parent / "qwen3_5_2b_4bit_palettized.yaml",
    "Qwen/Qwen3.5-4B": Path(__file__).resolve().parent / "qwen3_5_4b_4bit_palettized.yaml",
}
SUPPORTED_MODELS = tuple(IOS_COMPRESSION_CONFIGS)

PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Shipping context ladder: a sparse set of buckets. The runner picks the smallest bucket past
# the current position, so over-provisioning costs only the attention layers' KV — the DeltaNet
# state is fixed-size — while every extra bucket costs a whole program to compile and load.
SHIPPING_CONTEXT_LADDER = [256, 1024, 8192, 32768]

# Largest supported context, limited by the uint16 position_ids input.
IOS_MAX_CONTEXT_LENGTH = SHIPPING_CONTEXT_LADDER[-1]


# ===========================================================================
# Shared helpers
# ===========================================================================


def _resolve_eos_token_ids(hf_model_id: str, config: Any) -> list[int]:
    """Collect eos token ids from the generation config, tokenizer and model config."""
    ids: list[int] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                _add(v)
        elif isinstance(value, int) and value not in ids:
            ids.append(value)

    # A checkpoint without generation_config.json falls back to the tokenizer and model
    # config; any other failure (network, a corrupt file) raises, so a bundle never ships
    # missing the chat terminator.
    try:
        _add(GenerationConfig.from_pretrained(hf_model_id).eos_token_id)
    except OSError as exc:
        if not _is_missing_file_error(exc):
            raise
        logger.info("No generation config; taking eos tokens from the tokenizer and model config.")

    _add(AutoTokenizer.from_pretrained(hf_model_id).eos_token_id)

    _add(getattr(config, "eos_token_id", None))
    return ids


def _is_missing_file_error(exc: OSError) -> bool:
    """Whether ``exc`` is transformers reporting that the checkpoint lacks the file."""
    return isinstance(exc, FileNotFoundError) or "does not appear to have a file named" in str(exc)


def _ios_metadata_extras() -> dict[str, Any]:
    """Extra ``language`` metadata keys the iOS runner needs.

    ``states`` is read through the model class's export contract, so the pin the runner sees
    cannot drift from the states the graph declares.
    """
    return {"states": Qwen3_5ForCausalLMForiOS.export_state_classification()}


def _patch_language_metadata(
    bundle_path: Path,
    hf_model_id: str,
    config: Any,
    extras: dict[str, Any] | None = None,
) -> None:
    """Add Qwen3.5-specific keys to the ``language`` block of a written bundle.

    ``bundle_llm_asset`` writes the generic 0.2-schema metadata; the model-specific keys are
    merged in afterwards.
    """
    patch: dict[str, Any] = dict(extras or {})

    eos_token_ids = _resolve_eos_token_ids(hf_model_id, config)
    if eos_token_ids:
        patch["eos_token_ids"] = eos_token_ids

    if not patch:
        return

    metadata_path = bundle_path / "metadata.json"
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    metadata["language"].update(patch)
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Recorded {sorted(patch)} in {metadata_path}")


def _default_output_name(hf_model_id: str, compression: str) -> str:
    """Bundle name: lowercased HF tail + compression recipe + ``_static``.

    Mirrors ``coreai_models.export.pipeline._generate_output_name``.
    """
    tail = hf_model_id.split("/")[-1]
    base = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    if compression == "none":
        return f"{base}_static"
    already_prefixed = compression == base or compression.startswith(f"{base}_")
    return f"{compression if already_prefixed else f'{base}_{compression}'}_static"


def _resolve_bundle_paths(output_dir: str, output_name: str, overwrite: bool) -> tuple[Path, Path]:
    """Create the bundle directory and check for an existing asset. Returns (bundle, asset).

    An existing asset is only removed right before the new one is saved (see ``_export_ios``),
    so a rejected flag or a failed export keeps the previous bundle.
    """
    bundle_path = Path(output_dir) / output_name
    aimodel_path = bundle_path / f"{output_name}.aimodel"
    if aimodel_path.exists() and not overwrite:
        raise SystemExit(f"{aimodel_path} already exists. Use --overwrite to replace it.")
    bundle_path.mkdir(parents=True, exist_ok=True)
    return bundle_path, aimodel_path


def _load_compression_yaml(yaml_path: Path):  # type: ignore[no-untyped-def]
    """Load a coreai-opt compression YAML and validate it.

    iOS compresses via k-means palettization, so the YAML's single top-level key must be
    ``kmeans_palettization_config``. Returned as a prebuilt ``KMeansPalettizerConfig``.
    """
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or set(data) != {"kmeans_palettization_config"}:
        raise SystemExit(
            f"{yaml_path}: expected a single top-level key 'kmeans_palettization_config'."
        )
    return KMeansPalettizerConfig.from_dict(data)


# ===========================================================================
# The ladder
# ===========================================================================


def _int_list(raw: str) -> tuple[int, ...]:
    """argparse type for a comma-separated positive-int list."""
    try:
        values = tuple(int(x) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"values must be positive, got {list(values)}")
    return values


@dataclass(frozen=True)
class DevOverrides:
    """Developer-only narrowing of the ladder via the ``--dev-*`` flags."""

    extend_qlens: tuple[int, ...] | None = None
    prompt_qlens: tuple[int, ...] | None = None
    ladder_only: tuple[int, ...] | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DevOverrides":
        return cls(
            extend_qlens=args.dev_extend_qlens,
            prompt_qlens=args.dev_prompt_qlens,
            ladder_only=args.dev_ladder_only,
        )

    def log_if_set(self) -> None:
        """Log any narrowing, so a debug artifact is never mistaken for a shipping one."""
        for label, value in (
            ("extend query lengths", self.extend_qlens),
            ("prompt query lengths", self.prompt_qlens),
            ("context buckets", self.ladder_only),
        ):
            if value:
                logger.warning(f"DEV OVERRIDE: {label} restricted to {list(value)}")


# Shared default for the `dev` parameters: no narrowing. Safe to share since DevOverrides is frozen.
NO_DEV_OVERRIDES = DevOverrides()


def _extend_query_lengths(dev: DevOverrides = NO_DEV_OVERRIDES) -> list[int]:
    """Q-lengths to specialize the decode (``extend_*``) functions for.

    ``--dev-extend-qlens`` overrides; defaults to :data:`DECODE_QUERY_LEN`.
    """
    return list(dev.extend_qlens or (DECODE_QUERY_LEN,))


def _prompt_query_lengths(dev: DevOverrides = NO_DEV_OVERRIDES) -> list[int]:
    """Q-lengths to specialize the prefill (``prompt_opt_*``) functions for.

    ``--dev-prompt-qlens`` overrides; defaults to :data:`PREFILL_QUERY_LEN`.
    """
    return list(dev.prompt_qlens or (PREFILL_QUERY_LEN,))


def _flash_block_size(model: Qwen3_5ForCausalLMForiOS) -> int | None:
    """Flash chunk width the attention layers were built with, for the log line.

    ``None`` when ``--num-layers`` leaves no full-attention layer; the hybrid pattern leads
    with DeltaNet layers.
    """
    for module in model.extend.modules():
        if isinstance(module, BlockedSDPA):
            return module.block_size
    return None


def context_ladder(max_context_length: int, dev: DevOverrides = NO_DEV_OVERRIDES) -> list[int]:
    """Context buckets: the :data:`SHIPPING_CONTEXT_LADDER` entries below the smallest power of
    two covering ``max_context_length``, then that power of two.

    ``--dev-ladder-only`` emits only those buckets.
    """
    ctx_max = 1
    while ctx_max < max_context_length:
        ctx_max *= 2
    buckets = sorted({b for b in SHIPPING_CONTEXT_LADDER if b < ctx_max} | {ctx_max})

    if dev.ladder_only:
        buckets = [b for b in buckets if b in set(dev.ladder_only)]
    return buckets


def _ios_decomp_table():
    """iOS decomposition table: keep ``silu`` undecomposed."""
    decomp_table = torch.export.default_decompositions()
    decomp_table.pop(torch.ops.aten.silu.default)
    decomp_table.pop(torch.ops.aten.silu.out)
    return decomp_table


def _forward_inputs(
    model: Qwen3_5ForCausalLMForiOS, config: Any, ctx: int, query_len: int
) -> dict[str, torch.Tensor]:
    """Reference inputs for one ``(ctx, query_len)`` program, in ``forward`` order.

    Order must match ``Qwen3_5Extend.forward``; the converter binds names positionally.
    """
    cls = Qwen3_5ForCausalLMForiOS
    dims = cls._deltanet_dims(config)

    embedding_table = model.load_embeddings.embedding_table
    input_ids = torch.randint(1, config.vocab_size, (1, query_len), dtype=torch.int32)
    transformer_input = model.gather_embeddings(input_ids, embedding_table)

    # Only the full-attention layers cache K/V.
    key_cache = torch.zeros(
        dims["n_attn"], 1, cls._kv_embed_size(config), 1, ctx, dtype=torch.float16
    )
    in_step = torch.zeros((1,), dtype=torch.int32)

    return {
        TRANSFORMER_INPUT_NAME: transformer_input,
        # uint16 caps the supported context at 65535; see IOS_MAX_CONTEXT_LENGTH.
        POSITION_IDS_INPUT_NAME: torch.arange(query_len).to(torch.uint16).unsqueeze(0),
        IN_STEP_INPUT_NAME: in_step,
        CAUSAL_MASK_INPUT_NAME: torch.zeros(1, ctx, 1, query_len, dtype=torch.float16),
        SSM_UPDATE_FLAG_INPUT_NAME: torch.ones(1, query_len, dtype=torch.float16),
        CONV_STEP_INPUT_NAME: in_step.clone(),
        CONV_SEED_SELECTOR_INPUT_NAME: cls._seed_selector(config, query_len),
        KEY_CACHE_INPUT_NAME: key_cache,
        VALUE_CACHE_INPUT_NAME: key_cache.clone(),
        CONV_STATE_INPUT_NAME: cls._conv_state(config),
        RECURRENT_STATE_INPUT_NAME: cls._recurrent_state(config),
        EMBEDDING_TABLE_INPUT_NAME: embedding_table,
    }


EMBEDDING_TABLE_CONSTRAINT = HardwareConstraints(
    AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
)

# Both DeltaNet states are context-independent, so one constraint serves every bucket.
# DeltaNet states are rank 4, so the interleaved channel axis is index 1.
DELTA_STATE_CONSTRAINT = HardwareConstraints(
    AllocationType.IOSurface,
    interleave=[1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
    alignments=[1, 1, 1, 1, 1],
)


def forward_constraints(ctx: int) -> dict[str, HardwareConstraints]:
    """Layout constraints for one bucket's transformer entrypoint.

    The KV seq-stride alignment uses this bucket's ``ctx``, matching the runner's per-bucket
    allocation.
    """
    cache = HardwareConstraints(
        AllocationType.IOSurface,
        interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
        alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * ctx, 1],
    )
    return {
        EMBEDDING_TABLE_INPUT_NAME: EMBEDDING_TABLE_CONSTRAINT,
        KEY_CACHE_INPUT_NAME: cache,
        VALUE_CACHE_INPUT_NAME: cache,
        CONV_STATE_INPUT_NAME: DELTA_STATE_CONSTRAINT,
        RECURRENT_STATE_INPUT_NAME: DELTA_STATE_CONSTRAINT,
    }


def _export_ladder_programs(
    model: Qwen3_5ForCausalLMForiOS,
    config: Any,
    buckets: list[int],
    dev: DevOverrides = NO_DEV_OVERRIDES,
) -> dict[str, torch.export.ExportedProgram]:
    """Trace the transformer once per ``(bucket, family)``, fully static.

    Fully static: every loop is unrolled over concrete ``ctx`` and ``query_len``.
    """
    decomp_table = _ios_decomp_table()
    families = (
        (EXTEND_FUNCTION_NAME, _extend_query_lengths(dev), False),
        (PROMPT_OPT_FUNCTION_NAME, _prompt_query_lengths(dev), True),
    )

    programs: dict[str, torch.export.ExportedProgram] = {}
    with torch.no_grad():
        for ctx in buckets:
            for prefix, q_lens, prefill in families:
                for q_len in q_lens:
                    name = f"{prefix}_{ctx}_{q_len}"
                    logger.info(f"Exporting {name}...")
                    model.set_prefill_mode(prefill)
                    program = torch.export.export(
                        model.extend,
                        args=(),
                        kwargs=_forward_inputs(model, config, ctx, q_len),
                        dynamic_shapes=None,
                    )
                    program = program.run_decompositions(decomp_table)
                    remove_functionalization(program)
                    programs[name] = program
    model.set_prefill_mode(False)
    return programs


def _export_aux_programs(
    model: Qwen3_5ForCausalLMForiOS, config: Any, max_context_length: int
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram]:
    """Trace ``(gather_embeddings, load_embeddings)`` — both bucket-independent.

    These keep a dynamic sequence dim: they carry no context axis and no unrolled loop, so one
    program specialized over the query lengths serves the whole ladder.
    """
    embedding_table = model.load_embeddings.embedding_table
    input_ids = torch.randint(1, config.vocab_size, (1, DECODE_QUERY_LEN), dtype=torch.int32)
    seq_len_dim = torch.export.Dim("seq_len", max=max_context_length)
    with torch.no_grad():
        logger.info("Exporting gather_embeddings module...")
        gather_program = torch.export.export(
            model.gather_embeddings,
            args=(input_ids, embedding_table),
            dynamic_shapes={"input_ids": {1: seq_len_dim}, "embedding_table": None},
        )
        logger.info("Exporting load_embeddings module...")
        load_program = torch.export.export(model.load_embeddings, args=tuple())
    return gather_program, load_program


async def _convert_ladder_to_coreai(
    ladder_programs: dict[str, torch.export.ExportedProgram],
    gather_program: torch.export.ExportedProgram,
    load_program: torch.export.ExportedProgram,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert the ladder to a single multi-function ``AIProgram``.

    Each ladder program is registered under its own already-specialized name, so the
    transformer entrypoints take no static shape config — they are static by construction.
    They do take hardware constraints, and those are per-bucket: see
    :func:`forward_constraints`. All entrypoints share one copy of the weights.
    """
    mode = TorchConverter.Mode.DEBUG if include_debug_info else TorchConverter.Mode.RELEASE
    converter = TorchConverter(mode=mode)
    register_custom_torch_lowering(converter)

    # The graph I/O contract lives on the model class, so the recipe and the eager model
    # cannot drift apart. Each name hook is keyed by graph; the ladder registers every
    # bucket's pair of entrypoints against the single EXTEND entry, since all rungs share one
    # signature and differ only in shape.
    names = Qwen3_5ForCausalLMForiOS.export_input_names()
    states = Qwen3_5ForCausalLMForiOS.export_state_names()
    outputs = Qwen3_5ForCausalLMForiOS.export_output_names()

    converter.add_exported_program(
        load_program,
        input_names=list(names[LOAD_EMBEDDINGS_FUNCTION_NAME]),
        output_names=list(outputs[LOAD_EMBEDDINGS_FUNCTION_NAME]),
        entrypoint_name=LOAD_EMBEDDINGS_FUNCTION_NAME,
    )
    converter.add_exported_program(
        gather_program,
        input_names=list(names[GATHER_EMBEDDINGS_FUNCTION_NAME]),
        output_names=list(outputs[GATHER_EMBEDDINGS_FUNCTION_NAME]),
        entrypoint_name=GATHER_EMBEDDINGS_FUNCTION_NAME,
    )
    for name, program in ladder_programs.items():
        converter.add_exported_program(
            program,
            input_names=list(names[EXTEND_FUNCTION_NAME]),
            state_names=list(states[EXTEND_FUNCTION_NAME]),
            output_names=list(outputs[EXTEND_FUNCTION_NAME]),
            entrypoint_name=name,
        )

    coreai_program: AIProgram = converter.to_coreai()

    # Constraints first: setting a static shape config specializes the entrypoint, after
    # which it can no longer be addressed by its original name.
    for name in (LOAD_EMBEDDINGS_FUNCTION_NAME, GATHER_EMBEDDINGS_FUNCTION_NAME):
        coreai_program.set_hardware_constraints(
            name, {EMBEDDING_TABLE_INPUT_NAME: EMBEDDING_TABLE_CONSTRAINT}
        )
    q_lens = sorted({int(name.rsplit("_", 1)[1]) for name in ladder_programs})
    coreai_program.set_static_shape_config(
        GATHER_EMBEDDINGS_FUNCTION_NAME,
        {f'"{q}"': {TOKEN_IDS_INPUT_NAME: (1, q)} for q in q_lens},
    )
    for name in ladder_programs:
        ctx = int(name.rsplit("_", 2)[1])
        coreai_program.set_hardware_constraints(name, forward_constraints(ctx))

    return coreai_program


async def _export_ladder(
    model: Qwen3_5ForCausalLMForiOS,
    config: Any,
    max_context_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    dev: DevOverrides = NO_DEV_OVERRIDES,
) -> AIProgram:
    """Export Qwen3.5 as a per-bucket static ladder ``AIProgram``."""
    buckets = context_ladder(max_context_length, dev)
    dims = Qwen3_5ForCausalLMForiOS._deltanet_dims(config)
    logger.info(
        f"Static context ladder: contexts={buckets} "
        f"layers={dims['n_delta']} DeltaNet + {dims['n_attn']} full-attention "
        f"conv ring={dims['ring_width']} flash block_size={_flash_block_size(model)}"
    )

    ladder_programs = _export_ladder_programs(model, config, buckets, dev)
    gather_program, load_program = _export_aux_programs(model, config, max_context_length)
    logger.info(f"Traced {len(ladder_programs) + 2} functions.")

    return await _convert_ladder_to_coreai(
        ladder_programs=ladder_programs,
        gather_program=gather_program,
        load_program=load_program,
        include_debug_info=include_debug_info,
    )


def _build_palettization_inputs(
    model: Qwen3_5ForCausalLMForiOS, config: Any, max_context_length: int
) -> tuple[Any, ...]:
    """Example args for the torch palettizer, in ``model.forward`` order.

    Palettization runs on the eager top-level module, before tracing, so it wants ``forward``'s
    own signature, which starts from token ids and carries no embedding table. The cache is
    sized at the smallest bucket.
    """
    ctx = min(SHIPPING_CONTEXT_LADDER[0], max_context_length)
    graph_inputs = _forward_inputs(model, config, ctx, DECODE_QUERY_LEN)
    input_ids = torch.randint(1, config.vocab_size, (1, DECODE_QUERY_LEN), dtype=torch.int32)
    return (
        input_ids,
        graph_inputs[POSITION_IDS_INPUT_NAME],
        graph_inputs[IN_STEP_INPUT_NAME],
        graph_inputs[CAUSAL_MASK_INPUT_NAME],
        graph_inputs[SSM_UPDATE_FLAG_INPUT_NAME],
        graph_inputs[CONV_STEP_INPUT_NAME],
        graph_inputs[CONV_SEED_SELECTOR_INPUT_NAME],
        graph_inputs[KEY_CACHE_INPUT_NAME],
        graph_inputs[VALUE_CACHE_INPUT_NAME],
        graph_inputs[CONV_STATE_INPUT_NAME],
        graph_inputs[RECURRENT_STATE_INPUT_NAME],
    )


async def _export_ios(args: argparse.Namespace) -> str:
    hf_model_id: str = args.model
    target_dtype = PRECISIONS[args.compute_precision]
    max_ctx = args.max_context_length or IOS_MAX_CONTEXT_LENGTH
    dev = DevOverrides.from_args(args)
    dev.log_if_set()
    if args.num_layers is not None:
        logger.warning(
            f"DEV OVERRIDE: exporting only the first {args.num_layers} decoder layers; "
            "the result is not the checkpoint."
        )

    # Resolved before any weights are loaded so a bad recipe or an existing bundle fails fast.
    palettization_config = None
    if args.compression_config is not None:
        palettization_config = _load_compression_yaml(args.compression_config)
        compression = args.compression_config.stem
    else:
        compression = "none"
    hf_config = AutoConfig.from_pretrained(hf_model_id)
    # The checkpoint is a VLM; report fields from its text config.
    text_config = (
        hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
    )
    native_max_ctx = getattr(text_config, "max_position_embeddings", None)
    if native_max_ctx is not None and max_ctx > native_max_ctx:
        raise SystemExit(
            f"--max-context-length ({max_ctx}) exceeds the model's "
            f"max_position_embeddings ({native_max_ctx}). Choose a value <= {native_max_ctx}."
        )

    output_name = args.output_name or _default_output_name(hf_model_id, compression)
    bundle_path, aimodel_path = _resolve_bundle_paths(args.output_dir, output_name, args.overwrite)

    logger.info(f"Loading {hf_model_id} (iOS, dtype={target_dtype}, max_ctx={max_ctx})...")

    # Move loaded weights to disk-backed mmap tensors so the OS can evict weight pages during
    # palettization and the long ladder conversion. The temp dir must outlive every read of the
    # weights, so it wraps the whole model lifetime (through ``del model``).
    with tempfile.TemporaryDirectory(prefix="qwen3_5_export_") as temp_dir:
        model = Qwen3_5ForCausalLMForiOS.from_hf(
            hf_model_id,
            max_context_length=max_ctx,
            target_dtype=target_dtype,
            mmap_path=os.path.join(temp_dir, "weights"),
            num_layers=args.num_layers,
        ).eval()

        # The bundle advertises the requested max context. The top bucket is that value rounded
        # up to a power of two (see `context_ladder`), so the runner never asks past a bucket.
        text_config.max_position_embeddings = max_ctx

        # All shapes are derived from the reauthored config (truncated by --num-layers).
        config = model.config

        # ---- Palettization (skipped for --compression none) ----
        if palettization_config is not None:
            logger.info(f"Applying palettization ({compression})...")
            inputs = _build_palettization_inputs(model, config, max_ctx)
            model = cast(
                Qwen3_5ForCausalLMForiOS,
                palettize_pytorch_model(model, inputs, palettization_config),
            )

        # ---- Ladder export ----
        coreai_program = await _export_ladder(
            model,
            config,
            max_ctx,
            include_debug_info=args.include_debug_info,
            dev=dev,
        )

        del model

        if aimodel_path.exists():
            shutil.rmtree(aimodel_path)
        logger.info(f"Saving model to {aimodel_path}...")
        await asyncio.to_thread(
            coreai_program.save_asset, aimodel_path, build_aimodel_metadata(hf_model_id)
        )

        bundle_llm_asset(
            bundle_path=bundle_path,
            hf_model_id=hf_model_id,
            hf_config=text_config,
            compression=compression,
            name=output_name,
        )
        _patch_language_metadata(bundle_path, hf_model_id, text_config, _ios_metadata_extras())

    logger.info(f"Export complete: {bundle_path}")
    return str(bundle_path)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Qwen3.5 hybrid DeltaNet decoder to a Core AI bundle (iOS).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID. Verified on: " + ", ".join(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--platform",
        choices=["iOS"],
        default="iOS",
        help="Target platform (iOS only).",
    )
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument(
        "--compression-config",
        type=Path,
        default=None,
        help="coreai-opt compression YAML: a 'kmeans_palettization_config' recipe "
        "(default: the recipe for --model, alongside this script)",
    )
    compression_group.add_argument(
        "--compression",
        choices=["none"],
        default=None,
        help="Only 'none' is accepted, which exports at full precision. Pass "
        "--compression-config <yaml> for a custom recipe.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float16"],
        default=None,
        help="Compute precision for the model weights (iOS requires float16)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help=f"Maximum context length (default and max: {IOS_MAX_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Export only the first N decoder layers. A dev-loop knob: it produces a model "
        "that is not the checkpoint, but exports in a fraction of the time.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "exports"),
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Custom bundle name (without extension)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing bundle")
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help="Embed debug information in the exported .aimodel for debugging a conversion. "
        "Default: off, which embeds minimum debug information and makes the exported asset "
        "smaller.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")

    # Developer-only narrowing. A full ladder traces every context bucket at every
    # query length; these cut that down so one rung can be debugged in minutes.
    # Shipping exports pass none of them; see `DevOverrides`.
    dev_group = parser.add_argument_group("developer overrides")
    dev_group.add_argument(
        "--dev-extend-qlens",
        type=_int_list,
        metavar="N[,N...]",
        help="Query lengths to specialize the decode (extend_*) functions for, "
        f"instead of the shipping [{DECODE_QUERY_LEN}].",
    )
    dev_group.add_argument(
        "--dev-prompt-qlens",
        type=_int_list,
        metavar="N[,N...]",
        help="Query lengths to specialize the prefill (prompt_opt_*) functions for, "
        f"instead of the shipping [{PREFILL_QUERY_LEN}].",
    )
    dev_group.add_argument(
        "--dev-ladder-only",
        type=_int_list,
        metavar="CTX[,CTX...]",
        help="Emit only these context buckets. Each must be in the ladder for the "
        "requested --max-context-length.",
    )
    return parser


def _resolve_platform_defaults(args: argparse.Namespace) -> None:
    """Fill in the defaults and reject invalid combinations."""
    if args.compute_precision is None:
        args.compute_precision = "float16"

    if args.compute_precision != "float16":
        raise SystemExit(
            f"--platform iOS requires --compute-precision float16 (got '{args.compute_precision}')."
        )
    if args.max_context_length is not None and args.max_context_length > IOS_MAX_CONTEXT_LENGTH:
        raise SystemExit(
            f"--platform iOS supports at most {IOS_MAX_CONTEXT_LENGTH} tokens "
            f"(got {args.max_context_length}); `position_ids` is a uint16 input, so a larger "
            "context needs precomputed RoPE inputs instead."
        )

    dev = DevOverrides.from_args(args)
    if dev.ladder_only:
        ladder = context_ladder(args.max_context_length or IOS_MAX_CONTEXT_LENGTH)
        unknown = sorted(set(dev.ladder_only) - set(ladder))
        if unknown:
            raise SystemExit(
                f"--dev-ladder-only {unknown} not in the context ladder {ladder} for "
                "this --max-context-length."
            )
    for flag, q_lens in (
        ("--dev-extend-qlens", _extend_query_lengths(dev)),
        ("--dev-prompt-qlens", _prompt_query_lengths(dev)),
    ):
        for q_len in q_lens:
            if q_len > MAX_QUERY_LEN:
                # The conv ring is sized from MAX_QUERY_LEN, because that bounds how many
                # already-processed tokens a block-aligned start can re-send.
                raise SystemExit(
                    f"{flag}={q_len} exceeds MAX_QUERY_LEN={MAX_QUERY_LEN}; raise it in "
                    "primitives/ios/ssm_cache.py so the conv ring keeps up."
                )

    if args.compression is None and args.compression_config is None:
        default = IOS_COMPRESSION_CONFIGS.get(args.model)
        if default is None:
            raise SystemExit(
                f"No default compression recipe for '{args.model}'; the shipping recipes are "
                f"per checkpoint ({', '.join(SUPPORTED_MODELS)}). Pass --compression-config "
                "<yaml>, or --compression none to export at full precision."
            )
        args.compression_config = default

    if args.compression_config is not None and not args.compression_config.is_file():
        raise SystemExit(f"--compression-config: file not found: {args.compression_config}")


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    _resolve_platform_defaults(args)

    if args.model not in SUPPORTED_MODELS:
        logger.warning(
            f"{args.model} is not one of the verified checkpoints "
            f"({', '.join(SUPPORTED_MODELS)}). Exporting anyway, but the accuracy and "
            "performance of the result have not been verified."
        )

    asyncio.run(_export_ios(args))


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise SystemExit("Qwen3.5 export requires Python 3.11+")
    main()
