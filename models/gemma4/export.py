#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Standalone export recipe for the Gemma 4 PLE variants (E2B / E4B), iOS.

The Per-Layer-Embeddings variants don't fit the generic ``coreai.llm.export``
pipeline, for two reasons: each context bucket needs its own statically traced
graph, and the PLE table has to be dumped as a sidecar. So they ship as a standalone
recipe (see ``models/gemma4/README.md``):

    cd models/gemma4
    uv run export.py --model google/gemma-4-E2B-it --max-context-length 32768

The export follows ``coreai_models.export.ios``, except that each transformer
(context bucket, query length) pair is traced as its own fully static program,
``extend_{ctx}_{q}`` / ``prompt_opt_{ctx}_{q}``: ``BlockedSDPA`` unrolls its block
loop, so the graph depends on the context length. The gather is traced and
specialized as ``ios.py`` does it. A flat global KV cache pairs with a fixed-depth
sliding-window ring, RoPE arrives precomputed as ``rope_cos``/``rope_sin``, and the
INT8 Per-Layer Embeddings table is written as a sidecar next to the asset.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from coreai.authoring import AIProgram
from coreai_torch import TorchConverter
from transformers import AutoConfig, GenerationConfig

from coreai_models._constants import (
    DEFAULT_INCLUDE_DEBUG_INFO,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    PROMPT_OPT_FUNCTION_NAME,
    TRANSFORMER_INPUT_NAME,
)
from coreai_models.export.bundle import bundle_llm_asset
from coreai_models.export.compression import (
    palettize_pytorch_model,
)
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.export.pipeline import ExportConfig, _generate_output_name
from coreai_models.llm.export import _load_compression_config_object
from coreai_models.models.base import TraceSpec
from coreai_models.models.ios.gemma4_text import (
    Gemma4ForCausalLMForiOS,
)

logger = logging.getLogger("gemma4.export")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IOS_COMPRESSION_CONFIG = (
    Path(__file__).resolve().parent / "4bit_palettized.yaml"
)

# Checkpoints whose exports have been accuracy- and performance-verified. These
# are the ones the model card advertises as supported. Other Gemma 4 checkpoints
# are accepted with a warning -- they may well trace and export, but nothing about
# the resulting artifact has been verified.
SUPPORTED_MODELS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
)

# Gemma 4 checkpoints are multimodal: the text decoder is nested under
# `text_config`.
HF_CONFIG_ATTR = "text_config"


# ===========================================================================
# Shared helpers
# ===========================================================================


def _resolve_eos_token_ids(hf_model_id: str, text_config: Any) -> list[int]:
    """Collect end-of-generation token ids from the generation config.

    The tokenizer exposes only a single ``eos_token`` (``<eos>``), but Gemma chat
    models also stop on ``<end_of_turn>``. Carry the full ``generation_config``
    eos list into the bundle metadata so the runner halts cleanly. Falls back to
    the model config's ``eos_token_id``. Always returns a de-duplicated list.
    """
    ids: list[int] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                _add(v)
        elif isinstance(value, int) and value not in ids:
            ids.append(value)

    try:
        _add(GenerationConfig.from_pretrained(hf_model_id).eos_token_id)
    except OSError as exc:  # no generation_config.json in the checkpoint
        logger.warning(f"Could not load generation config for eos tokens: {exc}")

    _add(getattr(text_config, "eos_token_id", None))
    return ids


def _ios_metadata_extras(text_config: Any) -> dict[str, Any]:
    """Extra ``language`` metadata the iOS runner needs, grouped under ``overrides``.

    * ``sliding_window`` -- the runner builds the windowed mask for the sliding
      KV cache from it.
    * ``rope`` -- the iOS graph takes precomputed ``rope_cos``/``rope_sin``
      instead of ``position_ids`` (a 131k position overflows a 16-bit position input,
      and a 32-bit one feeding the RoPE gather fails to
      compile), so the runner needs
      the two head dims, the two RoPE bases, and the global partial-rotary factor
      to build the combined table rows.
    * ``final_logit_softcapping`` -- ``tanh`` is best run on the CPU rather than in the
      graph, so the iOS graph omits the ``c * tanh(logits / c)`` cap and the runner
      applies it on the CPU between reading the logits and sampling.
    """
    try:
        rope = text_config.rope_parameters
        overrides: dict[str, Any] = {
            "sliding_window": text_config.sliding_window,
            "rope": {
                "sliding_head_dim": text_config.head_dim,
                "global_head_dim": text_config.global_head_dim,
                "sliding_rope_theta": rope["sliding_attention"]["rope_theta"],
                "global_rope_theta": rope["full_attention"]["rope_theta"],
                "partial_rotary_factor": rope["full_attention"][
                    "partial_rotary_factor"
                ],
            },
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Gemma 4 config is missing a field the iOS runner needs: {exc!r}"
        ) from exc

    softcap = getattr(text_config, "final_logit_softcapping", None)
    if softcap:
        overrides["final_logit_softcapping"] = float(softcap)
    return {"overrides": overrides}


def _patch_language_metadata(
    bundle_path: Path,
    hf_model_id: str,
    text_config: Any,
    extras: Optional[dict[str, Any]] = None,
    auxiliary_assets: Optional[dict[str, Any]] = None,
) -> None:
    """Add Gemma4-specific keys to a written bundle.

    ``bundle_llm_asset`` writes the generic 0.2-schema metadata; the keys added
    here are Gemma4-only, so they are merged in afterwards rather than
    special-cased inside the shared bundler. ``extras`` lands in the ``language``
    block; ``auxiliary_assets`` lands in the top-level
    ``auxiliary_assets`` role map. ``assets`` is reserved for Core AI models, so
    sidecar artifacts such as the PLE table belong here instead.
    """
    patch: dict[str, Any] = dict(extras or {})
    asset_patch: dict[str, Any] = dict(auxiliary_assets or {})

    eos_token_ids = _resolve_eos_token_ids(hf_model_id, text_config)
    if eos_token_ids:
        patch["eos_token_ids"] = eos_token_ids

    if not patch and not asset_patch:
        return

    metadata_path = bundle_path / "metadata.json"
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    metadata["language"].update(patch)
    metadata.setdefault("auxiliary_assets", {}).update(asset_patch)
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Recorded {sorted(patch) + sorted(asset_patch)} in {metadata_path}")


def _text_config(hf_model_id: str) -> Any:
    """Load the Gemma 4 text-decoder sub-config."""
    raw_config = AutoConfig.from_pretrained(hf_model_id)
    return getattr(raw_config, HF_CONFIG_ATTR, raw_config)


def _resolve_bundle_paths(
    output_dir: str, output_name: str, overwrite: bool
) -> tuple[Path, Path]:
    """Create the bundle directory, failing fast on an existing asset without
    ``--overwrite``. Returns (bundle, asset). The existing asset is only removed just
    before the new one is saved, so a failed export leaves it in place."""
    bundle_path = Path(output_dir) / output_name
    aimodel_path = bundle_path / f"{output_name}.aimodel"
    if aimodel_path.exists() and not overwrite:
        raise SystemExit(
            f"{aimodel_path} already exists. Use --overwrite to replace it."
        )
    bundle_path.mkdir(parents=True, exist_ok=True)
    return bundle_path, aimodel_path


# ===========================================================================
# iOS: per-context blocked ladder of statically-shaped programs
# ===========================================================================

# Shipping per-function-type query lengths. Prefill (``prompt_opt``) runs at q=64
# and decode (``extend``) at q=8. The runner prefills in q=64 chunks while more than
# 64 tokens remain and runs the rest, including a prompt of 64 tokens or fewer, as
# q=8 ``extend`` steps. Shipping ``extend`` at q=8 only keeps the extend function
# count at one per context bucket, which keeps the program under the accelerator's
# per-program I/O cap.
SHIPPING_EXTEND_QLENS = [8]
SHIPPING_PROMPT_QLENS = [64]

# Shipping context ladder: a SPARSE set of buckets (<=4 up to 131072). A dense
# power-of-two ladder produces too many functions and blows past the accelerator's
# per-program I/O cap; this sparse ladder keeps every bucket (including 131072)
# accelerator-resident, at the cost of coarser decode-speed tiering.
SHIPPING_CONTEXT_LADDER = [1024, 8192, 32768, 131072]

# Largest context the iOS path supports. `context_ladder` rounds the requested
# length up to the next power of two and emits that as its own bucket, so asking
# for more than the top shipping bucket would synthesize an untested bucket (and
# push the program past the accelerator's per-program I/O cap). This is also the
# default, so an export covers the model's full context unless `--max-context-length`
# asks for less.
IOS_MAX_CONTEXT_LENGTH = SHIPPING_CONTEXT_LADDER[-1]


def context_ladder(max_context_length: int) -> list[int]:
    """Context-length buckets for the flat-cache flash ladder: the sparse
    :data:`SHIPPING_CONTEXT_LADDER` entries that fit under the smallest power of two
    covering ``max_context_length`` (always including that cap).

    Each context size is its own statically-shaped program (the flash block loop is
    unrolled, so ``ceil(ctx / block_size)`` is baked into the graph).
    """
    ctx_max = 1
    while ctx_max < max_context_length:
        ctx_max *= 2
    buckets = [b for b in SHIPPING_CONTEXT_LADDER if b < ctx_max]
    buckets.append(ctx_max)
    buckets = sorted(set(buckets))

    return buckets


def _ios_decomp_table():
    """iOS decomposition table: keep ``silu`` as-is (the accelerator has a fused op)."""
    decomp_table = torch.export.default_decompositions()
    decomp_table.pop(torch.ops.aten.silu.default)
    decomp_table.pop(torch.ops.aten.silu.out)
    return decomp_table


def _reference_inputs(
    model: Gemma4ForCausalLMForiOS, config, spec: TraceSpec
) -> tuple[dict[str, dict], dict[str, Any]]:
    """The model's reference inputs and dynamic shapes for one rung, checked against
    its contract."""
    reference_inputs = model.build_reference_inputs(config, torch.float16, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    model.validate_export_contract(reference_inputs, dynamic_shapes)
    return reference_inputs, dynamic_shapes


def _export_programs(
    model: Gemma4ForCausalLMForiOS,
    config,
    buckets: list[int],
) -> list[tuple[str, str, int, torch.export.ExportedProgram]]:
    """Trace one program per emitted entrypoint.

    Mirrors ``coreai_models.export.ios._export_programs``, except that every
    transformer (context bucket, query length) pair is its own fully static
    program, named ``{entrypoint}_{ctx}_{q}``. Returns
    ``(entrypoint name, contract graph, context bucket, program)`` tuples.

    Every trace is bounded by the top bucket rather than ``--max-context-length``:
    the ladder rounds that up to a power of two, so the top bucket can exceed it.
    """
    ladder_max = buckets[-1]
    decomp_table = _ios_decomp_table()
    block_size = model.extend.model.kv_block_size

    def trace(
        module, kwargs: dict, dynamic_shapes=None
    ) -> torch.export.ExportedProgram:
        return torch.export.export(
            module, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )

    programs: list[tuple[str, str, int, torch.export.ExportedProgram]] = []
    with torch.no_grad():
        logger.info(f"Exporting {LOAD_EMBEDDINGS_FUNCTION_NAME}...")
        programs.append(
            (
                LOAD_EMBEDDINGS_FUNCTION_NAME,
                LOAD_EMBEDDINGS_FUNCTION_NAME,
                buckets[0],
                trace(model.load_embeddings, {}),
            )
        )
        logger.info(f"Exporting {GATHER_EMBEDDINGS_FUNCTION_NAME}...")
        ref, dynamic_shapes = _reference_inputs(
            model,
            config,
            TraceSpec(
                max_context_length=ladder_max,
                cache_seq_len=buckets[0],
                query_len=model.IOS_QUERY_LEN,
            ),
        )
        programs.append(
            (
                GATHER_EMBEDDINGS_FUNCTION_NAME,
                GATHER_EMBEDDINGS_FUNCTION_NAME,
                buckets[0],
                trace(
                    model.gather_embeddings,
                    ref[GATHER_EMBEDDINGS_FUNCTION_NAME],
                    dynamic_shapes[GATHER_EMBEDDINGS_FUNCTION_NAME],
                ),
            )
        )

        # The transformer entry is used for both emitted entrypoints.
        for ctx in buckets:
            for entrypoint, prefill, qlens in (
                (EXTEND_FUNCTION_NAME, False, SHIPPING_EXTEND_QLENS),
                (PROMPT_OPT_FUNCTION_NAME, True, SHIPPING_PROMPT_QLENS),
            ):
                model.set_prefill_mode(prefill)
                for q in qlens:
                    name = f"{entrypoint}_{ctx}_{q}"
                    logger.info(
                        f"Exporting {name} "
                        f"(flash chunks={(ctx + block_size - 1) // block_size})..."
                    )
                    ref, _ = _reference_inputs(
                        model,
                        config,
                        TraceSpec(
                            max_context_length=ladder_max,
                            cache_seq_len=ctx,
                            query_len=q,
                        ),
                    )
                    program = trace(model.extend, ref[EXTEND_FUNCTION_NAME])
                    program = program.run_decompositions(decomp_table)
                    remove_functionalization(program)
                    programs.append((name, EXTEND_FUNCTION_NAME, ctx, program))

    return programs


async def _convert_to_coreai(
    model: Gemma4ForCausalLMForiOS,
    programs: list[tuple[str, str, int, torch.export.ExportedProgram]],
    config,
    gather_qlens: set[int],
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert the traced programs to one AIProgram with iOS constraints.

    Mirrors ``coreai_models.export.ios._convert_to_coreai``. Static shapes and
    hardware constraints come from the model's contract for each program's bucket;
    the gather's specializations are narrowed to ``gather_qlens``, the query lengths
    the ladder emits.
    """
    inputs = model.export_input_names()
    states = model.export_state_names()
    outputs = model.export_output_names()

    mode = (
        TorchConverter.Mode.DEBUG if include_debug_info else TorchConverter.Mode.RELEASE
    )
    converter = TorchConverter(mode=mode)
    register_custom_torch_lowering(converter)
    for name, graph, _, program in programs:
        converter.add_exported_program(
            program,
            input_names=list(inputs[graph]),
            state_names=list(states[graph]),
            output_names=list(outputs[graph]),
            entrypoint_name=name,
        )

    coreai_program: AIProgram = converter.to_coreai()

    # Hardware constraints must be set before a graph's static shape config.
    for name, graph, ctx, _ in programs:
        constraints = model.export_hardware_constraints(ctx)[graph]
        if constraints:
            coreai_program.set_hardware_constraints(name, constraints)
        static_shapes = model.export_static_shape_configs(config, ctx)[graph]
        if graph == GATHER_EMBEDDINGS_FUNCTION_NAME:
            static_shapes = {
                label: shapes
                for label, shapes in static_shapes.items()
                if int(label.strip('"')) in gather_qlens
            }
        if static_shapes:
            coreai_program.set_static_shape_config(name, static_shapes)

    return coreai_program


async def _export_blocked_ladder(
    model: Gemma4ForCausalLMForiOS,
    config,
    max_context_length: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Export the Gemma4 model as a per-context blocked-flash ladder AIProgram."""
    buckets = context_ladder(max_context_length)
    logger.info(f"iOS context ladder: {buckets}")
    programs = _export_programs(model, config, buckets)
    gather_qlens = set(SHIPPING_EXTEND_QLENS) | set(SHIPPING_PROMPT_QLENS)
    return await _convert_to_coreai(
        model, programs, config, gather_qlens, include_debug_info
    )


def _palettization_inputs(
    model: Gemma4ForCausalLMForiOS, config, max_context_length: int
) -> tuple:
    """The palettizer's calibration inputs: the smallest rung's reference inputs,
    as ``model.forward``'s positional arguments.

    K-means palettization works on the weights; these only drive the trace, which is
    why zero RoPE rows and random ids and PLE rows are fine.
    """
    ctx = min(SHIPPING_CONTEXT_LADDER[0], max_context_length)
    ref, _ = _reference_inputs(
        model,
        config,
        TraceSpec(
            max_context_length=max_context_length,
            cache_seq_len=ctx,
            query_len=model.IOS_QUERY_LEN,
        ),
    )
    forward_inputs = {
        "input_ids": ref[GATHER_EMBEDDINGS_FUNCTION_NAME]["input_ids"],
        **{
            k: v
            for k, v in ref[EXTEND_FUNCTION_NAME].items()
            if k not in (TRANSFORMER_INPUT_NAME, EMBEDDING_TABLE_INPUT_NAME)
        },
    }
    return model.reference_inputs_as_args(forward_inputs)


async def _export_ios(args: argparse.Namespace) -> str:
    hf_model_id: str = args.model
    target_dtype = torch.float16
    max_ctx = args.max_context_length or IOS_MAX_CONTEXT_LENGTH

    # Palettization comes from a coreai-opt YAML: DEFAULT_IOS_COMPRESSION_CONFIG
    # unless --compression-config overrides it. `--compression none` leaves it
    # unset. Resolved before any weights are loaded so a bad recipe fails fast.
    palettization_config = None
    if args.compression_config is not None:
        palettization_config = _load_compression_config_object(
            args.compression_config, "iOS"
        )
        compression = args.compression_config.stem
    else:
        compression = "none"

    output_name = args.output_name or _generate_output_name(
        ExportConfig(
            hf_model_id=hf_model_id,
            variant="iOS",
            compression=compression,
            compression_config_object=palettization_config,
        )
    )
    bundle_path, aimodel_path = _resolve_bundle_paths(
        args.output_dir, output_name, args.overwrite
    )

    logger.info(
        f"Loading {hf_model_id} (iOS, dtype={target_dtype}, max_ctx={max_ctx})..."
    )

    # Move loaded weights to disk-backed mmap tensors so the OS can evict weight
    # pages during palettization and the long blocked-ladder conversion. The temp
    # dir must outlive every read of the weights, so it wraps the whole model
    # lifetime (through ``del model``); it is cleaned up on scope exit.
    with tempfile.TemporaryDirectory(prefix="gemma4_export_") as temp_dir:
        hf_config = _text_config(hf_model_id)

        # The model's own limits are only known once its config is loaded; check them
        # before any weights are.
        if hf_config.sliding_window != Gemma4ForCausalLMForiOS.SLIDING_WINDOW:
            raise SystemExit(
                f"{hf_model_id} has sliding_window={hf_config.sliding_window}, but the "
                f"iOS export is built for {Gemma4ForCausalLMForiOS.SLIDING_WINDOW}."
            )
        native_max_ctx = getattr(hf_config, "max_position_embeddings", None)
        if native_max_ctx is not None and max_ctx > native_max_ctx:
            raise SystemExit(
                f"--max-context-length ({max_ctx}) exceeds the model's "
                f"max_position_embeddings ({native_max_ctx}). "
                f"Choose a value <= {native_max_ctx}."
            )

        model = Gemma4ForCausalLMForiOS.from_hf(
            hf_model_id,
            max_context_length=max_ctx,
            target_dtype=target_dtype,
            mmap_path=os.path.join(temp_dir, "weights"),
        ).eval()
        hf_config.max_position_embeddings = max_ctx

        # Write the PLE sidecar straight away and drop the table: it is multiple
        # gigabytes and not a module weight, so mmap can't evict it.
        logger.info("Dumping Per-Layer Embeddings (PLE) artifact...")
        ple_path = model.dump_ple_embedding(str(bundle_path), output_name)
        logger.info(f"Wrote PLE artifact to {ple_path}")
        del model._ple_weight

        # ---- Palettization (skipped for --compression none) ----
        if palettization_config is not None:
            logger.info(f"Applying palettization ({compression})...")
            inputs = _palettization_inputs(model, hf_config, max_ctx)
            model = palettize_pytorch_model(model, inputs, palettization_config)

        # ---- Blocked-ladder export ----
        coreai_program = await _export_blocked_ladder(
            model,
            hf_config,
            max_ctx,
            include_debug_info=args.include_debug_info,
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
            hf_config=hf_config,
            compression=compression,
            name=output_name,
        )
        # The iOS runner needs the sliding window, the dual-RoPE table parameters
        # and the soft cap from `language.overrides`; the PLE sidecar is declared under
        # `auxiliary_assets`; `assets` is reserved for Core AI models.
        extras = _ios_metadata_extras(hf_config)
        auxiliary_assets = {"per_layer_embeddings": Path(ple_path).name}
        _patch_language_metadata(
            bundle_path, hf_model_id, hf_config, extras, auxiliary_assets=auxiliary_assets
        )

    logger.info(f"Export complete: {bundle_path}")
    return str(bundle_path)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Gemma 4 text decoder to a Core AI bundle (iOS).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID. Verified on: " + ", ".join(SUPPORTED_MODELS),
    )
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument(
        "--compression-config",
        type=Path,
        default=None,
        help="coreai-opt compression YAML: a 'kmeans_palettization_config' recipe "
        f"(default: {DEFAULT_IOS_COMPRESSION_CONFIG.name}, alongside this script)",
    )
    compression_group.add_argument(
        "--compression",
        choices=["none"],
        default=None,
        help="Only 'none' is accepted, which exports at full precision. The named "
        "presets are not used for Gemma 4: the shipped YAML recipes are "
        "mixed-precision, which the presets cannot express. Pass "
        "--compression-config <yaml> for a custom recipe.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Maximum context length, a power of two "
        f"(default and max: {IOS_MAX_CONTEXT_LENGTH})",
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
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing bundle"
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help="Embed debug information in the exported .aimodel for debugging a conversion. "
        "Default: off, which embeds minimum debug information and makes the exported "
        "asset smaller.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )

    return parser


def _resolve_defaults(args: argparse.Namespace) -> None:
    """Fill in the defaults and reject invalid combinations.

    Compression defaults to the ``kmeans_palettization_config`` YAML shipped
    alongside this script. ``--compression-config <yaml>`` swaps in a custom
    recipe. ``--compression`` only accepts ``none``, which skips compression
    entirely.
    """
    if (
        args.max_context_length is not None
        and args.max_context_length > IOS_MAX_CONTEXT_LENGTH
    ):
        raise SystemExit(
            f"--max-context-length supports at most {IOS_MAX_CONTEXT_LENGTH} tokens "
            f"(got {args.max_context_length}); the static-shape ladder tops out "
            "at that bucket."
        )

    # The top context bucket is --max-context-length itself, so it must be one.
    if args.max_context_length is not None and (
        args.max_context_length <= 0
        or args.max_context_length & (args.max_context_length - 1)
    ):
        raise SystemExit(
            f"--max-context-length must be a power of two (got {args.max_context_length})."
        )

    # A rung is traced at its query length plus two positions (TraceSpec).
    widest_prompt = max(SHIPPING_PROMPT_QLENS)
    if args.max_context_length is not None and args.max_context_length <= widest_prompt:
        raise SystemExit(
            f"--max-context-length must exceed the prefill query length {widest_prompt} "
            f"(got {args.max_context_length})."
        )

    if args.compression is None and args.compression_config is None:
        args.compression_config = DEFAULT_IOS_COMPRESSION_CONFIG


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    _resolve_defaults(args)

    if args.model not in SUPPORTED_MODELS:
        logger.warning(
            f"{args.model} is not one of the verified checkpoints "
            f"({', '.join(SUPPORTED_MODELS)}). Exporting anyway, but the accuracy and "
            "performance of the result have not been verified."
        )

    print(f"Export complete: {asyncio.run(_export_ios(args))}")


if __name__ == "__main__":
    main()
