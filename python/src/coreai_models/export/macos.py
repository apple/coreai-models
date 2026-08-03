# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
macOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram via:
torch.export -> decompose -> defunctionalize -> TorchConverter -> optimize.
"""

import logging

import coreai_torch
import coreai_torch.composite_ops
import torch
from coreai.authoring import AIProgram

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.primitives.macos.cache import KVCache

logger = logging.getLogger(__name__)

# Composite ops that are externalized (kept as named composites in the MLIR graph
# rather than being inlined/decomposed).
_EXTERNALIZE_SPECS = [
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.GatherMM,
        composite_op_name="gather_mm",
        composite_attrs=["num_batch_axes"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RoPE,
        composite_op_name="rope",
        composite_attrs=["scale", "base", "dims", "interleaved"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.GatedDeltaUpdate,
        composite_op_name="gated_delta_update",
        composite_attrs=[],
    ),
]


def _build_reference_inputs(
    model: torch.nn.Module,
    config,
    target_dtype: torch.dtype,
    max_context_length: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build reference inputs and dynamic shapes for macOS model export.

    Args:
        model: The PyTorch model (used only to read config).
        config: HuggingFace model config.
        target_dtype: Data type for cache tensors.
        max_context_length: Maximum context length for the model.

    Returns:
        Tuple of (reference_inputs dict, dynamic_shapes dict).
    """
    batch_size = 1
    vocab_size = config.vocab_size

    input_ids = torch.randint(1, vocab_size, (batch_size, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
    position_ids = (
        torch.arange(QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET)
    )

    # Clamp `max_position_embeddings` so KVCache.create_cache_tensors doesn't
    # allocate a full-context cache for huge models
    saved_max_pos = config.max_position_embeddings
    config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    # Gemma4-style dual-cache models with a genuine sliding window preallocate
    # the primary cache to exactly `sliding_window` tokens and keep it a fixed
    # size (see KVCache.update_and_fetch_windowed) instead of letting it grow
    # to the full context length.
    sliding_window = (
        getattr(config, "sliding_window", None)
        if hasattr(model, "_build_sliding_kv_cache_tensors")
        else None
    )
    if hasattr(model, "_build_sliding_kv_cache_tensors"):
        # Models with dual KV caches (e.g. Gemma4) size the primary cache by
        # their sliding-attention layer count rather than num_hidden_layers.
        k_cache, v_cache = model._build_sliding_kv_cache_tensors(
            config, sliding_window or TRACE_KV_CACHE_SEQ_LEN, target_dtype
        )
    else:
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=target_dtype)
    config.max_position_embeddings = saved_max_pos

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }

    seq_ids_max = max_context_length - 2
    if sliding_window:
        # A bounded sliding-window cache requires each call's chunk (query_len)
        # to fit within the window -- see the torch._check(query_len <= capacity())
        # in KVCache.update_and_fetch_windowed.
        seq_ids_max = min(seq_ids_max, sliding_window)
    seq_ids_dim = torch.export.Dim("seq_ids", max=seq_ids_max)
    dynamic_shapes: dict[str, dict[int, torch.export.Dim] | None] = {
        "input_ids": {1: seq_ids_dim},
        "position_ids": {
            1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_context_length - 1)
        },
    }
    if sliding_window:
        # Bounded window: the cache is preallocated at its final size and never
        # grows, so its sequence dimension is fully static.
        dynamic_shapes["k_cache"] = None
        dynamic_shapes["v_cache"] = None
    else:
        dynamic_shapes["k_cache"] = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        }
        dynamic_shapes["v_cache"] = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        }

    # Models with dual KV caches (e.g. Gemma4) expose a _build_full_kv_cache_tensors
    # classmethod that returns the extra (full-attention) k/v tensors.
    if hasattr(model, "_build_full_kv_cache_tensors"):
        k_full, v_full = model._build_full_kv_cache_tensors(
            config, TRACE_KV_CACHE_SEQ_LEN, target_dtype
        )
        reference_inputs["k_cache_full"] = k_full
        reference_inputs["v_cache_full"] = v_full
        dynamic_shapes["k_cache_full"] = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_full_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        }
        dynamic_shapes["v_cache_full"] = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_full_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        }

    # Models with an externalized Per-Layer Embeddings (PLE) table (e.g. Gemma4)
    # accept pre-gathered INT8 rows as a graph input rather than an in-graph
    # embedding lookup -- see `Gemma4TextModel.forward`/`dump_ple_embedding`.
    if hasattr(model, "_ple_weight"):
        ple_dim = config.hidden_size_per_layer_input
        ple_total_dim = config.num_hidden_layers * ple_dim
        reference_inputs["ple_embeddings"] = torch.randint(
            -128, 127, (batch_size, QUANT_TRACE_QUERY_LEN, ple_total_dim), dtype=torch.int8
        )
        dynamic_shapes["ple_embeddings"] = {1: seq_ids_dim}

    return reference_inputs, dynamic_shapes


def export_to_coreai(
    model: torch.nn.Module,
    reference_inputs: dict[str, torch.Tensor],
    dynamic_shapes: dict | None = None,
    input_names: tuple[str, ...] | None = None,
    output_names: tuple[str, ...] | None = None,
    state_names: tuple[str, ...] | None = None,
) -> AIProgram:
    """Export a stateful macOS model to a AIProgram.

    Low-level building block under `export_macos_model` (text-only LLMs). Use
    that when possible; reach for this directly only when you need
    component-specific input/output names that `export_macos_model`'s
    text-only defaults don't fit.

    This is the core export function that handles:
    1. torch.export with no_grad
    2. Decomposition via coreai_torch decomp table
    3. Defunctionalization (replacing auto-functionalized ops with immutable variants)
    4. TorchConverter with externalized composite modules
    5. Custom MLIR lowering registration

    Args:
        model: The PyTorch model to export (must be in eval mode).
        reference_inputs: Dict of reference input tensors (keyword args to forward).
        dynamic_shapes: Dynamic shape specifications for torch.export.
        input_names: Names for the model inputs in the exported graph. If both
            ``input_names`` and ``state_names`` are ``None``, the names default
            to ``reference_inputs.keys()``.
        output_names: Names for the model outputs in the exported graph.
        state_names: Names of inputs that are state (i.e. mutated in place by
            the forward pass and surfaced via the runtime ``state=`` kwarg
            rather than as regular inputs/outputs).

    Returns:
        A AIProgram ready for optimization and compilation.
    """
    # If the caller didn't pass input_names explicitly, derive them from
    # ``reference_inputs.keys()`` while excluding any name the caller declared
    # as state. This keeps the call to ``add_pytorch_module`` predictable
    # regardless of whether ``state_names`` is also set.
    if input_names is None:
        state_names_set = set(state_names or ())
        input_names = tuple(k for k in reference_inputs if k not in state_names_set)

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            aten_exported_program = torch.export.export(
                module,
                args=(),
                kwargs=reference_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        coreai_decomp_table = coreai_torch.get_decomp_table()
        coreaten_exported_program = aten_exported_program.run_decompositions(coreai_decomp_table)
        remove_functionalization(coreaten_exported_program)
        return coreaten_exported_program

    model.eval()
    converter = coreai_torch.TorchConverter()
    converter.add_pytorch_module(
        model,
        export_fn=export_fn,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=input_names,
        output_names=output_names,
        state_names=state_names,
    )
    register_custom_torch_lowering(converter)
    return converter.to_coreai()


def export_macos_model(
    model: torch.nn.Module,
    config,
    export_config,
) -> AIProgram:
    """Export a macOS model to a AIProgram.

    This is the main entry point for macOS model export. It:
    1. Builds reference inputs and dynamic shapes from the model config
    2. Exports the model through torch.export -> TorchConverter
    3. Optimizes the resulting AIProgram

    Args:
        model: A loaded PyTorch model (already in the correct dtype).
        config: HuggingFace model config (used for cache dimensions, vocab size, etc.).
        export_config: An ExportConfig instance (used for max_context_length, etc.).

    Returns:
        An optimized AIProgram ready for MLIR quantization and compilation.
    """
    max_context_length = getattr(export_config, "max_context_length", None)
    if max_context_length is None:
        max_context_length = getattr(config, "max_position_embeddings", 2048)

    # Determine target dtype from the model parameters
    target_dtype = next(model.parameters()).dtype

    logger.info(
        f"Exporting macOS model (dtype={target_dtype}, max_context_length={max_context_length})"
    )

    reference_inputs, dynamic_shapes = _build_reference_inputs(
        model, config, target_dtype, max_context_length
    )

    input_names: tuple[str, ...] = ("input_ids", "position_ids")
    if hasattr(model, "_ple_weight"):
        input_names = (*input_names, "ple_embeddings")
    output_names = ("logits",)
    # Models declare their own full state-name tuple (matching the exact order
    # of their forward()'s state args) via `_state_names` when the default
    # single-cache (KEY_CACHE_NAME, VALUE_CACHE_NAME) mapping doesn't apply --
    # e.g. Gemma4's dual-cache forward(..., k_cache, v_cache, k_cache_full, v_cache_full)
    # needs (slidingKeyCache, slidingValueCache, keyCache, valueCache) since the
    # Swift runner expects keyCache/valueCache to name the full-attention cache.
    state_names = getattr(model, "_state_names", (KEY_CACHE_NAME, VALUE_CACHE_NAME))

    logger.info("Exporting model to Core AI dialect...")
    coreai_program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        state_names=state_names,
    )

    logger.info("Optimizing AIProgram...")
    coreai_program.optimize()
    # coreai_program._mlir_module.dump()
    return coreai_program
