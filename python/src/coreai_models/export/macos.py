# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
macOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram via:
torch.export -> decompose -> defunctionalize -> TorchConverter -> optimize.

The result carries one entrypoint, ``main``, or two when the model opts into
``exports_prefill_graph``: ``main`` for decode, ``prefill`` for the prompt. An eager
module is traced twice for that -- the second time with prefill mode on, so it returns
nothing and the LM head drops out. A flattened graph module has no prefill mode left to
set, so the one trace is staged twice instead, the second time with its non-state
outputs trimmed off, which leaves the LM head dead.
"""

import copy
import dataclasses
import logging
from typing import Any

import coreai_torch
import torch
from coreai.authoring import AIProgram
from torch.export.graph_signature import ExportGraphSignature, OutputKind

from coreai_models._constants import (
    DEFAULT_INCLUDE_DEBUG_INFO,
    DRAFT_GRAPH_NAME,
    INJECT_KV_GRAPH_NAME,
    MAIN_GRAPH_NAME,
    PREFILL_GRAPH_NAME,
    TRACE_KV_CACHE_SEQ_LEN,
)
from coreai_models.export.externalize import (
    EXTERNALIZE_SPECS,
    subexport_and_restore,
)
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.models.base import BaseForCausalLM, TraceSpec

logger = logging.getLogger(__name__)


def _build_reference_inputs(
    model: BaseForCausalLM,
    config,
    target_dtype: torch.dtype,
    max_context_length: int,
) -> tuple[dict[str, Any], dict]:
    """Reference inputs and dynamic shapes for macOS export.

    Thin wrapper over the model's export-contract hooks, where the per-model variation
    lives. Returns ``(reference_inputs, dynamic_shapes)``.
    """
    # The trace cache length only bounds peak memory, so cap it at the context it serves.
    spec = TraceSpec(
        max_context_length=max_context_length,
        cache_seq_len=min(TRACE_KV_CACHE_SEQ_LEN, max_context_length),
    )
    reference_inputs = model.build_reference_inputs(config, target_dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    model.validate_export_contract(reference_inputs, dynamic_shapes)
    # A macOS model has exactly one traced signature; a `prefill` entrypoint, when the
    # model asks for one, is that same signature traced again in prefill mode.
    return reference_inputs[MAIN_GRAPH_NAME], dynamic_shapes[MAIN_GRAPH_NAME]


def _set_prefill_mode(module: torch.nn.Module, prefill: bool) -> None:
    """Toggle a model's prefill mode, if it has one.

    ``export_to_coreai`` also serves plain ``nn.Module`` components that know nothing
    about prefill, so this is a no-op for anything without the hook.

    Both traces share this one flag, so it is only correct while the converter exports
    entrypoints serially -- which it does today, and ``export_fn`` sets the mode as its
    first act. If that ever changes, the two traces would race and the prefill graph
    could silently come out identical to ``main``.
    """
    setter = getattr(module, "set_prefill_mode", None)
    if callable(setter):
        setter(prefill)


def _retype_inputs_to_quantized_graph(
    reference_inputs: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Cast reference inputs to dtypes that the quantized graph declares. Specifically to
    handle KV Cache quantization.

    Placeholders are paired with ``reference_inputs`` **positionally**, which is also the
    property ``export_to_coreai`` already relies on when it exports a graph-mode module
    with ``pass_inputs_as_kwargs=False``.

    Args:
        reference_inputs: The contract's reference inputs, keyed by parameter name in
            forward-signature order.
        model: The compressed model to be exported. A non-``GraphModule`` (eager mode or no
            compression) remains untouched.

    Returns:
        ``reference_inputs`` with each tensor in the dtype the graph declares. The
        reference caches are all-zeros, so the casts are exact.

    Raises:
        ValueError: If the graph's tensor placeholders and ``reference_inputs`` disagree
            in count, i.e. the positional pairing this depends on no longer holds.
    """
    if not isinstance(model, torch.fx.GraphModule):
        return reference_inputs

    placeholders = [
        node
        for node in model.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    if len(placeholders) != len(reference_inputs):
        raise ValueError(
            f"Quantized graph declares {len(placeholders)} tensor input(s) "
            f"{[n.name for n in placeholders]} but the export contract supplied "
            f"{len(reference_inputs)} {list(reference_inputs)}. These are paired "
            "positionally, so the export cannot proceed."
        )

    retyped: dict[str, Any] = {}
    for (name, value), node in zip(reference_inputs.items(), placeholders, strict=True):
        if node.name != name:
            # Positional pairing is authoritative
            logger.debug(f"Graph placeholder '{node.name}' pairs with contract input '{name}'")
        dtype = node.meta["val"].dtype
        if value.dtype != dtype:
            logger.info(
                f"Reference input '{name}': {value.dtype} -> {dtype} to match the quantized graph"
            )
            value = value.to(dtype)
        retyped[name] = value
    return retyped


def _drop_user_outputs(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Return the same program with everything but its state writes trimmed off.

    A decode trace ends in a tuple of the KV cache updates (``USER_INPUT_MUTATION``
    entries, which the converter binds as state) and ``logits``. Dropping the
    ``USER_OUTPUT`` entries and running DCE leaves the cache writes and kills whatever
    only fed ``logits`` -- the LM head and the tail of the model -- which is exactly the
    prefill graph. This is the eager ``prefill_mode`` trace expressed as a graph edit,
    for a module that was flattened before prefill mode could be set.

    The graph is copied, not edited in place; the state dict rides along by reference, so
    this costs a graph, not a second set of weights.
    """
    specs = exported_program.graph_signature.output_specs
    keep = [i for i, spec in enumerate(specs) if spec.kind != OutputKind.USER_OUTPUT]
    if len(keep) == len(specs):
        return exported_program

    graph = copy.deepcopy(exported_program.graph)
    output_node = next(node for node in reversed(graph.nodes) if node.op == "output")
    output_node.args = (tuple(output_node.args[0][i] for i in keep),)
    graph.eliminate_dead_code()
    graph_module = torch.fx.GraphModule(exported_program.graph_module, graph)

    return exported_program._update(
        graph_module,
        ExportGraphSignature(
            input_specs=list(exported_program.graph_signature.input_specs),
            output_specs=[specs[i] for i in keep],
        ),
    )


def _rename_for_second_entrypoint(externalized_programs: list, graph: torch.fx.Graph) -> list:
    """Re-badge externalized composite sub-programs for a second entrypoint.

    Each entry becomes one named ``coreai.graph`` in the module, and the name is fixed
    when the model is patched, so handing the same entries to two entrypoints emits the
    same symbol twice and the module fails to verify. The eager path re-patches per
    entrypoint and gets fresh names that way; a flattened model can't be re-patched
    (its call sites were baked in at capture), so rename the entries instead. Entries
    whose call sites the trim killed are dropped -- nothing would invoke them.
    """
    live = {node.name for node in graph.nodes}
    return [
        dataclasses.replace(ext, name=f"{ext.name}_{PREFILL_GRAPH_NAME}")
        for ext in externalized_programs
        if any(node in live for node in ext.source_nodes)
    ]


def export_to_coreai(
    model: torch.nn.Module,
    reference_inputs: dict[str, Any],
    dynamic_shapes: dict | None = None,
    input_names: tuple[str, ...] | None = None,
    output_names: tuple[str, ...] | None = None,
    state_names: tuple[str, ...] | None = None,
    externalized_model: torch.nn.Module | None = None,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    export_prefill_graph: bool = False,
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
        externalized_model: The eager module whose composite-op submodules were marked
            by ``patch_model_for_externalization`` before ``model`` was produced.
            Required when ``model`` is a flattened ``torch.fx.GraphModule``, and
            unused when it is an eager module.
        include_debug_info: When True, the converter runs in ``DEBUG`` mode and embeds debug
            information in the exported ``.aimodel``. Defaults to ``RELEASE`` mode,
            which embeds minimum debug information and makes the exported asset smaller.
        export_prefill_graph: When True, stage a second ``prefill`` entrypoint beside
            ``main``. It shares ``input_names``, ``state_names``, ``reference_inputs`` and
            ``dynamic_shapes`` with ``main`` and declares no outputs, so everything the KV
            cache writes don't feed is dropped. An eager ``model`` is traced a second time
            with prefill mode on; a flattened ``model``, which has no prefill mode left to
            set, reuses the decode trace and relies on the trimmed outputs alone.

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

    def make_export_fn(prefill: bool):
        def export_fn(
            module: torch.nn.Module, pass_inputs_as_kwargs: bool = True
        ) -> torch.export.ExportedProgram:
            # Set the mode here rather than around the staging call: the converter
            # re-exports the module during externalization, long after staging, so
            # whichever mode is set at that point is the one that would stick.
            _set_prefill_mode(module, prefill)
            # A module unlifted from an ExportedProgram only accepts the calling convention
            # it was captured with, and graph-mode compression captures positionally.
            # `reference_inputs` is insertion-ordered to match the forward signature.
            export_args = () if pass_inputs_as_kwargs else tuple(reference_inputs.values())
            export_kwargs = reference_inputs if pass_inputs_as_kwargs else None
            with torch.no_grad():
                aten_exported_program = torch.export.export(
                    module,
                    args=export_args,
                    kwargs=export_kwargs,
                    dynamic_shapes=dynamic_shapes,
                )
            coreai_decomp_table = coreai_torch.get_decomp_table()
            coreaten_exported_program = aten_exported_program.run_decompositions(
                coreai_decomp_table
            )
            remove_functionalization(coreaten_exported_program)
            return coreaten_exported_program

        return export_fn

    mode = (
        coreai_torch.TorchConverter.Mode.DEBUG
        if include_debug_info
        else coreai_torch.TorchConverter.Mode.RELEASE
    )
    converter = coreai_torch.TorchConverter(mode=mode)

    # GraphModule subclasses nn.Module, so this specific check has to come first
    if isinstance(model, torch.fx.GraphModule):
        if externalized_model is None:
            raise ValueError(
                "A flattened torch.fx.GraphModule needs an externalized_model handle. "
                "Call patch_model_for_externalization on the model before quantization."
            )
        exported_program = make_export_fn(prefill=False)(model, pass_inputs_as_kwargs=False)
        externalized_programs = subexport_and_restore(externalized_model, exported_program)

        converter.add_exported_program(
            exported_program,
            input_names=input_names,
            output_names=output_names,
            state_names=state_names,
            entrypoint_name=MAIN_GRAPH_NAME,
            _externalized_exported_programs=externalized_programs,  # type: ignore[call-arg]
        )
        if export_prefill_graph:
            # The flattened graph resolved `prefill_mode` when it was captured, so there is
            # no flag left to flip and no second trace to take. Stage the same program with
            # its non-state outputs trimmed instead: the KV cache writes survive as state,
            # and the LM head that only fed `logits` -- an output the runner rejects on a
            # prefill graph -- goes dead with it.
            logger.info(
                f"Exporting prefill entrypoint {PREFILL_GRAPH_NAME!r} from the flattened "
                "decode graph with its non-state outputs trimmed."
            )
            prefill_program = _drop_user_outputs(exported_program)
            converter.add_exported_program(
                prefill_program,
                input_names=input_names,
                output_names=(),
                state_names=state_names,
                entrypoint_name=PREFILL_GRAPH_NAME,
                _externalized_exported_programs=_rename_for_second_entrypoint(  # type: ignore[call-arg]
                    externalized_programs, prefill_program.graph
                ),
            )
    elif isinstance(model, torch.nn.Module):
        model.eval()
        converter.add_pytorch_module(
            model,
            export_fn=make_export_fn(prefill=False),
            externalize_modules=EXTERNALIZE_SPECS,
            input_names=input_names,
            output_names=output_names,
            state_names=state_names,
            entrypoint_name=MAIN_GRAPH_NAME,
        )
        if export_prefill_graph:
            logger.info(
                f"Exporting prefill entrypoint {PREFILL_GRAPH_NAME!r} (its trace ends at the "
                "last cache write, so 'skipping unused submodule' warnings for the tail of "
                "the model are expected)..."
            )
            converter.add_pytorch_module(
                model,
                export_fn=make_export_fn(prefill=True),
                externalize_modules=EXTERNALIZE_SPECS,
                input_names=input_names,
                output_names=(),
                state_names=state_names,
                entrypoint_name=PREFILL_GRAPH_NAME,
            )
    else:
        raise TypeError(
            "model must be a torch.nn.Module (eager-mode) or torch.fx.GraphModule "
            f"(graph-mode), got {type(model).__name__}."
        )

    register_custom_torch_lowering(converter)
    try:
        return converter.to_coreai()
    finally:
        # Don't leave a shared model in prefill mode for whatever runs next, including
        # when a trace raises -- callers reuse one model across quantize and export.
        _set_prefill_mode(model, False)


def export_macos_model(
    model: BaseForCausalLM,
    config,
    export_config,
    externalized_model: BaseForCausalLM | None = None,
) -> AIProgram:
    """Export a macOS model to a AIProgram.

    This is the main entry point for macOS model export. It:
    1. Builds reference inputs and dynamic shapes from the model config
    2. Exports the model through torch.export -> TorchConverter
    3. Optimizes the resulting AIProgram

    Models that set ``exports_prefill_graph`` get a second ``prefill`` entrypoint with no
    LM head, from the same module. The Swift runner uses it for prefill when it is present
    and falls back to ``main`` when it is not.

    Args:
        model: A loaded PyTorch model (already in the correct dtype). Under
            graph-mode quantization this is the flattened ``torch.fx.GraphModule``,
            and the contract is read off ``externalized_model`` instead.
        config: HuggingFace model config (used for cache dimensions, vocab size, etc.).
        export_config: An ExportConfig instance (used for max_context_length, etc.).
        externalized_model: The eager module marked by
            ``patch_model_for_externalization`` before ``model`` was produced.
            See ``export_to_coreai``.

    Returns:
        An optimized AIProgram ready for MLIR quantization and compilation.
    """
    max_context_length = getattr(export_config, "max_context_length", None)
    if max_context_length is None:
        max_context_length = getattr(config, "max_position_embeddings", 2048)

    # Graph-mode quantization flattens the model into a torch.fx.GraphModule, which
    # carries none of the export-contract hooks. `externalized_model` is the eager
    # module that graph was captured from, so query the contract there.
    contract_model = model if externalized_model is None else externalized_model

    from coreai_models.export.pipeline import _resolve_precision

    target_dtype = _resolve_precision(export_config.compute_precision)

    logger.info(
        f"Exporting macOS model (dtype={target_dtype}, max_context_length={max_context_length})"
    )

    reference_inputs, dynamic_shapes = _build_reference_inputs(
        contract_model, config, target_dtype, max_context_length
    )

    reference_inputs = _retype_inputs_to_quantized_graph(reference_inputs, model)

    logger.info("Exporting model to Core AI dialect...")
    coreai_program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=contract_model.export_input_names()[MAIN_GRAPH_NAME],
        output_names=contract_model.export_output_names()[MAIN_GRAPH_NAME],
        state_names=contract_model.export_state_names()[MAIN_GRAPH_NAME],
        include_debug_info=getattr(export_config, "include_debug_info", DEFAULT_INCLUDE_DEBUG_INFO),
        externalized_model=externalized_model,
        export_prefill_graph=contract_model.exports_prefill_graph,
    )

    logger.info("Optimizing AIProgram...")
    coreai_program.optimize()

    return coreai_program


class _InjectKVEntrypointModule(torch.nn.Module):
    """``nn.Module`` wrapper around the drafter's ``inject_kv_graph`` entrypoint.

    ``torch.export.export`` requires an ``nn.Module``, not a bound method. Registering the
    drafter as a submodule lifts its parameters and buffers into the exported program, and
    ``forward`` delegates to the (state-only, empty-output) ``inject_kv_graph`` entrypoint.
    The forward param names match the graph's input/state names so a per-name
    ``dynamic_shapes`` dict stays valid.
    """

    def __init__(self, drafter: torch.nn.Module) -> None:
        super().__init__()
        self.drafter = drafter

    def forward(self, features, position_ids, sliding_k_cache, sliding_v_cache):
        return self.drafter.inject_kv_graph(
            features, position_ids, sliding_k_cache, sliding_v_cache
        )


class _DraftEntrypointModule(torch.nn.Module):
    """``nn.Module`` wrapper around the drafter's ``draft_graph`` entrypoint (see
    :class:`_InjectKVEntrypointModule`); ``forward`` returns the draft ``logits``."""

    def __init__(self, drafter: torch.nn.Module) -> None:
        super().__init__()
        self.drafter = drafter

    def forward(self, input_ids, position_ids, sliding_k_cache, sliding_v_cache):
        return self.drafter.draft_graph(input_ids, position_ids, sliding_k_cache, sliding_v_cache)


_DFLASH_ENTRYPOINT_WRAPPERS = {
    INJECT_KV_GRAPH_NAME: _InjectKVEntrypointModule,
    DRAFT_GRAPH_NAME: _DraftEntrypointModule,
}


def _make_dflash_export_fn(graph: str, ref_inputs: dict[str, Any], dyn_shapes: dict | None):
    """Build an ``export_fn`` that traces a DFlash drafter entrypoint via an nn.Module.

    ``add_pytorch_module`` hands the patched (externalization-marked) model to the
    returned ``export_fn`` as ``module``. We wrap that same model in the entrypoint's
    ``nn.Module`` wrapper -- ``torch.export.export`` rejects the bound entrypoint method
    directly (it demands an ``nn.Module``), and the wrapper's ``forward`` delegates to it.
    Because the wrapper holds the very model ``add_pytorch_module`` patched, the
    composite-op submodules the entrypoint invokes are the patched ones, mirroring
    ``export_to_coreai``'s ``make_export_fn`` (minus the prefill toggle).

    ``ref_inputs`` is insertion-ordered to match the entrypoint signature and is passed as
    kwargs so ``dyn_shapes`` keys line up by parameter name; a falsy ``dyn_shapes`` passes
    ``None`` (e.g. an all-static entrypoint).
    """
    wrapper_cls = _DFLASH_ENTRYPOINT_WRAPPERS[graph]

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        wrapper = wrapper_cls(module).eval()
        with torch.no_grad():
            aten_exported_program = torch.export.export(
                wrapper,
                args=(),
                kwargs=ref_inputs,
                dynamic_shapes=dyn_shapes or None,
            )
        coreaten_exported_program = aten_exported_program.run_decompositions(
            coreai_torch.get_decomp_table()
        )
        remove_functionalization(coreaten_exported_program)
        return coreaten_exported_program

    return export_fn


def export_dflash_drafter(model: BaseForCausalLM, config, export_config) -> AIProgram:
    """Export the DFlash drafter's two macOS entrypoints to a single AIProgram.

    Unlike ``export_macos_model`` (one traced signature, plus an optional ``prefill``
    twin of it), the DFlash drafter has two *distinct* bound methods sharing one ring-KV
    state: ``inject_kv`` writes the target features into ``slidingKeyCache`` /
    ``slidingValueCache`` (no outputs), and ``draft`` reads them back and emits draft
    ``logits``. Because both entrypoints declare the *same* state names, the converter
    binds one physical ring buffer across them -- the macOS ``main``/``prefill`` and iOS
    ``extend``/``prompt_opt`` shared-KV mechanism.

    This mirrors the eager block of ``export_to_coreai`` (one ``add_pytorch_module`` per
    entrypoint on the same eager model) and iOS's ``_convert_to_coreai`` contract loop
    (one entrypoint per named bound method). ``export_to_coreai`` and
    ``export_macos_model`` are left untouched.

    Args:
        model: The eager ``MuseGlimmerDFlashDrafterForCausalLM`` (already in the target
            dtype). Never graph-mode-quantized in the pipeline.
        config: HuggingFace model config (cache dimensions, vocab size, etc.).
        export_config: An ExportConfig instance (max_context_length, compute_precision,
            include_debug_info).

    Returns:
        An optimized AIProgram carrying the ``inject_kv`` and ``draft`` entrypoints.
    """
    max_context_length = getattr(export_config, "max_context_length", None) or getattr(
        config, "max_position_embeddings", 2048
    )

    from coreai_models.export.pipeline import _resolve_precision

    target_dtype = _resolve_precision(export_config.compute_precision)

    logger.info(
        f"Exporting DFlash drafter (dtype={target_dtype}, max_context_length={max_context_length})"
    )

    # The trace cache length only bounds peak memory, so cap it at the context it serves.
    spec = TraceSpec(
        max_context_length=max_context_length,
        cache_seq_len=min(TRACE_KV_CACHE_SEQ_LEN, max_context_length),
    )
    reference_inputs = model.build_reference_inputs(config, target_dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    # Keep all graphs -- unlike the single-graph macOS path, we do not unwrap [MAIN].
    model.validate_export_contract(reference_inputs, dynamic_shapes)

    inputs = model.export_input_names()
    states = model.export_state_names()
    outputs = model.export_output_names()

    # The two entrypoints are traced through nn.Module wrappers
    # (_DFLASH_ENTRYPOINT_WRAPPERS), so only the graph names are needed here.
    entry_graphs = (INJECT_KV_GRAPH_NAME, DRAFT_GRAPH_NAME)
    if set(entry_graphs) != set(inputs):
        raise ValueError(
            "DFlash drafter entrypoints "
            f"{sorted(entry_graphs)} disagree with the export contract graphs "
            f"{sorted(inputs)}; expected exactly {{{INJECT_KV_GRAPH_NAME!r}, "
            f"{DRAFT_GRAPH_NAME!r}}}."
        )

    # The drafter is exported eagerly; graph-mode quantization is never applied to it in
    # the pipeline (and the per-method externalization re-patching below relies on it).
    if isinstance(model, torch.fx.GraphModule):
        raise TypeError(
            "export_dflash_drafter only supports the eager drafter module, got a "
            "flattened torch.fx.GraphModule."
        )

    mode = (
        coreai_torch.TorchConverter.Mode.DEBUG
        if getattr(export_config, "include_debug_info", DEFAULT_INCLUDE_DEBUG_INFO)
        else coreai_torch.TorchConverter.Mode.RELEASE
    )
    converter = coreai_torch.TorchConverter(mode=mode)
    model.eval()

    for graph in entry_graphs:
        # Calling add_pytorch_module twice on the SAME eager model is already proven by
        # the prefill path (main/prefill above). The eager path re-patches externalization
        # per entrypoint, so there is no symbol-name collision (that only afflicts the
        # flattened path via _rename_for_second_entrypoint).
        #
        # torch.export.export rejects a bound method (it demands an nn.Module), so
        # _make_dflash_export_fn traces a small nn.Module wrapper that delegates to the
        # entrypoint; the wrapper holds the model add_pytorch_module patched, so
        # externalization still sees the patched composite-op submodules.
        converter.add_pytorch_module(
            model,
            export_fn=_make_dflash_export_fn(graph, reference_inputs[graph], dynamic_shapes[graph]),
            externalize_modules=EXTERNALIZE_SPECS,
            input_names=inputs[graph],
            output_names=outputs[graph],
            state_names=states[graph],
            entrypoint_name=graph,
        )

    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    logger.info("Optimizing AIProgram...")
    program.optimize()

    return program
