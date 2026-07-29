# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
MLIR custom op definitions and lowerings for the export pipeline.

Provides:
- ``immutable_slice_update`` custom op (export-specific, non-mutating variant)
- Custom MLIR lowerings for slice update, composite ops, and dequantization
- ``remove_functionalization`` pass for replacing auto-functionalized ops
- ``register_custom_torch_lowering`` to register all lowerings on a converter
"""

import operator
from collections.abc import Callable
from typing import Annotated

import numpy as np
import torch
from coreai._compiler.dialects import coreai, coreaix
from coreai._compiler.ir import Location, OpResultList, Value
from coreai.authoring.types import AllocationType, HardwareConstraints, TensorSpec
from coreai_torch._utils import generate_composite_decl
from torch import Tensor, fx
from torch._higher_order_ops.auto_functionalize import (
    AutoFunctionalized,
    AutoFunctionalizedV2,
)
from torch.fx.node import Argument

# Re-export mutable_slice_update so callers can import from a single place.
# The canonical definition lives in coreai_models.primitives._ops.
from coreai_models.primitives._ops import mutable_slice_update  # noqa: F401

# ---------------------------------------------------------------------------
# Immutable slice update (export-specific)
# ---------------------------------------------------------------------------


@torch.library.custom_op("coreai::immutable_slice_update", mutates_args=[])
def immutable_slice_update(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    """
    Immutable slice update operation.

    Similar to mutable_slice_update but doesn't mutate the input tensor.
    Used during graph transformations where mutation isn't allowed.

    Args:
        x: The tensor to update
        update: The update values to insert
        begin: Tensor containing start indices for each dimension
        end: Tensor containing end indices for each dimension

    Returns:
        A new tensor with the update applied
    """
    result = x.clone()
    result[
        begin[0] : end[0],
        begin[1] : end[1],
        begin[2] : end[2],
        begin[3] : end[3],
        begin[4] : end[4],
    ] = update
    return result


@immutable_slice_update.register_fake
def immutable_slice_update_meta(  # type: ignore[no-untyped-def]
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    """Fake implementation for tracing/meta operations."""
    return torch.empty(x.shape, dtype=x.dtype)


# ---------------------------------------------------------------------------
# MLIR operand helpers
# ---------------------------------------------------------------------------


def _get_operand(
    values_map: dict[str, Value],
    node: fx.Node,
    idx: int,
    loc: Location | None = None,
) -> Value:
    """Return the MLIR Value for node.args[idx], converting scalars/tensors/lists to constants."""
    assert 0 <= idx < len(node.args), (
        f"get_operand: idx {idx} out of range for node {node} with {len(node.args)} args"
    )
    arg: Argument = node.args[idx]
    if isinstance(arg, fx.Node):
        return values_map[arg.name]
    if isinstance(arg, list) and any(isinstance(e, fx.Node) for e in arg):
        # Mixed list: resolve fx.Node elements via values_map, keep ints as constants.
        dim_vals = [
            values_map[e.name] if isinstance(e, fx.Node) else coreai.constant([e]) for e in arg
        ]
        return coreai.concat(0, dim_vals) if len(dim_vals) > 1 else dim_vals[0]
    if isinstance(arg, bool | int | float | Tensor | list):
        data = arg.detach().cpu().numpy() if isinstance(arg, Tensor) else arg
        return coreai.constant(data, loc=loc)
    raise ValueError(f"Unsupported arg type {type(arg)} in node {node}: {arg}")


def _get_operands(
    values_map: dict[str, Value],
    node: fx.Node,
    indices: list[int],
    loc: Location | None = None,
) -> list[Value]:
    """Get multiple operand Values from an FX node's arguments by index."""
    return [_get_operand(values_map, node, i, loc) for i in indices]


# ---------------------------------------------------------------------------
# Graph rewrite: remove auto-functionalization
# ---------------------------------------------------------------------------


def generate_node(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.fx.Node:
    """
    Create an FX node with the correct target name for a custom op.

    For some reason ``graph.call_function(the_function=target_fn)`` does not
    preserve the correct name — this tracing-based workaround does.
    """

    def internal_func(x):  # type: ignore[no-untyped-def]
        return target_fn(x)

    g = torch.fx.symbolic_trace(internal_func)

    for node in g.graph.nodes:
        if hasattr(node.target, "name") and node.target.name() == target_fn._qualname:
            return node  # type: ignore[return-value]

    raise RuntimeError(f"Unable to find {target_fn} in generated function")


def _autofunc_base_tensor(autofunc_node: fx.Node) -> Argument:
    """Return the mutated base tensor an auto-functionalized node wraps.

    ``AutoFunctionalizedV2`` indirects the mutated tensor through
    ``_all_bases[_x_base_index]``; v1 passes it directly as the ``x`` kwarg.
    """
    if isinstance(autofunc_node.target, AutoFunctionalizedV2):
        base_idx = autofunc_node.kwargs["_x_base_index"]
        all_bases = autofunc_node.kwargs["_all_bases"]
        return all_bases[base_idx]
    return autofunc_node.kwargs["x"]


def _copy_node_provenance(dst: fx.Node, src: fx.Node) -> None:
    """Copy ``val`` plus the provenance metadata from ``src`` onto ``dst``."""
    dst.meta["val"] = src.meta["val"]
    dst.meta["nn_module_stack"] = src.meta.get("nn_module_stack", {})
    dst.meta["stack_trace"] = src.meta.get("stack_trace", "")
    dst.meta["source_fn_stack"] = src.meta.get("source_fn_stack", [])
    dst.stack_trace = src.stack_trace


def _partition_autofunc_nodes(
    graph: torch.fx.Graph,
) -> tuple[dict[str, fx.Node], dict[str, fx.Node]]:
    """Split auto-functionalized nodes by the mutable op they wrap.

    Returns ``(slice_update_autofuncs, cache_update_autofuncs)``, each keyed by
    node name. Raises ``AssertionError`` for any other auto-functionalized op,
    or for a node with a consumer the rewrite paths can't handle.
    """
    slice_update_autofuncs: dict[str, fx.Node] = {}
    cache_update_autofuncs: dict[str, fx.Node] = {}

    for node in graph.nodes:
        if not isinstance(node.target, AutoFunctionalized | AutoFunctionalizedV2):
            continue
        assert len(node.args) == 1, (
            f"Expected 1 arg, found {len(node.args)} on {node.format_node()}"
        )
        non_getitem_users = [u for u in node.users if u.target is not operator.getitem]
        assert not non_getitem_users, (
            f"Auto-functionalized node {node.name} has non-getitem users "
            f"{[u.name for u in non_getitem_users]}; remove_functionalization only "
            "knows how to rewrite getitem consumers."
        )
        op_name = node.args[0].name()
        if op_name == "coreai::mutable_slice_update":
            slice_update_autofuncs[node.name] = node
        elif op_name == "coreai::mutable_cache_update_and_fetch":
            cache_update_autofuncs[node.name] = node
        else:
            raise AssertionError(
                f"Expected mutable_slice_update or mutable_cache_update_and_fetch, found {op_name}"
            )

    return slice_update_autofuncs, cache_update_autofuncs


def _replace_slice_update_autofuncs(
    graph: torch.fx.Graph,
    slice_update_autofuncs: dict[str, fx.Node],
    get_items: list[fx.Node],
    get_item_replacements: dict[str, fx.Node],
) -> None:
    """Replace each ``mutable_slice_update`` getitem with an ``immutable_slice_update``.

    Appends the retired getitem nodes to ``get_items`` and records each
    getitem-name → replacement-node mapping in ``get_item_replacements`` for the
    caller's teardown pass.
    """
    autofunc_replacements: dict[str, fx.Node] = {}
    # Snapshot the node list: the loop body inserts nodes via node_copy().
    for getitem_node in list(graph.nodes):
        # Find if there is an autofunc node that feeds into this node
        autofunc_node = None
        for input_node in getitem_node.all_input_nodes:
            if input_node.name in slice_update_autofuncs:
                autofunc_node = slice_update_autofuncs[input_node.name]
                break

        if autofunc_node is None:
            continue

        assert isinstance(autofunc_node, torch.fx.Node)
        if autofunc_node.name in autofunc_replacements:
            # Duplicate getitem — reuse the first slice_update node
            slice_node = autofunc_replacements[autofunc_node.name]
        else:
            with graph.inserting_before(getitem_node):
                # Create immutable version as a replacement
                slice_node = generate_node(target_fn=immutable_slice_update)
                slice_node = graph.node_copy(slice_node)

                # Extract args differently for v1 vs v2
                if isinstance(autofunc_node.target, AutoFunctionalizedV2):
                    slice_node.args = (
                        _autofunc_base_tensor(autofunc_node),
                        autofunc_node.kwargs["update"],
                        autofunc_node.kwargs["begin"],
                        autofunc_node.kwargs["end"],
                    )
                else:
                    # v1 kwargs: x, update, begin, end
                    slice_node.args = tuple(autofunc_node.kwargs.values())

                _copy_node_provenance(slice_node, getitem_node)

            autofunc_replacements[autofunc_node.name] = slice_node

        # Replace the getitem node with the slice_update node
        getitem_node.replace_all_uses_with(slice_node)
        get_items.append(getitem_node)
        get_item_replacements[getitem_node.name] = slice_node


def _replace_cache_update_autofuncs(
    graph: torch.fx.Graph,
    cache_update_autofuncs: dict[str, fx.Node],
    get_items: list[fx.Node],
    get_item_replacements: dict[str, fx.Node],
) -> None:
    """Decompose each ``mutable_cache_update_and_fetch`` into aten + immutable ops.

    Emits ``aten.unsqueeze`` → ``immutable_slice_update`` (the 5D mutated cache,
    getitem index 1) → ``aten.slice`` (layer) → optional ``aten.slice`` (seq) →
    ``aten.squeeze`` (the 4D fetched slice, getitem index 0).

    Appends the retired getitem nodes to ``get_items`` and records each
    getitem-name → replacement-node mapping in ``get_item_replacements`` for the
    caller's teardown pass.
    """
    for autofunc_node in cache_update_autofuncs.values():
        # Map getitem index -> getitem node.
        getitem_by_idx: dict[int, fx.Node] = {}
        for user in autofunc_node.users:
            if user.target is operator.getitem:
                idx = user.args[1]
                assert idx not in getitem_by_idx, (
                    f"{autofunc_node.name} has multiple getitem users at index {idx} "
                    f"({getitem_by_idx[idx].name}, {user.name}); expected one per index."
                )
                getitem_by_idx[idx] = user

        getitem_fetched = getitem_by_idx.get(0)  # 4D fetched slice -> SDPA
        getitem_cache = getitem_by_idx.get(1)  # 5D mutated cache -> handle
        assert getitem_cache is not None, (
            f"{autofunc_node.name}: no getitem at index 1 for the mutated cache; "
            f"the cache write would be dead-code-eliminated. Found {sorted(getitem_by_idx)}."
        )

        with graph.inserting_before(autofunc_node):
            update = autofunc_node.kwargs["update"]

            unsqueeze_node = graph.call_function(
                torch.ops.aten.unsqueeze.default,
                args=(update, 0),
            )
            if "val" in update.meta:
                unsqueeze_node.meta["val"] = update.meta["val"].unsqueeze(0)

        # immutable_slice_update (5D), inserted after the unsqueeze_node.
        with graph.inserting_after(unsqueeze_node):
            isu_node = generate_node(target_fn=immutable_slice_update)
            isu_node = graph.node_copy(isu_node)
            isu_node.args = (
                _autofunc_base_tensor(autofunc_node),
                unsqueeze_node,
                autofunc_node.kwargs["begin"],
                autofunc_node.kwargs["end"],
            )
            # 5D meta comes from the mutated-cache getitem.
            _copy_node_provenance(isu_node, getitem_cache)

        # narrow(0, layer_idx, 1) -> [narrow(seq_dim, 0, seq_len)] -> squeeze(0).
        # The seq narrow is skipped when seq_len is None.
        # layer_idx is a Python int (loop unrolled at trace time);
        # seq_len is either None or a SymInt -> fx.Node in the exported graph.
        layer_idx = autofunc_node.kwargs["layer_idx"]
        seq_dim = autofunc_node.kwargs["seq_dim"]
        seq_len = autofunc_node.kwargs["seq_len"]

        with graph.inserting_after(isu_node):
            narrow_layer = graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(isu_node, 0, layer_idx, layer_idx + 1),
            )
            if "val" in isu_node.meta:
                narrow_layer.meta["val"] = isu_node.meta["val"].narrow(0, layer_idx, 1)

        # The pre-squeeze fetch tensor: narrow_seq if we emitted one, else narrow_layer.
        pre_squeeze = narrow_layer
        if seq_len is not None:
            with graph.inserting_after(narrow_layer):
                narrow_seq = graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    args=(narrow_layer, seq_dim, 0, seq_len),
                )
                if "val" in narrow_layer.meta:
                    seq_len_val = seq_len.meta["val"] if isinstance(seq_len, fx.Node) else seq_len
                    narrow_seq.meta["val"] = narrow_layer.meta["val"].narrow(
                        seq_dim, 0, seq_len_val
                    )
            pre_squeeze = narrow_seq

        with graph.inserting_after(pre_squeeze):
            squeeze_op = graph.call_function(
                torch.ops.aten.squeeze.dims,
                args=(pre_squeeze, [0]),
            )
            # 4D meta comes from the fetched-slice getitem (what SDPA expects).
            if getitem_fetched is not None:
                _copy_node_provenance(squeeze_op, getitem_fetched)

        if getitem_fetched is not None:
            getitem_fetched.replace_all_uses_with(squeeze_op)
            get_items.append(getitem_fetched)
            get_item_replacements[getitem_fetched.name] = squeeze_op

        getitem_cache.replace_all_uses_with(isu_node)
        get_items.append(getitem_cache)
        get_item_replacements[getitem_cache.name] = isu_node


def _erase_autofunc_nodes(
    program: torch.export.ExportedProgram,
    graph: torch.fx.Graph,
    get_items: list[fx.Node],
    get_item_replacements: dict[str, fx.Node],
    autofunc_nodes: tuple[fx.Node, ...],
) -> None:
    """Erase the retired getitem + autofunc nodes and repoint output specs.

    Any graph-signature output that named a retired getitem is repointed at that
    getitem's replacement node so the exported program stays consistent.
    """
    signature_replacements: dict[int, fx.Node] = {}
    for getitem_node in get_items:
        graph.erase_node(getitem_node)
        for i, spec in enumerate(program.graph_signature.output_specs):
            if spec.arg.name == getitem_node.name:
                signature_replacements[i] = get_item_replacements[getitem_node.name]

    for auto_func_node in autofunc_nodes:
        graph.erase_node(auto_func_node)

    for idx, repl_node in signature_replacements.items():
        program.graph_signature.output_specs[idx].arg.name = repl_node.name


def remove_functionalization(program: torch.export.ExportedProgram) -> None:
    """
    Remove auto-functionalization wrappers inserted by torch.export.

    torch.export inserts functionalization and getitem ops in place of every
    mutable custom op. Since Core AI doesn't support higher-order ops (the
    functionalization op has the custom op function as an argument), this
    removes those nodes and replaces them with the immutable custom op directly.

    Two mutable ops are handled:

    - ``coreai::mutable_slice_update`` → a single ``immutable_slice_update``
      (used by ``SSMState`` and the iOS cache).
    - ``coreai::mutable_cache_update_and_fetch`` → ``aten.unsqueeze`` followed by
      ``immutable_slice_update``  (the 5D mutated cache) followed by
      ``aten.slice`` (layer) + ``aten.slice`` (seq) + ``aten.squeeze``
      (the 4D fetched slice feeding SDPA). The fused op has two consumers via getitem:
      index 0 is the fetched slice, index 1 is the mutated cache.

    Note: Any other auto-functionalized custom op will raise an assertion error.
    """
    graph = program.graph_module.graph
    assert isinstance(graph, torch.fx.Graph)

    slice_update_autofuncs, cache_update_autofuncs = _partition_autofunc_nodes(graph)

    get_items: list[fx.Node] = []
    get_item_replacements: dict[str, fx.Node] = {}

    _replace_slice_update_autofuncs(graph, slice_update_autofuncs, get_items, get_item_replacements)
    _replace_cache_update_autofuncs(graph, cache_update_autofuncs, get_items, get_item_replacements)

    _erase_autofunc_nodes(
        program,
        graph,
        get_items,
        get_item_replacements,
        (*slice_update_autofuncs.values(), *cache_update_autofuncs.values()),
    )

    # Recompile the program
    program.graph_module.recompile()


# ---------------------------------------------------------------------------
# Custom MLIR lowerings
# ---------------------------------------------------------------------------


def custom_lowering_slice_update(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower immutable_slice_update to coreai.slice_update."""
    x, update, begin, end = _get_operands(values_map, node, [0, 1, 2, 3])
    strides = [1] * x.type.rank
    return coreai.slice_update(x, begin, end, strides, update)


def custom_lowering_composite_op_inputs(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower CompositeOps::label_tensor_as_input to a passthrough."""
    return _get_operand(values_map, node, 0)


def custom_lowering_composite_op_outputs(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower CompositeOps::label_tensor_as_output to a passthrough."""
    return _get_operand(values_map, node, 0)


def custom_lowering_dequantize_per_tensor(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower dequantize_per_tensor to coreai.dequantize."""
    input, scale, zp = _get_operands(values_map, node, [0, 1, 2])
    offset2 = coreai.constant(0, dtype=scale.type.element_type)
    return coreai.blockwise_shift_scale(input, scale, zp, offset2)


def custom_lowering_fused_gather_dequant(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower coreai::fused_dequant_gather_reshape to a composite op."""
    emb_table, input_ids, scale = _get_operands(values_map, node, [0, 1, 2])

    hidden_size = emb_table.type.shape[-1]
    input_id_shape = coreai.get_shape(input_ids)
    final_shape = coreai.concat(
        0,
        [
            input_id_shape,
            coreai.constant([1, hidden_size], dtype=input_id_shape.type.element_type),
        ],
    )

    scale_shape = coreai.slice_(final_shape, [1], [4], [1])

    input_names = ["embedding_table", "input_ids", "scale", "final_shape"]
    output_names = ["output"]
    op_attributes: dict = {}
    composite_decl = generate_composite_decl(
        emb_table.context,
        "fused_interleaved_embedding_gather_dequant_reshape",
        input_names,
        output_names,
        op_attributes,
    )
    emb_enc = HardwareConstraints(
        AllocationType.IOSurface, alignments=[1, 1, 1, 1], interleave=[8, 1, 1]
    )

    if emb_table.type.encoding is None or emb_table.type.encoding != emb_enc.to_mlir():
        emb_table = coreaix.copy_with_constraints(emb_table, emb_enc)

    @coreai.graph(no_inline=True, composite_decl=composite_decl)
    def fused_interleaved_embedding_gather_dequant_reshape(
        embedding_table: Annotated[Value, TensorSpec(encoding=emb_enc)],
        input_ids: Value,
        scale: Value,
        final_shape: Value,
    ) -> Value:
        embedding_table = coreaix.copy_discarding_constraints(embedding_table)
        gathered = coreai.gather_nd(embedding_table, coreai.expand_dims(input_ids, [2]))
        dequantized = coreai.blockwise_shift_scale(
            gathered,
            coreai.reshape(coreai.slice_(scale, [0, 0, 0], [1, 1, 1], [1, 1, 1]), []),
            coreai.constant(0, dtype=np.int8),
            coreai.constant(0, dtype=np.float16),
        )
        return dequantized

    return fused_interleaved_embedding_gather_dequant_reshape(
        emb_table, input_ids, coreai.broadcast_to(scale, scale_shape), final_shape
    )[0]


def custom_lowering_rope_gather_cached_cos_sin(values_map, node, location):  # type: ignore[no-untyped-def]
    """Lower coreai::rope_gather_cached_cos_sin to a composite gather op with IOSurface
    constraints."""
    pos_ids, cos, sin = _get_operands(values_map, node, [0, 1, 2])

    constraints = HardwareConstraints(
        AllocationType.IOSurface, alignments=[1, 1, 32, 1], interleave=[1, 1, 1]
    )
    input_names = ["pos_ids", "cos_cache", "sin_cache"]
    output_names = ["gathered_cos", "gathered_sin"]
    op_attributes: dict = {}
    composite_decl = generate_composite_decl(
        pos_ids.context,
        "rope_cached_cos_sin_gather",
        input_names,
        output_names,
        op_attributes,
    )

    @coreai.graph(no_inline=True, composite_decl=composite_decl)
    def rope_gather(pos_ids: Value, cos_cache: Value, sin_cache: Value) -> OpResultList:
        pos_ids = coreai.cast(pos_ids, dtype=np.int32)
        pos_ids = coreai.expand_dims(pos_ids, [2])
        gathered_cos = coreai.gather_nd(cos_cache, pos_ids)
        gathered_sin = coreai.gather_nd(sin_cache, pos_ids)
        gathered_cos = coreaix.copy_with_constraints(gathered_cos, constraints)
        gathered_sin = coreaix.copy_with_constraints(gathered_sin, constraints)
        return gathered_cos, gathered_sin

    gathered_cos, gathered_sin = rope_gather(pos_ids, cos, sin)
    g_cos = coreaix.copy_discarding_constraints(gathered_cos)
    g_sin = coreaix.copy_discarding_constraints(gathered_sin)
    return g_cos, g_sin


def register_custom_torch_lowering(converter) -> None:  # type: ignore[no-untyped-def]
    """Register all custom MLIR lowerings on the given TorchImporter converter."""
    converter.register_torch_lowering("coreai::immutable_slice_update.default")(
        custom_lowering_slice_update
    )
    converter.register_torch_lowering("CompositeOps::label_tensor_as_input.default")(
        custom_lowering_composite_op_inputs
    )
    converter.register_torch_lowering("CompositeOps::label_tensor_as_output.default")(
        custom_lowering_composite_op_outputs
    )
    converter.register_torch_lowering("coreai::dequantize_per_tensor.default")(
        custom_lowering_dequantize_per_tensor
    )
    converter.register_torch_lowering("coreai::fused_dequant_gather_reshape.default")(
        custom_lowering_fused_gather_dequant
    )
    converter.register_torch_lowering("coreai::rope_gather_cached_cos_sin.default")(
        custom_lowering_rope_gather_cached_cos_sin
    )
