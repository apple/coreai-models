# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared utilities for the parity tests.

Exposes test base classes (``ForCausalLMTestBase``), reference helpers
(``compare_state_dict``, ``load_state_dict_from_ref_model``,
``create_test_inputs``, ``assert_close``), end-to-end run helpers
(``run_compare_coreai``, ``run_compare_coreai_explicit_kv_cache``,
``run_torch_prompt_extend_test``), and layer-count utilities
(``get_layer_counts``, ``assert_layer_counts``). Several optional
dependencies (``coreai.authoring.AIProgram``,
``coreai.runtime.NDArray``, the stateful exporter) are imported defensively
so this module can be imported for ``pytest --collect-only`` even when those
symbols are unavailable in the current environment.
"""

from __future__ import annotations

import asyncio
import collections.abc
import inspect
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import pytest
import torch
import torch.nn as nn
import transformers
from coreai_torch import TorchConverter

# ``coreai_models.export.mlir_ops`` goes through
# ``coreai_models/export/__init__.py``, which eagerly imports
# ``pipeline`` -> ``macos`` -> ``coreai.authoring.AIProgram``. On
# ``coreai-core`` builds where ``AIProgram`` is unavailable, that fails.
# To keep this module importable for ``pytest --collect-only`` we defer
# those imports until they're actually called.
if TYPE_CHECKING:
    from coreai_models.export.mlir_ops import (
        register_custom_torch_lowering,
        remove_functionalization,
    )
else:
    try:
        from coreai_models.export.mlir_ops import (
            register_custom_torch_lowering,
            remove_functionalization,
        )
    except ImportError:
        # Fallback stubs raise only when an actual test invokes them. The
        # collection-only gate doesn't trigger these code paths.
        def register_custom_torch_lowering(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            raise ImportError(
                "coreai_models.export.mlir_ops.register_custom_torch_lowering "
                "is unavailable in this environment"
            )

        def remove_functionalization(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            raise ImportError(
                "coreai_models.export.mlir_ops.remove_functionalization is "
                "unavailable in this environment"
            )


from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.macos.cache import KVCache

from ._deps import _HAS_COREAI, _hf_hub_reachable
from .common.names import (
    key_cache_swift_name,
    value_cache_swift_name,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram
    from coreai.runtime import NDArray
else:
    if _HAS_COREAI:
        from coreai.authoring import AIProgram
        from coreai.runtime import NDArray
    else:
        AIProgram = Any
        NDArray = Any

if _HAS_COREAI:
    from .export.exporters.coreai_exporter import CoreaiStatefulExporter
else:
    CoreaiStatefulExporter = None  # type: ignore[assignment]

TensorOrArray: TypeAlias = torch.Tensor | np.ndarray
TensorOrArrayCollection: TypeAlias = TensorOrArray | list[TensorOrArray] | tuple[TensorOrArray, ...]


def compare_state_dict(ref_model: torch.nn.Module, model: torch.nn.Module) -> None:
    ref_state_dict = ref_model.state_dict()
    model_state_dict = model.state_dict()
    assert set(ref_state_dict.keys()) == set(model_state_dict.keys()), (
        "State dictionaries do not have the same keys"
    )
    for key in ref_state_dict:
        assert torch.allclose(ref_state_dict[key], model_state_dict[key])


def load_state_dict_from_ref_model(
    model: BaseForCausalLM,
    ref_model: transformers.modeling_utils.PreTrainedModel,
) -> None:
    """
    Load state dict from reference model into target model.

    This utility handles the common pattern of:
    1. Getting state_dict from ref_model
    2. Calling _mutate_state_dict on model
    3. Loading the state_dict into model

    Args:
        model: Target model to load state dict into
        ref_model: Reference model to get state dict from
    """
    state_dict = ref_model.state_dict()
    if not isinstance(state_dict, collections.abc.MutableMapping):
        # some HF models uses immutable state dict
        # (e.g. GPT-OSS uses collections.OrderedDict)
        # so we make a shallow copy into a mutable dict
        state_dict = dict(state_dict)
    model._mutate_state_dict(state_dict)
    model.load_state_dict(state_dict)


def create_test_inputs(
    config,
    batch_size: int = 1,
    seq_len: int = 2048,
    offset: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create test input_ids and position_ids for Core AI tests.

    Args:
        config: Model configuration with vocab_size attribute
        batch_size: Batch size for inputs (default: 1)
        seq_len: Sequence length for input_ids (default: 2048)
        offset: Offset to add to position_ids (default: 1024)

    Returns:
        Tuple of (input_ids, position_ids)
    """
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
    position_ids = (
        torch.arange(seq_len + offset, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, seq_len + offset)
    )
    return input_ids, position_ids


def create_dynamic_shapes_for_explicit_kv_coreai_test(
    max_seq_len: int,
    min_seq_len: int = 2048,
    kv_max_seq_len: int | None = None,
) -> dict:
    """Create dynamic shapes for Core AI tests with explicit KV cache inputs.

    Returns dynamic shapes dict for (input_ids, position_ids, k_cache, v_cache).

    Args:
        max_seq_len: Max sequence length for input_ids/position_ids dynamic dims.
        min_seq_len: Min sequence length for position_ids and KV cache dynamic dims.
        kv_max_seq_len: Max sequence length for KV cache dynamic dim. If None,
            defaults to max_seq_len.
    """
    if kv_max_seq_len is None:
        kv_max_seq_len = max_seq_len
    return {
        "input_ids": {1: torch.export.Dim("seq_ids", max=max_seq_len - 2)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=min_seq_len, max=max_seq_len - 1)},
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=min_seq_len, max=kv_max_seq_len
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=min_seq_len, max=kv_max_seq_len
            )
        },
    }


def get_torch_export_graph(
    model: torch.nn.Module,
    batch_size: int = 1,
    query_len: int = 8,
    max_context_length: int = 16,
) -> torch.export.ExportedProgram:
    """
    Export a CausalLM model to torch ExportedProgram.

    Args:
        model: PyTorch model
        batch_size: Batch size for inputs
        query_len: Query sequence length
        max_context_length: Maximum context length for dynamic shapes

    Returns:
        ExportedProgram with decompositions applied
    """
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 32000

    input_ids = torch.randint(1, vocab_size, (batch_size, query_len), dtype=torch.int32)
    offset = 4  # position_ids must be longer than input_ids so torch.export sees independent dims
    position_ids = (
        torch.arange(query_len + offset, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, query_len + offset)
    )
    k_cache, v_cache = KVCache.create_cache_tensors(model.config)
    inputs = (input_ids, position_ids, k_cache, v_cache)
    dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(
        max_context_length,
        min_seq_len=3,
        kv_max_seq_len=k_cache.shape[KVCache.seq_len_dim()],
    )

    with torch.no_grad():
        exported_program = torch.export.export(model, args=inputs, dynamic_shapes=dynamic_shapes)
        exported_program = exported_program.run_decompositions()

    return exported_program


def _verify_and_get_next_token(
    output: torch.Tensor,
    output_hf: torch.Tensor,
    rtol: float,
    atol: float,
    strict_compare_numerical: bool = True,
    is_extend: bool = False,
) -> torch.Tensor:
    predicted_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
    predicted_token_hf = torch.argmax(output_hf[:, -1, :], dim=-1, keepdim=True).to(torch.int32)

    assert predicted_token.shape == (1, 1)
    assert predicted_token_hf.shape == (1, 1)
    assert torch.equal(predicted_token, predicted_token_hf)

    if strict_compare_numerical:
        if is_extend:
            output_hf = output_hf[:, -1:, :]
        assert_close(output, output_hf, rtol=rtol, atol=atol)

    return predicted_token


def run_torch_prompt_extend_test(
    model: torch.nn.Module,
    hf_model: torch.nn.Module,
    precision: torch.dtype,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    extend_steps: int = 3,
    strict_compare_numerical: bool = True,
    skip_dtype_cast: bool = False,
) -> None:
    # set models precision
    if not skip_dtype_cast:
        model = model.to(precision)
        hf_model = hf_model.to(precision)

    # prompt
    batch_size = 1
    seq_len = 1024
    config = hf_model.config
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)

    k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=precision)
    output = model(input_ids, position_ids, k_cache, v_cache)

    output_hf = hf_model(input_ids).logits

    new_ids = _verify_and_get_next_token(
        output,
        output_hf,
        rtol,
        atol,
        strict_compare_numerical,
        is_extend=False,
    )

    # extend
    hf_inputs = input_ids
    for step in range(extend_steps):
        print(f"step {step}")
        new_position_id = torch.tensor([[position_ids.shape[-1]]]).expand(batch_size, 1)
        position_ids = torch.concat([position_ids, new_position_id], axis=-1)
        hf_inputs = torch.concat([hf_inputs, new_ids], dim=-1)

        output = model(new_ids, position_ids, k_cache, v_cache)

        output_hf = hf_model(hf_inputs).logits

        new_ids = _verify_and_get_next_token(
            output, output_hf, rtol, atol, strict_compare_numerical, is_extend=True
        )


def _construct_causal_mask(
    max_position_embeddings: int,
    seq_len: int,
    offset: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a causal attention mask for iOS models.

    Builds a lower-triangular mask of shape ``(1, max_position_embeddings, 1, seq_len)``
    where positions that should not be attended to are filled with ``-inf``. For each
    query position ``i``, key positions beyond ``offset+i`` are blocked. This also blocks
    empty cache positions beyond the sequence length.
    """
    causal_mask = torch.zeros(
        (1, max_position_embeddings, 1, seq_len),
        dtype=dtype,
    )
    for i in range(seq_len):
        causal_mask[:, offset + i + 1 :, :, i] = float("-inf")
    return causal_mask


def run_torch_prompt_extend_test_ios(
    model: torch.nn.Module,
    hf_model: torch.nn.Module,
    precision: torch.dtype,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    extend_steps: int = 3,
    strict_compare_numerical: bool = True,
    use_additional_transpose: bool = False,
) -> None:
    # Set models to specified precision
    model = model.to(precision)
    hf_model = hf_model.to(precision)

    # Setup initial prompt
    batch_size = 1
    seq_len = 100
    config = hf_model.config

    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    cache_offset = torch.tensor([0], dtype=torch.int32)

    # Initialize KV cache
    k_cache, v_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=precision)

    # Generate causal mask and run prompt processing
    causal_mask = _construct_causal_mask(
        config.max_position_embeddings, seq_len, cache_offset[0], precision
    )
    output = model(input_ids, position_ids, cache_offset, causal_mask, k_cache, v_cache)

    if use_additional_transpose:
        output = output.transpose(2, 1)

    output = output.squeeze(2)
    output_hf = hf_model(input_ids).logits

    predicted_token = _verify_and_get_next_token(
        output,
        output_hf,
        rtol,
        atol,
        strict_compare_numerical,
        is_extend=False,
    )

    # Test token extension
    hf_context = input_ids
    for step in range(extend_steps):
        print(f"Extension step {step}")

        # Update position tracking
        cache_offset += position_ids.shape[-1]
        position_ids = cache_offset.expand(batch_size, 1)

        # Generate next token
        causal_mask = _construct_causal_mask(
            config.max_position_embeddings, 1, cache_offset[0], precision
        )
        output = model(predicted_token, position_ids, cache_offset, causal_mask, k_cache, v_cache)
        output = output.squeeze(2)

        # Update HuggingFace context and generate
        hf_context = torch.concat([hf_context, predicted_token], dim=-1)
        output_hf = hf_model(hf_context).logits

        # Verify outputs and get next token
        predicted_token = _verify_and_get_next_token(
            output, output_hf, rtol, atol, strict_compare_numerical, is_extend=True
        )


def run_torch_prompt_extend_static_test(
    model: torch.nn.Module,
    hf_model: torch.nn.Module,
    precision: torch.dtype,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    extend_steps: int = 3,
    strict_compare_numerical: bool = True,
) -> None:
    # set models precision
    model = model.to(precision)
    hf_model = hf_model.to(precision)

    # prompt
    batch_size = 1
    seq_len = 2048
    config = hf_model.config
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=precision)
    output = model(input_ids, position_ids, k_cache, v_cache)
    output_hf = hf_model(input_ids).logits

    # the sample is should be the same
    new_ids = torch.argmax(output[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
    new_hf_ids = torch.argmax(output_hf[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
    assert new_ids.shape == (1, 1)
    assert new_hf_ids.shape == (1, 1)
    assert torch.equal(new_ids, new_hf_ids)

    if strict_compare_numerical:
        assert_close(output, output_hf, rtol=rtol, atol=atol)

    # extend
    hf_inputs = input_ids
    for step in range(extend_steps):
        print(f"step {step}")
        new_position_id = torch.tensor([[position_ids.shape[-1]]]).expand(batch_size, 1)
        position_ids = torch.concat([position_ids, new_position_id], axis=-1)
        hf_inputs = torch.concat([hf_inputs, new_ids], dim=-1)
        output = model(new_ids, position_ids, k_cache, v_cache)
        output_hf = hf_model(hf_inputs).logits

        new_ids = _verify_and_get_next_token(
            output, output_hf, rtol, atol, strict_compare_numerical
        )


def assert_close(
    a: TensorOrArrayCollection,
    b: TensorOrArrayCollection,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    def sanitize_input(t: TensorOrArrayCollection) -> list[np.ndarray]:
        if not isinstance(t, (list, tuple)):
            t = (t,)

        res = []
        for v in t:
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                res.append(v.detach().numpy())
            elif type(v).__name__ == "_DLTensor":
                v = torch.from_dlpack(v)
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                res.append(v.detach().numpy())
            elif hasattr(v, "numpy"):
                # Handle NDArray objects
                arr = v.numpy()
                if arr.dtype == np.float16:  # Convert bfloat16 equivalent
                    arr = arr.astype(np.float32)
                res.append(arr)
            else:
                res.append(v)

        return res

    a, b = sanitize_input(a), sanitize_input(b)
    assert len(a) == len(b)

    for v1, v2 in zip(a, b, strict=True):
        assert v1.shape == v2.shape, f"{v1.shape} vs {v2.shape}."

        # get the idx of max abs and relative error
        abs_err = np.abs(v1 - v2)
        rel_err = abs_err / np.maximum(np.abs(v2), 1e-12)
        idx_abs = np.unravel_index(np.argmax(abs_err), abs_err.shape)
        idx_rel = np.unravel_index(np.argmax(rel_err), rel_err.shape)

        # Fetch the corresponding values
        err_msg = (
            f"max abs error {abs_err[idx_abs]} with ({v1[idx_abs]},{v2[idx_abs]})."
            + f"max rel error {rel_err[idx_rel]} with ({v1[idx_rel]},{v2[idx_rel]})."
        )
        if not np.allclose(v1, v2, rtol=rtol, atol=atol):
            print(err_msg)
            np.testing.assert_allclose(v1, v2, rtol=rtol, atol=atol)
            raise ValueError(err_msg)


def _construct_dynamic_shape(
    model: torch.nn.Module,
    dynamic_shape_range: list[dict[str, tuple[int, int]]],
) -> dict[str, dict[str, torch.export.Dim]]:
    sig = inspect.signature(model.forward)
    arg_names = [name for name, _ in sig.parameters.items()]
    assert len(dynamic_shape_range) == len(arg_names)

    # construct the dynamic shape
    res = {}
    for name, shape in zip(arg_names, dynamic_shape_range, strict=True):
        tmp = {}
        for k, v in shape.items():
            tmp[k] = torch.export.Dim(name=f"{name}_dim_{k}", min=v[0], max=v[1])
        res[name] = tmp
    return res


def _sanitize_inputs(
    inputs: list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    elif isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    elif not isinstance(inputs, tuple):
        err = f"Invalid inputs type {type(inputs)}"
        raise ValueError(err)
    return inputs


def _convert_inputs_to_fp16(inputs: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    res = []
    for val in inputs:
        if val.dtype == torch.float32:
            res.append(val.to(torch.float16))
        else:
            res.append(val)
    return tuple(res)


def _expand_inputs_for_coreai(inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    """
    Expand inputs for Core AI runner by flattening nested tuples/lists.

    Args:
        inputs: Input tensors (possibly nested in tuples/lists)

    Returns:
        Flattened list of tensors
    """
    coreai_input_as_list = []
    for val in inputs:
        if isinstance(val, (list, tuple)):
            for t in val:
                coreai_input_as_list.append(t)
        else:
            coreai_input_as_list.append(val)
    return coreai_input_as_list


async def _async_get_coreai_program(
    *,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: list[dict[str, tuple[int, int]]] | None = None,
    use_fp16_precision: bool = False,
) -> AIProgram:
    """
    Export model and return the Core AI program.

    This is the core export function used by both layer count tests and
    numerical comparison tests.

    Args:
        model: PyTorch model to export
        inputs: Input tensors for the model
        dynamic_shapes: Optional dynamic shape specifications
        use_fp16_precision: Whether to convert to FP16

    Returns:
        AIProgram containing the Core AI MLIR
    """
    # sanitize dynamic shape
    if dynamic_shapes is not None and isinstance(dynamic_shapes, list):
        dynamic_shapes = _construct_dynamic_shape(
            model,
            dynamic_shapes,
        )

    # sanitize inputs
    inputs = _sanitize_inputs(inputs)

    # convert the inputs and torch model to fp16
    if use_fp16_precision:
        inputs = _convert_inputs_to_fp16(inputs)
        model = model.half()

    # torch.export
    with torch.no_grad():
        exported_program = torch.export.export(
            model,
            args=inputs,
            dynamic_shapes=dynamic_shapes,
        )
        exported_program = exported_program.run_decompositions()
        remove_functionalization(exported_program)

    # import into Core AI and run essential optimization passes
    assert not hasattr(model, KVCache.HF_K_BUFFER_NAME), (
        "caches should not be registered as buffers"
    )
    coreai_input_as_list = _expand_inputs_for_coreai(inputs)
    input_names = [f"input_{i}" for i in range(len(coreai_input_as_list))]
    output_names = None

    importer = TorchConverter().add_exported_program(
        exported_program,
        input_names=input_names,
        output_names=output_names,
    )
    register_custom_torch_lowering(importer)
    coreai_program = importer.to_coreai()

    coreai_program.optimize()

    return coreai_program


async def _asyn_run_compare_coreai(
    *,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: list[dict[str, tuple[int, int]]] | None = None,
    use_fp16_precision: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    assert not hasattr(model, KVCache.HF_K_BUFFER_NAME), "caches should not registered as buffers."

    # sanitize inputs for later use
    inputs = _sanitize_inputs(inputs)

    # convert the inputs and torch model to fp16
    if use_fp16_precision:
        inputs = _convert_inputs_to_fp16(inputs)
        model = model.half()

    # Get the Core AI program using shared export function
    coreai_program = await _async_get_coreai_program(
        model=model,
        inputs=inputs,
        dynamic_shapes=dynamic_shapes,
        use_fp16_precision=False,  # Already converted above
    )

    # expand the inputs for the runner
    coreai_input_as_list = _expand_inputs_for_coreai(inputs)

    # compare output
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmpdir:
        # save and load the model via the AIModel API
        asset = coreai_program.save_asset(Path(tmpdir))
        async with asset.executable() as aimodel:
            function = aimodel.load_function("main")

            # determine io names from function descriptor
            input_names = function.desc.input_names
            output_names = function.desc.output_names
            kv_buffer_names = (KVCache.HF_K_BUFFER_NAME, KVCache.HF_V_BUFFER_NAME)
            assert len(input_names) == len(coreai_input_as_list)
            coreai_inputs = {
                name: NDArray(data=tensor.contiguous())
                for name, tensor in zip(input_names, coreai_input_as_list, strict=True)
            }
            coreai_outputs = await function(coreai_inputs)

            # torch predicts
            torch_output = model(*inputs)

            # compare the numerical results for non-stateful output
            runtime_outputs = [
                coreai_outputs[v].numpy() for v in output_names if v not in kv_buffer_names
            ]
            if not isinstance(torch_output, (list, tuple)):
                torch_output = (torch_output,)

            assert len(torch_output) == len(runtime_outputs)
            for a, b in zip(torch_output, runtime_outputs, strict=True):
                assert_close(a, b, atol=atol, rtol=rtol)


def run_compare_coreai(
    *,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: list[dict[str, tuple[int, int]]] | None = None,
    use_fp16_precision: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    asyncio.run(
        _asyn_run_compare_coreai(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            use_fp16_precision=use_fp16_precision,
            atol=atol,
            rtol=rtol,
        )
    )


async def _async_run_compare_coreai_explicit_kv_cache(
    *,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: dict,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Shared async helper for testing models with explicit KV cache inputs through Core AI.

    Handles: export -> convert -> Core AI runtime -> compare torch vs Core AI outputs.
    Assumes inputs are (input_ids, position_ids, k_cache, v_cache).
    """
    input_ids, position_ids, k_cache, v_cache = inputs

    coreai_stateful_exporter = CoreaiStatefulExporter(
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=(key_cache_swift_name, value_cache_swift_name),
    )
    coreai_program = await coreai_stateful_exporter._async_export_and_optimize(
        model,
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "k_cache": k_cache,
            "v_cache": v_cache,
        },
        dynamic_shapes=dynamic_shapes,
    )

    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmpdir:
        asset = coreai_program.save_asset(Path(tmpdir))
        async with asset.executable() as aimodel:
            function = aimodel.load_function("main")

            k_cache.zero_()
            v_cache.zero_()
            coreai_inputs = {
                "input_ids": NDArray(data=input_ids.contiguous()),
                "position_ids": NDArray(data=position_ids.contiguous()),
            }
            state = {
                key_cache_swift_name: NDArray(data=k_cache),
                value_cache_swift_name: NDArray(data=v_cache),
            }
            coreai_results = await function(coreai_inputs, state=state)

    # Torch reference
    k_cache.zero_()
    v_cache.zero_()
    with torch.no_grad():
        torch_logits = model(*inputs)

    assert_close(torch_logits, coreai_results["logits"], atol=atol, rtol=rtol)
    assert_close(k_cache, state[key_cache_swift_name].numpy(), atol=atol, rtol=rtol)
    assert_close(v_cache, state[value_cache_swift_name].numpy(), atol=atol, rtol=rtol)


def run_compare_coreai_explicit_kv_cache(
    *,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: dict,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Test a model with explicit KV cache inputs through the Core AI pipeline.

    Assumes inputs are (input_ids, position_ids, k_cache, v_cache).
    Exports, converts, runs Core AI runtime, and compares outputs against torch.
    """
    asyncio.run(
        _async_run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=atol,
            rtol=rtol,
        )
    )


class ForCausalLMTestBase:
    """
    Base class for ForCausalLM model tests.

    Subclasses should define:
    - _toy_model_id: str - HuggingFace model ID for testing
    - _model_class: Type[BaseForCausalLM] - Model class to instantiate
    - _test_weights_tying: bool - Whether to test weight tying (default: False)

    This class provides common test methods for dtype conversion and casting.
    """

    _toy_model_id: str
    _model_class: type[BaseForCausalLM]
    _test_weights_tying: bool = False
    _test_kv_cache: bool = True
    _test_weight_activation_quantization: bool = False
    _test_eager_activation_quantization: bool = True

    @pytest.fixture(autouse=True)
    def _skip_if_hf_unreachable(self) -> None:
        """Skip tests in this class when the HF Hub cannot be reached.

        Every test in ``ForCausalLMTestBase`` ultimately calls
        ``_model_class.from_hf(self._toy_model_id)`` (or
        ``transformers.AutoConfig.from_pretrained``), so a Hub block produces
        the same proxy/tunnel error in every case. We probe ``model_info``
        once per test with a short timeout so sandboxed environments skip
        cleanly instead of churning through retries.
        """
        if not _hf_hub_reachable(self._toy_model_id):
            pytest.skip(
                f"HuggingFace Hub unreachable for {self._toy_model_id!r}; "
                "skipping network-dependent ForCausalLM test"
            )

    def _assert_floating_point_dtype(self, model: torch.nn.Module, dtype: torch.dtype) -> None:
        """Assert all floating-point parameters and state dict entries match the expected dtype."""
        for name, v in model.named_parameters():
            if v.is_floating_point():
                assert v.dtype == dtype, f"param {name}: expected {dtype}, got {v.dtype}"
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                assert v.dtype == dtype, f"state {k}: expected {dtype}, got {v.dtype}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_to_diff_dtype(self, dtype: torch.dtype) -> None:
        """Test model conversion using .to() method."""
        model = self._model_class.from_hf(self._toy_model_id)
        model = model.to(dtype)
        self._assert_floating_point_dtype(model, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_from_hf_dtype(self, dtype: torch.dtype) -> None:
        model = self._model_class.from_hf(self._toy_model_id, target_dtype=dtype)
        self._assert_floating_point_dtype(model, dtype)

    def test_cast(self) -> None:
        """Test model casting using .half(), .bfloat16(), and .float() methods."""
        model = self._model_class.from_hf(self._toy_model_id)

        # fp16
        model = model.half()
        self._assert_floating_point_dtype(model, torch.float16)

        # bfp16
        model = model.bfloat16()
        self._assert_floating_point_dtype(model, torch.bfloat16)

        # fp32
        model = model.float()
        self._assert_floating_point_dtype(model, torch.float32)

    @pytest.mark.parametrize("random_initialization", [True, False])
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_weights_tying(self, random_initialization) -> None:
        """
        Test that weight tying is correctly implemented.

        Verifies that:
        1. The config has tie_word_embeddings set to True
        2. lm_head and embed_tokens share the same weight tensor
        3. The exported graph only contains lm_head weight (not embed_tokens)
        4. The lm_head weight is used twice in the graph

        Only runs if _test_weights_tying is True for the test class.
        """
        if not self._test_weights_tying:
            pytest.skip("Weight tying test not enabled for this model")

        if random_initialization:
            config = transformers.AutoConfig.from_pretrained(self._toy_model_id)
            # Handle models where config has a text_config attribute (e.g., Gemma3)
            model_config = config.text_config if hasattr(config, "text_config") else config
            model = self._model_class(model_config)
        else:
            model = self._model_class.from_hf(self._toy_model_id)

        # Ensure tie_word_embeddings is set (toy models may not have it configured)
        if not model.config.tie_word_embeddings:
            model.config.tie_word_embeddings = True
            if model.lm_head.weight is not model.model.embed_tokens.weight:
                model.lm_head.weight = model.model.embed_tokens.weight

        assert model.config.tie_word_embeddings
        assert model.lm_head.weight is model.model.embed_tokens.weight

        exported_program = get_torch_export_graph(model)

        lm_head_node = next(
            (node for node in exported_program.graph.nodes if "p_lm_head_weight" in node.name),
            None,
        )
        embed_tokens_node = next(
            (
                node
                for node in exported_program.graph.nodes
                if "p_model_embed_tokens_weight" in node.name
            ),
            None,
        )

        # With tied weights, torch.export may leave a dead embed_tokens
        # placeholder (num_users=0) that DCE doesn't remove. The real
        # deduplication check is that lm_head_weight is used for both
        # embedding and the final linear (num_users=2), and embed_tokens
        # has no users if it exists.
        # This is a bug due to the upgrade of torch
        if embed_tokens_node is not None:
            assert len(embed_tokens_node.users) == 0, (
                f"embed_tokens_weight should have 0 users (dead placeholder) "
                f"but has {len(embed_tokens_node.users)}"
            )
        assert lm_head_node is not None
        assert len(lm_head_node.users) == 2

    @pytest.mark.parametrize("quantization_mode", ["eager", "graph"])
    @pytest.mark.parametrize("activation_quantization", [True, False])
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_weight_activation_quantization(
        self, activation_quantization: bool, quantization_mode: Literal["eager", "graph"]
    ) -> None:
        """
        Test that weight and weight + activation quantization produces a mlirb model
        through Core AI export, in both eager- and graph-mode quantization.
        Only runs if _test_weight_activation_quantization is True for the test class.
        """
        if not self._test_weight_activation_quantization:
            pytest.skip("Weight/Activation Quantization test not enabled for this model")

        if (
            quantization_mode == "eager"
            and activation_quantization
            and not self._test_eager_activation_quantization
        ):
            pytest.skip("Eager-mode activation quantization test not enabled for this model")

        # Same building blocks as `_async_export_model`: load HF ->
        # `quantize_for_export` -> `export_macos_model`. Calling those rather than
        # reimplementing the calibration trace keeps this from drifting from
        # production, as it had. The asset write is skipped; we only confirm
        # quantize + export produces a non-None AIProgram.
        from coreai_models.export.compression import quantize_for_export
        from coreai_models.export.externalize import patch_model_for_externalization
        from coreai_models.export.macos import export_macos_model
        from coreai_models.export.pipeline import ExportConfig

        graph_mode = quantization_mode == "graph"

        hf_config = transformers.AutoConfig.from_pretrained(self._toy_model_id)
        is_gemma = "gemma" in self._model_class.__name__.lower()
        if is_gemma and hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        weight_qspec_dict = {
            "weight": {
                "dtype": "int4",
                "qscheme": "symmetric_with_clipping",
                "granularity": {
                    "type": "per_block",
                    "block_size": 8,
                    "axis": 1,
                },
            }
        }
        activation_qspec_dict = {
            "*": {
                "dtype": "int8",
                "qscheme": "symmetric_with_clipping",
                "granularity": {
                    "type": "per_tensor",
                },
            }
        }
        rms_norm_cls = (
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne"
            if is_gemma
            else "coreai_models.primitives.macos.rms_norm.RMSNorm"
        )
        torch_quantization_config = {
            "global_config": {
                "op_state_spec": weight_qspec_dict,
                "op_input_spec": activation_qspec_dict if activation_quantization else None,
                "op_output_spec": activation_qspec_dict if activation_quantization else None,
            },
            "module_type_configs": {
                "coreai_models.primitives.macos.sdpa.SDPA": None,
                "coreai_models.primitives.macos.rope.RoPE": None,
                rms_norm_cls: None,
            },
            "execution_mode": quantization_mode,
            "calibrate_activations": activation_quantization,
        }

        def calibration_data_fn() -> list[torch.Tensor]:
            return [
                torch.randint(1, hf_config.vocab_size, (1, 16), dtype=torch.int32) for _ in range(2)
            ]

        max_context_length = 4096
        target_dtype = torch.float16
        hf_state_dict_prefix = "language_model." if is_gemma else ""
        hf_config_attr = "text_config" if is_gemma else None

        with tempfile.TemporaryDirectory() as tmpdir:
            layer_mmap_dir = f"{tmpdir}/layers"

            os.makedirs(layer_mmap_dir, exist_ok=True)
            model = self._model_class.from_hf_memory_efficient(
                self._toy_model_id,
                max_context_length=max_context_length,
                target_dtype=target_dtype,
                mmap_path=layer_mmap_dir,
                hf_config_attr=hf_config_attr,
                hf_state_dict_prefix=hf_state_dict_prefix,
            ).eval()

            quantizer_mmap_dir: str | None = None
            # coreai-opt only supports mmap-backed finalization in eager mode
            if not graph_mode:
                quantizer_mmap_dir = f"{tmpdir}/quantized"
                os.makedirs(quantizer_mmap_dir, exist_ok=True)

            externalized_model = None
            if graph_mode:
                patch_model_for_externalization(model)
                externalized_model = model

            model = quantize_for_export(
                model,
                hf_config,
                target_dtype,
                dict(torch_quantization_config),
                calibration_data_fn=calibration_data_fn if activation_quantization else None,
                mmap_dir=quantizer_mmap_dir,
            )

            export_config = ExportConfig(
                hf_model_id=self._toy_model_id,
                max_context_length=max_context_length,
                quantization_mode=quantization_mode,
            )
            coreai_program = export_macos_model(
                model, hf_config, export_config, externalized_model=externalized_model
            )

            assert coreai_program is not None, "export_macos_model returned None, conversion failed"


"""
The layers below are used only for the unittest purpose, in order to match the HF numerical.
"""


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class SparseMoeBlockaScatter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(dim=dim, hidden_dim=hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x = x.view(-1, dim)
        router_logits = self.gate(x)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch * seq_len, dim), dtype=x.dtype)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = x[None, top_x].reshape(-1, dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch, seq_len, dim)

        return final_hidden_states


def switch_block_to_scatter(model: torch.nn.Module) -> torch.nn.Module:
    from coreai_models.models.macos.mixtral import SparseMoeBlock

    # Recursively search for SparseMoeBlock instances in the model
    for name, module in model.named_children():
        if isinstance(module, SparseMoeBlock):
            # Create a new SparseMoeBlockaScatter with the same parameters
            scatter_block = SparseMoeBlockaScatter(
                dim=module.gate.weight.shape[1],
                hidden_dim=module.switch_mlp.gate_proj.weight.shape[1],
                num_experts=module.switch_mlp.gate_proj.weight.shape[0],
                top_k=module.top_k,
                norm_topk_prob=getattr(module, "norm_topk_prob", True),
            )

            # Copy the gate weights
            scatter_block.gate.weight = torch.nn.Parameter(module.gate.weight.clone())

            # Create experts and copy weights from switch_mlp
            for i in range(module.switch_mlp.gate_proj.weight.shape[0]):
                # w1 corresponds to gate_proj
                scatter_block.experts[i].w1.weight = torch.nn.Parameter(
                    module.switch_mlp.gate_proj.weight[i].clone()
                )

                # w2 corresponds to down_proj
                scatter_block.experts[i].w2.weight = torch.nn.Parameter(
                    module.switch_mlp.down_proj.weight[i].clone()
                )

                # w3 corresponds to up_proj
                scatter_block.experts[i].w3.weight = torch.nn.Parameter(
                    module.switch_mlp.up_proj.weight[i].clone()
                )

            # Replace the module
            setattr(model, name, scatter_block)
        else:
            # Recursively process child modules
            switch_block_to_scatter(module)

    return model


# =============================================================================
# Layer Count Verification Utilities
# =============================================================================


@dataclass
class LayerCountResult:
    """Result of layer count comparison."""

    actual_counts: dict[str, int]
    expected_counts: dict[str, int]
    mlir_str: str

    def get_diff(self) -> dict[str, tuple[int, int]]:
        """Returns dict of {op_name: (expected, actual)} for mismatches."""
        all_ops = set(self.actual_counts.keys()) | set(self.expected_counts.keys())
        diff = {}
        for op in all_ops:
            expected = self.expected_counts.get(op, 0)
            actual = self.actual_counts.get(op, 0)
            if expected != actual:
                diff[op] = (expected, actual)
        return diff


def count_coreai_operations(mlir_str: str) -> dict[str, int]:
    """
    Parse MLIR string and count Core AI operations.

    Core AI operations appear in the format: coreai.operation_name
    Examples: coreai.linear, coreai.rms_norm, coreai.batch_matmul

    Args:
        mlir_str: MLIR assembly string

    Returns:
        Dictionary mapping operation names to counts
    """
    pattern = r"coreai\.([a-zA-Z_][a-zA-Z0-9_\.]*)"
    matches = re.findall(pattern, mlir_str)
    return dict(Counter(matches))


def get_layer_counts(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    dynamic_shapes: list[dict[str, tuple[int, int]]] | None = None,
    use_fp16_precision: bool = False,
) -> LayerCountResult:
    """
    Export model to Core AI and count the MLIR operations.

    Uses CoreaiStatefulExporter to match the production export path, including
    composite op externalization (RMSNorm, RoPE, SDPA, etc.).

    Args:
        model: PyTorch model to analyze
        inputs: Input tensor(s) for the model
        dynamic_shapes: Optional dynamic shape specifications
        use_fp16_precision: Whether to convert to FP16

    Returns:
        LayerCountResult with actual counts and the MLIR string
    """
    from .export.exporters.coreai_exporter import (
        CoreaiStatefulExporter,
    )

    inputs = _sanitize_inputs(inputs)

    if use_fp16_precision:
        inputs = _convert_inputs_to_fp16(inputs)
        model = model.half()

    sig = inspect.signature(model.forward)
    param_names = [
        name for name, p in sig.parameters.items() if p.default is inspect.Parameter.empty
    ]
    if len(param_names) < len(inputs):
        param_names = list(sig.parameters.keys())[: len(inputs)]
    reference_inputs = dict(zip(param_names, inputs, strict=False))

    async def _run():
        exporter = CoreaiStatefulExporter()
        program = await exporter._async_export_and_optimize(
            model, reference_inputs, dynamic_shapes=dynamic_shapes
        )
        return program

    coreai_program = asyncio.run(_run())

    mlir_str = coreai_program.module.operation.get_asm(
        large_elements_limit=0,  # Don't print tensor values
        large_resource_limit=0,  # Don't print resources
        enable_debug_info=False,
        pretty_debug_info=False,
        print_generic_op_form=False,
        use_local_scope=False,
        assume_verified=False,
    )

    actual_counts = count_coreai_operations(mlir_str)

    return LayerCountResult(
        actual_counts=actual_counts,
        expected_counts={},  # Will be set by caller
        mlir_str=mlir_str,
    )


def assert_layer_counts(
    result: LayerCountResult,
    expected_counts: dict[str, int],
    strict: bool = True,
) -> None:
    """
    Assert that actual layer counts match expected counts.

    Args:
        result: LayerCountResult from get_layer_counts
        expected_counts: Dict of expected operation counts
        strict: If True (default), fail on extra ops not in expected_counts.
                If False, only check ops listed in expected_counts.
                Use False for tests where some ops may vary (e.g., platform-specific).
    """
    result.expected_counts = expected_counts
    diff = result.get_diff()

    if not strict:
        # Remove ops that are in actual but not in expected
        diff = {k: v for k, v in diff.items() if k in expected_counts}

    if diff:
        error_lines = ["Layer count mismatch:"]
        for op, (expected, actual) in sorted(diff.items()):
            error_lines.append(f"  {op}: expected {expected}, got {actual}")
        raise AssertionError("\n".join(error_lines))
