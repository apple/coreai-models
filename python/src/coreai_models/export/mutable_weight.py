# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export and run modules that hold mutable weights (``MutableWeight``).

``export_mutable_weight_module`` writes a ``.aimodel`` whose mutable weights the runtime
holds on device between calls. ``MutableWeightRunner`` loads it and runs one call at a
time, so the weights persist, change and can be replaced without re-exporting.

How the weights are held is internal to this module: the exporter currently declares each
one as a runtime state, the mechanism the KV cache uses, and the runner passes those states
on every call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.primitives.mutable_weight import MutableWeight

if TYPE_CHECKING:
    from coreai.authoring import AIProgram
    from coreai.runtime import InferenceFunction, NDArray, SpecializationOptions


def export_mutable_weight_module(
    module: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    output_path: Path | str,
    weight_names: Sequence[str] | None = None,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> tuple[AIProgram, Path]:
    """Export ``module`` to a ``.aimodel`` whose mutable weights stay on device.

    The module's ``forward`` must write each of its mutable weights, with
    ``MutableWeight.update`` or, for a weight it only uses, ``MutableWeight.read``. Each
    weight is declared to the runtime under its ``MutableWeight.name``: the exporter maps
    every buffer the graph writes to the ``MutableWeight`` that owns it, so the binding
    does not depend on the order of ``weight_names``.

    Args:
        module: The module to export.
        sample_inputs: Example ``forward`` arguments for tracing; their shapes and
            dtypes matter, not their values.
        input_names: Names of the ``forward`` arguments other than the mutable weights.
        output_names: Names of the ``forward`` return values.
        output_path: Where to write the ``.aimodel``.
        weight_names: The ``MutableWeight.name`` of every weight to keep on device, in
            any order. Defaults to every ``MutableWeight`` in ``module``.
        include_debug_info: Convert in ``DEBUG`` mode rather than ``RELEASE``; see
            ``export_to_coreai``.

    Returns:
        The converted program and ``output_path``.

    Raises:
        ValueError: If ``weight_names`` names something that is not a ``MutableWeight``
            of ``module``, or a name twice; if ``forward`` does not write a weight it
            names, which happens when the weight is only read with
            ``MutableWeight.value``; if ``forward`` writes a weight it does not name, or
            a buffer or input that is not a ``MutableWeight``; or if two
            ``MutableWeight`` s in ``module`` share a name.
    """
    from coreai_models.export.macos import export_and_decompose, make_torch_converter
    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    module = module.eval()
    output_path = Path(output_path)

    exported = export_and_decompose(module, tuple(sample_inputs))
    # Checked here rather than in the converter's export_fn, which wraps every error.
    state_names = _state_names_by_owner(module, exported, weight_names)

    converter = make_torch_converter(include_debug_info)
    converter.add_pytorch_module(
        module,
        export_fn=lambda _: exported,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        state_names=state_names,
    )
    register_custom_torch_lowering(converter)
    program: AIProgram = converter.to_coreai()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(output_path)
    return program, output_path


def _state_names_by_owner(
    module: nn.Module,
    exported: torch.export.ExportedProgram,
    weight_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """The ``MutableWeight.name`` of each buffer ``exported`` writes, in graph order.

    The converter binds ``state_names`` by position to the graph's mutated buffer inputs,
    so the names are listed in that order, each taken from the weight that owns the buffer.
    """
    owner_of_buffer: dict[str, str] = {}
    path_of_name: dict[str, str] = {}
    for path, submodule in module.named_modules():
        if not isinstance(submodule, MutableWeight):
            continue
        if submodule.name in path_of_name:
            raise ValueError(
                f"MutableWeights {path_of_name[submodule.name]!r} and {path!r} share the "
                f"name {submodule.name!r}; each needs its own runtime name"
            )
        path_of_name[submodule.name] = path
        owner_of_buffer[f"{path}._data" if path else "_data"] = submodule.name

    signature = exported.graph_signature
    if signature.user_inputs_to_mutate:
        raise ValueError(
            f"forward writes its input(s) {sorted(signature.user_inputs_to_mutate.values())}; "
            "only MutableWeights can be written"
        )
    mutated = set(signature.buffers_to_mutate.values())
    written: list[str] = []
    for spec in signature.input_specs:
        if spec.kind is not InputKind.BUFFER or spec.target not in mutated:
            continue
        if spec.target not in owner_of_buffer:
            raise ValueError(
                f"forward writes the buffer {spec.target!r}, which is not a MutableWeight's"
            )
        written.append(owner_of_buffer[spec.target])

    expected = list(path_of_name) if weight_names is None else list(weight_names)
    duplicates = sorted({name for name in expected if expected.count(name) > 1})
    if duplicates:
        raise ValueError(f"weight_names lists {duplicates} more than once")
    unknown = [name for name in expected if name not in path_of_name]
    if unknown:
        raise ValueError(
            f"weight_names {unknown} are not MutableWeights of the module; "
            f"it has {list(path_of_name)}"
        )
    unwritten = [name for name in expected if name not in written]
    if unwritten:
        raise ValueError(
            f"forward does not write the mutable weight(s) {unwritten}. A mutable weight "
            "stays on device only when forward writes it; use MutableWeight.read() for a "
            "weight forward uses without changing."
        )
    unlisted = [name for name in written if name not in expected]
    if unlisted:
        raise ValueError(f"forward writes the mutable weight(s) {unlisted}, not in weight_names")
    return tuple(written)


class MutableWeightRunner:
    """Runs an exported mutable-weight module, keeping its weights on device.

    Each :meth:`step` runs one function of the model once; the mutable weights persist
    and change between calls, and the host passes only the other inputs.
    :meth:`set_weight` replaces a weight's value between calls, and
    :meth:`weights_snapshot` reads the current values. Every call runs on one event
    loop, the one the model was loaded on. Names, shapes and dtypes are checked against
    the states the loaded function declares.

    Args:
        aimodel_path: The exported ``.aimodel``.
        weights_init: The initial value of every mutable weight, keyed by name.
        options: Specialization options for loading, such as the preferred compute
            unit; the runtime's default when omitted.
        function_name: The function :meth:`step` runs.

    Raises:
        ValueError: If ``weights_init`` misses one of the function's mutable weights or
            names one it does not have, or a value's shape or dtype differs from the
            weight's.
    """

    def __init__(
        self,
        aimodel_path: Path | str,
        weights_init: dict[str, np.ndarray | torch.Tensor],
        *,
        options: SpecializationOptions | None = None,
        function_name: str = "main",
    ) -> None:
        self._path = Path(aimodel_path)
        self._fn: InferenceFunction
        self._weights: dict[str, NDArray] = {}
        self._exit_stack = AsyncExitStack()
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._async_init(options, function_name))
            declared = self._fn.desc.state_names
            missing = [name for name in declared if name not in weights_init]
            unknown = [name for name in weights_init if name not in declared]
            if missing or unknown:
                raise ValueError(
                    f"weights_init must hold every mutable weight of {function_name!r}, "
                    f"{list(declared)}; missing {missing}, unknown {unknown}"
                )
            for name in declared:
                self._weights[name] = self._checked(name, weights_init[name])
        except Exception:
            self.close()
            raise

    async def _async_init(self, options: SpecializationOptions | None, function_name: str) -> None:
        from coreai.authoring.asset import AIModelAsset

        asset = AIModelAsset.load(self._path)
        model = await self._exit_stack.enter_async_context(asset.executable(options))
        self._fn = model.load_function(function_name)

    async def _async_step(
        self, inputs: dict[str, np.ndarray | torch.Tensor]
    ) -> dict[str, np.ndarray]:
        from coreai.runtime import NDArray

        arrays = {name: NDArray(data=_as_tensor(value)) for name, value in inputs.items()}
        results = await self._fn(dict(arrays), state=dict(self._weights))
        return {
            name: np.asarray(cast(NDArray, value).numpy()).copy() for name, value in results.items()
        }

    def step(self, inputs: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
        """Run the function once on ``inputs`` and return its outputs by name."""
        return self._loop.run_until_complete(self._async_step(inputs))

    @property
    def weight_names(self) -> tuple[str, ...]:
        """The mutable weights the loaded function declares, in its state order."""
        return tuple(self._fn.desc.state_names)

    def set_weight(self, name: str, value: np.ndarray | torch.Tensor) -> None:
        """Replace the value of the mutable weight ``name`` before the next call.

        Raises:
            KeyError: If ``name`` is not one of the model's mutable weights.
            ValueError: If ``value``'s shape or dtype differs from the weight's.
        """
        self._weights[name] = self._checked(name, value)

    def _checked(self, name: str, value: np.ndarray | torch.Tensor) -> NDArray:
        """``value`` as an ``NDArray``, checked against the function's state ``name``."""
        from coreai.runtime import NDArray

        if name not in self._fn.desc.state_names:
            raise KeyError(f"no mutable weight named {name!r}; the model has {self.weight_names}")
        descriptor = self._fn.desc.state_descriptor(name)
        array = NDArray(data=_as_tensor(value))
        expected = (tuple(descriptor.shape), str(descriptor.dtype))
        actual = (tuple(array.shape), str(array.dtype))
        if actual != expected:
            raise ValueError(
                f"mutable weight {name!r} is {expected[0]} {expected[1]}, "
                f"got {actual[0]} {actual[1]}"
            )
        return array

    def weights_snapshot(self) -> dict[str, np.ndarray]:
        """A copy of every mutable weight's current value, keyed by name."""
        return {name: np.asarray(value.numpy()).copy() for name, value in self._weights.items()}

    def close(self) -> None:
        """Release the loaded model and its event loop."""
        try:
            self._loop.run_until_complete(self._exit_stack.aclose())
        finally:
            self._loop.close()

    def __enter__(self) -> MutableWeightRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _as_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return torch.as_tensor(np.ascontiguousarray(value))
