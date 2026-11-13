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

if TYPE_CHECKING:
    from coreai.authoring import AIProgram
    from coreai.runtime import InferenceFunction, SpecializationOptions


def export_mutable_weight_module(
    module: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    weight_names: Sequence[str],
    output_path: Path | str,
) -> tuple[AIProgram, Path]:
    """Export ``module`` to a ``.aimodel`` whose mutable weights stay on device.

    The module's ``forward`` must write each of its mutable weights, with
    ``MutableWeight.update`` or, for a weight it only uses, ``MutableWeight.read``.

    Args:
        module: The module to export.
        sample_inputs: Example ``forward`` arguments for tracing; their shapes and
            dtypes matter, not their values.
        input_names: Names of the ``forward`` arguments other than the mutable weights.
        output_names: Names of the ``forward`` return values.
        weight_names: Names of the mutable weights, matching their ``MutableWeight.name``.
        output_path: Where to write the ``.aimodel``.

    Returns:
        The converted program and ``output_path``.

    Raises:
        ValueError: If ``forward`` writes fewer mutable weights than ``weight_names``
            names, which happens when a weight is only read with ``MutableWeight.value``.
    """
    import coreai_torch

    module = module.eval()
    output_path = Path(output_path)

    with torch.no_grad():
        exported = torch.export.export(module, args=tuple(sample_inputs))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    # Checked here rather than in the converter's export_fn, which wraps every error.
    written = len(exported.graph_signature.buffers_to_mutate)
    if written < len(weight_names):
        raise ValueError(
            f"forward writes {written} mutable weight(s) but weight_names has "
            f"{len(weight_names)}: {list(weight_names)}. A mutable weight stays on device only "
            "when forward writes it; use MutableWeight.read() for a weight forward uses "
            "without changing."
        )

    converter = coreai_torch.TorchConverter()
    converter.add_pytorch_module(
        module,
        export_fn=lambda _: exported,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        state_names=tuple(weight_names),
    )
    program: AIProgram = converter.to_coreai()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(output_path)
    return program, output_path


class MutableWeightRunner:
    """Runs an exported mutable-weight module, keeping its weights on device.

    Each :meth:`step` runs one function of the model once; the mutable weights persist
    and change between calls, and the host passes only the other inputs.
    :meth:`set_weight` replaces a weight's value between calls, and
    :meth:`weights_snapshot` reads the current values. Every call runs on one event
    loop, the one the model was loaded on.

    Args:
        aimodel_path: The exported ``.aimodel``.
        weights_init: The initial value of every mutable weight, keyed by name.
        options: Specialization options for loading, such as the preferred compute
            unit; the runtime's default when omitted.
        function_name: The function :meth:`step` runs.
    """

    def __init__(
        self,
        aimodel_path: Path | str,
        weights_init: dict[str, np.ndarray | torch.Tensor],
        *,
        options: SpecializationOptions | None = None,
        function_name: str = "main",
    ) -> None:
        from coreai.runtime import NDArray

        self._path = Path(aimodel_path)
        self._fn: InferenceFunction
        self._exit_stack = AsyncExitStack()
        self._loop = asyncio.new_event_loop()
        self._weights: dict[str, NDArray] = {
            name: NDArray(data=_as_tensor(value)) for name, value in weights_init.items()
        }
        self._loop.run_until_complete(self._async_init(options, function_name))

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

    def set_weight(self, name: str, value: np.ndarray | torch.Tensor) -> None:
        """Replace the value of the mutable weight ``name`` before the next call.

        Raises:
            KeyError: If ``name`` is not one of the model's mutable weights.
        """
        from coreai.runtime import NDArray

        if name not in self._weights:
            raise KeyError(f"no mutable weight named {name!r}")
        self._weights[name] = NDArray(data=_as_tensor(value))

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
