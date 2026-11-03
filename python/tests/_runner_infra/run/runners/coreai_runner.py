# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import asyncio
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self, override

from ...common.types.dependency_types import Tensor
from ...common.utils.coreai.tensor import ndarray_to_torch_tensor
from .runner import Runner

if TYPE_CHECKING:
    from coreai.authoring import AIProgram
    from coreai.runtime import InferenceFunction, NDArray
else:
    # NDArray is needed at runtime inside this module to construct inputs.
    from coreai.runtime import InferenceFunction, NDArray


class CoreaiRuntime:
    async def _async_load_model(
        self: Self,
        coreai_program: AIProgram,
        asset_path: Path,
    ) -> InferenceFunction:
        # `asset.executable()` requires the asset directory to end in `.aimodel`.
        aimodel_path = asset_path.parent / "model.aimodel"
        if aimodel_path.exists():
            shutil.rmtree(aimodel_path)
        asset = coreai_program.save_asset(aimodel_path)
        # Keep the executable context open for the lifetime of this runtime so
        # subsequent forward() calls reuse the loaded AIModel. The context is
        # released when self._exit_stack is closed (or at process exit).
        ai_model = await self._exit_stack.enter_async_context(asset.executable())
        return ai_model.load_function("main")

    def __init__(
        self: Self,
        coreai_program: AIProgram,
        asset_path: Path,
        output_names: tuple[str] | None = None,
    ) -> None:
        self._coreai_program = coreai_program
        self._output_names = output_names
        self._exit_stack = AsyncExitStack()
        try:
            self._function = asyncio.run(self._async_load_model(coreai_program, asset_path))
        except RuntimeError as error:
            asyncio.run(self._exit_stack.aclose())
            # Imported here: testing_utils pulls in the whole export stack.
            from ...testing_utils import skip_if_neural_engine_refused

            skip_if_neural_engine_refused(error)
            raise

    async def _async_forward(
        self: Self, named_inputs: dict[str, Tensor]
    ) -> dict[str, torch.Tensor]:
        coreai_inputs: dict[str, NDArray] = {}
        for name, tensor in named_inputs.items():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                # DLPack capsules cannot capture all of PyTorch semantics
                tensor = tensor.detach()
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int64:
                tensor = tensor.to(torch.int32)
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            coreai_inputs[name] = NDArray(data=tensor)

        coreai_outputs: dict[str, NDArray] = await self._function(coreai_inputs)

        outputs: dict[str, torch.Tensor] = {
            name: ndarray_to_torch_tensor(tensor) for name, tensor in coreai_outputs.items()
        }
        if self._output_names is not None:
            # reorder outputs according to specified output names
            outputs = {output_name: outputs[output_name] for output_name in self._output_names}
        return outputs

    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        return asyncio.run(self._async_forward(named_inputs))

    async def aclose(self: Self) -> None:
        """Release the asset.executable() context held in ``_exit_stack``.

        Callers that drive ``CoreaiRuntime`` from an async context should
        ``await runtime.aclose()`` (or use it as an async context manager via
        ``async with`` / ``__aexit__``) when they're done. Without this, the
        AIModel resources held by the executable() context leak until process
        exit.
        """
        await self._exit_stack.aclose()

    def close(self: Self) -> None:
        """Synchronous counterpart to :py:meth:`aclose` for sync callers."""
        asyncio.run(self.aclose())

    async def __aenter__(self: Self) -> Self:
        return self

    async def __aexit__(self: Self, *exc_info: object) -> None:
        await self.aclose()


class CoreaiRunner(Runner):
    def __init__(
        self: Self,
        coreai_program: AIProgram,
        asset_path: Path,
        output_names: tuple[str] | None = None,
    ) -> None:
        super().__init__()
        self._runtime = CoreaiRuntime(coreai_program, asset_path, output_names)

    @override
    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        return self._runtime.forward(named_inputs)

    def close(self: Self) -> None:
        """Release runtime resources (forwards to ``CoreaiRuntime.close``)."""
        self._runtime.close()

    async def aclose(self: Self) -> None:
        await self._runtime.aclose()
