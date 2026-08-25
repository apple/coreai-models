# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import coreai_torch
import torch
from typing_extensions import Self, final, override

from coreai_models.export.externalize import EXTERNALIZE_SPECS
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)

if TYPE_CHECKING:
    from coreai.authoring import AIProgram


class CoreaiExporter:
    def __init__(
        self: Self,
        input_names: tuple[str, ...] | None = None,
        output_names: tuple[str, ...] | None = None,
    ) -> None:
        self._input_names = input_names
        self._output_names = output_names

    async def _async_export(
        self: Self,
        torch_module: torch.nn.Module,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
    ) -> AIProgram:
        input_names = (
            self._input_names if self._input_names is not None else tuple(reference_inputs.keys())
        )

        torch_module.eval()
        converter = coreai_torch.TorchConverter()
        converter.add_pytorch_module(
            torch_module,
            export_fn=lambda m: torch.export.export(
                m,
                args=(),
                kwargs=reference_inputs,
                dynamic_shapes=dynamic_shapes,
            ).run_decompositions(coreai_torch.get_decomp_table()),
            externalize_modules=EXTERNALIZE_SPECS,
            input_names=input_names,
            output_names=self._output_names,
        )
        return converter.to_coreai()

    async def _async_optimize(self: Self, coreai_program: AIProgram) -> None:
        coreai_program.optimize()

    @final
    async def _async_export_and_optimize(
        self: Self,
        torch_module: torch.nn.Module,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
    ) -> AIProgram:
        coreai_program = await self._async_export(
            torch_module,
            reference_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        await self._async_optimize(coreai_program)
        return coreai_program

    @final
    def export(
        self: Self,
        torch_module: torch.nn.Module,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
    ) -> AIProgram:
        return asyncio.run(
            self._async_export_and_optimize(
                torch_module,
                reference_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        )


class CoreaiStatefulExporter(CoreaiExporter):
    def __init__(
        self: Self,
        input_names: tuple[str, ...] | None = None,
        output_names: tuple[str, ...] | None = None,
        state_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            input_names=input_names,
            output_names=output_names,
        )
        self._state_names = state_names

    @override
    async def _async_export(
        self: Self,
        torch_module: torch.nn.Module,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
    ) -> AIProgram:
        def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
            with torch.no_grad():
                aten_exported_program = torch.export.export(
                    module,
                    args=(),
                    kwargs=reference_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
            coreai_decomp_table = coreai_torch.get_decomp_table()
            coreaten_exported_program = aten_exported_program.run_decompositions(
                coreai_decomp_table
            )
            remove_functionalization(coreaten_exported_program)
            return coreaten_exported_program

        torch_module.eval()
        converter = coreai_torch.TorchConverter()
        # Default ``input_names`` only when the caller has declared
        # ``state_names`` -- in that case we know which reference inputs
        # are state vs regular and can derive a safe non-state input list.
        # When both are None, pass ``input_names=None`` straight through so
        # ``add_pytorch_module`` derives names from the exported program's
        # signature (this is the path callers like ``get_layer_counts`` rely
        # on, where torch.export DCEs in-place-mutated cache tensors and the
        # graph signature is the only source of truth for live inputs).
        if self._input_names is None and self._state_names is not None:
            state_names_set = set(self._state_names)
            input_names = tuple(k for k in reference_inputs if k not in state_names_set)
        else:
            input_names = self._input_names
        converter.add_pytorch_module(
            torch_module,
            export_fn=export_fn,
            externalize_modules=EXTERNALIZE_SPECS,
            input_names=input_names,
            output_names=self._output_names,
            state_names=self._state_names,
        )
        register_custom_torch_lowering(converter)
        return converter.to_coreai()

    @override
    async def _async_optimize(self: Self, coreai_program: AIProgram) -> None:
        coreai_program.optimize()
