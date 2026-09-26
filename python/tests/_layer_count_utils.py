# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Layer count verification utilities for model tests.

These helpers export a PyTorch module through the same pipeline used by the
production macOS exporter, then count the resulting `coreai.<op>` operations
in the emitted MLIR. The op counts act as a parity contract: divergence
here means the model implementation has drifted from the expected MLIR
shape.
"""

import asyncio
import inspect
import re
from collections import Counter
from dataclasses import dataclass

import torch

from ._runner_infra.export.exporters.coreai_exporter import CoreaiStatefulExporter


@dataclass
class LayerCountResult:
    """Result of layer count comparison."""

    actual_counts: dict[str, int]
    mlir_str: str

    def get_diff(self, expected_counts: dict[str, int]) -> dict[str, tuple[int, int]]:
        """Return dict of {op_name: (expected, actual)} for mismatches.

        Args:
            expected_counts: Mapping of op name -> expected count to compare
                against ``self.actual_counts``.
        """
        all_ops = set(self.actual_counts.keys()) | set(expected_counts.keys())
        diff: dict[str, tuple[int, int]] = {}
        for op in all_ops:
            expected = expected_counts.get(op, 0)
            actual = self.actual_counts.get(op, 0)
            if expected != actual:
                diff[op] = (expected, actual)
        return diff


def count_coreai_operations(mlir_str: str) -> dict[str, int]:
    """Parse an MLIR string and count Core AI operations.

    Core AI operations appear in the form ``coreai.<op_name>``.
    """
    pattern = r"coreai\.([a-zA-Z_][a-zA-Z0-9_\.]*)"
    matches = re.findall(pattern, mlir_str)
    return dict(Counter(matches))


def _sanitize_inputs(
    inputs: list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if isinstance(inputs, list):
        return tuple(inputs)
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    if isinstance(inputs, tuple):
        return inputs
    err = f"Invalid inputs type {type(inputs)}"
    raise ValueError(err)


def _convert_inputs_to_fp16(
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    res = []
    for val in inputs:
        if val.dtype == torch.float32:
            res.append(val.to(torch.float16))
        else:
            res.append(val)
    return tuple(res)


def get_layer_counts(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    dynamic_shapes: list[dict[str, tuple[int, int]]] | None = None,
    use_fp16_precision: bool = False,
) -> LayerCountResult:
    """Export ``model`` through the macOS pipeline and count MLIR operations.

    Uses ``CoreaiStatefulExporter`` to match the production export path,
    including composite op externalization (RMSNorm, RoPE, SDPA, etc.) and
    auto-classification of mutated KV-cache buffers as state.
    """
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

    if len(param_names) != len(inputs):
        raise ValueError(
            f"Got {len(inputs)} input(s) but {type(model).__name__}.forward "
            f"resolves to {len(param_names)} parameter name(s): {param_names}. "
            f"Provide exactly one input per required positional parameter "
            f"(or up to the total number of parameters including defaults)."
        )
    reference_inputs = dict(zip(param_names, inputs, strict=True))

    async def _run() -> "AIProgram":  # noqa: F821
        exporter = CoreaiStatefulExporter()
        return await exporter._async_export_and_optimize(
            model, reference_inputs, dynamic_shapes=dynamic_shapes
        )

    coreai_program = asyncio.run(_run())

    mlir_str = coreai_program._module._mlir_module.operation.get_asm(
        large_elements_limit=0,
        large_resource_limit=0,
        enable_debug_info=False,
        pretty_debug_info=False,
        print_generic_op_form=False,
        use_local_scope=False,
        assume_verified=False,
    )

    actual_counts = count_coreai_operations(mlir_str)

    return LayerCountResult(
        actual_counts=actual_counts,
        mlir_str=mlir_str,
    )


def assert_layer_counts(
    result: LayerCountResult,
    expected_counts: dict[str, int],
    strict: bool = True,
) -> None:
    """Assert that actual layer counts match expected counts.

    Args:
        result: ``LayerCountResult`` from ``get_layer_counts``.
        expected_counts: Mapping of expected op name -> count.
        strict: If True, fail on extra ops not in ``expected_counts``.
    """
    diff = result.get_diff(expected_counts)

    if not strict:
        diff = {k: v for k, v in diff.items() if k in expected_counts}

    if diff:
        error_lines = ["Layer count mismatch:"]
        for op, (expected, actual) in sorted(diff.items()):
            error_lines.append(f"  {op}: expected {expected}, got {actual}")
        raise AssertionError("\n".join(error_lines))
