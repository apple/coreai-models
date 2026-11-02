# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS SwitchLinear and SwitchGLU primitives."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from typing_extensions import Self, override

from coreai_models.primitives.macos.switch import (
    SwiGLU as CoreaiTorchSwiGLU,
)
from coreai_models.primitives.macos.switch import (
    SwitchGLU as CoreaiTorchSwitchGLU,
)
from coreai_models.primitives.macos.switch import (
    SwitchLinear as CoreaiTorchSwitchLinear,
)
from tests._runner_infra._deps import _HAS_MLX, _MSG_MLX_NOT_FOUND
from tests._runner_infra.common.types.dependency_types import (
    PRECISION_IN_SOURCE,
    SourceModel,
)
from tests._runner_infra.common.types.export_types import (
    Backend,
    Frontend,
)
from tests._runner_infra.common.types.run_types import RunConfig
from tests._runner_infra.common.types.source_types import (
    Author,
    Precision,
    Source,
    SourceConfig,
)
from tests._runner_infra.testing_utils import uses_os_coreai
from tests.test_model_units.test_primitives.test_macos._random_input_models import (
    RandomInputModel,
    RandomInputWithIndicesModel,
)

if _HAS_MLX:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from mlx_lm.models.switch_layers import SwiGLU as MlxSwiGLU
    from mlx_lm.models.switch_layers import SwitchGLU as MlxSwitchGLU
    from mlx_lm.models.switch_layers import SwitchLinear as MlxSwitchLinear


F16_MACOS_27_XFAIL = pytest.param(
    Precision.f16,
    marks=pytest.mark.xfail(uses_os_coreai(), reason="known issue on macOS 27", strict=False),
)


try:
    import coreai_torch  # noqa: F401

    HAS_COREAI = True
except ImportError:
    HAS_COREAI = False


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestSwitchLinear:
    """Test SwitchLinear with small dummy inputs."""

    def test_basic_forward(self):
        """SwitchLinear should produce output without errors."""
        from coreai_models.primitives.macos.switch import SwitchLinear

        input_dims, output_dims = 32, 16
        num_weight_sets, num_experts = 1, 4
        num_active_experts = 2
        batch_seq = 3

        switch = SwitchLinear(
            input_dims=input_dims,
            output_dims=output_dims,
            num_weight_sets=num_weight_sets,
            num_experts=num_experts,
            bias=True,
        )

        x = torch.randn(batch_seq, 1, 1, input_dims)
        indices = torch.randint(0, num_experts, (batch_seq, num_active_experts), dtype=torch.int32)

        out = switch(x, indices)
        # Expected: num_weight_sets x batch_seq x num_active_experts x 1 x output_dims
        assert out.shape == (num_weight_sets, batch_seq, num_active_experts, 1, output_dims)

    def test_no_bias(self):
        """SwitchLinear without bias should work."""
        from coreai_models.primitives.macos.switch import SwitchLinear

        switch = SwitchLinear(
            input_dims=16, output_dims=8, num_weight_sets=1, num_experts=2, bias=False
        )
        x = torch.randn(2, 1, 1, 16)
        indices = torch.zeros(2, 1, dtype=torch.int32)
        out = switch(x, indices)
        assert out.shape == (1, 2, 1, 1, 8)


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestSwiGLU:
    """Test SwiGLU activation."""

    def test_forward(self):
        """SwiGLU should combine SiLU(gate) * up."""
        from coreai_models.primitives.macos.switch import SwiGLU

        swiglu = SwiGLU()
        up = torch.randn(2, 4, 16)
        gate = torch.randn(2, 4, 16)
        out = swiglu(up, gate)
        assert out.shape == up.shape

        # Verify manually
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestSwitchGLU:
    """Test SwitchGLU end-to-end."""

    def test_basic_forward(self):
        """SwitchGLU should produce output with correct shape."""
        from coreai_models.primitives.macos.switch import SwitchGLU

        hidden_size, moe_intermediate_size = 32, 64
        num_experts, num_active = 4, 2
        batch_size, query_length = 1, 4

        switch_glu = SwitchGLU(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            bias=False,
        )

        x = torch.randn(batch_size, query_length, hidden_size)
        indices = torch.randint(
            0, num_experts, (batch_size, query_length, num_active), dtype=torch.int32
        )

        out = switch_glu(x, indices)
        assert out.shape == (batch_size, query_length, num_active, hidden_size)


# =============================================================================
# Functional-parity tests
# =============================================================================
#
# The classes below cover four parity axes:
#
# * Naive PyTorch reference parity
#   (``oss_torch_config`` vs ``coreai_torch_eager_config``)
# * MLX parity (``oss_mlx_config`` vs ``coreai_torch_eager_config``), gated by
#   ``_HAS_MLX``
# * ``torch.export`` parity
#   (``coreai_torch_export_config`` vs ``coreai_torch_eager_config``)
# * Core AI / Core AI-backend parity
#   (``coreai_torch_export_coreai_coreai_torch_config`` vs
#   ``coreai_torch_export_config``)


# ---------------------------------------------------------------------------
# Naive reference implementations (no GatherMM, just explicit loops)
# ---------------------------------------------------------------------------


class NaiveSwitchLinear(torch.nn.Module):
    """Loop-based reference for SwitchLinear."""

    def __init__(
        self: Self,
        input_dims: int,
        output_dims: int,
        num_weight_sets: int,
        num_experts: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.rand(num_weight_sets, num_experts, output_dims, input_dims)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(num_weight_sets, num_experts, output_dims))

    def forward(
        self: Self,
        x: torch.Tensor,
        indices: torch.IntTensor,
    ) -> torch.Tensor:
        # x: (batch, 1, 1, input_dims)
        # indices: (batch, num_active_experts)
        # output: (num_weight_sets, batch, num_active, 1, output_dims)
        weight_T = self.weight.transpose(-1, -2)
        num_ws = weight_T.shape[0]
        batch = x.shape[0]
        num_active = indices.shape[1]
        out_dims = weight_T.shape[3]
        result = torch.zeros(
            num_ws,
            batch,
            num_active,
            1,
            out_dims,
            dtype=x.dtype,
            device=x.device,
        )
        for ws in range(num_ws):
            for b in range(batch):
                for k in range(num_active):
                    eidx = indices[b, k]
                    result[ws, b, k, 0] = x[b, 0, 0] @ weight_T[ws, eidx]
                    if hasattr(self, "bias"):
                        result[ws, b, k, 0] += self.bias[ws, eidx]
        return result


class NaiveSwiGLU(torch.nn.Module):
    """Plain-PyTorch reference for SwiGLU (no custom ops)."""

    def forward(self: Self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up


class NaiveSwitchGLU(torch.nn.Module):
    """Scatter-based reference for SwitchGLU."""

    def __init__(
        self: Self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj_weight = torch.nn.Parameter(
            torch.rand(1, num_experts, moe_intermediate_size, hidden_size)
        )
        self.up_proj_weight = torch.nn.Parameter(
            torch.rand(1, num_experts, moe_intermediate_size, hidden_size)
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.rand(1, num_experts, hidden_size, moe_intermediate_size)
        )
        if bias:
            self.gate_proj_bias = torch.nn.Parameter(
                torch.rand(1, num_experts, moe_intermediate_size)
            )
            self.up_proj_bias = torch.nn.Parameter(
                torch.rand(1, num_experts, moe_intermediate_size)
            )
            self.down_proj_bias = torch.nn.Parameter(torch.rand(1, num_experts, hidden_size))

    def forward(
        self: Self,
        x: torch.Tensor,
        indices: torch.IntTensor,
    ) -> torch.Tensor:
        # x: (batch, seq_len, hidden)
        # indices: (batch, seq_len, num_active_experts)
        # output: (batch, seq_len, num_active, hidden)
        batch, seq_len, hidden = x.shape
        num_active = indices.shape[-1]
        result = torch.zeros(
            batch,
            seq_len,
            num_active,
            hidden,
            dtype=x.dtype,
            device=x.device,
        )
        for b in range(batch):
            for s in range(seq_len):
                for k in range(num_active):
                    eidx = indices[b, s, k]
                    token = x[b, s]
                    gate = token @ self.gate_proj_weight[0, eidx].T
                    up = token @ self.up_proj_weight[0, eidx].T
                    if hasattr(self, "gate_proj_bias"):
                        gate = gate + self.gate_proj_bias[0, eidx]
                        up = up + self.up_proj_bias[0, eidx]
                    activated = torch.nn.functional.silu(gate) * up
                    down = activated @ self.down_proj_weight[0, eidx].T
                    if hasattr(self, "down_proj_bias"):
                        down = down + self.down_proj_bias[0, eidx]
                    result[b, s, k] = down
        return result


# ---------------------------------------------------------------------------
# MLX wrappers
# ---------------------------------------------------------------------------

if _HAS_MLX:

    class _MlxSwitchLinearWrapper(mlx_nn.Module):
        """Adds num_weight_sets=1 leading dim to match coreai output."""

        def __init__(self: Self, inner: "MlxSwitchLinear") -> None:
            super().__init__()
            self.inner = inner

        def __call__(self: Self, x: "mx.array", indices: "mx.array") -> "mx.array":
            return mx.expand_dims(self.inner(x, indices), 0)

    class _MlxSwiGLUWrapper(mlx_nn.Module):
        """Adapts MLX SwiGLU(x, gate) to coreai convention (up, gate)."""

        def __call__(self: Self, up: "mx.array", gate: "mx.array") -> "mx.array":
            return MlxSwiGLU()(up, gate)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class SwitchLinear(RandomInputWithIndicesModel):
    _model_name = "SwitchLinear"

    def __init__(
        self: Self,
        root_path: Path,
        input_dims: int = 4,
        output_dims: int = 6,
        num_weight_sets: int = 1,
        num_experts: int = 4,
        num_active_experts: int = 2,
        batch_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__(root_path=root_path)
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._num_weight_sets = num_weight_sets
        self._num_experts = num_experts
        self._num_active_experts = num_active_experts
        self._batch_size = batch_size
        self._bias = bias
        # Pre-generate weights for sharing across implementations
        self._weight = torch.rand(num_weight_sets, num_experts, output_dims, input_dims)
        self._bias_param = torch.rand(num_weight_sets, num_experts, output_dims) if bias else None

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchSwitchLinear(
                self._input_dims,
                self._output_dims,
                self._num_weight_sets,
                self._num_experts,
                bias=self._bias,
            )
            model.weight = torch.nn.Parameter(self._weight.clone())
            if self._bias:
                model.bias = torch.nn.Parameter(self._bias_param.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = NaiveSwitchLinear(
                self._input_dims,
                self._output_dims,
                self._num_weight_sets,
                self._num_experts,
                bias=self._bias,
            )
            model.weight = torch.nn.Parameter(self._weight.clone())
            if self._bias:
                model.bias = torch.nn.Parameter(self._bias_param.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            assert self._num_weight_sets == 1, "MLX SwitchLinear has no num_weight_sets"
            inner = MlxSwitchLinear(
                self._input_dims,
                self._output_dims,
                self._num_experts,
                bias=self._bias,
            )
            inner.weight = mx.array(self._weight[0].numpy()).astype(dtype)
            if self._bias:
                inner.bias = mx.array(self._bias_param[0].numpy()).astype(dtype)
            model = _MlxSwitchLinearWrapper(inner)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        return {"x": (self._batch_size, 1, 1, self._input_dims)}

    @property
    @override
    def named_index_input_specs(
        self: Self,
    ) -> dict[str, tuple[tuple[int, ...], int]]:
        return {
            "indices": (
                (self._batch_size, self._num_active_experts),
                self._num_experts,
            ),
        }


class SwiGLU(RandomInputModel):
    _model_name = "SwiGLU"

    def __init__(
        self: Self,
        root_path: Path,
        batch_size: int = 3,
        num_active_experts: int = 4,
        intermediate_size: int = 5,
    ) -> None:
        super().__init__(root_path=root_path)
        self._batch_size = batch_size
        self._num_active_experts = num_active_experts
        self._intermediate_size = intermediate_size

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
            model = CoreaiTorchSwiGLU()
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = NaiveSwiGLU()
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            model = _MlxSwiGLUWrapper()
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        shape = (
            self._batch_size,
            self._num_active_experts,
            self._intermediate_size,
        )
        return {"up": shape, "gate": shape}


class SwitchGLU(RandomInputWithIndicesModel):
    _model_name = "SwitchGLU"

    def __init__(
        self: Self,
        root_path: Path,
        hidden_size: int = 4,
        moe_intermediate_size: int = 6,
        num_experts: int = 4,
        num_active_experts: int = 2,
        batch_size: int = 2,
        query_length: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__(root_path=root_path)
        self._hidden_size = hidden_size
        self._moe_intermediate_size = moe_intermediate_size
        self._num_experts = num_experts
        self._num_active_experts = num_active_experts
        self._batch_size = batch_size
        self._query_length = query_length
        self._bias = bias
        # Pre-generate weights for sharing across implementations
        self._gate_proj_weight = torch.rand(1, num_experts, moe_intermediate_size, hidden_size)
        self._up_proj_weight = torch.rand(1, num_experts, moe_intermediate_size, hidden_size)
        self._down_proj_weight = torch.rand(1, num_experts, hidden_size, moe_intermediate_size)
        if bias:
            self._gate_proj_bias = torch.rand(1, num_experts, moe_intermediate_size)
            self._up_proj_bias = torch.rand(1, num_experts, moe_intermediate_size)
            self._down_proj_bias = torch.rand(1, num_experts, hidden_size)

    def _load_coreai_torch_weights(self: Self, model: CoreaiTorchSwitchGLU) -> None:
        model.gate_proj.weight = torch.nn.Parameter(self._gate_proj_weight.clone())
        model.up_proj.weight = torch.nn.Parameter(self._up_proj_weight.clone())
        model.down_proj.weight = torch.nn.Parameter(self._down_proj_weight.clone())
        if self._bias:
            model.gate_proj.bias = torch.nn.Parameter(self._gate_proj_bias.clone())
            model.up_proj.bias = torch.nn.Parameter(self._up_proj_bias.clone())
            model.down_proj.bias = torch.nn.Parameter(self._down_proj_bias.clone())

    def _load_naive_torch_weights(self: Self, model: NaiveSwitchGLU) -> None:
        model.gate_proj_weight = torch.nn.Parameter(self._gate_proj_weight.clone())
        model.up_proj_weight = torch.nn.Parameter(self._up_proj_weight.clone())
        model.down_proj_weight = torch.nn.Parameter(self._down_proj_weight.clone())
        if self._bias:
            model.gate_proj_bias = torch.nn.Parameter(self._gate_proj_bias.clone())
            model.up_proj_bias = torch.nn.Parameter(self._up_proj_bias.clone())
            model.down_proj_bias = torch.nn.Parameter(self._down_proj_bias.clone())

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchSwitchGLU(
                self._hidden_size,
                self._moe_intermediate_size,
                self._num_experts,
                bias=self._bias,
            )
            self._load_coreai_torch_weights(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = NaiveSwitchGLU(
                self._hidden_size,
                self._moe_intermediate_size,
                self._num_experts,
                bias=self._bias,
            )
            self._load_naive_torch_weights(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            model = MlxSwitchGLU(
                self._hidden_size,
                self._moe_intermediate_size,
                self._num_experts,
                bias=self._bias,
            )
            # Set weights (squeeze num_weight_sets=1 dim via [0])
            model.gate_proj.weight = mx.array(self._gate_proj_weight[0].numpy()).astype(dtype)
            model.up_proj.weight = mx.array(self._up_proj_weight[0].numpy()).astype(dtype)
            model.down_proj.weight = mx.array(self._down_proj_weight[0].numpy()).astype(dtype)
            if self._bias:
                model.gate_proj.bias = mx.array(self._gate_proj_bias[0].numpy()).astype(dtype)
                model.up_proj.bias = mx.array(self._up_proj_bias[0].numpy()).astype(dtype)
                model.down_proj.bias = mx.array(self._down_proj_bias[0].numpy()).astype(dtype)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        return {
            "x": (self._batch_size, self._query_length, self._hidden_size),
        }

    @property
    @override
    def named_index_input_specs(
        self: Self,
    ) -> dict[str, tuple[tuple[int, ...], int]]:
        return {
            "indices": (
                (
                    self._batch_size,
                    self._query_length,
                    self._num_active_experts,
                ),
                self._num_experts,
            ),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSwitch:
    @staticmethod
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("num_weight_sets", [1, 2])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_switch_linear(bias: bool, num_weight_sets: int, precision: Precision) -> None:
        """Verify Core AI Torch / Core AI SwitchLinear matches OSS."""
        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        oss_mlx_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.mlx),
            precision=precision,
            backend=cast("Backend", Backend.mlx),
        )
        coreai_torch_eager_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        coreai_torch_export_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_export),
        )
        coreai_torch_export_coreai_coreai_torch_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            frontend=cast("Frontend", Frontend.torch_export),
            backend=cast("Backend", Backend.coreai),
        )
        rtol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = SwitchLinear(
                Path(temp_directory),
                bias=bias,
                num_weight_sets=num_weight_sets,
            )
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
            if _HAS_MLX and num_weight_sets == 1:
                model.validate(
                    coreai_torch_eager_config,
                    oss_mlx_config,
                    rtol=rtol,
                    atol=atol,
                )
            elif not _HAS_MLX:
                msg = f"{_MSG_MLX_NOT_FOUND} so cannot validate coreai torch authoring vs mlx-lm"
                warnings.warn(msg, stacklevel=2)
            model.validate(
                coreai_torch_export_config,
                coreai_torch_eager_config,
                rtol=rtol,
                atol=atol,
            )
            model.validate(
                coreai_torch_export_coreai_coreai_torch_config,
                coreai_torch_export_config,
                rtol=rtol,
                atol=atol,
            )

    @staticmethod
    @pytest.mark.parametrize("precision", [Precision.f32, F16_MACOS_27_XFAIL, Precision.bf16])
    def test_swiglu(precision: Precision) -> None:
        """Verify Core AI Torch / Core AI SwiGLU matches OSS (naive and MLX-LM)."""
        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        oss_mlx_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.mlx),
            precision=precision,
            backend=cast("Backend", Backend.mlx),
        )
        coreai_torch_eager_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        coreai_torch_export_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_export),
        )
        coreai_torch_export_coreai_coreai_torch_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            frontend=cast("Frontend", Frontend.torch_export),
            backend=cast("Backend", Backend.coreai),
        )
        rtol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = SwiGLU(Path(temp_directory))
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
            if _HAS_MLX:
                model.validate(
                    coreai_torch_eager_config,
                    oss_mlx_config,
                    rtol=rtol,
                    atol=atol,
                )
            else:
                msg = f"{_MSG_MLX_NOT_FOUND} so cannot validate coreai torch authoring vs mlx-lm"
                warnings.warn(msg, stacklevel=2)
            model.validate(
                coreai_torch_export_config,
                coreai_torch_eager_config,
                rtol=rtol,
                atol=atol,
            )
            model.validate(
                coreai_torch_export_coreai_coreai_torch_config,
                coreai_torch_export_config,
                rtol=rtol,
                atol=atol,
            )

    @staticmethod
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("num_active_experts", [1, 2])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_switch_glu(bias: bool, num_active_experts: int, precision: Precision) -> None:
        """Verify Core AI Torch / Core AI SwitchGLU matches OSS."""
        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        oss_mlx_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.mlx),
            precision=precision,
            backend=cast("Backend", Backend.mlx),
        )
        coreai_torch_eager_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
        )
        coreai_torch_export_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_export),
        )
        coreai_torch_export_coreai_coreai_torch_config = RunConfig(
            author=cast("Author", Author.coreai),
            source=cast("Source", Source.torch),
            precision=precision,
            frontend=cast("Frontend", Frontend.torch_export),
            backend=cast("Backend", Backend.coreai),
        )
        rtol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 1e-3, Precision.bf16: 1e-2}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = SwitchGLU(
                Path(temp_directory),
                bias=bias,
                num_active_experts=num_active_experts,
            )
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
            if _HAS_MLX:
                model.validate(
                    coreai_torch_eager_config,
                    oss_mlx_config,
                    rtol=rtol,
                    atol=atol,
                )
            else:
                msg = f"{_MSG_MLX_NOT_FOUND} so cannot validate coreai torch authoring vs mlx-lm"
                warnings.warn(msg, stacklevel=2)
            model.validate(
                coreai_torch_export_config,
                coreai_torch_eager_config,
                rtol=rtol,
                atol=atol,
            )
            model.validate(
                coreai_torch_export_coreai_coreai_torch_config,
                coreai_torch_export_config,
                rtol=rtol,
                atol=atol,
            )
