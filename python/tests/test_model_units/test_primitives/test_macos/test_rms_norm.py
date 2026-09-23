# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for macOS RMSNorm primitive against a manual reference implementation."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm as HFGemma3RMSNorm,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextRMSNormGated as HFQwen3NextRMSNormGated,
)
from typing_extensions import Self, override

from coreai_models.primitives.macos.rms_norm import (
    RMSNormGated as CoreaiTorchRMSNormGated,
)
from coreai_models.primitives.macos.rms_norm import (
    RMSNormPlusOne as CoreaiTorchRMSNormPlusOne,
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
from tests.test_model_units.test_primitives.test_macos._random_input_models import (
    RandomInputModel,
)

if _HAS_MLX:
    import mlx
    import mlx.core
    import mlx.nn
    from mlx_lm.models.gemma3_text import RMSNorm as MlxlmGemma3RMSNorm
    from mlx_lm.models.qwen3_next import (
        Qwen3NextRMSNormGated as _MlxlmQwen3NextRMSNormGated,
    )

    class _MlxRMSNormGated(mlx.nn.Module):
        """Wraps MlxlmQwen3NextRMSNormGated (hidden_states, gate) to accept (x, gate)."""

        def __init__(self, dims: int, eps: float) -> None:
            super().__init__()
            self.norm = _MlxlmQwen3NextRMSNormGated(dims, eps=eps)

        def __call__(self, x, gate=None):  # type: ignore[no-untyped-def]
            return self.norm(x, gate)


try:
    import coreai_torch  # noqa: F401

    HAS_COREAI = True
except ImportError:
    HAS_COREAI = False


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Manual RMSNorm reference: x / sqrt(mean(x^2) + eps) * weight."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return (normed * weight).to(x.dtype)


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestmacOSRMSNorm:
    """Compare coreai_models macOS RMSNorm against a manual reference."""

    def test_basic_forward(self):
        """RMSNorm output should match manual reference."""
        from coreai_models.primitives.macos.rms_norm import RMSNorm

        dim = 64
        eps = 1e-5
        torch.manual_seed(42)

        norm = RMSNorm(dim=dim, eps=eps)
        # Set non-trivial weights
        norm.weight.data = torch.randn(dim)

        x = torch.randn(1, 8, dim)
        our_out = norm(x)
        ref_out = _reference_rms_norm(x, norm.weight.data, eps)

        torch.testing.assert_close(our_out, ref_out, rtol=1e-3, atol=1e-3)

    def test_output_shape(self):
        """Output shape should match input shape."""
        from coreai_models.primitives.macos.rms_norm import RMSNorm

        dim = 64
        norm = RMSNorm(dim=dim)
        x = torch.randn(2, 16, dim)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_norm_plus_one(self):
        """RMSNormPlusOne should apply weight + 1.0 offset."""
        from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne

        dim = 64
        eps = 1e-5
        torch.manual_seed(42)

        norm = RMSNormPlusOne(dim=dim, eps=eps)
        norm.weight.data = torch.randn(dim)

        x = torch.randn(1, 8, dim)
        out = norm(x)
        ref_out = _reference_rms_norm(x, norm.weight.data.float() + 1.0, eps)

        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


# =============================================================================
# Functional-parity tests
# =============================================================================
#
# The classes below cover four parity axes:
#
# * HF eager parity (``oss_torch_config`` vs ``coreai_torch_eager_config``)
# * MLX parity (``oss_mlx_config`` vs ``coreai_torch_eager_config``), gated by
#   ``_HAS_MLX``
# * ``torch.export`` parity
#   (``coreai_torch_export_config`` vs ``coreai_torch_eager_config``)
# * Core AI / Core AI-backend parity
#   (``coreai_torch_export_coreai_coreai_torch_config`` vs
#   ``coreai_torch_export_config``)


# ---------------------------------------------------------------------------
# HF reference wrappers
# ---------------------------------------------------------------------------


class _HFRMSNormGated(torch.nn.Module):
    """Wraps HFQwen3NextRMSNormGated (hidden_states, gate) to accept (x, gate)."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.norm = HFQwen3NextRMSNormGated(dims, eps=eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(x, gate)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class RMSNormPlusOne(RandomInputModel):
    _model_name = "RMSNormPlusOne"

    def __init__(
        self: Self,
        root_path: Path,
        # RMSNorm specification
        dims: int = 32,
        eps: float = 1e-5,
        # reference inputs dimension
        batch_size: int = 3,
    ) -> None:
        super().__init__(root_path=root_path)
        # RMSNorm specification
        self._dims = dims
        self._eps = eps
        # reference inputs dimension
        self._batch_size = batch_size

        # set same weights -- all three implementations add +1 at forward time
        self._weight = torch.rand(dims, dtype=torch.float32)

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchRMSNormPlusOne(self._dims, eps=self._eps)
            model.weight = torch.nn.Parameter(self._weight.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            # HF Gemma3RMSNorm: forward does (1.0 + weight.float()) * norm(x)
            # forward(self, x) -- param name matches
            model = HFGemma3RMSNorm(self._dims, eps=self._eps)
            model.weight = torch.nn.Parameter(self._weight.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            # MLX Gemma3 RMSNorm: __call__ does rms_norm(x, 1.0 + self.weight)
            # __call__(self, x) -- param name matches
            model = MlxlmGemma3RMSNorm(self._dims, eps=self._eps)
            model.weight = mlx.core.array(self._weight)
            model.set_dtype(dtype)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        x_shape = (self._batch_size, self._dims)
        return {"x": x_shape}


class RMSNormGated(RandomInputModel):
    _model_name = "RMSNormGated"

    def __init__(
        self: Self,
        root_path: Path,
        # RMSNorm specification
        dims: int = 32,
        eps: float = 1e-6,
        # reference inputs dimension
        batch_size: int = 3,
    ) -> None:
        super().__init__(root_path=root_path)
        # RMSNorm specification
        self._dims = dims
        self._eps = eps
        # reference inputs dimension
        self._batch_size = batch_size

        # set same weights
        self._weight = torch.rand(dims, dtype=torch.float32)

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchRMSNormGated(self._dims, eps=self._eps)
            model.weight = torch.nn.Parameter(self._weight.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFRMSNormGated(self._dims, eps=self._eps)
            model.norm.weight = torch.nn.Parameter(self._weight.clone())
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            # MLX Qwen3NextRMSNormGated: __call__(hidden_states, gate) -- needs wrapper
            model = _MlxRMSNormGated(self._dims, eps=self._eps)
            model.norm.weight = mlx.core.array(self._weight)
            model.set_dtype(dtype)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        shape = (self._batch_size, self._dims)
        return {"x": shape, "gate": shape}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRMSNorm:
    @staticmethod
    @pytest.mark.parametrize("model_class", [RMSNormPlusOne, RMSNormGated])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_rms_norm(model_class: type[RandomInputModel], precision: Precision) -> None:
        """Verify Core AI Torch / Core AI RMS Norm matches OSS (HF and MLX-LM)."""
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
            model = model_class(Path(temp_directory))
            model.validate(coreai_torch_eager_config, oss_torch_config, rtol=rtol, atol=atol)
            if _HAS_MLX:
                model.validate(coreai_torch_eager_config, oss_mlx_config, rtol=rtol, atol=atol)
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
