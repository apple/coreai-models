# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for macOS RoPE primitive."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssRotaryEmbedding,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    apply_rotary_pos_emb as gpt_oss_apply_rotary_pos_emb,
)
from typing_extensions import Self, override

from coreai_models.primitives.macos.rope import (
    YarnRoPE as CoreaiTorchYarnRoPE,
)
from coreai_models.primitives.macos.rope import (
    initialize_rope as coreai_torch_initialize_rope,
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
from tests._runner_infra.common.utils.torch.tensor import (
    torch_tensor_to_numpy_array,
)
from tests.test_model_units.test_primitives.test_macos._random_input_models import (
    RandomInputModel,
)

if _HAS_MLX:
    from mlx_lm.models.rope_utils import (
        YarnRoPE as MlxlmYarnRoPE,
    )
    from mlx_lm.models.rope_utils import (
        initialize_rope as oss_mlx_initialize_rope,
    )

    from tests._runner_infra.common.utils.mlx.tensor import (
        mlx_array_to_numpy_array,
    )


try:
    import coreai_torch  # noqa: F401

    HAS_COREAI = True
except ImportError:
    HAS_COREAI = False


@pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")
class TestmacOSRoPE:
    """Test coreai_models macOS RoPE primitive."""

    def test_basic_forward(self):
        """RoPE should produce output of the same shape without errors."""
        from coreai_models.primitives.macos.rope import RoPE

        dims = 32
        rope = RoPE(dims=dims)

        batch_size, n_heads, seq_len, head_dim = 1, 4, 8, 32
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        offset = torch.tensor([0], dtype=torch.int32)

        out = rope(x, offset=offset)
        assert out.shape == x.shape

    def test_different_offsets_produce_different_results(self):
        """Different position offsets should produce different embeddings."""
        from coreai_models.primitives.macos.rope import RoPE

        dims = 32
        rope = RoPE(dims=dims)

        x = torch.randn(1, 4, 1, 32)
        out0 = rope(x, offset=torch.tensor([0], dtype=torch.int32))
        out5 = rope(x, offset=torch.tensor([5], dtype=torch.int32))

        assert not torch.allclose(out0, out5), "Different offsets should produce different results"

    def test_initialize_rope_default(self):
        """initialize_rope with default config should return a RoPE instance."""
        from coreai_models.primitives.macos.rope import initialize_rope

        rope = initialize_rope(dims=32, base=10000.0)
        x = torch.randn(1, 4, 8, 32)
        out = rope(x, offset=torch.tensor([0], dtype=torch.int32))
        assert out.shape == x.shape

    @pytest.mark.parametrize("dims", [16, 8])
    def test_decomposed_rope_applies_given_cos_sin(self, dims, monkeypatch):
        """Under FORCE_DECOMPOSED_ROPE, given cos/sin rotate exactly as the composite does.

        ``x`` is ``[B, L, H, D]``, so positions inferred from ``shape[-2]`` would be wrong.
        """
        from coreai_models.primitives.macos.rope import DecomposedRoPE, RoPE, initialize_rope

        x = torch.randn(2, 5, 3, 16)
        angles = torch.randn(1, 5, 1, dims // 2)
        cos, sin = angles.cos(), angles.sin()
        expected = RoPE(dims=dims)(x, cos=cos, sin=sin)

        monkeypatch.setenv("FORCE_DECOMPOSED_ROPE", "1")
        rope = initialize_rope(dims=dims)
        assert isinstance(rope, DecomposedRoPE)
        assert torch.equal(rope(x, cos=cos, sin=sin), expected)


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


class _HFYarnRoPE(torch.nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.rotary = GptOssRotaryEmbedding(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _heads, seq_len, _head_dim = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary(x, position_ids)
        x_rotated, _ = gpt_oss_apply_rotary_pos_emb(x, x, cos, sin)
        return x_rotated


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class YarnRoPE(RandomInputModel):
    _model_name = "YarnRoPE"

    def __init__(
        self: Self,
        root_path: Path,
        # RoPE specification
        head_dim: int = 32,
        rope_theta: float = 150000.0,
        rope_scaling: dict | None = None,
        # reference inputs dimension
        batch_size: int = 3,
        num_attention_heads: int = 32,
        query_length: int = 4,
    ) -> None:
        super().__init__(root_path=root_path)
        # RoPE specification
        self._head_dim = head_dim
        self._rope_theta = rope_theta
        self._rope_scaling = rope_scaling or {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
        }
        # reference inputs dimension
        self._batch_size = batch_size
        self._num_attention_heads = num_attention_heads
        self._query_length = query_length

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = coreai_torch_initialize_rope(
                dims=self._head_dim,
                base=self._rope_theta,
                scaling_config=self._rope_scaling,
            )
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            config = GptOssConfig(
                hidden_size=self._head_dim * self._num_attention_heads,
                num_attention_heads=self._num_attention_heads,
                head_dim=self._head_dim,
                rope_theta=self._rope_theta,
                max_position_embeddings=131072,
                rope_scaling=self._rope_scaling,
            )
            model = _HFYarnRoPE(config)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            model = oss_mlx_initialize_rope(
                dims=self._head_dim,
                base=self._rope_theta,
                traditional=False,
                scaling_config=self._rope_scaling,
            )
            model.set_dtype(dtype)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @property
    @override
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        input_shape = (
            self._batch_size,
            self._num_attention_heads,
            self._query_length,
            self._head_dim,
        )
        return {"x": input_shape}

    @override
    def validate(
        self: Self,
        run_config: RunConfig = RunConfig(),  # noqa: B008
        reference_run_config: RunConfig | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        snr_threshold: float = 15.0,
        psnr_threshold: float = 29.5,
    ) -> None:
        if (
            run_config.author == Author.coreai
            and run_config.source == Source.torch
            and reference_run_config is not None
            and reference_run_config.author == Author.oss
            and reference_run_config.source == Source.torch
        ):
            # additionally validate source between coreai torch vs oss torch (HF gpt_oss)
            coreai_torch_yarn_rope = self.source_model(run_config)
            oss_torch_yarn_rope = self.source_model(reference_run_config)
            assert isinstance(coreai_torch_yarn_rope, CoreaiTorchYarnRoPE)
            np.testing.assert_allclose(
                torch_tensor_to_numpy_array(coreai_torch_yarn_rope._freqs),
                torch_tensor_to_numpy_array(oss_torch_yarn_rope.rotary.inv_freq),
                rtol=rtol,
                atol=atol,
            )
        elif (
            run_config.author == Author.coreai
            and run_config.source == Source.torch
            and reference_run_config is not None
            and reference_run_config.author == Author.oss
            and reference_run_config.source == Source.mlx
        ):
            # MLX-LM's YarnRoPE hardcodes floor/ceil on the correction range
            # bounds (`mlx_lm/models/rope_utils.py:166-168`) -- it can only
            # represent `truncate=True` semantics. When this test runs with
            # `truncate=False` (gpt-oss's real config), MLX physically can't
            # match on the YaRN ramp dims. Skip MLX comparison in that case;
            # the truncate=False path is still validated against HF above
            # and end-to-end via TestGptOssEndtoEnd.
            if not self._rope_scaling.get("truncate", True):
                warnings.warn(
                    "Skipping MLX YaRN comparison because rope_scaling.truncate=False; "
                    "MLX-LM cannot represent that config. HF comparison still ran.",
                    stacklevel=2,
                )
                return
            # additionally validate source between coreai torch vs oss mlx
            coreai_torch_yarn_rope = self.source_model(run_config)
            oss_mlx_yarn_rope = self.source_model(reference_run_config)
            assert isinstance(coreai_torch_yarn_rope, CoreaiTorchYarnRoPE)
            assert isinstance(oss_mlx_yarn_rope, MlxlmYarnRoPE)
            np.testing.assert_allclose(
                torch_tensor_to_numpy_array(coreai_torch_yarn_rope._freqs),
                # MLX RoPE "frequency" is actually period, i.e. frequency = 1.0 / period
                1.0 / mlx_array_to_numpy_array(oss_mlx_yarn_rope._freqs),
                rtol=1e-5,
                atol=1e-5,
            )
        super().validate(
            run_config,
            reference_run_config=reference_run_config,
            rtol=rtol,
            atol=atol,
            snr_threshold=snr_threshold,
            psnr_threshold=psnr_threshold,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoPE:
    @staticmethod
    @pytest.mark.parametrize("model_class", [YarnRoPE])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_rope(model_class: type[RandomInputModel], precision: Precision) -> None:
        """Verify Core AI Torch / Core AI RoPE matches OSS (HF and MLX-LM)."""
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
        rtol = {
            Precision.f32: 1e-5,
            Precision.f16: 2e-3 if model_class is YarnRoPE else 1e-3,
            Precision.bf16: 2e-2 if model_class is YarnRoPE else 1e-2,
        }[precision]
        atol = {
            Precision.f32: 1e-5,
            Precision.f16: 2e-3 if model_class is YarnRoPE else 1e-3,
            Precision.bf16: 2e-2 if model_class is YarnRoPE else 1e-2,
        }[precision]
        if precision == Precision.f16:
            torch.random.manual_seed(42)
        with tempfile.TemporaryDirectory() as temp_directory:
            model = model_class(Path(temp_directory))
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
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
