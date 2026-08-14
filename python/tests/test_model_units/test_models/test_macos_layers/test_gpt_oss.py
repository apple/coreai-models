# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS GPT-OSS model."""

import functools
import os
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM as HFGptOssForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.macos.gpt_oss import GptOssForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from tests._runner_infra._deps import _HAS_MLX, _MSG_MLX_NOT_FOUND
from tests._runner_infra.common.types.dependency_types import (
    PRECISION_IN_SOURCE,
    SourceModel,
    Tensor,
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

if _HAS_MLX:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from mlx_lm.models.gpt_oss import AttentionBlock as MLXAttention
    from mlx_lm.models.gpt_oss import MLPBlock as MLXMoeMlp
    from mlx_lm.models.gpt_oss import ModelArgs as MLXGptOssModelArgs
    from mlx_lm.models.gpt_oss import SwiGLU as MLXGptOssSwiGLU
    from mlx_lm.models.gpt_oss import TransformerBlock as MLXTransformer
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention as HFGptOssAttention,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer as HFGptOssDecoderLayer,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssMLP as HFGptOssMLP,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssRotaryEmbedding,
)

from coreai_models.models.macos.gpt_oss import (
    Attention as CoreaiTorchAttention,
)
from coreai_models.models.macos.gpt_oss import (
    GptOssSwiGLU as CoreaiTorchGptOssSwiGLU,
)
from coreai_models.models.macos.gpt_oss import (
    MoeMlp as CoreaiTorchMoeMlp,
)
from coreai_models.models.macos.gpt_oss import (
    # ``coreai-models`` exports the transformer block as ``TransformerBlock``;
    # keep the local alias used below.
    TransformerBlock as CoreaiTorchTransformer,
)
from tests._runner_infra.models.model import Model
from tests._runner_infra.testing_utils import ForCausalLMTestBase


def _make_gpt_oss_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
    head_dim: int = 16,
    num_local_experts: int = 4,
    num_experts_per_tok: int = 2,
) -> GptOssConfig:
    config = GptOssConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        # GptOssConfig defaults to YaRN scaling, which the reauthored model does
        # not support. `rope_scaling` is an alias of `rope_parameters` as of
        # transformers 5.0, so setting it to None would drop the RoPE config
        # entirely instead of just disabling scaling.
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    return config


class TestmacOSGptOssForCausalLM:
    """Test macOS GptOssForCausalLM."""

    def test_forward_parity_single_token(self):
        config = _make_gpt_oss_config()

        hf_model = HFGptOssForCausalLM(config).to(torch.float32).eval()

        our_model = GptOssForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        config = _make_gpt_oss_config()
        our_model = GptOssForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab)


# =============================================================================
# Functional-parity tests
# =============================================================================
#
# The classes below cover four parity axes:
# * HF eager parity
# * MLX parity (gated by ``_HAS_MLX``)
# * ``torch.export`` parity
# * Core AI / Core AI-backend parity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def use_hf_impl():
    """Override conftest to disable HuggingFace implementation for GPT-OSS tests."""
    original = os.environ.get("USE_HF_IMPL")
    os.environ["USE_HF_IMPL"] = "false"
    yield
    if original is None:
        os.environ.pop("USE_HF_IMPL", None)
    else:
        os.environ["USE_HF_IMPL"] = original


# ---------------------------------------------------------------------------
# HF reference wrappers
# ---------------------------------------------------------------------------


class _HFGptOssAttention(torch.nn.Module):
    """Wrapper around HF GptOssAttention that accepts (x, position_ids)."""

    def __init__(self: Self, config: GptOssConfig, layer_idx: int = 0) -> None:
        super().__init__()
        config._attn_implementation = "eager"
        self.inner = HFGptOssAttention(config=config, layer_idx=layer_idx)
        self.rotary = GptOssRotaryEmbedding(config)

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        cos, sin = self.rotary(x, position_ids)
        output = self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )[0]
        return output


class _HFGptOssMLP(torch.nn.Module):
    """Wrapper around HF GptOssMLP that drops router_scores from output."""

    def __init__(self: Self, config: GptOssConfig) -> None:
        super().__init__()
        self.inner = HFGptOssMLP(config)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        routed_out, _router_scores = self.inner(x)
        return routed_out


class _HFGptOssDecoderLayer(torch.nn.Module):
    """Wrapper around HF GptOssDecoderLayer that accepts (x, position_ids)."""

    def __init__(self: Self, config: GptOssConfig, layer_idx: int = 0) -> None:
        super().__init__()
        config._attn_implementation = "eager"
        self.inner = HFGptOssDecoderLayer(config=config, layer_idx=layer_idx)
        self.rotary = GptOssRotaryEmbedding(config)

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        cos, sin = self.rotary(x, position_ids)
        output = self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )
        return output


# ---------------------------------------------------------------------------
# MLX wrappers
# ---------------------------------------------------------------------------

if _HAS_MLX:

    class _MlxGptOssSwiGLU(mlx_nn.Module):
        """Wraps MLX SwiGLU to accept (up, gate) matching coreai-torch signature."""

        def __init__(self: Self) -> None:
            super().__init__()
            self.inner = MLXGptOssSwiGLU()

        def __call__(self: Self, up: "mx.array", gate: "mx.array") -> "mx.array":
            return self.inner(up, gate)

    class _MlxGptOssAttention(mlx_nn.Module):
        """Wraps MLX AttentionBlock to accept (x, position_ids)."""

        def __init__(self: Self, config: "MLXGptOssModelArgs") -> None:
            super().__init__()
            self.inner = MLXAttention(config)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)

    class _MlxGptOssTransformer(mlx_nn.Module):
        """Wraps MLX TransformerBlock to accept (x, position_ids)."""

        def __init__(self: Self, config: "MLXGptOssModelArgs") -> None:
            super().__init__()
            self.inner = MLXTransformer(config)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DICT = {
    "model_type": "gpt_oss",
    "num_hidden_layers": 2,
    "num_local_experts": 32,
    "num_experts_per_tok": 4,
    "vocab_size": 201088,
    "rms_norm_eps": 1e-05,
    "hidden_size": 32,
    "intermediate_size": 64,
    "head_dim": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "sliding_window": 128,
    "rope_theta": 150000,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": False,
    },
    "layer_types": [
        "sliding_attention",
        "full_attention",
    ],
}


def _make_layers_gpt_oss_config(**overrides: object) -> GptOssConfig:
    """Single source of truth for test config."""
    d = {**_DEFAULT_CONFIG_DICT, **overrides}
    return GptOssConfig(**d)


def _make_mlx_gpt_oss_args(**overrides: object) -> "MLXGptOssModelArgs":
    d = {**_DEFAULT_CONFIG_DICT, **overrides}
    return MLXGptOssModelArgs(**d)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class GptOssSwiGLUModel(Model):
    """SwiGLU activation — no parameters, two-input forward(up, gate)."""

    _model_name = "GptOssSwiGLU"

    def __init__(
        self: Self,
        root_path: Path,
        batch_size: int = 2,
        seq_len: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(root_path=root_path)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._hidden_dim = hidden_dim

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchGptOssSwiGLU()
            dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
            model.to(dtype)
            return model
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            model = _MlxGptOssSwiGLU()
            # SwiGLU has no parameters, but set_dtype for consistency
            return model
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)

    @override
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if source_config == SourceConfig():
            assert source_config.source == Source.torch
            assert source_config.precision == Precision.f32
            return {
                "up": torch.rand(
                    (self._batch_size, self._seq_len, self._hidden_dim),
                    dtype=torch.float32,
                ),
                "gate": torch.rand(
                    (self._batch_size, self._seq_len, self._hidden_dim),
                    dtype=torch.float32,
                ),
            }
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_config)
                    dtype = PRECISION_IN_SOURCE[cast("Source", Source.torch)][
                        source_config.precision
                    ]
                    return {
                        name: t.to(dtype) if t.is_floating_point() else t
                        for name, t in named_inputs_f32.items()
                    }
                case Source.mlx:
                    torch_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_config)
                    import mlx.core

                    return {name: mlx.core.array(t) for name, t in named_inputs_torch.items()}
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)


class GptOssAttentionModel(Model):
    """Attention with biases on all projections and sinks parameter."""

    _model_name = "GptOssAttention"

    def __init__(
        self: Self,
        root_path: Path,
        batch_size: int = 2,
        seq_len: int = 4,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        config = _make_layers_gpt_oss_config()
        self._config = config
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        # Pre-generate shared weights (separate q/k/v, with biases)
        self._q_proj_weight = torch.randn(num_heads * head_dim, hidden_size)
        self._q_proj_bias = torch.randn(num_heads * head_dim)
        self._k_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        self._k_proj_bias = torch.randn(num_kv_heads * head_dim)
        self._v_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        self._v_proj_bias = torch.randn(num_kv_heads * head_dim)
        self._o_proj_weight = torch.randn(hidden_size, num_heads * head_dim)
        self._o_proj_bias = torch.randn(hidden_size)

        # Pre-generate sinks
        self._sinks = torch.randn(num_heads)

    def _load_torch_weights(self: Self, attn: torch.nn.Module) -> None:
        """Load pre-generated weights into coreai-torch Attention."""
        attn.q_proj.weight = torch.nn.Parameter(self._q_proj_weight.clone())
        attn.q_proj.bias = torch.nn.Parameter(self._q_proj_bias.clone())
        attn.k_proj.weight = torch.nn.Parameter(self._k_proj_weight.clone())
        attn.k_proj.bias = torch.nn.Parameter(self._k_proj_bias.clone())
        attn.v_proj.weight = torch.nn.Parameter(self._v_proj_weight.clone())
        attn.v_proj.bias = torch.nn.Parameter(self._v_proj_bias.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        attn.o_proj.bias = torch.nn.Parameter(self._o_proj_bias.clone())
        attn.sinks = torch.nn.Parameter(self._sinks.clone())

    def _load_torch_weights_hf(self: Self, hf_attn: torch.nn.Module) -> None:
        """Load pre-generated weights into HF GptOssAttention."""
        hf_attn.q_proj.weight = torch.nn.Parameter(self._q_proj_weight.clone())
        hf_attn.q_proj.bias = torch.nn.Parameter(self._q_proj_bias.clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(self._k_proj_weight.clone())
        hf_attn.k_proj.bias = torch.nn.Parameter(self._k_proj_bias.clone())
        hf_attn.v_proj.weight = torch.nn.Parameter(self._v_proj_weight.clone())
        hf_attn.v_proj.bias = torch.nn.Parameter(self._v_proj_bias.clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        hf_attn.o_proj.bias = torch.nn.Parameter(self._o_proj_bias.clone())
        hf_attn.sinks = torch.nn.Parameter(self._sinks.clone())

    def _load_mlx_weights(self: Self, mlx_attn: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX AttentionBlock."""
        inner = mlx_attn.inner
        dtype = inner.q_proj.weight.dtype

        inner.q_proj.weight = mx.array(self._q_proj_weight.numpy()).astype(dtype)
        inner.q_proj.bias = mx.array(self._q_proj_bias.numpy()).astype(dtype)
        inner.k_proj.weight = mx.array(self._k_proj_weight.numpy()).astype(dtype)
        inner.k_proj.bias = mx.array(self._k_proj_bias.numpy()).astype(dtype)
        inner.v_proj.weight = mx.array(self._v_proj_weight.numpy()).astype(dtype)
        inner.v_proj.bias = mx.array(self._v_proj_bias.numpy()).astype(dtype)
        inner.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)
        inner.o_proj.bias = mx.array(self._o_proj_bias.numpy()).astype(dtype)
        inner.sinks = mx.array(self._sinks.numpy()).astype(dtype)

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchAttention(config=self._config, layer_idx=0)
            self._load_torch_weights(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFGptOssAttention(config=self._config, layer_idx=0)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_gpt_oss_args()
            model = _MlxGptOssAttention(mlx_args)
            model.inner.set_dtype(dtype)
            self._load_mlx_weights(model)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @override
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if source_config == SourceConfig():
            assert source_config.source == Source.torch
            assert source_config.precision == Precision.f32
            named_inputs = {
                "x": torch.rand(
                    (self._batch_size, self._seq_len, self._config.hidden_size),
                    dtype=torch.float32,
                ),
            }
            named_inputs["position_ids"] = self._offset + torch.arange(
                self._seq_len, dtype=torch.int32
            ).unsqueeze(0).expand(self._batch_size, -1)
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_config)
                    dtype = PRECISION_IN_SOURCE[cast("Source", Source.torch)][
                        source_config.precision
                    ]
                    named_inputs = {}
                    for name, tensor in named_inputs_f32.items():
                        if tensor.is_floating_point():
                            named_inputs[name] = tensor.to(dtype)
                        else:
                            named_inputs[name] = tensor
                case Source.mlx:
                    torch_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_config)
                    import mlx.core

                    named_inputs = {
                        name: mlx.core.array(t) for name, t in named_inputs_torch.items()
                    }
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class GptOssMoeMlpModel(Model):
    """MoE MLP with router bias and expert biases."""

    _model_name = "GptOssMoeMlp"

    def __init__(
        self: Self,
        root_path: Path,
        batch_size: int = 2,
        seq_len: int = 4,
    ) -> None:
        super().__init__(root_path=root_path)
        config = _make_layers_gpt_oss_config()
        self._config = config
        self._batch_size = batch_size
        self._seq_len = seq_len

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_experts = config.num_local_experts

        # Pre-generate router weights + bias
        self._router_weight = torch.randn(num_experts, hidden_size)
        self._router_bias = torch.randn(num_experts)

        # Pre-generate expert weights + biases
        # coreai-torch SwitchGLU uses shape (1, num_experts, out, in)
        self._experts_gate_proj_weight = torch.randn(1, num_experts, intermediate_size, hidden_size)
        self._experts_up_proj_weight = torch.randn(1, num_experts, intermediate_size, hidden_size)
        self._experts_down_proj_weight = torch.randn(1, num_experts, hidden_size, intermediate_size)
        self._experts_gate_proj_bias = torch.randn(1, num_experts, intermediate_size)
        self._experts_up_proj_bias = torch.randn(1, num_experts, intermediate_size)
        self._experts_down_proj_bias = torch.randn(1, num_experts, hidden_size)

    def _load_torch_weights(self: Self, moe: torch.nn.Module) -> None:
        """Load pre-generated weights into coreai-torch MoeMlp."""
        moe.router.weight = torch.nn.Parameter(self._router_weight.clone())
        moe.router.bias = torch.nn.Parameter(self._router_bias.clone())
        moe.experts.gate_proj.weight = torch.nn.Parameter(self._experts_gate_proj_weight.clone())
        moe.experts.up_proj.weight = torch.nn.Parameter(self._experts_up_proj_weight.clone())
        moe.experts.down_proj.weight = torch.nn.Parameter(self._experts_down_proj_weight.clone())
        moe.experts.gate_proj.bias = torch.nn.Parameter(self._experts_gate_proj_bias.clone())
        moe.experts.up_proj.bias = torch.nn.Parameter(self._experts_up_proj_bias.clone())
        moe.experts.down_proj.bias = torch.nn.Parameter(self._experts_down_proj_bias.clone())

    def _load_torch_weights_hf(self: Self, hf_mlp: torch.nn.Module) -> None:
        """Load pre-generated weights into HF GptOssMLP (interleaved gate_up layout)."""
        num_experts = self._config.num_local_experts
        intermediate_size = self._config.intermediate_size
        hidden_size = self._config.hidden_size

        # Router: HF GptOssTopKRouter has raw weight/bias Parameters
        hf_mlp.router.weight = torch.nn.Parameter(self._router_weight.clone())
        hf_mlp.router.bias = torch.nn.Parameter(self._router_bias.clone())

        # Experts: interleave gate/up into gate_up_proj
        # Our layout: (1, E, intermediate, hidden) -> squeeze -> (E, intermediate, hidden)
        # -> transpose -> (E, hidden, intermediate)
        gate_w = self._experts_gate_proj_weight[0].transpose(1, 2)  # (E, hidden, inter)
        up_w = self._experts_up_proj_weight[0].transpose(1, 2)  # (E, hidden, inter)
        # HF gate_up_proj: (E, hidden, 2*inter) with gate at [::2], up at [1::2]
        gate_up_w = torch.zeros(num_experts, hidden_size, 2 * intermediate_size, dtype=gate_w.dtype)
        gate_up_w[..., ::2] = gate_w
        gate_up_w[..., 1::2] = up_w
        hf_mlp.experts.gate_up_proj = torch.nn.Parameter(gate_up_w)

        # gate_up_proj_bias: interleave (1, E, inter) gate and up biases
        gate_b = self._experts_gate_proj_bias[0]  # (E, inter)
        up_b = self._experts_up_proj_bias[0]  # (E, inter)
        gate_up_b = torch.zeros(num_experts, 2 * intermediate_size, dtype=gate_b.dtype)
        gate_up_b[..., ::2] = gate_b
        gate_up_b[..., 1::2] = up_b
        hf_mlp.experts.gate_up_proj_bias = torch.nn.Parameter(gate_up_b)

        # down_proj: (1, E, hidden, inter) -> squeeze -> (E, hidden, inter)
        # -> transpose -> (E, inter, hidden) to match HF layout
        down_w = self._experts_down_proj_weight[0].transpose(1, 2)  # (E, inter, hidden)
        hf_mlp.experts.down_proj = torch.nn.Parameter(down_w)

        # down_proj_bias: (1, E, hidden) -> squeeze -> (E, hidden)
        hf_mlp.experts.down_proj_bias = torch.nn.Parameter(self._experts_down_proj_bias[0].clone())

    def _load_mlx_weights(self: Self, mlx_moe: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX MLPBlock."""
        dtype = mlx_moe.router.weight.dtype

        mlx_moe.router.weight = mx.array(self._router_weight.numpy()).astype(dtype)
        mlx_moe.router.bias = mx.array(self._router_bias.numpy()).astype(dtype)
        # MLX SwitchGLU uses (num_experts, out, in) — squeeze the leading 1
        mlx_moe.experts.gate_proj.weight = mx.array(
            self._experts_gate_proj_weight[0].numpy()
        ).astype(dtype)
        mlx_moe.experts.up_proj.weight = mx.array(self._experts_up_proj_weight[0].numpy()).astype(
            dtype
        )
        mlx_moe.experts.down_proj.weight = mx.array(
            self._experts_down_proj_weight[0].numpy()
        ).astype(dtype)
        mlx_moe.experts.gate_proj.bias = mx.array(self._experts_gate_proj_bias[0].numpy()).astype(
            dtype
        )
        mlx_moe.experts.up_proj.bias = mx.array(self._experts_up_proj_bias[0].numpy()).astype(dtype)
        mlx_moe.experts.down_proj.bias = mx.array(self._experts_down_proj_bias[0].numpy()).astype(
            dtype
        )

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchMoeMlp(config=self._config)
            self._load_torch_weights(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFGptOssMLP(config=self._config)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_gpt_oss_args()
            model = MLXMoeMlp(mlx_args)
            model.set_dtype(dtype)
            self._load_mlx_weights(model)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @override
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if source_config == SourceConfig():
            assert source_config.source == Source.torch
            assert source_config.precision == Precision.f32
            return {
                "x": torch.rand(
                    (self._batch_size, self._seq_len, self._config.hidden_size),
                    dtype=torch.float32,
                ),
            }
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_config)
                    dtype = PRECISION_IN_SOURCE[cast("Source", Source.torch)][
                        source_config.precision
                    ]
                    return {
                        name: t.to(dtype) if t.is_floating_point() else t
                        for name, t in named_inputs_f32.items()
                    }
                case Source.mlx:
                    torch_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_config)
                    import mlx.core

                    return {name: mlx.core.array(t) for name, t in named_inputs_torch.items()}
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)


class GptOssTransformerModel(Model):
    """Transformer block — attention + MoE MLP + layernorms."""

    _model_name = "GptOssTransformer"

    def __init__(
        self: Self,
        root_path: Path,
        batch_size: int = 2,
        seq_len: int = 4,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        config = _make_layers_gpt_oss_config()
        self._config = config
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size
        num_experts = config.num_local_experts

        # -- Attention weights + biases --
        self._q_proj_weight = torch.randn(num_heads * head_dim, hidden_size)
        self._q_proj_bias = torch.randn(num_heads * head_dim)
        self._k_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        self._k_proj_bias = torch.randn(num_kv_heads * head_dim)
        self._v_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        self._v_proj_bias = torch.randn(num_kv_heads * head_dim)
        self._o_proj_weight = torch.randn(hidden_size, num_heads * head_dim)
        self._o_proj_bias = torch.randn(hidden_size)
        self._sinks = torch.randn(num_heads)

        # -- MoE MLP weights + biases --
        self._router_weight = torch.randn(num_experts, hidden_size)
        self._router_bias = torch.randn(num_experts)
        self._experts_gate_proj_weight = torch.randn(1, num_experts, intermediate_size, hidden_size)
        self._experts_up_proj_weight = torch.randn(1, num_experts, intermediate_size, hidden_size)
        self._experts_down_proj_weight = torch.randn(1, num_experts, hidden_size, intermediate_size)
        self._experts_gate_proj_bias = torch.randn(1, num_experts, intermediate_size)
        self._experts_up_proj_bias = torch.randn(1, num_experts, intermediate_size)
        self._experts_down_proj_bias = torch.randn(1, num_experts, hidden_size)

        # -- Layernorm weights --
        self._input_ln_weight = torch.randn(hidden_size)
        self._post_attn_ln_weight = torch.randn(hidden_size)

    def _load_torch_weights(self: Self, block: torch.nn.Module) -> None:
        """Load pre-generated weights into coreai-torch Transformer."""
        attn = block.self_attn
        attn.q_proj.weight = torch.nn.Parameter(self._q_proj_weight.clone())
        attn.q_proj.bias = torch.nn.Parameter(self._q_proj_bias.clone())
        attn.k_proj.weight = torch.nn.Parameter(self._k_proj_weight.clone())
        attn.k_proj.bias = torch.nn.Parameter(self._k_proj_bias.clone())
        attn.v_proj.weight = torch.nn.Parameter(self._v_proj_weight.clone())
        attn.v_proj.bias = torch.nn.Parameter(self._v_proj_bias.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        attn.o_proj.bias = torch.nn.Parameter(self._o_proj_bias.clone())
        attn.sinks = torch.nn.Parameter(self._sinks.clone())

        mlp = block.mlp
        mlp.router.weight = torch.nn.Parameter(self._router_weight.clone())
        mlp.router.bias = torch.nn.Parameter(self._router_bias.clone())
        mlp.experts.gate_proj.weight = torch.nn.Parameter(self._experts_gate_proj_weight.clone())
        mlp.experts.up_proj.weight = torch.nn.Parameter(self._experts_up_proj_weight.clone())
        mlp.experts.down_proj.weight = torch.nn.Parameter(self._experts_down_proj_weight.clone())
        mlp.experts.gate_proj.bias = torch.nn.Parameter(self._experts_gate_proj_bias.clone())
        mlp.experts.up_proj.bias = torch.nn.Parameter(self._experts_up_proj_bias.clone())
        mlp.experts.down_proj.bias = torch.nn.Parameter(self._experts_down_proj_bias.clone())

        block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_torch_weights_hf(self: Self, hf_block: torch.nn.Module) -> None:
        """Load pre-generated weights into HF GptOssDecoderLayer."""
        num_experts = self._config.num_local_experts
        intermediate_size = self._config.intermediate_size
        hidden_size = self._config.hidden_size

        # Attention weights (same layout as ours)
        attn = hf_block.self_attn
        attn.q_proj.weight = torch.nn.Parameter(self._q_proj_weight.clone())
        attn.q_proj.bias = torch.nn.Parameter(self._q_proj_bias.clone())
        attn.k_proj.weight = torch.nn.Parameter(self._k_proj_weight.clone())
        attn.k_proj.bias = torch.nn.Parameter(self._k_proj_bias.clone())
        attn.v_proj.weight = torch.nn.Parameter(self._v_proj_weight.clone())
        attn.v_proj.bias = torch.nn.Parameter(self._v_proj_bias.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        attn.o_proj.bias = torch.nn.Parameter(self._o_proj_bias.clone())
        attn.sinks = torch.nn.Parameter(self._sinks.clone())

        # MoE MLP weights (interleaved gate_up layout)
        mlp = hf_block.mlp
        mlp.router.weight = torch.nn.Parameter(self._router_weight.clone())
        mlp.router.bias = torch.nn.Parameter(self._router_bias.clone())

        gate_w = self._experts_gate_proj_weight[0].transpose(1, 2)
        up_w = self._experts_up_proj_weight[0].transpose(1, 2)
        gate_up_w = torch.zeros(num_experts, hidden_size, 2 * intermediate_size, dtype=gate_w.dtype)
        gate_up_w[..., ::2] = gate_w
        gate_up_w[..., 1::2] = up_w
        mlp.experts.gate_up_proj = torch.nn.Parameter(gate_up_w)

        gate_b = self._experts_gate_proj_bias[0]
        up_b = self._experts_up_proj_bias[0]
        gate_up_b = torch.zeros(num_experts, 2 * intermediate_size, dtype=gate_b.dtype)
        gate_up_b[..., ::2] = gate_b
        gate_up_b[..., 1::2] = up_b
        mlp.experts.gate_up_proj_bias = torch.nn.Parameter(gate_up_b)

        down_w = self._experts_down_proj_weight[0].transpose(1, 2)
        mlp.experts.down_proj = torch.nn.Parameter(down_w)
        mlp.experts.down_proj_bias = torch.nn.Parameter(self._experts_down_proj_bias[0].clone())

        # Layernorm weights
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_mlx_weights(self: Self, mlx_block: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX TransformerBlock."""
        inner = mlx_block.inner
        attn = inner.self_attn
        dtype = attn.q_proj.weight.dtype

        attn.q_proj.weight = mx.array(self._q_proj_weight.numpy()).astype(dtype)
        attn.q_proj.bias = mx.array(self._q_proj_bias.numpy()).astype(dtype)
        attn.k_proj.weight = mx.array(self._k_proj_weight.numpy()).astype(dtype)
        attn.k_proj.bias = mx.array(self._k_proj_bias.numpy()).astype(dtype)
        attn.v_proj.weight = mx.array(self._v_proj_weight.numpy()).astype(dtype)
        attn.v_proj.bias = mx.array(self._v_proj_bias.numpy()).astype(dtype)
        attn.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)
        attn.o_proj.bias = mx.array(self._o_proj_bias.numpy()).astype(dtype)
        attn.sinks = mx.array(self._sinks.numpy()).astype(dtype)

        mlp = inner.mlp
        mlp.router.weight = mx.array(self._router_weight.numpy()).astype(dtype)
        mlp.router.bias = mx.array(self._router_bias.numpy()).astype(dtype)
        mlp.experts.gate_proj.weight = mx.array(self._experts_gate_proj_weight[0].numpy()).astype(
            dtype
        )
        mlp.experts.up_proj.weight = mx.array(self._experts_up_proj_weight[0].numpy()).astype(dtype)
        mlp.experts.down_proj.weight = mx.array(self._experts_down_proj_weight[0].numpy()).astype(
            dtype
        )
        mlp.experts.gate_proj.bias = mx.array(self._experts_gate_proj_bias[0].numpy()).astype(dtype)
        mlp.experts.up_proj.bias = mx.array(self._experts_up_proj_bias[0].numpy()).astype(dtype)
        mlp.experts.down_proj.bias = mx.array(self._experts_down_proj_bias[0].numpy()).astype(dtype)

        inner.input_layernorm.weight = mx.array(self._input_ln_weight.numpy()).astype(dtype)
        inner.post_attention_layernorm.weight = mx.array(self._post_attn_ln_weight.numpy()).astype(
            dtype
        )

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchTransformer(config=self._config, layer_idx=0)
            self._load_torch_weights(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFGptOssDecoderLayer(config=self._config, layer_idx=0)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_gpt_oss_args()
            model = _MlxGptOssTransformer(mlx_args)
            model.inner.set_dtype(dtype)
            self._load_mlx_weights(model)
        else:
            msg = f"Does not support {source_config}"
            raise NotImplementedError(msg)
        return model

    @override
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if source_config == SourceConfig():
            assert source_config.source == Source.torch
            assert source_config.precision == Precision.f32
            named_inputs = {
                "x": torch.rand(
                    (self._batch_size, self._seq_len, self._config.hidden_size),
                    dtype=torch.float32,
                ),
            }
            named_inputs["position_ids"] = self._offset + torch.arange(
                self._seq_len, dtype=torch.int32
            ).unsqueeze(0).expand(self._batch_size, -1)
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_config)
                    dtype = PRECISION_IN_SOURCE[cast("Source", Source.torch)][
                        source_config.precision
                    ]
                    named_inputs = {}
                    for name, tensor in named_inputs_f32.items():
                        if tensor.is_floating_point():
                            named_inputs[name] = tensor.to(dtype)
                        else:
                            named_inputs[name] = tensor
                case Source.mlx:
                    torch_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_config)
                    import mlx.core

                    named_inputs = {
                        name: mlx.core.array(t) for name, t in named_inputs_torch.items()
                    }
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGptOssLayers:
    @staticmethod
    @pytest.mark.parametrize(
        "model_class",
        [
            GptOssSwiGLUModel,
            GptOssAttentionModel,
            GptOssMoeMlpModel,
            GptOssTransformerModel,
        ],
    )
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_gpt_oss_layers(
        model_class: type[Model],
        precision: Precision,
    ) -> None:
        """Verify coreai-torch GPT-OSS layers match HF, MLX, and Core AI export."""
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

        # Due to we cannot use hf impl for sdpa sink reason
        # we need relaxed tolerance
        rtol = {Precision.f32: 1e-2, Precision.f16: 3e-1, Precision.bf16: 1e1}[precision]
        atol = {Precision.f32: 1e-2, Precision.f16: 3e-1, Precision.bf16: 1e1}[precision]

        if model_class == GptOssTransformerModel and precision == Precision.f16:
            pytest.xfail("Unstable config")

        with tempfile.TemporaryDirectory() as temp_directory:
            model = model_class(Path(temp_directory))
            if model_class is not GptOssSwiGLUModel:
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


# ---------------------------------------------------------------------------
# ForCausalLM test (unchanged)
# ---------------------------------------------------------------------------


class TestGptOssForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/gpt-oss-tiny-random"
    _model_class = GptOssForCausalLM
