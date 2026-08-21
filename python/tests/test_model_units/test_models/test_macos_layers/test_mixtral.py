# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Mixtral model parity with HuggingFace."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.mixtral.modeling_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM as HFMixtralForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.macos.mixtral import MixtralForCausalLM
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
    from mlx_lm.models.mixtral import MixtralAttention as MlxMixtralAttentionInner
    from mlx_lm.models.mixtral import MixtralDecoderLayer as MlxMixtralDecoderLayer
    from mlx_lm.models.mixtral import (
        MixtralSparseMoeBlock as MlxMixtralSparseMoeBlockInner,
    )
    from mlx_lm.models.mixtral import ModelArgs as MlxMixtralModelArgs
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention as HFMixtralAttention,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralSparseMoeBlock as HFMixtralSparseMoeBlock,
)

from coreai_models.models.macos import mixtral as mixtral_module
from coreai_models.models.macos.mixtral import (
    Attention as CoreaiTorchAttention,
)
from coreai_models.models.macos.mixtral import (
    MixtralForCausalLM as CoreaiTorchMixtralForCausalLM,
)
from coreai_models.models.macos.mixtral import (
    SparseMoeBlock as CoreaiTorchSparseMoeBlock,
)
from coreai_models.models.macos.mixtral import (
    TransformerBlock as CoreaiTorchTransformerBlock,
)
from tests._runner_infra.models.model import Model
from tests._runner_infra.testing_utils import ForCausalLMTestBase


def _make_mixtral_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    num_local_experts: int = 4,
    num_experts_per_tok: int = 2,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
) -> MixtralConfig:
    config = MixtralConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        # Must be passed to the constructor: as of transformers 5.0 the theta
        # lives in `rope_parameters`, and assigning `config.rope_theta`
        # afterwards only sets an attribute that the HF model ignores.
        rope_theta=10000.0,
    )
    return config


def _load_hf_moe_weights(
    hf_moe: torch.nn.Module,
    gate_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> None:
    """Load SwitchGLU-layout expert weights into an HF MixtralSparseMoeBlock.

    transformers >= 5.0 stores the experts of a layer in two fused 3D
    parameters on ``MixtralExperts`` rather than per-expert w1/w2/w3 linears,
    with w1 (gate) and w3 (up) concatenated along the output dim and w2 as
    ``down_proj``.

    The projection weights use the SwitchGLU layout, i.e. a leading singleton
    dim in front of the expert dim.
    """
    hf_moe.gate.weight = torch.nn.Parameter(gate_weight.clone())
    hf_moe.experts.gate_up_proj = torch.nn.Parameter(
        torch.cat([gate_proj_weight[0], up_proj_weight[0]], dim=1)
    )
    hf_moe.experts.down_proj = torch.nn.Parameter(down_proj_weight[0].clone())


class TestmacOSMixtralForCausalLM:
    """Test macOS MixtralForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_mixtral_config()

        hf_model = HFMixtralForCausalLM(config).to(torch.float32).eval()

        our_model = MixtralForCausalLM(config, model_device="cpu")
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

    def test_forward_parity_multi_token(self):
        seq_len = 8
        config = _make_mixtral_config()

        hf_model = HFMixtralForCausalLM(config).to(torch.float32).eval()

        our_model = MixtralForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        config = _make_mixtral_config()
        our_model = MixtralForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab)

    def test_mutate_state_dict_stacks_moe_experts(self):
        config = _make_mixtral_config()
        our_model = MixtralForCausalLM(config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = hidden // n_heads
        num_experts = 4
        intermediate = 128

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.block_sparse_moe.gate.weight"] = torch.randn(num_experts, hidden)
        for e in range(num_experts):
            sd[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(
                intermediate, hidden
            )
            sd[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(
                hidden, intermediate
            )
            sd[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(
                intermediate, hidden
            )

        our_model._mutate_state_dict(sd)

        # MoE experts should be stacked
        assert "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight" in sd
        assert sd["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"].shape == (
            1,
            num_experts,
            intermediate,
            hidden,
        )
        assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in sd

    def test_mutate_state_dict_splits_fused_moe_experts(self):
        """Fused (transformers >= 5.0) `mlp` experts become SwitchGLU weights."""
        config = _make_mixtral_config()
        our_model = MixtralForCausalLM(config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = hidden // n_heads
        num_experts = 4
        intermediate = 128

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)
        router_weight = torch.randn(num_experts, hidden)
        sd["model.layers.0.mlp.gate.weight"] = router_weight
        gate_up_proj = torch.randn(num_experts, 2 * intermediate, hidden)
        sd["model.layers.0.mlp.experts.gate_up_proj"] = gate_up_proj
        sd["model.layers.0.mlp.experts.down_proj"] = torch.randn(num_experts, hidden, intermediate)

        our_model._mutate_state_dict(sd)

        prefix = "model.layers.0.block_sparse_moe"
        for proj, expected_shape in (
            ("gate_proj", (1, num_experts, intermediate, hidden)),
            ("up_proj", (1, num_experts, intermediate, hidden)),
            ("down_proj", (1, num_experts, hidden, intermediate)),
        ):
            key = f"{prefix}.switch_mlp.{proj}.weight"
            assert key in sd
            assert sd[key].shape == expected_shape

        # w1 (gate) and w3 (up) are packed in that order along the output dim
        torch.testing.assert_close(
            sd[f"{prefix}.switch_mlp.gate_proj.weight"][0], gate_up_proj[:, :intermediate]
        )
        torch.testing.assert_close(
            sd[f"{prefix}.switch_mlp.up_proj.weight"][0], gate_up_proj[:, intermediate:]
        )

        # The router moves to our module name
        torch.testing.assert_close(sd[f"{prefix}.gate.weight"], router_weight)

        assert "model.layers.0.mlp.gate.weight" not in sd
        assert "model.layers.0.mlp.experts.gate_up_proj" not in sd
        assert "model.layers.0.mlp.experts.down_proj" not in sd


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
# HF reference wrappers
# ---------------------------------------------------------------------------


class _HFMixtralAttention(torch.nn.Module):
    """Wrapper around HF MixtralAttention that accepts (x, position_ids)."""

    def __init__(self: Self, config: MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        self.inner = HFMixtralAttention(config=config, layer_idx=layer_idx)
        self.rotary = MixtralRotaryEmbedding(config)

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        # Build causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Compute RoPE embeddings
        cos, sin = self.rotary(x, position_ids)
        output = self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )[0]
        return output


class _HFMixtralTransformerBlock(torch.nn.Module):
    """Wrapper around HF MixtralDecoderLayer that accepts (x, position_ids)."""

    def __init__(self: Self, config: MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        self.inner = MixtralDecoderLayer(config=config, layer_idx=layer_idx)
        self.rotary = MixtralRotaryEmbedding(config)

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        # Build causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Compute RoPE embeddings
        cos, sin = self.rotary(x, position_ids)
        output = self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )
        return output


class _HFMixtralSparseMoeBlockWrapper(torch.nn.Module):
    """Wrapper around HF MixtralSparseMoeBlock.

    transformers >= 5.0 returns the hidden states directly instead of a
    ``(hidden_states, router_logits)`` tuple.
    """

    def __init__(self: Self, config: MixtralConfig) -> None:
        super().__init__()
        self.inner = HFMixtralSparseMoeBlock(config)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


# ---------------------------------------------------------------------------
# MLX wrappers
# ---------------------------------------------------------------------------

if _HAS_MLX:

    class _MlxMixtralAttention(mlx_nn.Module):
        """Wraps mlx_lm Mixtral Attention to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxMixtralModelArgs") -> None:
            super().__init__()
            self.inner = MlxMixtralAttentionInner(args)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)

    class _MlxMixtralTransformerBlock(mlx_nn.Module):
        """Wraps mlx_lm Mixtral DecoderLayer to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxMixtralModelArgs") -> None:
            super().__init__()
            self.inner = MlxMixtralDecoderLayer(args)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)

    class _MlxMixtralSparseMoeBlock(mlx_nn.Module):
        """Wraps mlx_lm Mixtral SparseMoeBlock to accept (x) -> array."""

        def __init__(self: Self, args: "MlxMixtralModelArgs") -> None:
            super().__init__()
            self.inner = MlxMixtralSparseMoeBlockInner(args)

        def __call__(self: Self, x: "mx.array") -> "mx.array":
            return self.inner(x)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class MixtralAttention(Model):
    _model_name = "MixtralAttention"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 2,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        batch_size: int = 2,
        seq_len: int = 10,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._layer_idx = layer_idx
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        # additional config args derived from user args
        self._hidden_size = num_attention_heads * head_dim

        # Pre-generate shared weights (no bias for Mixtral)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

    def _load_torch_weights_ours(self: Self, attn: torch.nn.Module) -> None:
        """Load pre-generated weights into our fused-qkv Attention."""
        attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_attn: torch.nn.Module) -> None:
        """Load pre-generated weights into HF's separate q/k/v projections."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        hf_attn.q_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[:q_size].clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(
            self._qkv_proj_weight[q_size : q_size + k_size].clone()
        )
        hf_attn.v_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[q_size + k_size :].clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

    def _load_mlx_weights(self: Self, mlx_attn: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Mixtral Attention."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        dtype = mlx_attn.inner.q_proj.weight.dtype

        mlx_attn.inner.q_proj.weight = mx.array(self._qkv_proj_weight[:q_size].numpy()).astype(
            dtype
        )
        mlx_attn.inner.k_proj.weight = mx.array(
            self._qkv_proj_weight[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        mlx_attn.inner.v_proj.weight = mx.array(
            self._qkv_proj_weight[q_size + k_size :].numpy()
        ).astype(dtype)
        mlx_attn.inner.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)

    def _make_config(self: Self) -> MixtralConfig:
        config = MixtralConfig(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=6,
            num_local_experts=4,
            rms_norm_eps=9.87,
            rope_theta=1e5,
        )
        config._attn_implementation = "sdpa"
        return config

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        config = self._make_config()
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchAttention(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFMixtralAttention(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = MlxMixtralModelArgs(
                model_type="mixtral",
                hidden_size=self._hidden_size,
                num_hidden_layers=1,
                intermediate_size=6,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                num_local_experts=4,
                rms_norm_eps=9.87,
                vocab_size=1,
                rope_theta=1e5,
            )
            model = _MlxMixtralAttention(mlx_args)
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
                    (self._batch_size, self._seq_len, self._hidden_size),
                    dtype=torch.float32,
                )
            }
            # Generate position_ids starting from offset
            named_inputs["position_ids"] = self._offset + torch.arange(
                self._seq_len, dtype=torch.int32
            ).unsqueeze(0).expand(self._batch_size, -1)
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_source_config)
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
                    torch_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_source_config)
                    import mlx.core

                    named_inputs = {
                        name: mlx.core.array(input_torch)
                        for name, input_torch in named_inputs_torch.items()
                    }
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class MixtralTransformerBlock(Model):
    """MoE decoder layer -- every Mixtral layer uses MoE (no dense layers)."""

    _model_name = "MixtralTransformerBlock"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 2,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        num_local_experts: int = 4,
        intermediate_size: int = 6,
        num_experts_per_tok: int = 2,
        layer_idx: int = 0,
        batch_size: int = 1,
        seq_len: int = 10,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._num_local_experts = num_local_experts
        self._intermediate_size = intermediate_size
        self._num_experts_per_tok = num_experts_per_tok
        self._layer_idx = layer_idx
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        self._hidden_size = num_attention_heads * head_dim

        # Pre-generate shared attention weights (no bias, no q_norm/k_norm)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

        # Pre-generate MoE gate weight
        self._moe_gate_weight = torch.randn(num_local_experts, self._hidden_size)

        # Pre-generate SwitchGLU weights (optimized layout)
        # HF per-expert: experts[i].w1 -> gate, w3 -> up, w2 -> down
        self._switch_gate_proj_weight = torch.randn(
            1, num_local_experts, intermediate_size, self._hidden_size
        )
        self._switch_up_proj_weight = torch.randn(
            1, num_local_experts, intermediate_size, self._hidden_size
        )
        self._switch_down_proj_weight = torch.randn(
            1, num_local_experts, self._hidden_size, intermediate_size
        )

        # Pre-generate shared layernorm weights
        self._input_ln_weight = torch.randn(self._hidden_size)
        self._post_attn_ln_weight = torch.randn(self._hidden_size)

    def _load_torch_weights_ours(self: Self, block: torch.nn.Module) -> None:
        """Load pre-generated weights into our TransformerBlock (MoE)."""
        # Attention weights (fused qkv, no bias, no q_norm/k_norm)
        block.self_attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        block.self_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        # MoE weights (accessed via block_sparse_moe)
        block.block_sparse_moe.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        block.block_sparse_moe.switch_mlp.gate_proj.weight = torch.nn.Parameter(
            self._switch_gate_proj_weight.clone()
        )
        block.block_sparse_moe.switch_mlp.up_proj.weight = torch.nn.Parameter(
            self._switch_up_proj_weight.clone()
        )
        block.block_sparse_moe.switch_mlp.down_proj.weight = torch.nn.Parameter(
            self._switch_down_proj_weight.clone()
        )
        # Layernorm weights
        block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_torch_weights_hf(self: Self, hf_block: torch.nn.Module) -> None:
        """Load pre-generated weights into HF MixtralDecoderLayer."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        # Attention weights (separate q/k/v, no bias)
        hf_attn = hf_block.self_attn
        hf_attn.q_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[:q_size].clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(
            self._qkv_proj_weight[q_size : q_size + k_size].clone()
        )
        hf_attn.v_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[q_size + k_size :].clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

        # MoE weights (fused expert layout). transformers >= 5.0 renamed the
        # submodule from `block_sparse_moe` to `mlp`.
        _load_hf_moe_weights(
            hf_block.mlp,
            self._moe_gate_weight,
            self._switch_gate_proj_weight,
            self._switch_up_proj_weight,
            self._switch_down_proj_weight,
        )

        # Layernorm weights
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_mlx_weights(self: Self, mlx_block: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Mixtral DecoderLayer."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        inner = mlx_block.inner
        dtype = inner.self_attn.q_proj.weight.dtype

        # Attention weights (no bias, no q_norm/k_norm)
        inner.self_attn.q_proj.weight = mx.array(self._qkv_proj_weight[:q_size].numpy()).astype(
            dtype
        )
        inner.self_attn.k_proj.weight = mx.array(
            self._qkv_proj_weight[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        inner.self_attn.v_proj.weight = mx.array(
            self._qkv_proj_weight[q_size + k_size :].numpy()
        ).astype(dtype)
        inner.self_attn.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)

        # MoE weights
        # MLX SwitchGLU uses [num_experts, intermediate, hidden] (no leading dim 1)
        inner.block_sparse_moe.gate.weight = mx.array(self._moe_gate_weight.numpy()).astype(dtype)
        inner.block_sparse_moe.switch_mlp.gate_proj.weight = mx.array(
            self._switch_gate_proj_weight[0].numpy()
        ).astype(dtype)
        inner.block_sparse_moe.switch_mlp.up_proj.weight = mx.array(
            self._switch_up_proj_weight[0].numpy()
        ).astype(dtype)
        inner.block_sparse_moe.switch_mlp.down_proj.weight = mx.array(
            self._switch_down_proj_weight[0].numpy()
        ).astype(dtype)

        # Layernorm weights
        inner.input_layernorm.weight = mx.array(self._input_ln_weight.numpy()).astype(dtype)
        inner.post_attention_layernorm.weight = mx.array(self._post_attn_ln_weight.numpy()).astype(
            dtype
        )

    def _make_config(self: Self) -> MixtralConfig:
        config = MixtralConfig(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=self._intermediate_size,
            num_local_experts=self._num_local_experts,
            num_experts_per_tok=self._num_experts_per_tok,
            rms_norm_eps=9.87,
            rope_theta=1e5,
        )
        config._attn_implementation = "sdpa"
        return config

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        config = self._make_config()
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchTransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFMixtralTransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = MlxMixtralModelArgs(
                model_type="mixtral",
                hidden_size=self._hidden_size,
                num_hidden_layers=1,
                intermediate_size=self._intermediate_size,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                num_local_experts=self._num_local_experts,
                num_experts_per_tok=self._num_experts_per_tok,
                rms_norm_eps=9.87,
                vocab_size=1,
                rope_theta=1e5,
            )
            model = _MlxMixtralTransformerBlock(mlx_args)
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
                    (self._batch_size, self._seq_len, self._hidden_size),
                    dtype=torch.float32,
                )
            }
            named_inputs["position_ids"] = self._offset + torch.arange(
                self._seq_len, dtype=torch.int32
            ).unsqueeze(0).expand(self._batch_size, -1)
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_source_config)
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
                    torch_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_source_config)
                    import mlx.core

                    named_inputs = {
                        name: mlx.core.array(input_torch)
                        for name, input_torch in named_inputs_torch.items()
                    }
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class MixtralSparseMoeBlock(Model):
    """Standalone SparseMoeBlock (no attention, no layernorm)."""

    _model_name = "MixtralSparseMoeBlock"

    def __init__(
        self: Self,
        root_path: Path,
        hidden_size: int = 4,
        intermediate_size: int = 6,
        num_local_experts: int = 4,
        top_k: int = 2,
        batch_size: int = 2,
        seq_len: int = 10,
    ) -> None:
        super().__init__(root_path=root_path)
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_local_experts = num_local_experts
        self._top_k = top_k
        self._batch_size = batch_size
        self._seq_len = seq_len

        # Pre-generate MoE gate weight
        self._moe_gate_weight = torch.randn(num_local_experts, hidden_size)

        # Pre-generate SwitchGLU weights (optimized layout)
        self._switch_gate_proj_weight = torch.randn(
            1, num_local_experts, intermediate_size, hidden_size
        )
        self._switch_up_proj_weight = torch.randn(
            1, num_local_experts, intermediate_size, hidden_size
        )
        self._switch_down_proj_weight = torch.randn(
            1, num_local_experts, hidden_size, intermediate_size
        )

    def _load_torch_weights_ours(self: Self, moe: torch.nn.Module) -> None:
        """Load pre-generated weights into CoreaiTorchSparseMoeBlock."""
        moe.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        moe.switch_mlp.gate_proj.weight = torch.nn.Parameter(self._switch_gate_proj_weight.clone())
        moe.switch_mlp.up_proj.weight = torch.nn.Parameter(self._switch_up_proj_weight.clone())
        moe.switch_mlp.down_proj.weight = torch.nn.Parameter(self._switch_down_proj_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_moe: torch.nn.Module) -> None:
        """Load pre-generated weights into HF MixtralSparseMoeBlock (fused experts)."""
        _load_hf_moe_weights(
            hf_moe,
            self._moe_gate_weight,
            self._switch_gate_proj_weight,
            self._switch_up_proj_weight,
            self._switch_down_proj_weight,
        )

    def _load_mlx_weights(self: Self, mlx_moe: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX MixtralSparseMoeBlock."""
        dtype = mlx_moe.inner.gate.weight.dtype

        mlx_moe.inner.gate.weight = mx.array(self._moe_gate_weight.numpy()).astype(dtype)
        # MLX SwitchGLU uses [num_experts, intermediate, hidden] (no leading dim 1)
        mlx_moe.inner.switch_mlp.gate_proj.weight = mx.array(
            self._switch_gate_proj_weight[0].numpy()
        ).astype(dtype)
        mlx_moe.inner.switch_mlp.up_proj.weight = mx.array(
            self._switch_up_proj_weight[0].numpy()
        ).astype(dtype)
        mlx_moe.inner.switch_mlp.down_proj.weight = mx.array(
            self._switch_down_proj_weight[0].numpy()
        ).astype(dtype)

    def _make_config(self: Self) -> MixtralConfig:
        return MixtralConfig(
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            num_local_experts=self._num_local_experts,
            num_experts_per_tok=self._top_k,
        )

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchSparseMoeBlock(
                dim=self._hidden_size,
                hidden_dim=self._intermediate_size,
                num_experts=self._num_local_experts,
                top_k=self._top_k,
            )
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            config = self._make_config()
            model = _HFMixtralSparseMoeBlockWrapper(config)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = MlxMixtralModelArgs(
                model_type="mixtral",
                hidden_size=self._hidden_size,
                intermediate_size=self._intermediate_size,
                num_local_experts=self._num_local_experts,
                num_experts_per_tok=self._top_k,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=1,
            )
            model = _MlxMixtralSparseMoeBlock(mlx_args)
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
                "x": torch.randn(
                    (self._batch_size, self._seq_len, self._hidden_size),
                    dtype=torch.float32,
                )
            }
        else:
            match source_config.source:
                case Source.torch:
                    torch_f32_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs_f32 = self.reference_inputs(torch_f32_source_config)
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
                    torch_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=source_config.precision,
                    )
                    named_inputs_torch = self.reference_inputs(torch_source_config)
                    import mlx.core

                    named_inputs = {
                        name: mlx.core.array(input_torch)
                        for name, input_torch in named_inputs_torch.items()
                    }
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMixtralLayers:
    @staticmethod
    @pytest.mark.parametrize("model_class", [MixtralAttention, MixtralTransformerBlock])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    @pytest.mark.parametrize(
        "num_attention_heads, num_key_value_heads",
        [(1, 1), (8, 1), (8, 4), (8, 8)],
    )
    def test_mixtral_layers(
        model_class: type[Model],
        precision: Precision,
        num_attention_heads: int,
        num_key_value_heads: int,
    ) -> None:
        """Verify Core AI Torch Mixtral layers match HuggingFace and MLX."""
        if num_key_value_heads > num_attention_heads:
            pytest.skip("num_key_value_heads > num_attention_heads is invalid")

        # Disable fused KV for the entire test so Core AI model uses
        # separate projections (matching HF) during both construction
        # and forward.
        original_fused_kv = mixtral_module.USE_FUSED_KV
        mixtral_module.USE_FUSED_KV = False
        try:
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

            rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]
            atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]
            with tempfile.TemporaryDirectory() as temp_directory:
                model = model_class(
                    Path(temp_directory),
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
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
                    msg = (
                        f"{_MSG_MLX_NOT_FOUND} so cannot validate coreai torch authoring vs mlx-lm"
                    )
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
        finally:
            mixtral_module.USE_FUSED_KV = original_fused_kv

    @staticmethod
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_mixtral_sparse_block(
        top_k: int,
        precision: Precision,
    ) -> None:
        """Verify Core AI Torch SparseMoeBlock matches HuggingFace and MLX."""
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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]

        with tempfile.TemporaryDirectory() as temp_directory:
            model = MixtralSparseMoeBlock(Path(temp_directory), top_k=top_k)
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


@pytest.mark.slow
class TestMixtralForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/mixtral-8xtiny-random"
    _model_class = CoreaiTorchMixtralForCausalLM
