# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Qwen2 model parity with HuggingFace."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM
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
    from mlx_lm.models.qwen2 import Attention as MlxQwen2Attention
    from mlx_lm.models.qwen2 import ModelArgs as MlxQwen2ModelArgs
    from mlx_lm.models.qwen2 import TransformerBlock as MlxQwen2TransformerBlock
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention as HFQwen2Attention,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RotaryEmbedding,
)

from coreai_models.models.macos.qwen2 import (
    Attention as CoreaiTorchAttention,
)
from coreai_models.models.macos.qwen2 import (
    Qwen2ForCausalLM as CoreaiTorchQwen2ForCausalLM,
)
from coreai_models.models.macos.qwen2 import (
    TransformerBlock as CoreaiTorchTransformerBlock,
)
from tests._runner_infra.models.model import Model
from tests._runner_infra.testing_utils import ForCausalLMTestBase


def _make_qwen2_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
) -> Qwen2Config:
    config = Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    config.rope_scaling = None
    config.rope_theta = 10000.0
    return config


def _make_hf_qwen2_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
) -> Qwen2Config:
    return Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
    )


class TestmacOSQwen2ForCausalLM:
    """Test macOS Qwen2ForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        """Single-token decode: our macOS model matches HF logits."""
        hf_config = _make_hf_qwen2_config()
        our_config = _make_qwen2_config()

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()

        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_token(self):
        """Multi-token prefill: our macOS model matches HF logits."""
        seq_len = 8
        hf_config = _make_hf_qwen2_config()
        our_config = _make_qwen2_config()

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()

        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_two_layers(self):
        """Two-layer model: verify parity scales with depth."""
        hf_config = _make_hf_qwen2_config(num_hidden_layers=2)
        our_config = _make_qwen2_config(num_hidden_layers=2)

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float32).eval()

        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_float16(self):
        """Verify parity in float16 precision."""
        hf_config = _make_hf_qwen2_config()
        our_config = _make_qwen2_config()

        hf_model = HFQwen2ForCausalLM(hf_config).to(torch.float16).eval()

        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float16).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(0, 100, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-3, rtol=5e-3)

    def test_output_shape(self):
        """Output shape is (batch, seq_len, vocab_size)."""
        our_config = _make_qwen2_config()
        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab)

    def test_mutate_state_dict_fuses_qkv(self):
        """_mutate_state_dict fuses q/k/v projections and biases into qkv_proj."""
        our_config = _make_qwen2_config()
        our_model = Qwen2ForCausalLM(our_config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = hidden // n_heads

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.q_proj.bias"] = torch.randn(n_heads * head_dim)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.bias"] = torch.randn(n_kv_heads * head_dim)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.bias"] = torch.randn(n_kv_heads * head_dim)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.mlp.gate_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.up_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.down_proj.weight"] = torch.randn(hidden, 128)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)

        our_model._mutate_state_dict(sd)

        # q/k/v should be fused into qkv_proj
        assert "model.layers.0.self_attn.qkv_proj.weight" in sd
        assert "model.layers.0.self_attn.qkv_proj.bias" in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd
        assert "model.layers.0.self_attn.k_proj.weight" not in sd
        assert "model.layers.0.self_attn.v_proj.weight" not in sd

        # fused weight shape: (n_heads*hd + 2*n_kv_heads*hd, hidden)
        expected_rows = n_heads * head_dim + 2 * n_kv_heads * head_dim
        assert sd["model.layers.0.self_attn.qkv_proj.weight"].shape == (expected_rows, hidden)
        assert sd["model.layers.0.self_attn.qkv_proj.bias"].shape == (expected_rows,)


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


class _HFQwen2Attention(torch.nn.Module):
    """Wrapper around HF Qwen2Attention that accepts (x, position_ids)."""

    def __init__(self: Self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.inner = HFQwen2Attention(config=config, layer_idx=layer_idx)
        self.rotary = Qwen2RotaryEmbedding(config)

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


class _HFQwen2TransformerBlock(torch.nn.Module):
    """Wrapper around HF Qwen2DecoderLayer that accepts (x, position_ids)."""

    def __init__(self: Self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.inner = Qwen2DecoderLayer(config=config, layer_idx=layer_idx)
        self.rotary = Qwen2RotaryEmbedding(config)

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


# ---------------------------------------------------------------------------
# MLX wrappers
# ---------------------------------------------------------------------------

if _HAS_MLX:

    class _MlxQwen2Attention(mlx_nn.Module):
        """Wraps mlx_lm Qwen2 Attention to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxQwen2ModelArgs") -> None:
            super().__init__()
            self.inner = MlxQwen2Attention(args)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)

    class _MlxQwen2TransformerBlock(mlx_nn.Module):
        """Wraps mlx_lm Qwen2 TransformerBlock to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxQwen2ModelArgs") -> None:
            super().__init__()
            self.inner = MlxQwen2TransformerBlock(args)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class Qwen2Attention(Model):
    _model_name = "Qwen2Attention"

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

        # Pre-generate shared weights
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._qkv_proj_bias = torch.randn(qkv_total_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

    def _load_torch_weights_ours(self: Self, attn: torch.nn.Module) -> None:
        """Load pre-generated weights into our fused-qkv Attention."""
        attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        attn.qkv_proj.bias = torch.nn.Parameter(self._qkv_proj_bias.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_attn: torch.nn.Module) -> None:
        """Load pre-generated weights into HF's separate q/k/v projections."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        hf_attn.q_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[:q_size].clone())
        hf_attn.q_proj.bias = torch.nn.Parameter(self._qkv_proj_bias[:q_size].clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(
            self._qkv_proj_weight[q_size : q_size + k_size].clone()
        )
        hf_attn.k_proj.bias = torch.nn.Parameter(
            self._qkv_proj_bias[q_size : q_size + k_size].clone()
        )
        hf_attn.v_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[q_size + k_size :].clone())
        hf_attn.v_proj.bias = torch.nn.Parameter(self._qkv_proj_bias[q_size + k_size :].clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

    def _load_mlx_weights(self: Self, mlx_attn: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Qwen2 Attention."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        dtype = mlx_attn.inner.q_proj.weight.dtype

        mlx_attn.inner.q_proj.weight = mx.array(self._qkv_proj_weight[:q_size].numpy()).astype(
            dtype
        )
        mlx_attn.inner.q_proj.bias = mx.array(self._qkv_proj_bias[:q_size].numpy()).astype(dtype)
        mlx_attn.inner.k_proj.weight = mx.array(
            self._qkv_proj_weight[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        mlx_attn.inner.k_proj.bias = mx.array(
            self._qkv_proj_bias[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        mlx_attn.inner.v_proj.weight = mx.array(
            self._qkv_proj_weight[q_size + k_size :].numpy()
        ).astype(dtype)
        mlx_attn.inner.v_proj.bias = mx.array(
            self._qkv_proj_bias[q_size + k_size :].numpy()
        ).astype(dtype)
        mlx_attn.inner.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)

    def _make_config(self: Self) -> Qwen2Config:
        config = Qwen2Config(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=6,
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
            model = _HFQwen2Attention(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = MlxQwen2ModelArgs(
                model_type="qwen2",
                hidden_size=self._hidden_size,
                num_hidden_layers=1,
                intermediate_size=6,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                rms_norm_eps=9.87,
                vocab_size=1,
                rope_theta=1e5,
            )
            model = _MlxQwen2Attention(mlx_args)
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


class Qwen2TransformerBlock(Model):
    _model_name = "Qwen2TransformerBlock"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 2,
        intermediate_size: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        batch_size: int = 1,
        seq_len: int = 10,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._intermediate_size = intermediate_size
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._layer_idx = layer_idx
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        # additional config args derived from user args
        self._hidden_size = num_attention_heads * head_dim

        # Pre-generate shared attention weights
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._qkv_proj_bias = torch.randn(qkv_total_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

        # Pre-generate shared MLP weights
        self._gate_weight = torch.randn(intermediate_size, self._hidden_size)
        self._up_weight = torch.randn(intermediate_size, self._hidden_size)
        self._down_weight = torch.randn(self._hidden_size, intermediate_size)

        # Pre-generate shared layernorm weights
        self._input_ln_weight = torch.randn(self._hidden_size)
        self._post_attn_ln_weight = torch.randn(self._hidden_size)

    def _load_torch_weights_ours(self: Self, block: torch.nn.Module) -> None:
        """Load pre-generated weights into our TransformerBlock."""
        # Attention weights (fused qkv)
        block.self_attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        block.self_attn.qkv_proj.bias = torch.nn.Parameter(self._qkv_proj_bias.clone())
        block.self_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        # MLP weights
        block.mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        block.mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        block.mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())
        # Layernorm weights
        block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_torch_weights_hf(self: Self, hf_block: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Qwen2DecoderLayer."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        # Attention weights (separate q/k/v)
        hf_attn = hf_block.self_attn
        hf_attn.q_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[:q_size].clone())
        hf_attn.q_proj.bias = torch.nn.Parameter(self._qkv_proj_bias[:q_size].clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(
            self._qkv_proj_weight[q_size : q_size + k_size].clone()
        )
        hf_attn.k_proj.bias = torch.nn.Parameter(
            self._qkv_proj_bias[q_size : q_size + k_size].clone()
        )
        hf_attn.v_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[q_size + k_size :].clone())
        hf_attn.v_proj.bias = torch.nn.Parameter(self._qkv_proj_bias[q_size + k_size :].clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())

        # MLP weights
        hf_block.mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        hf_block.mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        hf_block.mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())

        # Layernorm weights
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_mlx_weights(self: Self, mlx_block: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Qwen2 TransformerBlock."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        inner = mlx_block.inner
        dtype = inner.self_attn.q_proj.weight.dtype

        # Attention weights
        inner.self_attn.q_proj.weight = mx.array(self._qkv_proj_weight[:q_size].numpy()).astype(
            dtype
        )
        inner.self_attn.q_proj.bias = mx.array(self._qkv_proj_bias[:q_size].numpy()).astype(dtype)
        inner.self_attn.k_proj.weight = mx.array(
            self._qkv_proj_weight[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        inner.self_attn.k_proj.bias = mx.array(
            self._qkv_proj_bias[q_size : q_size + k_size].numpy()
        ).astype(dtype)
        inner.self_attn.v_proj.weight = mx.array(
            self._qkv_proj_weight[q_size + k_size :].numpy()
        ).astype(dtype)
        inner.self_attn.v_proj.bias = mx.array(
            self._qkv_proj_bias[q_size + k_size :].numpy()
        ).astype(dtype)
        inner.self_attn.o_proj.weight = mx.array(self._o_proj_weight.numpy()).astype(dtype)

        # MLP weights
        inner.mlp.gate_proj.weight = mx.array(self._gate_weight.numpy()).astype(dtype)
        inner.mlp.up_proj.weight = mx.array(self._up_weight.numpy()).astype(dtype)
        inner.mlp.down_proj.weight = mx.array(self._down_weight.numpy()).astype(dtype)

        # Layernorm weights
        inner.input_layernorm.weight = mx.array(self._input_ln_weight.numpy()).astype(dtype)
        inner.post_attention_layernorm.weight = mx.array(self._post_attn_ln_weight.numpy()).astype(
            dtype
        )

    def _make_config(self: Self) -> Qwen2Config:
        config = Qwen2Config(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=self._intermediate_size,
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
            model = _HFQwen2TransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = MlxQwen2ModelArgs(
                model_type="qwen2",
                hidden_size=self._hidden_size,
                num_hidden_layers=1,
                intermediate_size=self._intermediate_size,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                rms_norm_eps=9.87,
                vocab_size=1,
                rope_theta=1e5,
            )
            model = _MlxQwen2TransformerBlock(mlx_args)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen2Layers:
    @staticmethod
    @pytest.mark.parametrize("model_class", [Qwen2Attention, Qwen2TransformerBlock])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    @pytest.mark.parametrize(
        "num_attention_heads, num_key_value_heads",
        [(1, 1), (8, 1), (8, 4), (8, 8)],
    )
    def test_qwen2_layers(
        model_class: type[Model],
        precision: Precision,
        num_attention_heads: int,
        num_key_value_heads: int,
    ) -> None:
        """Verify Core AI Torch Qwen2 layers match HuggingFace and MLX."""
        if num_key_value_heads > num_attention_heads:
            pytest.skip("num_key_value_heads > num_attention_heads is invalid")

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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 5e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 5e-1}[precision]
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


class TestQwen2ForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/qwen2.5-tiny-random"
    _model_class = CoreaiTorchQwen2ForCausalLM
    _test_weights_tying = True
    _test_weight_activation_quantization = True
    # opt out of eager mode quantization for toy qwen2 model
    _test_eager_activation_quantization = False
