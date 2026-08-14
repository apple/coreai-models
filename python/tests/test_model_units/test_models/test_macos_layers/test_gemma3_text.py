# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Gemma3 model parity with HuggingFace."""

import functools
import math
import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention as HFGemma3Attention,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3MLP as HFGemma3MLP,
)
from typing_extensions import Self, override

from coreai_models.models.macos import gemma3_text as gemma3_text_module
from coreai_models.models.macos.gemma3_text import (
    MLP as CoreaiTorchMLP,
)
from coreai_models.models.macos.gemma3_text import (
    Attention as CoreaiTorchAttention,
)
from coreai_models.models.macos.gemma3_text import (
    Embedding as CoreaiTorchEmbedding,
)
from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
from coreai_models.models.macos.gemma3_text import (
    Gemma3ForCausalLM as CoreaiTorchGemma3ForCausalLM,
)
from coreai_models.models.macos.gemma3_text import (
    TransformerBlock as CoreaiTorchTransformerBlock,
)
from coreai_models.primitives.macos.cache import KVCache
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
from tests._runner_infra.models.model import Model
from tests._runner_infra.testing_utils import ForCausalLMTestBase


def _make_gemma3_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
    head_dim: int = 16,
) -> Gemma3TextConfig:
    config = Gemma3TextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
        query_pre_attn_scalar=head_dim,
        sliding_window=16,
        # Must be passed to the constructor: `layer_types` is derived from it in
        # `__post_init__` and is what drives the attention type per layer.
        sliding_window_pattern=2,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        pad_token_id=0,
    )
    return config


class TestmacOSGemma3ForCausalLM:
    """Test macOS Gemma3ForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_gemma3_config()

        hf_model = HFGemma3ForCausalLM(config).to(torch.float32).eval()

        our_model = Gemma3ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(1, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_token(self):
        seq_len = 8
        config = _make_gemma3_config()

        hf_model = HFGemma3ForCausalLM(config).to(torch.float32).eval()

        our_model = Gemma3ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        input_ids = torch.randint(1, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        config = _make_gemma3_config()
        our_model = Gemma3ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, 100
        input_ids = torch.randint(1, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab)

    def test_mutate_state_dict_fuses_qkv_and_norms(self):
        config = _make_gemma3_config()
        our_model = Gemma3ForCausalLM(config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = 16

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.0.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.0.self_attn.q_norm.weight"] = torch.randn(head_dim)
        sd["model.layers.0.self_attn.k_norm.weight"] = torch.randn(head_dim)
        sd["model.layers.0.mlp.gate_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.up_proj.weight"] = torch.randn(128, hidden)
        sd["model.layers.0.mlp.down_proj.weight"] = torch.randn(hidden, 128)
        sd["model.layers.0.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_attention_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.pre_feedforward_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.0.post_feedforward_layernorm.weight"] = torch.randn(hidden)

        our_model._mutate_state_dict(sd)

        assert "model.layers.0.self_attn.qkv_proj.weight" in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd
        assert "model.layers.0.self_attn.qk_norm.weight" in sd
        assert "model.layers.0.self_attn.q_norm.weight" not in sd


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
# Autouse fixture: disable fused KV for HF parity tests
# ---------------------------------------------------------------------------


# Note: ``TestmacOSGemma3ForCausalLM.test_mutate_state_dict_fuses_qkv_and_norms``
# explicitly verifies the fused-KV path, so this fixture must NOT be
# module-scoped autouse — it would flip ``USE_FUSED_KV`` to False for that
# test as well. Instead, attach it as a class-scoped autouse fixture only on
# the layer-parity classes that need the non-fused HF reference path.
@pytest.fixture(scope="class")
def use_non_fused_kv():
    """Use non-fused KV for HuggingFace comparison tests."""
    original = gemma3_text_module.USE_FUSED_KV
    gemma3_text_module.USE_FUSED_KV = False
    yield
    gemma3_text_module.USE_FUSED_KV = original


# ---------------------------------------------------------------------------
# HF reference wrappers
# ---------------------------------------------------------------------------


def _build_gemma3_attention_mask(
    seq_len: int,
    layer_idx: int,
    sliding_window: int,
    sliding_window_pattern: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the causal (possibly sliding-window) attention mask."""
    if sliding_window <= seq_len and (layer_idx + 1) % sliding_window_pattern != 0:
        full_trues = torch.ones((seq_len, seq_len), dtype=torch.bool)
        causal_mask = full_trues.tril(diagonal=0)
        left_out_of_window = full_trues.tril(diagonal=-sliding_window)
        attn_mask = torch.logical_xor(causal_mask, left_out_of_window)
        full_inf = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        full_zero = torch.full((seq_len, seq_len), 0.0, dtype=dtype)
        causal_mask_float = torch.where(attn_mask, full_zero, full_inf)
    else:
        causal_mask_float = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=dtype), diagonal=1
        )
    return causal_mask_float.unsqueeze(0).unsqueeze(0)


class _HFGemma3Attention(torch.nn.Module):
    """Wrapper around HF Gemma3Attention that accepts (x, position_ids)."""

    def __init__(
        self: Self,
        config: Gemma3TextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.inner = HFGemma3Attention(config=config, layer_idx=layer_idx)
        # A single rotary embedding covers both layer types; it picks the local
        # or global theta from `config.rope_parameters[layer_type]`.
        self.rotary = Gemma3RotaryEmbedding(config)
        self._layer_type = config.layer_types[layer_idx]
        self._layer_idx = layer_idx
        self._sliding_window = config.sliding_window
        self._sliding_window_pattern = config._sliding_window_pattern

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        attention_mask = _build_gemma3_attention_mask(
            seq_len,
            self._layer_idx,
            self._sliding_window,
            self._sliding_window_pattern,
            x.dtype,
        )
        cos, sin = self.rotary(x, position_ids, layer_type=self._layer_type)
        output = self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )[0]
        return output


class _HFGemma3TransformerBlock(torch.nn.Module):
    """Wrapper around HF Gemma3DecoderLayer that accepts (x, position_ids)."""

    def __init__(
        self: Self,
        config: Gemma3TextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.inner = Gemma3DecoderLayer(config=config, layer_idx=layer_idx)
        self.rotary = Gemma3RotaryEmbedding(config)
        self._layer_type = config.layer_types[layer_idx]
        self._layer_idx = layer_idx
        self._sliding_window = config.sliding_window
        self._sliding_window_pattern = config._sliding_window_pattern
        self._seq_len: int | None = None

    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        attention_mask = _build_gemma3_attention_mask(
            seq_len,
            self._layer_idx,
            self._sliding_window,
            self._sliding_window_pattern,
            x.dtype,
        )
        cos, sin = self.rotary(x, position_ids, layer_type=self._layer_type)
        # transformers >= 5.0 takes one `position_embeddings` for the layer's own
        # attention type and returns the hidden states directly.
        return self.inner(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            cache_position=torch.arange(0, seq_len),
        )


class _HFGemma3MLP(torch.nn.Module):
    """Wrapper around HF Gemma3MLP that accepts (x,)."""

    def __init__(self: Self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.inner = HFGemma3MLP(config)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class _HFGemma3Embedding(torch.nn.Module):
    """Wrapper around HF Gemma3TextScaledWordEmbedding that accepts (input_ids,)."""

    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float,
    ) -> None:
        super().__init__()
        self.inner = Gemma3TextScaledWordEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            embed_scale=embed_scale,
        )

    def forward(self: Self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class Gemma3Attention(Model):
    _model_name = "Gemma3Attention"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        sliding_window: int = 4,
        sliding_window_pattern: int = 2,
        # Arbitrary non-default value; Gemma3TextConfig requires an int.
        query_pre_attn_scalar: int = 3,
        batch_size: int = 1,
        seq_len: int = 10,
        offset: int = 3,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._layer_idx = layer_idx
        self._sliding_window = sliding_window
        self._sliding_window_pattern = sliding_window_pattern
        self._query_pre_attn_scalar = query_pre_attn_scalar
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        # derived
        # Small fixed hidden_size for fast CI. Gemma3TextConfig validates that
        # hidden_size is a multiple of num_attention_heads, so this has to cover
        # the largest parameterized head count.
        self._hidden_size = 8

        # Pre-generate shared weights (no bias for Gemma3)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)
        self._q_norm_weight = torch.randn(head_dim)
        self._k_norm_weight = torch.randn(head_dim)

    def _load_torch_weights_ours(self: Self, attn: torch.nn.Module) -> None:
        """Load pre-generated weights into our fused-qkv Attention."""
        attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())

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
        hf_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        hf_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())

    def _make_config(self: Self) -> Gemma3TextConfig:
        config = Gemma3TextConfig(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=6,
            rms_norm_eps=9.87,
            query_pre_attn_scalar=self._query_pre_attn_scalar,
            hidden_activation="gelu_pytorch_tanh",
            sliding_window=self._sliding_window,
            sliding_window_pattern=self._sliding_window_pattern,
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
            model = _HFGemma3Attention(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
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
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class Gemma3TransformerBlock(Model):
    _model_name = "Gemma3TransformerBlock"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 16,
        intermediate_size: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        sliding_window: int = 4,
        sliding_window_pattern: int = 2,
        # Arbitrary non-default value; Gemma3TextConfig requires an int.
        query_pre_attn_scalar: int = 3,
        batch_size: int = 1,
        seq_len: int = 10,
        offset: int = 3,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._intermediate_size = intermediate_size
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._layer_idx = layer_idx
        self._sliding_window = sliding_window
        self._sliding_window_pattern = sliding_window_pattern
        self._query_pre_attn_scalar = query_pre_attn_scalar
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        # derived
        # Small fixed hidden_size for fast CI. Gemma3TextConfig validates that
        # hidden_size is a multiple of num_attention_heads, so this has to cover
        # the largest parameterized head count.
        self._hidden_size = 8

        # Pre-generate shared attention weights (no bias for Gemma3)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)
        self._q_norm_weight = torch.randn(head_dim)
        self._k_norm_weight = torch.randn(head_dim)

        # Pre-generate shared MLP weights
        self._gate_weight = torch.randn(intermediate_size, self._hidden_size)
        self._up_weight = torch.randn(intermediate_size, self._hidden_size)
        self._down_weight = torch.randn(self._hidden_size, intermediate_size)

        # Pre-generate shared layernorm weights (4 layer norms for Gemma3)
        self._input_ln_weight = torch.randn(self._hidden_size)
        self._post_attn_ln_weight = torch.randn(self._hidden_size)
        self._pre_ff_ln_weight = torch.randn(self._hidden_size)
        self._post_ff_ln_weight = torch.randn(self._hidden_size)

    def _load_torch_weights_ours(self: Self, block: torch.nn.Module) -> None:
        """Load pre-generated weights into our TransformerBlock."""
        # Attention weights (fused qkv, no bias)
        block.self_attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        block.self_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        block.self_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        block.self_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())
        # MLP weights
        block.mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        block.mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        block.mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())
        # Layernorm weights (4 layer norms)
        block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )
        block.pre_feedforward_layernorm.weight = torch.nn.Parameter(self._pre_ff_ln_weight.clone())
        block.post_feedforward_layernorm.weight = torch.nn.Parameter(
            self._post_ff_ln_weight.clone()
        )

    def _load_torch_weights_hf(self: Self, hf_block: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Gemma3DecoderLayer."""
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
        hf_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        hf_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())

        # MLP weights
        hf_block.mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        hf_block.mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        hf_block.mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())

        # Layernorm weights (4 layer norms)
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )
        hf_block.pre_feedforward_layernorm.weight = torch.nn.Parameter(
            self._pre_ff_ln_weight.clone()
        )
        hf_block.post_feedforward_layernorm.weight = torch.nn.Parameter(
            self._post_ff_ln_weight.clone()
        )

    def _make_config(self: Self) -> Gemma3TextConfig:
        config = Gemma3TextConfig(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=self._intermediate_size,
            rms_norm_eps=9.87,
            query_pre_attn_scalar=self._query_pre_attn_scalar,
            hidden_activation="gelu_pytorch_tanh",
            sliding_window=self._sliding_window,
            sliding_window_pattern=self._sliding_window_pattern,
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
            model = _HFGemma3TransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
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
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class Gemma3MLP(Model):
    _model_name = "Gemma3MLP"

    def __init__(
        self: Self,
        root_path: Path,
        hidden_size: int = 4,
        intermediate_size: int = 6,
        batch_size: int = 2,
        seq_len: int = 2,
    ) -> None:
        super().__init__(root_path=root_path)
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._batch_size = batch_size
        self._seq_len = seq_len

        # Pre-generate shared MLP weights
        self._gate_weight = torch.randn(intermediate_size, hidden_size)
        self._up_weight = torch.randn(intermediate_size, hidden_size)
        self._down_weight = torch.randn(hidden_size, intermediate_size)

    def _load_torch_weights_ours(self: Self, mlp: torch.nn.Module) -> None:
        """Load pre-generated weights into our MLP."""
        mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_mlp: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Gemma3MLP."""
        hf_mlp.gate_proj.weight = torch.nn.Parameter(self._gate_weight.clone())
        hf_mlp.up_proj.weight = torch.nn.Parameter(self._up_weight.clone())
        hf_mlp.down_proj.weight = torch.nn.Parameter(self._down_weight.clone())

    def _make_config(self: Self) -> Gemma3TextConfig:
        return Gemma3TextConfig(
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            hidden_activation="gelu_pytorch_tanh",
            # Attention is unused here, but the config validates that
            # hidden_size is a multiple of num_attention_heads.
            num_attention_heads=1,
            num_key_value_heads=1,
        )

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        config = self._make_config()
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchMLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFGemma3MLP(config)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
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
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


class Gemma3Embedding(Model):
    _model_name = "Gemma3Embedding"

    def __init__(
        self: Self,
        root_path: Path,
        num_embeddings: int = 3,
        embedding_dim: int = 4,
        padding_idx: int = 0,
        batch_size: int = 2,
        seq_len: int = 10,
    ) -> None:
        super().__init__(root_path=root_path)
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx
        self._batch_size = batch_size
        self._seq_len = seq_len

        self._embed_scale = 1.0 / math.sqrt(embedding_dim)

        # Pre-generate shared embedding weights
        self._embed_weight = torch.randn(num_embeddings, embedding_dim)

    def _load_torch_weights_ours(self: Self, embed: torch.nn.Module) -> None:
        """Load pre-generated weights into our Embedding."""
        embed.weight = torch.nn.Parameter(self._embed_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_embed: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Gemma3TextScaledWordEmbedding."""
        hf_embed.weight = torch.nn.Parameter(self._embed_weight.clone())

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchEmbedding(
                num_embeddings=self._num_embeddings,
                embedding_dim=self._embedding_dim,
                padding_idx=self._padding_idx,
                embed_scale=self._embed_scale,
            )
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            model = _HFGemma3Embedding(
                num_embeddings=self._num_embeddings,
                embedding_dim=self._embedding_dim,
                padding_idx=self._padding_idx,
                embed_scale=self._embed_scale,
            )
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
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
                "input_ids": torch.randint(
                    0,
                    self._num_embeddings,
                    (self._batch_size, self._seq_len),
                    dtype=torch.int32,
                )
            }
        else:
            match source_config.source:
                case Source.torch:
                    # Embedding inputs are integer — no dtype conversion needed
                    torch_f32_source_config = SourceConfig(
                        source=cast("Source", Source.torch),
                        precision=cast("Precision", Precision.f32),
                    )
                    named_inputs = dict(self.reference_inputs(torch_f32_source_config))
                case _:
                    msg = f"Source {source_config.source} has no reference inputs"
                    raise NotImplementedError(msg)
        return named_inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGemma3Layers:
    @pytest.fixture(autouse=True)
    def _use_non_fused_kv(self, use_non_fused_kv):
        """Activate non-fused KV path for HF parity tests in this class."""
        yield

    @staticmethod
    @pytest.mark.parametrize("model_class", [Gemma3Attention, Gemma3TransformerBlock])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    @pytest.mark.parametrize(
        "num_attention_heads, num_key_value_heads",
        [(1, 1), (8, 4), (8, 8)],
    )
    @pytest.mark.parametrize("sliding_window", [4, 10000])
    def test_gemma3_layers(
        model_class: type[Model],
        precision: Precision,
        num_attention_heads: int,
        num_key_value_heads: int,
        sliding_window: int,
    ) -> None:
        """Verify Core AI Torch Gemma3 attention/transformer layers match HuggingFace."""
        if num_key_value_heads > num_attention_heads:
            pytest.skip("num_key_value_heads > num_attention_heads is invalid")

        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = model_class(
                Path(temp_directory),
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                sliding_window=sliding_window,
            )
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
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
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_gemma3_mlp(precision: Precision) -> None:
        """Verify Core AI Torch Gemma3 MLP matches HuggingFace."""
        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = Gemma3MLP(Path(temp_directory))
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
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
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_gemma3_embedding(precision: Precision) -> None:
        """Verify Core AI Torch Gemma3 Embedding matches HuggingFace."""
        oss_torch_config = RunConfig(
            author=cast("Author", Author.oss),
            source=cast("Source", Source.torch),
            precision=precision,
            backend=cast("Backend", Backend.torch_eager),
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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 1e-1}[precision]
        with tempfile.TemporaryDirectory() as temp_directory:
            model = Gemma3Embedding(Path(temp_directory))
            model.validate(
                coreai_torch_eager_config,
                oss_torch_config,
                rtol=rtol,
                atol=atol,
            )
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


class TestGemma3TextForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/gemma-3-tiny-random"
    _model_class = CoreaiTorchGemma3ForCausalLM
    _test_weights_tying = True
    _test_weight_activation_quantization = True
