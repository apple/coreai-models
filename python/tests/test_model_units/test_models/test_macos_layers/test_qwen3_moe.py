# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Qwen3 MoE model parity with HuggingFace."""

import functools
import tempfile
import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.macos.qwen3_moe import Qwen3MoeForCausalLM
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
    from mlx_lm.models.qwen3_moe import Attention as MlxQwen3MoeAttention
    from mlx_lm.models.qwen3_moe import ModelArgs as MlxQwen3MoeModelArgs
    from mlx_lm.models.qwen3_moe import (
        Qwen3MoeDecoderLayer as MlxQwen3MoeDecoderLayer,
    )
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention as HFQwen3MoeAttention,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRotaryEmbedding,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)

from coreai_models.models.macos import qwen3_moe as qwen3_moe_module
from coreai_models.models.macos.qwen3_moe import (
    Attention as CoreaiTorchAttention,
)
from coreai_models.models.macos.qwen3_moe import (
    Qwen3MoeForCausalLM as CoreaiTorchQwen3MoeForCausalLM,
)
from coreai_models.models.macos.qwen3_moe import (
    SparseMoeBlock as CoreaiTorchSparseMoeBlock,
)
from coreai_models.models.macos.qwen3_moe import (
    TransformerBlock as CoreaiTorchTransformerBlock,
)
from tests._runner_infra.models.model import Model
from tests._runner_infra.testing_utils import ForCausalLMTestBase


def _make_qwen3_moe_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 2,
    intermediate_size: int = 128,
    moe_intermediate_size: int = 64,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    decoder_sparse_step: int = 2,
    norm_topk_prob: bool = True,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
    head_dim: int = 16,
) -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        decoder_sparse_step=decoder_sparse_step,
        norm_topk_prob=norm_topk_prob,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
    )
    config.rope_scaling = None
    config.rope_theta = 10000.0
    return config


def _make_hf_qwen3_moe_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 2,
    intermediate_size: int = 128,
    moe_intermediate_size: int = 64,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    decoder_sparse_step: int = 2,
    norm_topk_prob: bool = True,
    vocab_size: int = 100,
    max_position_embeddings: int = 32,
    head_dim: int = 16,
) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        decoder_sparse_step=decoder_sparse_step,
        norm_topk_prob=norm_topk_prob,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
    )


class TestmacOSQwen3MoeForCausalLM:
    """Test macOS Qwen3MoeForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        """Single-token decode: our macOS model matches HF logits."""
        hf_config = _make_hf_qwen3_moe_config()
        our_config = _make_qwen3_moe_config()

        hf_model = HFQwen3MoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")
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
        hf_config = _make_hf_qwen3_moe_config()
        our_config = _make_qwen3_moe_config()

        hf_model = HFQwen3MoeForCausalLM(hf_config).to(torch.float32).eval()

        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")
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

    def test_forward_parity_float16(self):
        """Verify parity in float16 precision."""
        hf_config = _make_hf_qwen3_moe_config()
        our_config = _make_qwen3_moe_config()

        hf_model = HFQwen3MoeForCausalLM(hf_config).to(torch.float16).eval()

        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")
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
        our_config = _make_qwen3_moe_config()
        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, 100
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(our_config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, vocab)

    def test_mutate_state_dict_fuses_qkv_and_norms(self):
        """_mutate_state_dict fuses q/k/v and q_norm/k_norm."""
        our_config = _make_qwen3_moe_config(num_hidden_layers=1)
        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")

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

        our_model._mutate_state_dict(sd)

        # q/k/v should be fused into qkv_proj
        assert "model.layers.0.self_attn.qkv_proj.weight" in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd

        # q_norm and k_norm should be fused into qk_norm
        assert "model.layers.0.self_attn.qk_norm.weight" in sd
        assert "model.layers.0.self_attn.q_norm.weight" not in sd
        assert "model.layers.0.self_attn.k_norm.weight" not in sd

    def test_mutate_state_dict_stacks_moe_experts(self):
        """_mutate_state_dict stacks per-expert weights into SwitchGLU layout."""
        our_config = _make_qwen3_moe_config(num_hidden_layers=2)
        our_model = Qwen3MoeForCausalLM(our_config, model_device="cpu")

        hidden = 64
        n_heads = 4
        n_kv_heads = 2
        head_dim = 16
        num_experts = 4
        moe_intermediate = 64

        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(100, hidden)
        sd["model.norm.weight"] = torch.randn(hidden)
        sd["lm_head.weight"] = torch.randn(100, hidden)

        # Layer 0: dense MLP (layer_idx=0, (0+1)%2 != 0)
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

        # Layer 1: MoE (layer_idx=1, (1+1)%2 == 0)
        sd["model.layers.1.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd["model.layers.1.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.1.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden)
        sd["model.layers.1.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd["model.layers.1.self_attn.q_norm.weight"] = torch.randn(head_dim)
        sd["model.layers.1.self_attn.k_norm.weight"] = torch.randn(head_dim)
        sd["model.layers.1.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        for e in range(num_experts):
            sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                moe_intermediate, hidden
            )
            sd[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                moe_intermediate, hidden
            )
            sd[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                hidden, moe_intermediate
            )
        sd["model.layers.1.input_layernorm.weight"] = torch.randn(hidden)
        sd["model.layers.1.post_attention_layernorm.weight"] = torch.randn(hidden)

        our_model._mutate_state_dict(sd)

        # MoE experts should be stacked into switch_mlp
        assert "model.layers.1.mlp.switch_mlp.gate_proj.weight" in sd
        assert sd["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (
            1,
            num_experts,
            moe_intermediate,
            hidden,
        )
        # Individual expert keys should be gone
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" not in sd


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


class _HFQwen3MoeAttention(torch.nn.Module):
    """Wrapper around HF Qwen3MoeAttention that accepts (x, position_ids)."""

    def __init__(self: Self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.inner = HFQwen3MoeAttention(config=config, layer_idx=layer_idx)
        self.rotary = Qwen3MoeRotaryEmbedding(config)

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


class _HFQwen3MoeTransformerBlock(torch.nn.Module):
    """Wrapper around HF Qwen3MoeDecoderLayer that accepts (x, position_ids)."""

    def __init__(self: Self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.inner = Qwen3MoeDecoderLayer(config=config, layer_idx=layer_idx)
        self.rotary = Qwen3MoeRotaryEmbedding(config)

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


class _HFQwen3MoeSparseMoeBlockWrapper(torch.nn.Module):
    """Wrapper that returns just hidden_states (drops router_logits)."""

    def __init__(self: Self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.inner = HFQwen3MoeSparseMoeBlock(config)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)[0]


# ---------------------------------------------------------------------------
# MLX wrappers
# ---------------------------------------------------------------------------

if _HAS_MLX:

    class _MlxQwen3MoeAttention(mlx_nn.Module):
        """Wraps mlx_lm Qwen3Moe Attention to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxQwen3MoeModelArgs") -> None:
            super().__init__()
            self.inner = MlxQwen3MoeAttention(args, layer_idx=0)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)

    class _MlxQwen3MoeTransformerBlock(mlx_nn.Module):
        """Wraps mlx_lm Qwen3Moe DecoderLayer to accept (x, position_ids)."""

        def __init__(self: Self, args: "MlxQwen3MoeModelArgs", layer_idx: int) -> None:
            super().__init__()
            self.inner = MlxQwen3MoeDecoderLayer(args, layer_idx=layer_idx)

        def __call__(self: Self, x: "mx.array", position_ids: "mx.array") -> "mx.array":
            seq_len = x.shape[1]
            mask: str | None = "causal" if seq_len > 1 else None
            return self.inner(x, mask=mask, cache=None)


# ---------------------------------------------------------------------------
# Common MoE config defaults
# ---------------------------------------------------------------------------

_DEFAULT_NUM_EXPERTS = 4
_DEFAULT_MOE_INTERMEDIATE_SIZE = 8
_DEFAULT_DECODER_SPARSE_STEP = 2


def _make_layers_qwen3_moe_config(
    hidden_size: int,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int = 6,
    num_experts: int = _DEFAULT_NUM_EXPERTS,
    moe_intermediate_size: int = _DEFAULT_MOE_INTERMEDIATE_SIZE,
    decoder_sparse_step: int = _DEFAULT_DECODER_SPARSE_STEP,
    num_experts_per_tok: int = 2,
    norm_topk_prob: bool = True,
    rms_norm_eps: float = 9.87,
    rope_theta: float = 1e5,
) -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
        decoder_sparse_step=decoder_sparse_step,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
    )
    config._attn_implementation = "sdpa"
    return config


def _make_mlx_qwen3_moe_args(
    hidden_size: int,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int = 6,
    num_experts: int = _DEFAULT_NUM_EXPERTS,
    moe_intermediate_size: int = _DEFAULT_MOE_INTERMEDIATE_SIZE,
    decoder_sparse_step: int = _DEFAULT_DECODER_SPARSE_STEP,
    num_experts_per_tok: int = 2,
    norm_topk_prob: bool = True,
    rms_norm_eps: float = 9.87,
    rope_theta: float = 1e5,
) -> "MlxQwen3MoeModelArgs":
    return MlxQwen3MoeModelArgs(
        model_type="qwen3_moe",
        hidden_size=hidden_size,
        num_hidden_layers=2,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=[],
        moe_intermediate_size=moe_intermediate_size,
        rms_norm_eps=rms_norm_eps,
        vocab_size=1,
        rope_theta=rope_theta,
        head_dim=head_dim,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        norm_topk_prob=norm_topk_prob,
    )


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class Qwen3MoeAttention(Model):
    _model_name = "Qwen3MoeAttention"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        batch_size: int = 1,
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

        self._hidden_size = 4  # Match old base_qwen3_moe_config

        # Pre-generate shared weights (no bias for Qwen3Moe)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

        # Pre-generate Q/K norm weights (head_dim sized)
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

    def _load_mlx_weights(self: Self, mlx_attn: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Qwen3Moe Attention."""
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
        mlx_attn.inner.q_norm.weight = mx.array(self._q_norm_weight.numpy()).astype(dtype)
        mlx_attn.inner.k_norm.weight = mx.array(self._k_norm_weight.numpy()).astype(dtype)

    def _make_config(self: Self) -> Qwen3MoeConfig:
        return _make_layers_qwen3_moe_config(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
        )

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
            model = _HFQwen3MoeAttention(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_qwen3_moe_args(
                hidden_size=self._hidden_size,
                head_dim=self._head_dim,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
            )
            model = _MlxQwen3MoeAttention(mlx_args)
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


class Qwen3MoeTransformerBlock(Model):
    """Dense (non-MoE) decoder layer, layer_idx=0 so MLP is regular."""

    _model_name = "Qwen3MoeTransformerBlock"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 16,
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

        self._hidden_size = 4  # Match old base_qwen3_moe_config

        # Pre-generate shared attention weights (no bias for Qwen3Moe)
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

        # Pre-generate Q/K norm weights (head_dim sized)
        self._q_norm_weight = torch.randn(head_dim)
        self._k_norm_weight = torch.randn(head_dim)

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
        block.self_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        block.self_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        block.self_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())
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
        """Load pre-generated weights into HF Qwen3MoeDecoderLayer."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        # Attention weights (separate q/k/v)
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

        # Layernorm weights
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_mlx_weights(self: Self, mlx_block: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Qwen3Moe TransformerBlock."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        inner = mlx_block.inner
        dtype = inner.self_attn.q_proj.weight.dtype

        # Attention weights
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
        inner.self_attn.q_norm.weight = mx.array(self._q_norm_weight.numpy()).astype(dtype)
        inner.self_attn.k_norm.weight = mx.array(self._k_norm_weight.numpy()).astype(dtype)

        # MLP weights
        inner.mlp.gate_proj.weight = mx.array(self._gate_weight.numpy()).astype(dtype)
        inner.mlp.up_proj.weight = mx.array(self._up_weight.numpy()).astype(dtype)
        inner.mlp.down_proj.weight = mx.array(self._down_weight.numpy()).astype(dtype)

        # Layernorm weights
        inner.input_layernorm.weight = mx.array(self._input_ln_weight.numpy()).astype(dtype)
        inner.post_attention_layernorm.weight = mx.array(self._post_attn_ln_weight.numpy()).astype(
            dtype
        )

    def _make_config(self: Self) -> Qwen3MoeConfig:
        return _make_layers_qwen3_moe_config(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            intermediate_size=self._intermediate_size,
        )

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
            model = _HFQwen3MoeTransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_qwen3_moe_args(
                hidden_size=self._hidden_size,
                head_dim=self._head_dim,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                intermediate_size=self._intermediate_size,
            )
            model = _MlxQwen3MoeTransformerBlock(mlx_args, layer_idx=self._layer_idx)
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


class Qwen3MoeTransformerBlockMoE(Model):
    """MoE decoder layer, layer_idx=1 so (1+1) % decoder_sparse_step == 0."""

    _model_name = "Qwen3MoeTransformerBlockMoE"

    def __init__(
        self: Self,
        root_path: Path,
        head_dim: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        num_experts: int = _DEFAULT_NUM_EXPERTS,
        moe_intermediate_size: int = _DEFAULT_MOE_INTERMEDIATE_SIZE,
        num_experts_per_tok: int = 2,
        layer_idx: int = 1,
        batch_size: int = 1,
        seq_len: int = 10,
        offset: int = 0,
    ) -> None:
        super().__init__(root_path=root_path)
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads
        self._num_experts = num_experts
        self._moe_intermediate_size = moe_intermediate_size
        self._num_experts_per_tok = num_experts_per_tok
        self._layer_idx = layer_idx
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._offset = offset

        self._hidden_size = 4  # Match old base_qwen3_moe_config

        # Pre-generate shared attention weights
        qkv_total_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self._qkv_proj_weight = torch.randn(qkv_total_size, self._hidden_size)
        self._o_proj_weight = torch.randn(self._hidden_size, num_attention_heads * head_dim)

        # Pre-generate Q/K norm weights
        self._q_norm_weight = torch.randn(head_dim)
        self._k_norm_weight = torch.randn(head_dim)

        # Pre-generate MoE gate weight
        self._moe_gate_weight = torch.randn(num_experts, self._hidden_size)

        # Pre-generate SwitchGLU weights (optimized layout)
        self._switch_gate_proj_weight = torch.randn(
            1, num_experts, moe_intermediate_size, self._hidden_size
        )
        self._switch_up_proj_weight = torch.randn(
            1, num_experts, moe_intermediate_size, self._hidden_size
        )
        self._switch_down_proj_weight = torch.randn(
            1, num_experts, self._hidden_size, moe_intermediate_size
        )

        # Pre-generate shared layernorm weights
        self._input_ln_weight = torch.randn(self._hidden_size)
        self._post_attn_ln_weight = torch.randn(self._hidden_size)

    def _load_torch_weights_ours(self: Self, block: torch.nn.Module) -> None:
        """Load pre-generated weights into our TransformerBlock (MoE layer)."""
        # Attention weights (fused qkv)
        block.self_attn.qkv_proj.weight = torch.nn.Parameter(self._qkv_proj_weight.clone())
        block.self_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        block.self_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        block.self_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())
        # MoE weights
        block.mlp.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        block.mlp.switch_mlp.gate_proj.weight = torch.nn.Parameter(
            self._switch_gate_proj_weight.clone()
        )
        block.mlp.switch_mlp.up_proj.weight = torch.nn.Parameter(
            self._switch_up_proj_weight.clone()
        )
        block.mlp.switch_mlp.down_proj.weight = torch.nn.Parameter(
            self._switch_down_proj_weight.clone()
        )
        # Layernorm weights
        block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_torch_weights_hf(self: Self, hf_block: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Qwen3MoeDecoderLayer (MoE layer)."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim

        # Attention weights (separate q/k/v)
        hf_attn = hf_block.self_attn
        hf_attn.q_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[:q_size].clone())
        hf_attn.k_proj.weight = torch.nn.Parameter(
            self._qkv_proj_weight[q_size : q_size + k_size].clone()
        )
        hf_attn.v_proj.weight = torch.nn.Parameter(self._qkv_proj_weight[q_size + k_size :].clone())
        hf_attn.o_proj.weight = torch.nn.Parameter(self._o_proj_weight.clone())
        hf_attn.q_norm.weight = torch.nn.Parameter(self._q_norm_weight.clone())
        hf_attn.k_norm.weight = torch.nn.Parameter(self._k_norm_weight.clone())

        # MoE weights (per-expert layout)
        hf_block.mlp.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        for i in range(self._num_experts):
            hf_block.mlp.experts[i].gate_proj.weight = torch.nn.Parameter(
                self._switch_gate_proj_weight[0, i].clone()
            )
            hf_block.mlp.experts[i].up_proj.weight = torch.nn.Parameter(
                self._switch_up_proj_weight[0, i].clone()
            )
            hf_block.mlp.experts[i].down_proj.weight = torch.nn.Parameter(
                self._switch_down_proj_weight[0, i].clone()
            )

        # Layernorm weights
        hf_block.input_layernorm.weight = torch.nn.Parameter(self._input_ln_weight.clone())
        hf_block.post_attention_layernorm.weight = torch.nn.Parameter(
            self._post_attn_ln_weight.clone()
        )

    def _load_mlx_weights(self: Self, mlx_block: "mlx_nn.Module") -> None:
        """Load pre-generated weights into MLX Qwen3Moe TransformerBlock (MoE)."""
        q_size = self._num_attention_heads * self._head_dim
        k_size = self._num_key_value_heads * self._head_dim
        inner = mlx_block.inner
        dtype = inner.self_attn.q_proj.weight.dtype

        # Attention weights
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
        inner.self_attn.q_norm.weight = mx.array(self._q_norm_weight.numpy()).astype(dtype)
        inner.self_attn.k_norm.weight = mx.array(self._k_norm_weight.numpy()).astype(dtype)

        # MoE weights
        inner.mlp.gate.weight = mx.array(self._moe_gate_weight.numpy()).astype(dtype)
        # MLX SwitchGLU uses stacked expert weights
        inner.mlp.switch_mlp.gate_proj.weight = mx.array(
            self._switch_gate_proj_weight[0].numpy()
        ).astype(dtype)
        inner.mlp.switch_mlp.up_proj.weight = mx.array(
            self._switch_up_proj_weight[0].numpy()
        ).astype(dtype)
        inner.mlp.switch_mlp.down_proj.weight = mx.array(
            self._switch_down_proj_weight[0].numpy()
        ).astype(dtype)

        # Layernorm weights
        inner.input_layernorm.weight = mx.array(self._input_ln_weight.numpy()).astype(dtype)
        inner.post_attention_layernorm.weight = mx.array(self._post_attn_ln_weight.numpy()).astype(
            dtype
        )

    def _make_config(self: Self) -> Qwen3MoeConfig:
        return _make_layers_qwen3_moe_config(
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            num_attention_heads=self._num_attention_heads,
            num_key_value_heads=self._num_key_value_heads,
            num_experts=self._num_experts,
            moe_intermediate_size=self._moe_intermediate_size,
            num_experts_per_tok=self._num_experts_per_tok,
        )

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
            model = _HFQwen3MoeTransformerBlock(config=config, layer_idx=self._layer_idx)
            self._load_torch_weights_hf(model.inner)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.mlx:
            mlx_args = _make_mlx_qwen3_moe_args(
                hidden_size=self._hidden_size,
                head_dim=self._head_dim,
                num_attention_heads=self._num_attention_heads,
                num_key_value_heads=self._num_key_value_heads,
                num_experts=self._num_experts,
                moe_intermediate_size=self._moe_intermediate_size,
                num_experts_per_tok=self._num_experts_per_tok,
            )
            model = _MlxQwen3MoeTransformerBlock(mlx_args, layer_idx=self._layer_idx)
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


class Qwen3MoeSparseMoeBlock(Model):
    """Standalone SparseMoeBlock (no attention, no layernorm)."""

    _model_name = "Qwen3MoeSparseMoeBlock"

    def __init__(
        self: Self,
        root_path: Path,
        hidden_size: int = 4,
        moe_intermediate_size: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        norm_topk_prob: bool = True,
        batch_size: int = 2,
        seq_len: int = 10,
    ) -> None:
        super().__init__(root_path=root_path)
        self._hidden_size = hidden_size
        self._moe_intermediate_size = moe_intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._norm_topk_prob = norm_topk_prob
        self._batch_size = batch_size
        self._seq_len = seq_len

        # Pre-generate MoE gate weight
        self._moe_gate_weight = torch.randn(num_experts, hidden_size)

        # Pre-generate SwitchGLU weights (optimized layout)
        self._switch_gate_proj_weight = torch.randn(
            1, num_experts, moe_intermediate_size, hidden_size
        )
        self._switch_up_proj_weight = torch.randn(
            1, num_experts, moe_intermediate_size, hidden_size
        )
        self._switch_down_proj_weight = torch.randn(
            1, num_experts, hidden_size, moe_intermediate_size
        )

    def _load_torch_weights_ours(self: Self, moe: torch.nn.Module) -> None:
        """Load pre-generated weights into CoreaiTorchSparseMoeBlock."""
        moe.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        moe.switch_mlp.gate_proj.weight = torch.nn.Parameter(self._switch_gate_proj_weight.clone())
        moe.switch_mlp.up_proj.weight = torch.nn.Parameter(self._switch_up_proj_weight.clone())
        moe.switch_mlp.down_proj.weight = torch.nn.Parameter(self._switch_down_proj_weight.clone())

    def _load_torch_weights_hf(self: Self, hf_moe: torch.nn.Module) -> None:
        """Load pre-generated weights into HF Qwen3MoeSparseMoeBlock (per-expert)."""
        hf_moe.gate.weight = torch.nn.Parameter(self._moe_gate_weight.clone())
        for i in range(self._num_experts):
            hf_moe.experts[i].gate_proj.weight = torch.nn.Parameter(
                self._switch_gate_proj_weight[0, i].clone()
            )
            hf_moe.experts[i].up_proj.weight = torch.nn.Parameter(
                self._switch_up_proj_weight[0, i].clone()
            )
            hf_moe.experts[i].down_proj.weight = torch.nn.Parameter(
                self._switch_down_proj_weight[0, i].clone()
            )

    def _make_config(self: Self) -> Qwen3MoeConfig:
        return Qwen3MoeConfig(
            hidden_size=self._hidden_size,
            intermediate_size=self._moe_intermediate_size,
            num_experts=self._num_experts,
            num_experts_per_tok=self._top_k,
            norm_topk_prob=self._norm_topk_prob,
        )

    @override
    @functools.cache  # noqa: B019
    def source_model(self: Self, source_config: SourceConfig = SourceConfig()) -> SourceModel:  # noqa: B008
        dtype = PRECISION_IN_SOURCE[source_config.source][source_config.precision]
        if source_config.author == Author.coreai and source_config.source == Source.torch:
            model = CoreaiTorchSparseMoeBlock(
                dim=self._hidden_size,
                hidden_dim=self._moe_intermediate_size,
                num_experts=self._num_experts,
                top_k=self._top_k,
                norm_topk_prob=self._norm_topk_prob,
            )
            self._load_torch_weights_ours(model)
            model.to(dtype)
        elif source_config.author == Author.oss and source_config.source == Source.torch:
            config = self._make_config()
            model = _HFQwen3MoeSparseMoeBlockWrapper(config)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen3MoeLayers:
    @staticmethod
    @pytest.mark.parametrize(
        "model_class",
        [Qwen3MoeAttention, Qwen3MoeTransformerBlock, Qwen3MoeTransformerBlockMoE],
    )
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    @pytest.mark.parametrize(
        "num_attention_heads, num_key_value_heads",
        [(1, 1), (8, 1), (8, 4), (8, 8)],
    )
    def test_qwen3_moe_layers(
        model_class: type[Model],
        precision: Precision,
        num_attention_heads: int,
        num_key_value_heads: int,
    ) -> None:
        """Verify Core AI Torch Qwen3Moe layers match HuggingFace and MLX."""
        if num_key_value_heads > num_attention_heads:
            pytest.skip("num_key_value_heads > num_attention_heads is invalid")

        # Disable fused KV for the entire test so Core AI model uses
        # separate q_norm/k_norm (matching HF) during both construction
        # and forward.
        original_fused_kv = qwen3_moe_module.USE_FUSED_KV
        qwen3_moe_module.USE_FUSED_KV = False
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
            qwen3_moe_module.USE_FUSED_KV = original_fused_kv

    @staticmethod
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("precision", [Precision.f32, Precision.f16, Precision.bf16])
    def test_qwen3_moe_sparse_block(
        top_k: int,
        precision: Precision,
    ) -> None:
        """Verify Core AI Torch SparseMoeBlock matches HuggingFace."""
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

        rtol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]
        atol = {Precision.f32: 1e-5, Precision.f16: 5e-2, Precision.bf16: 2e-1}[precision]

        with tempfile.TemporaryDirectory() as temp_directory:
            model = Qwen3MoeSparseMoeBlock(Path(temp_directory), top_k=top_k)
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


@pytest.mark.slow
class TestQwen3MoeForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/qwen3-moe-tiny-random"
    _model_class = CoreaiTorchQwen3MoeForCausalLM
