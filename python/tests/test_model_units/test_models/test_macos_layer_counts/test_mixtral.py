# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Layer count tests for the macOS Mixtral model.

These verify that this repo's Mixtral implementation produces the expected
MLIR op counts. The ``EXPECTED_COUNTS`` dict is the parity contract --
divergence here means the implementation has drifted.
"""

import pytest
import torch
from transformers.models.mixtral.modeling_mixtral import MixtralConfig

from coreai_models.models.macos import mixtral
from coreai_models.primitives.macos.cache import KVCache
from tests._layer_count_utils import assert_layer_counts, get_layer_counts

# =============================================================================
# EXPECTED COUNTS
# =============================================================================
#
# These counts verify that critical model optimizations are applied correctly.
# Each test ensures that the PyTorch model compiles to an efficient MLIR<Core AI>
# representation with proper fusion and optimization patterns.
#
# NOTE: When composite_declaration ops are present, they indicate that patterns
# like RMSNorm, RoPE, or SDPA have been fused into single composite operations.
# This is expected and indicates proper fusion.

EXPECTED_COUNTS = {
    # RMSNorm Optimization Pattern:
    # - CRITICAL: composite_declaration indicates the RMSNorm is fused into a single operation
    # - Pattern: x * rsqrt(mean(x^2) + eps) * scale
    # - Verifies normalization is efficient without intermediate materialization
    #
    # GRAPH/INVOKE BREAKDOWN (graph=2, invoke=1):
    #   - Graph 1: @rms_norm composite - Fused RMSNorm implementation
    #   - Graph 2: @main - Entry point
    #   - Invoke 1: main calls rms_norm
    "RMSNorm": {
        "composite_declaration": 1,
        "constant": 3,
        "decomposable.broadcasting_add": 1,
        "decomposable.broadcasting_mul": 3,
        "graph": 2,
        "invoke": 1,
        "name": 5,
        "output": 2,
        "reduce_mean": 1,
        "rsqrt": 1,
    },
    # SparseMoeBlock Optimization Pattern (Mixtral-specific):
    # - Mixture of Experts routing and expert computation
    # - CRITICAL operations to verify:
    #   * Router softmax for expert selection
    #   * Top-k selection via argsort/sort operations
    #   * gather_along_axis for token routing to experts
    #   * Expert-specific batch_matmul operations (multiple experts)
    #   * reduce_sum for combining expert outputs
    # - This is a placeholder - specific counts depend on expert configuration
    "SparseMoeBlock": {},
    # Attention Optimization Pattern:
    # - CRITICAL: 4 batch_matmul operations verify Q/K/V projections + attention computation
    # - RoPE (Rotary Position Embedding) pattern: cos/sin operations + gather_along_axis
    # - Softmax should appear exactly once for attention scores normalization
    # - Verifies efficient attention without unnecessary intermediate operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=3, invoke=2):
    #   - Graph 1: @rope composite - Rotary Position Embedding
    #   - Graph 2: @scaled_dot_product_attention composite - SDPA
    #   - Graph 3: @main - Entry point
    #   - Invoke 1: main calls rope for position encoding
    #   - Invoke 2: main calls scaled_dot_product_attention
    "Attention": {
        "cast": 1,
        "composite_declaration": 2,
        "concat": 1,
        "constant": 32,
        "cos": 1,
        "decomposable.broadcasting_add": 2,
        "decomposable.broadcasting_batch_matmul": 4,
        "decomposable.broadcasting_mul": 6,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "graph": 3,
        "invoke": 2,
        "name": 10,
        "output": 3,
        "reshape": 8,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 5,
    },
    # TransformerBlock Optimization Pattern (Mixtral MoE-specific):
    # - Combines one Attention + SparseMoeBlock + two RMSNorm layers
    # - CRITICAL: MoE-specific operations: argsort, sort, gather_along_axis for expert routing
    # - 2 softmax = 1 (attention) + 1 (MoE router)
    # - reduce_sum for combining expert outputs
    # - Verifies efficient MoE routing and computation without redundant operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=6, invoke=7):
    #   - Graph 1: @rms_norm composite
    #   - Graph 2: @rope composite (from attention)
    #   - Graph 3: @scaled_dot_product_attention composite (from attention)
    #   - Graph 4: @gather_mm composite (MoE expert-1)
    #   - Graph 5: @gather_mm composite (MoE expert-2)
    #   - Graph 6: @main - Entry point
    #   - Invoke 1: main calls rms_norm (pre-attention)
    #   - Invoke 2: main calls rope (via attention)
    #   - Invoke 3: main calls scaled_dot_product_attention (via attention)
    #   - Invoke 4: main calls rms_norm (post-attention)
    #   - Invoke 5: main calls gather_mm (MoE expert-1)
    #   - Invoke 6: main calls gather_mm (MoE expert-2)
    #   - Invoke 7: main calls rms_norm (MoE routing norm)
    "TransformerBlock": {
        "argsort": 1,
        "broadcast_in_dims": 3,
        "cast": 1,
        "composite_declaration": 5,
        "concat": 1,
        "constant": 61,
        "cos": 1,
        "decomposable.broadcasting_add": 5,
        "decomposable.broadcasting_batch_matmul": 7,
        "decomposable.broadcasting_divide": 0,
        "decomposable.broadcasting_mul": 11,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 4,
        "graph": 6,
        "invoke": 7,
        "name": 21,
        "output": 6,
        "reduce_mean": 1,
        "reduce_sum": 1,
        "reshape": 16,
        "rsqrt": 1,
        "silu": 1,
        "sin": 1,
        "slice": 8,
        "softmax": 2,
        "sort": 1,
        "transpose": 6,
    },
    # ForCausalLM Optimization Pattern (Mixtral MoE-specific):
    # - Complete model: Embedding + 1 TransformerBlock (with MoE) + LM head
    # - CRITICAL: MoE operations: argsort, sort, gather_along_axis preserved in full model
    # - KV cache operations: create_token, handle, read_handle, write_handle, slice_update
    # - 2 softmax = 1 (attention) + 1 (MoE router)
    # - Verifies end-to-end MoE model with stateful KV cache for efficient autoregressive generation
    #
    # GRAPH/INVOKE BREAKDOWN (graph=6, invoke=8):
    #   - Graph 1-5: Same composites as TransformerBlock
    #   - Graph 6: @main - Entry point
    #   - Invoke 1-7: Same as TransformerBlock
    #   - Invoke 8: main calls rms_norm (final, before LM head)
    "ForCausalLM": {
        "argsort": 1,
        "broadcast_in_dims": 3,
        "cast": 1,
        "composite_declaration": 5,
        "concat": 1,
        "constant": 70,
        "cos": 1,
        "create_token": 1,
        "decomposable.broadcasting_add": 5,
        "decomposable.broadcasting_batch_matmul": 8,
        "decomposable.broadcasting_divide": 0,
        "decomposable.broadcasting_mul": 11,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 4,
        "gather_nd": 1,
        "graph": 6,
        "handle": 2,
        "invoke": 8,
        "name": 25,
        "output": 6,
        "read_handle": 4,
        "reduce_mean": 1,
        "reduce_sum": 1,
        "reshape": 21,
        "rsqrt": 1,
        "silu": 1,
        "sin": 1,
        "slice": 10,
        "slice_update": 2,
        "softmax": 2,
        "sort": 1,
        "token": 17,
        "transpose": 7,
        "write_handle": 2,
    },
}


# =============================================================================
# CONFIG FIXTURE
# =============================================================================


@pytest.fixture
def mixtral_config() -> MixtralConfig:
    """Create a small MixtralConfig for testing."""
    return MixtralConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        num_local_experts=4,
        num_experts_per_tok=2,
    )


# =============================================================================
# TESTS
# =============================================================================


class TestMixtralLayerCounts:
    """Layer count tests for Mixtral model components."""

    def test_rmsnorm_layer_counts(self) -> None:
        """RMSNorm exports to expected Core AI operations."""
        model = mixtral.RMSNorm(64, eps=1e-5)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["RMSNorm"])

    def test_attention_layer_counts(self, mixtral_config: MixtralConfig) -> None:
        """Attention exports to expected Core AI operations."""
        model = mixtral.Attention(config=mixtral_config, layer_idx=0)
        x = torch.randn(2, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0).expand(2, -1)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["Attention"])

    def test_transformer_block_layer_counts(self, mixtral_config: MixtralConfig) -> None:
        """TransformerBlock exports to expected Core AI operations."""
        model = mixtral.TransformerBlock(config=mixtral_config, layer_idx=0)
        x = torch.randn(1, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["TransformerBlock"])

    def test_for_causal_lm_layer_counts(self, mixtral_config: MixtralConfig) -> None:
        """MixtralForCausalLM exports to expected Core AI operations."""
        mixtral_config.num_hidden_layers = 1
        mixtral_config.vocab_size = 100

        model = mixtral.MixtralForCausalLM(mixtral_config, model_device="cpu")
        input_ids = torch.randint(0, mixtral_config.vocab_size, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(mixtral_config)

        result = get_layer_counts(model=model, inputs=(input_ids, position_ids, k_cache, v_cache))
        assert_layer_counts(result, EXPECTED_COUNTS["ForCausalLM"])
