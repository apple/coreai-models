# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Layer count tests for the macOS Gemma3 model.

These verify that this repo's Gemma3 implementation produces the expected
MLIR op counts. The ``EXPECTED_COUNTS`` dict is the parity contract --
divergence here means the implementation has drifted.
"""

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from coreai_models.models.macos import gemma3_text
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
    # MLP Optimization Pattern (Gemma3-specific):
    # - CRITICAL: 3 batch_matmul operations (one for each w1, w2, w3 linear layer)
    # - Pattern: w2(gelu(w1(x)) * w3(x)) - GeGLU activation (Gemma uses GELU instead of SiLU)
    # - Verifies linear layers are properly lowered without extra operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=1, invoke=0):
    #   - Graph 1: @main - Entry point (all operations inline, no composites)
    "MLP": {
        "constant": 4,
        "decomposable.broadcasting_batch_matmul": 3,
        "decomposable.broadcasting_mul": 1,
        "gelu": 1,
        "graph": 1,
        "name": 2,
        "output": 1,
        "transpose": 3,
    },
    # Attention Optimization Pattern (Gemma3-specific):
    # - CRITICAL: 4 batch_matmul operations verify Q/K/V projections + attention computation
    # - RoPE (Rotary Position Embedding) pattern: cos/sin operations + gather_along_axis
    # - Softmax should appear exactly once for attention scores normalization
    # - EXTRA RMSNorm ops (rsqrt, reduce_mean) indicate query pre-attention normalization
    # - Verifies efficient attention without unnecessary intermediate operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=4, invoke=3):
    #   - Graph 1: @rope composite - Rotary Position Embedding
    #   - Graph 2: @rms_norm composite - Q-norm (query normalization before attention)
    #   - Graph 3: @scaled_dot_product_attention composite - SDPA
    #   - Graph 4: @main - Entry point
    #   - Invoke 1: main calls rope
    #   - Invoke 2: main calls rms_norm (Q-norm)
    #   - Invoke 3: main calls scaled_dot_product_attention
    "Attention": {
        "cast": 1,
        "composite_declaration": 3,
        "concat": 1,
        "constant": 35,
        "cos": 1,
        "decomposable.broadcasting_add": 3,
        "decomposable.broadcasting_batch_matmul": 4,
        "decomposable.broadcasting_mul": 9,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "graph": 4,
        "invoke": 3,
        "name": 13,
        "output": 4,
        "reduce_mean": 1,
        "reshape": 8,
        "rsqrt": 1,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 5,
    },
    # TransformerBlock Optimization Pattern (Gemma3-specific):
    # - Combines one Attention + one MLP + multiple RMSNorm layers (input, post-attention,
    #   post-feedforward, pre-feedforward, and query norm)
    # - CRITICAL: 7 batch_matmul = 4 (attention) + 3 (MLP)
    # - Verifies the full transformer block composition without redundant operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=5, invoke=7):
    #   - Graph 1: @rms_norm composite
    #   - Graph 2: @rope composite (from attention)
    #   - Graph 3: @scaled_dot_product_attention composite (from attention)
    #   - Graph 4: @rms_norm composite (Q-norm, from attention)
    #   - Graph 5: @main - Entry point
    #   - Invoke 1-2: RMSNorm calls (pre-attention, post-attention)
    #   - Invoke 3: rope (via attention)
    #   - Invoke 4: rms_norm Q-norm (via attention)
    #   - Invoke 5: scaled_dot_product_attention (via attention)
    #   - Invoke 6-7: RMSNorm calls (pre-feedforward, post-feedforward)
    "TransformerBlock": {
        "cast": 1,
        "composite_declaration": 4,
        "concat": 1,
        "constant": 38,
        "cos": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 7,
        "decomposable.broadcasting_mul": 13,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "gelu": 1,
        "graph": 5,
        "invoke": 7,
        "name": 16,
        "output": 5,
        "reduce_mean": 2,
        "reshape": 5,
        "rsqrt": 2,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 8,
    },
    # ForCausalLM Optimization Pattern (Gemma3-specific):
    # - Complete model: Embedding + 1 TransformerBlock + LM head
    # - CRITICAL: 8 batch_matmul = 7 (TransformerBlock) + 1 (LM head projection)
    # - KV cache operations: create_token, handle, read_handle, write_handle, slice_update
    # - Verifies end-to-end model with stateful KV cache for efficient autoregressive generation
    #
    # GRAPH/INVOKE BREAKDOWN (graph=5, invoke=8):
    #   - Graph 1-4: Same composites as TransformerBlock
    #   - Graph 5: @main - Entry point
    #   - Invoke 1-7: Same as TransformerBlock
    #   - Invoke 8: main calls rms_norm (final, before LM head)
    "ForCausalLM": {
        "cast": 1,
        "composite_declaration": 4,
        "concat": 1,
        "constant": 47,
        "cos": 1,
        "create_token": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 8,
        "decomposable.broadcasting_mul": 14,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "gather_nd": 1,
        "gelu": 1,
        "graph": 5,
        "handle": 2,
        "invoke": 8,
        "name": 20,
        "output": 5,
        "read_handle": 4,
        "reduce_mean": 2,
        "reshape": 10,
        "rsqrt": 2,
        "sin": 1,
        "slice": 8,
        "slice_update": 2,
        "softmax": 1,
        "token": 17,
        "transpose": 9,
        "write_handle": 2,
    },
}


# =============================================================================
# CONFIG FIXTURE
# =============================================================================


@pytest.fixture
def gemma3_config() -> Gemma3TextConfig:
    """Create a small Gemma3TextConfig for testing."""
    return Gemma3TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        sliding_window=128,
        query_pre_attn_scalar=1,
    )


# =============================================================================
# TESTS
# =============================================================================


class TestGemma3LayerCounts:
    """Layer count tests for Gemma3 model components."""

    def test_rmsnorm_layer_counts(self) -> None:
        """RMSNorm exports to expected Core AI operations."""
        # The Gemma3-specific RMSNorm is RMSNormPlusOne, re-exported as
        # ``Gemma3RMSNorm`` inside ``gemma3_text``.
        model = gemma3_text.Gemma3RMSNorm(64, eps=1e-5)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["RMSNorm"])

    def test_mlp_layer_counts(self) -> None:
        """MLP exports to expected Core AI operations."""
        model = gemma3_text.MLP(dim=64, hidden_dim=128)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["MLP"])

    def test_attention_layer_counts(self, gemma3_config: Gemma3TextConfig) -> None:
        """Attention exports to expected Core AI operations."""
        model = gemma3_text.Attention(config=gemma3_config, layer_idx=0)
        x = torch.randn(2, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0).expand(2, -1)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["Attention"])

    def test_transformer_block_layer_counts(self, gemma3_config: Gemma3TextConfig) -> None:
        """TransformerBlock exports to expected Core AI operations."""
        model = gemma3_text.TransformerBlock(config=gemma3_config, layer_idx=0)
        x = torch.randn(1, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["TransformerBlock"])

    def test_for_causal_lm_layer_counts(self, gemma3_config: Gemma3TextConfig) -> None:
        """Gemma3ForCausalLM exports to expected Core AI operations."""
        gemma3_config.num_hidden_layers = 1
        gemma3_config.vocab_size = 100

        model = gemma3_text.Gemma3ForCausalLM(gemma3_config, model_device="cpu")
        input_ids = torch.randint(0, gemma3_config.vocab_size, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(gemma3_config)

        result = get_layer_counts(model=model, inputs=(input_ids, position_ids, k_cache, v_cache))
        assert_layer_counts(result, EXPECTED_COUNTS["ForCausalLM"])
