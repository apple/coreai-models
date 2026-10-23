# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for iOS RoPE."""

import pytest
import torch
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb as mistral_apply_rotary_pos_emb,
)
from typing_extensions import Self

from coreai_models.primitives.ios.rope import RoPECache as NERoPE
from coreai_models.primitives.ios.rope import RoPECache as iOSRoPE
from coreai_models.primitives.ios.rope import apply_rope
from tests._runner_infra.testing_utils import (
    assert_close,
    run_compare_coreai,
)


class TestNERoPE:
    """Test iOS RoPE primitive."""

    def test_basic_forward(self):
        """iOS RoPE should produce output of the same shape."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        batch_size, n_heads, seq_len = 1, 4, 8
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = rope.gather_cos_sin(position_ids)

        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_different_positions_differ(self):
        """Different position_ids should produce different outputs."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        x = torch.randn(1, 2, 1, head_dim)
        cos0, sin0 = rope.gather_cos_sin(torch.tensor([[0]]))
        cos5, sin5 = rope.gather_cos_sin(torch.tensor([[5]]))

        out0 = apply_rope(x, cos0, sin0)
        out5 = apply_rope(x, cos5, sin5)

        assert not torch.allclose(out0, out5), "Different positions should differ"

    def test_cos_sin_cache_shape(self):
        """cos/sin caches should have shape (max_cache_size, head_dim)."""
        head_dim = 16
        max_cache_size = 32
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        assert rope.cos_cached.shape == (max_cache_size, head_dim)
        assert rope.sin_cached.shape == (max_cache_size, head_dim)

    def test_rope_equivariance(self):
        """RoPE(x, pos=p) should equal RoPE(x, pos=p) -- deterministic."""
        head_dim = 32
        rope = NERoPE(head_dim=head_dim, max_cache_size=64)

        x = torch.randn(1, 2, 4, head_dim)
        pos = torch.arange(4).unsqueeze(0)
        cos, sin = rope.gather_cos_sin(pos)
        out1 = apply_rope(x, cos, sin)
        out2 = apply_rope(x, cos, sin)
        torch.testing.assert_close(out1, out2)

    def test_gather_cos_sin_shape(self):
        """gather_cos_sin should return two tensors with shape (batch, seq_len, head_dim)."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        batch_size, seq_len = 1, 8
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

        rope_cos, rope_sin = rope.gather_cos_sin(position_ids)

        assert rope_cos.shape == (batch_size, seq_len, head_dim)
        assert rope_sin.shape == (batch_size, seq_len, head_dim)

    def test_gather_cos_sin_values_match_cache(self):
        """gather_cos_sin should return values sliced from the cos/sin cache."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        position_ids = torch.tensor([[3, 7, 1]], dtype=torch.int32)
        rope_cos, rope_sin = rope.gather_cos_sin(position_ids)

        # Manually index the cache for comparison
        expected_cos = rope.cos_cached[position_ids]
        expected_sin = rope.sin_cached[position_ids]

        torch.testing.assert_close(rope_cos, expected_cos)
        torch.testing.assert_close(rope_sin, expected_sin)

    def test_apply_rope_shape(self):
        """apply_rope should return a tensor of the same shape as input."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        batch_size, n_heads, seq_len = 1, 4, 8
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        rope_cos, rope_sin = rope.gather_cos_sin(position_ids)

        out = apply_rope(x, rope_cos, rope_sin)
        assert out.shape == x.shape

    def test_apply_rope_matches_forward(self):
        """apply_rope(x, gather_cos_sin(pos)) should match rope.forward(x, pos)."""
        head_dim = 32
        max_cache_size = 64
        rope = NERoPE(head_dim=head_dim, max_cache_size=max_cache_size)

        batch_size, n_heads, seq_len = 1, 4, 6
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        cos, sin = rope.gather_cos_sin(position_ids)

        # Reference: forward does gather+apply internally
        expected = apply_rope(x, cos, sin)

        # New API: gather then apply_rope separately
        rope_cos, rope_sin = rope.gather_cos_sin(position_ids)
        result = apply_rope(x, rope_cos, rope_sin)

        torch.testing.assert_close(result, expected)


# Base test class to avoid duplication
class BaseRoPETest:
    def _run_test_hf(self, precision: torch.dtype, **kwargs) -> None:
        """Helper method to run HuggingFace tests."""
        # Get custom atol if provided by subclass
        atol = getattr(self, "_get_hf_atol", lambda p: 1e-5)(precision)

        model, inputs, expected_outputs = self.get_model_asset(precision=precision, **kwargs)
        query, key, position_ids = inputs
        for x, x_hf in zip((query, key), expected_outputs, strict=True):
            x = model(x, position_ids)
            assert_close(x, x_hf, atol=atol)

    def _run_test_coreai(self, precision: torch.dtype, **kwargs) -> None:
        """Helper method to run Core AI tests."""
        model, inputs, _ = self.get_model_asset(precision=precision, **kwargs)
        query, key, position_ids = inputs
        for x in (query, key):
            run_compare_coreai(
                model=model,
                inputs=(x, position_ids),
                atol=5e-3 if precision == torch.float16 else 1e-4,
            )


# iOS RoPE wrapper classes
class iOSRoPEWrapper(iOSRoPE):
    def forward(self: Self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self.gather_cos_sin(position_ids)
        return apply_rope(x, cos, sin)


@pytest.mark.parametrize(
    "config_class,rotary_embedding_class,apply_rotary_fn",
    [
        (MistralConfig, MistralRotaryEmbedding, mistral_apply_rotary_pos_emb),
    ],
)
@pytest.mark.parametrize("head_is_different", [True, False])
class TestiOSStandardRope(BaseRoPETest):
    @staticmethod
    def get_model_asset(
        config_class,
        rotary_embedding_class,
        apply_rotary_fn,
        head_is_different: bool = False,
        precision: torch.dtype = torch.float32,
    ) -> tuple[
        torch.nn.Module,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # important to set the batch to 2 not 1, this caught a bug in the iOS rope.
        batch = 2
        seq_len = 10
        head_dim = 8
        hidden_size = 16
        n_heads = 2
        n_kv_heads = 4
        theta = 314159
        max_position_embeddings = 123456

        # head_dim is set to hidden_size // n_heads.
        # however, we need to make sure that if the numbers are different,
        # the rope implementation should respect the head_dim
        head_dim = head_dim if head_is_different else hidden_size // n_heads
        config = config_class(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            head_dim=head_dim,
            rope_theta=theta,
            max_position_embeddings=max_position_embeddings,
        )

        # inputs
        offset = 3
        x = torch.rand(batch, seq_len, 123)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch, -1
        )
        query = torch.rand(batch, n_heads, seq_len, head_dim)
        key = torch.rand(batch, n_kv_heads, seq_len, head_dim)

        # iOS rope
        hf_rotary = rotary_embedding_class(config)
        rope = iOSRoPEWrapper(
            head_dim=head_dim,
            max_cache_size=max_position_embeddings,
            base=theta,
        )

        # update all precision
        x = x.to(precision)
        hf_rotary = hf_rotary.to(precision)
        query = query.to(precision)
        key = key.to(precision)
        rope = rope.to(precision)

        # HF ground truth
        hf_cos, hf_sin = hf_rotary(x, position_ids)
        query_hf, key_hf = apply_rotary_fn(query, key, hf_cos, hf_sin)

        return rope, (query, key, position_ids), (query_hf, key_hf)

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
    def test_hf(
        self,
        config_class,
        rotary_embedding_class,
        apply_rotary_fn,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> None:
        """Test against HuggingFace implementation."""
        self._run_test_hf(
            precision=precision,
            config_class=config_class,
            rotary_embedding_class=rotary_embedding_class,
            apply_rotary_fn=apply_rotary_fn,
            head_is_different=head_is_different,
        )

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_coreai(
        self,
        config_class,
        rotary_embedding_class,
        apply_rotary_fn,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> None:
        """Test Core AI compilation and execution."""
        self._run_test_coreai(
            precision=precision,
            config_class=config_class,
            rotary_embedding_class=rotary_embedding_class,
            apply_rotary_fn=apply_rotary_fn,
            head_is_different=head_is_different,
        )


def test_apply_rope_rejects_ndim_not_dividing_last_dim():
    x = torch.randn(1, 2, 3, 6)
    cos = torch.randn(1, 3, 6)
    sin = torch.randn(1, 3, 6)
    with pytest.raises((RuntimeError, ValueError)):
        apply_rope(x, cos, sin, ndim=4)
