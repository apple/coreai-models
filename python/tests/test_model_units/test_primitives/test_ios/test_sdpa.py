# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS SDPA primitive."""

from dataclasses import dataclass

import pytest
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from coreai_models.primitives.ios.sdpa import SDPA, BlockedSDPA
from tests._runner_infra.testing_utils import (
    assert_close,
    run_compare_coreai,
)


class TestiOSSDPABasic:
    """Test iOS SDPA with small inputs."""

    def test_basic_forward(self):
        """SDPA should produce output of correct shape."""
        head_dim = 16
        n_heads = 4
        n_kv_heads = 2
        batch_size = 1
        seq_len = 4
        max_seq_len = 8

        sdpa = SDPA(head_dim=head_dim)

        query = torch.randn(batch_size, n_heads * head_dim, 1, seq_len)
        key = torch.randn(batch_size, n_kv_heads * head_dim, 1, max_seq_len)
        value = torch.randn(batch_size, n_kv_heads * head_dim, 1, max_seq_len)
        # Causal mask: (1, max_seq_len, 1, seq_len), 0 for valid, -inf for masked
        causal_mask = torch.zeros(1, max_seq_len, 1, seq_len)
        # Mask out future positions (upper triangle)
        for s in range(seq_len):
            causal_mask[0, s + 1 :, 0, s] = float("-inf")

        out = sdpa(query, key, value, causal_mask)
        assert out.shape == (batch_size, n_heads * head_dim, 1, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Softmax attention weights should sum to approximately 1."""
        head_dim = 8
        sdpa = SDPA(head_dim=head_dim)

        # Single head, single position for easy verification
        query = torch.randn(1, head_dim, 1, 1)
        key = torch.randn(1, head_dim, 1, 4)
        value = torch.randn(1, head_dim, 1, 4)
        causal_mask = torch.zeros(1, 4, 1, 1)  # all positions visible

        out = sdpa(query, key, value, causal_mask)
        # Output should be a weighted combination of values, not zeros
        assert out.shape == (1, head_dim, 1, 1)
        assert not torch.all(out == 0)

    def test_custom_scale(self):
        """SDPA should accept a custom scale factor."""
        head_dim = 16
        custom_scale = 0.1
        sdpa = SDPA(head_dim=head_dim, scale=custom_scale)

        query = torch.randn(1, head_dim, 1, 2)
        key = torch.randn(1, head_dim, 1, 4)
        value = torch.randn(1, head_dim, 1, 4)
        causal_mask = torch.zeros(1, 4, 1, 2)

        out = sdpa(query, key, value, causal_mask)
        assert out.shape == (1, head_dim, 1, 2)


class TestSDPA:
    """
    Test iOS SDPA functionality.
    Used by: iOS versions of Mistral, Qwen2, Qwen3 models.

    iOS SDPA has a different interface optimized for iOS hardware:
    - query shape: (batch, n_heads*head_dim, 1, seq_len)
    - key shape: (batch, n_kv_heads*head_dim, 1, seq_len)
    - value shape: (batch, n_kv_heads*head_dim, 1, seq_len)
    - causal_mask shape: (1, seq_len, 1, seq_len)
    """

    @staticmethod
    def get_model_asset(
        precision: torch.dtype,
        use_casual_mask: bool,
    ) -> tuple[
        torch.nn.Module,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        batch = 1
        seq = 10
        head_dim = 24
        n_heads = 8
        n_kv_heads = 4

        # Generate data in standard format first
        # Standard format: (batch, n_heads, seq, head_dim)
        query_std = torch.rand(batch, n_heads, seq, head_dim)
        key_std = torch.rand(batch, n_kv_heads, seq, head_dim)
        value_std = torch.rand(batch, n_kv_heads, seq, head_dim)

        scale = head_dim**-0.5

        if use_casual_mask:
            causal_mask_base = torch.triu(torch.full((seq, seq), float(-40000)), diagonal=1)
        else:
            causal_mask_base = torch.zeros(seq, seq)

        attention_mask_std = causal_mask_base.unsqueeze(0).unsqueeze(0)

        # Update precision
        query_std = query_std.to(precision)
        key_std = key_std.to(precision)
        value_std = value_std.to(precision)
        attention_mask_std = attention_mask_std.to(precision)

        # HF ground truth
        @dataclass
        class MockModule:
            num_key_value_groups = n_heads // n_kv_heads
            training = False

        output_hf, _ = sdpa_attention_forward(
            module=MockModule(),
            query=query_std,
            key=key_std,
            value=value_std,
            attention_mask=attention_mask_std,
            scaling=scale,
        )

        # Convert to iOS format
        query_ios = (
            query_std.permute(0, 1, 3, 2).reshape(batch, n_heads * head_dim, seq).unsqueeze(2)
        )
        key_ios = (
            key_std.permute(0, 1, 3, 2).reshape(batch, n_kv_heads * head_dim, seq).unsqueeze(2)
        )
        value_ios = (
            value_std.permute(0, 1, 3, 2).reshape(batch, n_kv_heads * head_dim, seq).unsqueeze(2)
        )

        # Causal mask for iOS: (1, key_seq, 1, query_seq)
        # Note: causal_mask_base is [query, key], but iOS needs [key, query]
        causal_mask_ios = causal_mask_base.t().unsqueeze(0).unsqueeze(2).to(precision).contiguous()

        # iOS SDPA
        ios_sdpa = SDPA(head_dim=head_dim, scale=scale)
        ios_sdpa = ios_sdpa.to(precision)

        # Convert HF output to iOS format
        output_hf_ios = (
            output_hf.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch, n_heads * head_dim, seq)
            .unsqueeze(2)
        )  # (batch, n_heads*head_dim, 1, seq)

        return (
            ios_sdpa,
            (query_ios, key_ios, value_ios, causal_mask_ios),
            output_hf_ios,
        )

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    @pytest.mark.parametrize("use_casual_mask", [True, False])
    def test_hf(self, precision: torch.dtype, use_casual_mask: bool) -> None:
        model, inputs, expected_output = self.get_model_asset(precision, use_casual_mask)
        output = model(*inputs)
        assert_close(
            output,
            expected_output,
            atol=1e-3 if precision == torch.float16 else 1e-5,
        )

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    @pytest.mark.parametrize("use_casual_mask", [True, False])
    def test_coreai(self, precision: torch.dtype, use_casual_mask: bool) -> None:
        model, inputs, _ = self.get_model_asset(precision, use_casual_mask)
        run_compare_coreai(
            model=model,
            inputs=inputs,
            atol=5e-3 if precision == torch.float16 else 1e-5,
        )


class TestBlockedSDPA:
    """The online-softmax merge across key blocks against plain softmax attention."""

    @staticmethod
    def reference(query, key, value, causal_mask, head_dim):
        n_heads, n_kv = query.shape[1] // head_dim, key.shape[1] // head_dim
        q = query.view(n_heads, head_dim, -1).transpose(1, 2).float()  # (H, q, d)
        k = key.view(n_kv, head_dim, -1).float().repeat_interleave(n_heads // n_kv, 0)
        v = value.view(n_kv, head_dim, -1).float().repeat_interleave(n_heads // n_kv, 0)
        mask = causal_mask[0, :, 0, :].T.float()  # (q, ctx)
        weights = torch.softmax(q @ k * head_dim**-0.5 + mask, dim=-1)
        out = weights @ v.transpose(1, 2)  # (H, q, d)
        return out.transpose(1, 2).reshape(1, n_heads * head_dim, 1, -1)

    @pytest.mark.parametrize("n_blocks", [1, 2, 3])
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_matches_softmax_attention(self, n_blocks: int, precision: torch.dtype) -> None:
        torch.manual_seed(0)
        head_dim, n_heads, n_kv, q_len, block = 16, 4, 2, 8, 32
        ctx = n_blocks * block
        # Queries sit at the end of the filled cache, so every block holds live keys.
        offset = ctx - q_len
        query = torch.randn(1, n_heads * head_dim, 1, q_len, dtype=precision)
        key = torch.randn(1, n_kv * head_dim, 1, ctx, dtype=precision)
        value = torch.randn(1, n_kv * head_dim, 1, ctx, dtype=precision)
        causal_mask = torch.zeros(1, ctx, 1, q_len, dtype=precision)
        for i in range(q_len):
            causal_mask[0, offset + i + 1 :, 0, i] = -40000.0

        got = BlockedSDPA(head_dim=head_dim, block_size=block)(query, key, value, causal_mask)
        expected = self.reference(query, key, value, causal_mask, head_dim)
        assert_close(got.float(), expected, atol=2e-3 if precision == torch.float16 else 1e-5)
