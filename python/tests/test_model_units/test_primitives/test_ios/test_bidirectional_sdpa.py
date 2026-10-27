# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the SAM3 bidirectional SDPA primitive."""

import torch

from coreai_models.primitives.ios.bidirectional_sdpa import BidirectionalSDPA

_HEADS = 4
_HEAD_DIM = 8


def _qkv(query_seq: int, key_seq: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(1, _HEADS * _HEAD_DIM, 1, query_seq)
    k = torch.randn(1, _HEADS * _HEAD_DIM, 1, key_seq)
    v = torch.randn(1, _HEADS * _HEAD_DIM, 1, key_seq)
    return q, k, v


def _key_padding_bias(key_seq: int, num_valid: int) -> torch.Tensor:
    bias = torch.zeros(1, 1, 1, key_seq)
    bias[..., num_valid:] = -40000.0
    return bias


def test_chunked_forward_broadcasts_key_padding_mask():
    """A (B, 1, 1, K) mask must apply to every query chunk, not just the first."""
    query_seq, key_seq = 10, 6
    q, k, v = _qkv(query_seq, key_seq)
    bias = _key_padding_bias(key_seq, num_valid=2)

    chunked = BidirectionalSDPA(_HEADS, _HEAD_DIM, query_chunk_size=4)
    unchunked = BidirectionalSDPA(_HEADS, _HEAD_DIM, query_chunk_size=query_seq)
    expanded = bias.expand(1, _HEADS, query_seq, key_seq)

    out = chunked(q, k, v, attention_mask=bias)
    torch.testing.assert_close(out, unchunked(q, k, v, attention_mask=bias))
    torch.testing.assert_close(out, chunked(q, k, v, attention_mask=expanded))


def test_key_padding_mask_ignores_padded_keys():
    query_seq, key_seq, num_valid = 5, 6, 2
    q, k, v = _qkv(query_seq, key_seq)
    sdpa = BidirectionalSDPA(_HEADS, _HEAD_DIM, query_chunk_size=2)
    bias = _key_padding_bias(key_seq, num_valid)

    k_noisy, v_noisy = k.clone(), v.clone()
    k_noisy[..., num_valid:] = torch.randn_like(k_noisy[..., num_valid:])
    v_noisy[..., num_valid:] = torch.randn_like(v_noisy[..., num_valid:])

    torch.testing.assert_close(
        sdpa(q, k, v, attention_mask=bias), sdpa(q, k_noisy, v_noisy, attention_mask=bias)
    )
