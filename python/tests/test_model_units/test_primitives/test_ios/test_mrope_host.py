# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Host-side RoPE gather for multi-position (M-RoPE) models.

Covers the issue #280 workaround: the in-graph ``rope_cached_cos_sin_gather``
composite tolerates only one lookup per graph, so M-RoPE models precompute cos/sin on
the host and feed them in as graph inputs.
"""

import pytest
import torch

from coreai_models.primitives.ios.rope import RoPECache


def _interleaved_masks(head_dim: int, n_sections: int = 3) -> torch.Tensor:
    """Interleaved M-RoPE section masks (Qwen3.5 layout): freq f -> row f % n."""
    frequency = torch.arange(head_dim) % (head_dim // 2)
    return torch.stack([(frequency % n_sections == s).float() for s in range(n_sections)])


def _contiguous_masks(head_dim: int, n_sections: int = 3) -> torch.Tensor:
    """Contiguous M-RoPE section masks (Qwen2-VL ``mrope_section`` layout)."""
    band = head_dim // n_sections
    masks = torch.zeros(n_sections, head_dim)
    for s in range(n_sections):
        hi = head_dim if s == n_sections - 1 else (s + 1) * band
        masks[s, s * band : hi] = 1.0
    return masks


def test_gather_cos_sin_host_matches_composite_op():
    """Host gather must return the same values as the custom-op gather path."""
    rope = RoPECache(head_dim=32, max_cache_size=64)
    pos = torch.tensor([[3, 7, 1]], dtype=torch.int32)

    host_cos, host_sin = rope.gather_cos_sin_host(pos)
    op_cos, op_sin = rope.gather_cos_sin(pos)

    torch.testing.assert_close(host_cos, op_cos)
    torch.testing.assert_close(host_sin, op_sin)
    # And it is a plain index into the cache.
    torch.testing.assert_close(host_cos, rope.cos_cached[pos])
    torch.testing.assert_close(host_sin, rope.sin_cached[pos])


def test_gather_cos_sin_host_shape():
    rope = RoPECache(head_dim=32, max_cache_size=64)
    pos = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    cos, sin = rope.gather_cos_sin_host(pos)
    assert cos.shape == (1, 8, 32)
    assert sin.shape == (1, 8, 32)


@pytest.mark.parametrize("masks_fn", [_interleaved_masks, _contiguous_masks])
def test_mrope_identical_rows_collapse_to_1d(masks_fn):
    """Partitioning masks + identical rows == the ordinary single-row embedding."""
    head_dim = 36
    rope = RoPECache(head_dim=head_dim, max_cache_size=64)
    masks = masks_fn(head_dim)
    # Masks must partition the head dim for the collapse to hold.
    torch.testing.assert_close(masks.sum(0), torch.ones(head_dim))

    pos = torch.tensor([[0, 3, 7]], dtype=torch.int32)
    cos_m, sin_m = rope.mrope_cos_sin([pos, pos, pos], masks)
    cos_1, sin_1 = rope.gather_cos_sin_host(pos)

    torch.testing.assert_close(cos_m, cos_1)
    torch.testing.assert_close(sin_m, sin_1)


def test_mrope_selects_the_right_row_per_dim():
    """Each head-dim entry must take its value from the section its mask marks."""
    head_dim = 36
    rope = RoPECache(head_dim=head_dim, max_cache_size=64)
    masks = _interleaved_masks(head_dim)

    rows = [
        torch.tensor([[2, 5]], dtype=torch.int32),
        torch.tensor([[7, 1]], dtype=torch.int32),
        torch.tensor([[3, 9]], dtype=torch.int32),
    ]
    cos, sin = rope.mrope_cos_sin(rows, masks)
    per_row = [rope.gather_cos_sin_host(r) for r in rows]

    for d in range(head_dim):
        # Exactly one section owns this dim.
        assert masks[:, d].sum().item() == 1.0
        s = int(masks[:, d].argmax())
        torch.testing.assert_close(cos[..., d], per_row[s][0][..., d])
        torch.testing.assert_close(sin[..., d], per_row[s][1][..., d])


def test_mrope_row_count_mismatch_raises():
    rope = RoPECache(head_dim=36, max_cache_size=64)
    masks = _interleaved_masks(36)  # 3 sections
    pos = torch.tensor([[0, 1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="position rows"):
        rope.mrope_cos_sin([pos, pos], masks)
