# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for StaticVisionEncoder grid_t / num_frames handling."""

import pytest
import torch
import torch.nn as nn

from coreai_models.vlm.export import StaticVisionEncoder


class _FakeVisualModel(nn.Module):
    """Minimal stand-in for the HF visual backbone."""

    def __init__(self, hidden: int = 32, patch_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, hidden, bias=False)
        self.blocks = nn.ModuleList()
        self.merger = nn.Linear(hidden, hidden, bias=False)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 32)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 8)


_COMMON = dict(image_size=32, patch_size=16, spatial_merge_size=1, temporal_patch_size=2)


class TestStaticVisionEncoderGridT:
    def test_single_image_default(self):
        enc = StaticVisionEncoder(_FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON)
        assert enc.grid_t == 1
        assert enc.num_frames == 1

    def test_single_image_explicit(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        assert enc.grid_t == 1

    def test_multi_frame_divisible(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        assert enc.grid_t == 2
        assert enc.num_patches == 2 * 2 * 2  # grid_t * grid_h * grid_w

    def test_multi_frame_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            StaticVisionEncoder(
                _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=3
            )

    def test_patchify_single_image_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        pixels = torch.randn(1, 3, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)

    def test_patchify_multi_frame_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        pixels = torch.randn(1, 3 * 4, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)
