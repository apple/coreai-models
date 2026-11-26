# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS MLP primitive."""

import pytest
import torch

from coreai_models.primitives.ios.mlp import MLP


class TestiOSMLP:
    """Test iOS MLP primitive."""

    @pytest.fixture
    def mlp(self):
        """Create a small MLP for testing."""
        torch.manual_seed(42)
        return MLP(dim=16, hidden_dim=32).eval()

    def test_output_shape(self, mlp):
        """MLP should preserve the input shape."""
        x = torch.randn(1, 4, 1, 16)
        out = mlp(x)
        assert out.shape == x.shape

    def test_output_shape_single_token(self, mlp):
        """MLP should work with a single token (decode step)."""
        x = torch.randn(1, 1, 1, 16)
        out = mlp(x)
        assert out.shape == (1, 1, 1, 16)

    def test_output_shape_batched(self, mlp):
        """MLP should work with batch size > 1."""
        x = torch.randn(2, 4, 1, 16)
        out = mlp(x)
        assert out.shape == (2, 4, 1, 16)

    def test_per_token_independence(self, mlp):
        """Each token should be processed independently.

        Running 8 tokens together should give the same result as running
        each token separately — regardless of how tokens are batched.
        """
        x = torch.randn(1, 8, 1, 16)
        out_together = mlp(x)

        for t in range(8):
            token = x[:, t : t + 1, :, :]
            out_single = mlp(token)
            torch.testing.assert_close(
                out_together[:, t : t + 1, :, :],
                out_single,
                atol=1e-6,
                rtol=1e-5,
            )

    def test_matches_linear_reference(self, mlp):
        """MLP output should match an equivalent computation using Linear layers."""
        dim = 16
        x = torch.randn(1, 4, 1, dim)

        # Build equivalent Linear layers from the Conv2d weights.
        gate_w = mlp.gate_proj.weight.squeeze()  # (hidden_dim, dim)
        up_w = mlp.up_proj.weight.squeeze()
        down_w = mlp.down_proj.weight.squeeze()  # (dim, hidden_dim)

        x_flat = x.squeeze(-2)  # (1, 4, dim)
        gate = torch.nn.functional.silu(x_flat @ gate_w.T)
        up = x_flat @ up_w.T
        ref = (up * gate) @ down_w.T  # (1, 4, dim)

        out = mlp(x).squeeze(-2)  # (1, 4, dim)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_with_bias(self):
        """MLP with bias=True should still produce correct shapes."""
        torch.manual_seed(42)
        mlp = MLP(dim=16, hidden_dim=32, bias=True).eval()
        x = torch.randn(1, 4, 1, 16)
        out = mlp(x)
        assert out.shape == x.shape
