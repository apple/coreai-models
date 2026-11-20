"""Unit tests for the pure-math compression helpers.

Run from this directory: ``python3 -m pytest test_compression_metrics.py -q``.
These cover the coreai-opt-independent helpers only, so they need torch but
no prepared model and no checkpoint downloads.
"""

from __future__ import annotations

from torch import nn

from compression_metrics import (
    LayerSpec,
    check_divisibility,
    compression_ratio,
    compute_average_bitwidth,
    compute_compressed_size_mb,
    fp16_baseline_mb,
)


def _toy_model() -> nn.Module:
    # LayerNorm's weight is 1-D and used to crash check_divisibility with
    # IndexError on axis=1; it is not a type coreai-opt compresses, so it
    # must never be flagged at all.
    return nn.Sequential(nn.Linear(64, 48), nn.LayerNorm(48), nn.Linear(48, 10))


class TestCheckDivisibility:
    def test_skips_one_dimensional_weights(self):
        # axis=1 on a model containing LayerNorm must not raise.
        result = check_divisibility(_toy_model(), axis=1, block_size=32)
        # Linear(64->48): shape[1] == 64, divisible by 32.
        # Linear(48->10): shape[1] == 48, not divisible by 32.
        assert list(result.values()) == [(48, 32)]

    def test_reports_offending_dims_along_axis_0(self):
        result = check_divisibility(_toy_model(), axis=0, block_size=16)
        # Out-features: 48 % 16 == 0, 10 % 16 != 0; the norm is not a
        # compressible type and is not checked.
        assert list(result.values()) == [(10, 16)]

    def test_norms_are_never_flagged(self):
        # coreai-opt never palettizes or block-quantizes norms, so an
        # indivisible norm weight cannot cause a silently skipped layer
        # and must not appear in the report.
        model = nn.Sequential(nn.LayerNorm(60))
        assert check_divisibility(model, axis=0, block_size=8) == {}

    def test_conv_transpose_output_channels_resolve_to_dim_1(self):
        # ConvTranspose stores weight as (in, out, *kernel): its output
        # channels live on tensor dim 1. With axis=0 (output channels) and
        # group 8, ConvTranspose2d(64, 12, 3) must be reported for 12 —
        # the palettizer would silently skip it — not cleared for 64.
        model = nn.Sequential(nn.ConvTranspose2d(64, 12, 3))
        result = check_divisibility(model, axis=0, block_size=8)
        assert list(result.values()) == [(12, 8)]

    def test_conv_transpose_input_channels_resolve_to_dim_0(self):
        # The same layer along the input-channel axis resolves to dim 0
        # (64), which is divisible by 8.
        model = nn.Sequential(nn.ConvTranspose2d(64, 12, 3))
        assert check_divisibility(model, axis=1, block_size=8) == {}

    def test_embedding_is_checked(self):
        model = nn.Sequential(nn.Embedding(100, 32))
        result = check_divisibility(model, axis=0, block_size=8)
        assert list(result.values()) == [(100, 8)]

    def test_all_divisible_returns_empty(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        assert check_divisibility(model, axis=1, block_size=32) == {}

    def test_negative_axis_in_range_is_honored(self):
        model = nn.Sequential(nn.Linear(48, 10))
        result = check_divisibility(model, axis=-1, block_size=32)
        assert list(result.values()) == [(48, 32)]


class TestPureMathHelpers:
    def test_fp16_baseline_counts_all_parameters(self):
        model = nn.Linear(64, 48)  # 64*48 weights + 48 biases
        expected_mb = (64 * 48 + 48) * 2 / (1024 * 1024)
        assert abs(fp16_baseline_mb(model) - expected_mb) < 1e-9

    def test_uncompressed_spec_has_no_overhead(self):
        spec = LayerSpec(name="fc", weight_shape=(64, 48), n_bits=16)
        expected_mb = 64 * 48 * 2 / (1024 * 1024)
        assert abs(compute_compressed_size_mb([spec]) - expected_mb) < 1e-9

    def test_average_bitwidth_is_weighted_by_numel(self):
        specs = [
            LayerSpec(name="a", weight_shape=(10, 10), n_bits=4),
            LayerSpec(name="b", weight_shape=(30, 10), n_bits=16),
        ]
        # (100*4 + 300*16) / 400 = 13.0
        assert abs(compute_average_bitwidth(specs) - 13.0) < 1e-9

    def test_compression_ratio(self):
        assert abs(compression_ratio(100.0, 25.0) - 4.0) < 1e-9

    def test_torch_tensor_guard_ignores_non_tensor_weight(self):
        # A compressible type whose weight is not a tensor (e.g. swapped
        # out by an export wrapper) must be skipped, not crash the check.
        model = nn.Sequential(nn.Linear(32, 30))
        del model[0].weight
        model[0].weight = "not-a-tensor"
        assert check_divisibility(model, axis=1, block_size=32) == {}
