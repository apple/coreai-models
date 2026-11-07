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
    # LayerNorm's weight is 1-D: it has no in-features axis to block along,
    # and used to crash check_divisibility with IndexError.
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
        # Out-features: 48 % 16 == 0, 48 (norm, skipped), 10 % 16 != 0.
        assert list(result.values()) == [(10, 16)]

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
        class Odd(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = "not-a-tensor"

        model = nn.Sequential(nn.Linear(32, 32))
        model.add_module("odd", Odd())
        assert check_divisibility(model, axis=1, block_size=32) == {}
