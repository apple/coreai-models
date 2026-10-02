// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Deterministic random source for reproducible noise generation.
///
/// Multiple implementations exist to match the exact random sequences produced by
/// different Python frameworks. Using the matching source guarantees the same seed
/// produces the same image across platforms.
///
/// - ``TorchRandomSource``: Default. Matches PyTorch (`torch.manual_seed`). Used by Flux and most modern diffusion models.
/// - ``NumPyRandomSource``: Matches NumPy's legacy `numpy.random.RandomState`.
/// - ``NvRandomSource``: Matches NVIDIA cuRAND (Philox). Used by some ComfyUI/Automatic1111 workflows.
///
/// The implementations are direct ports of MT19937 / Philox; do not refactor the
/// bitwise logic without verifying output against the Python reference for multiple seeds.
public protocol RandomSource {
    mutating func nextNormal(mean: Double, stdev: Double) -> Double
    mutating func normalArray(_ shape: [Int], mean: Double, stdev: Double) -> [Float]
}
