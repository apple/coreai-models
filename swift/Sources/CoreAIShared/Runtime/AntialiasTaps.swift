// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// The filter taps of one axis of an antialiased separable resize, as PyTorch's
/// `aten/src/ATen/native/UpSample.h::_compute_weights_aa` computes them with
/// `align_corners=false`: when downscaling, the filter is stretched by the ratio.
///
/// The weights are not normalized. Callers divide by their sum, each at its own precision, since
/// the rounding of that division is part of matching a given PyTorch path.
package enum AntialiasTaps {
    package enum Filter: Sendable {
        /// `max(0, 1 - |x|)`, support 1: bilinear.
        case triangle
        /// Keys cubic with `a = -0.5`, support 2: bicubic.
        case keysCubic

        var support: Double {
            switch self {
            case .triangle: 1
            case .keysCubic: 2
            }
        }

        func callAsFunction(_ x: Double) -> Double {
            let x = abs(x)
            switch self {
            case .triangle:
                return max(0, 1 - x)
            case .keysCubic:
                let a = -0.5
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                return x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0
            }
        }
    }

    /// For each of `destinationSize` outputs, the first source index it reads and the
    /// unnormalized weights of the source indices from there on.
    package static func taps(
        sourceSize: Int, destinationSize: Int, filter: Filter
    ) -> [(start: Int, weights: [Double])] {
        let scale = Double(sourceSize) / Double(destinationSize)
        let support = scale >= 1 ? filter.support * scale : filter.support
        let inverseScale = scale >= 1 ? 1 / scale : 1
        return (0..<destinationSize).map { index in
            let center = scale * (Double(index) + 0.5)
            let start = max(0, Int(center - support + 0.5))
            let end = min(sourceSize, Int(center + support + 0.5))
            let weights = (start..<max(start, end)).map { filter((Double($0) - center + 0.5) * inverseScale) }
            return (start, weights)
        }
    }
}
