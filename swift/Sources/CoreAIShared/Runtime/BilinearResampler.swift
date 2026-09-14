// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Separable bilinear resampling of a planar `Float` image, matching
/// `torch.nn.functional.interpolate(mode: "bilinear", align_corners: false)`.
///
/// Hand-written rather than delegating to `vImageScale_PlanarF`, whose high-quality path is a
/// different kernel and drifts from PyTorch. Construct once per (source, destination) pair
/// and reuse; the weight tables are the only setup cost.
///
/// `antialias` widens the filter when downsampling, the way PIL and
/// `torchvision.transforms.Resize` do. Upsampling ignores it.
public struct BilinearResampler: Sendable {
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let destinationWidth: Int
    public let destinationHeight: Int

    private let horizontal: Weights
    private let vertical: Weights

    public init(
        sourceWidth: Int, sourceHeight: Int,
        destinationWidth: Int, destinationHeight: Int,
        antialias: Bool = false
    ) {
        precondition(
            sourceWidth > 0 && sourceHeight > 0 && destinationWidth > 0 && destinationHeight > 0,
            "BilinearResampler dimensions must be positive")
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.destinationWidth = destinationWidth
        self.destinationHeight = destinationHeight
        self.horizontal = Weights(
            sourceSize: sourceWidth, destinationSize: destinationWidth, antialias: antialias)
        self.vertical = Weights(
            sourceSize: sourceHeight, destinationSize: destinationHeight, antialias: antialias)
    }

    /// True when the resampler would copy its input unchanged.
    public var isIdentity: Bool {
        sourceWidth == destinationWidth && sourceHeight == destinationHeight
    }

    /// Resample a row-major `sourceHeight × sourceWidth` buffer.
    public func resample(_ source: [Float]) -> [Float] {
        precondition(
            source.count >= sourceWidth * sourceHeight,
            "BilinearResampler expected \(sourceWidth * sourceHeight) samples, got \(source.count)")
        if isIdentity { return Array(source.prefix(sourceWidth * sourceHeight)) }

        // Horizontal first, so the vertical pass runs over the smaller of the two widths
        // when downscaling and the taps stay whole contiguous rows either way.
        var intermediate = [Float](repeating: 0, count: sourceHeight * destinationWidth)
        source.withUnsafeBufferPointer { input in
            intermediate.withUnsafeMutableBufferPointer { output in
                horizontal.applyAcrossRows(
                    input: input.baseAddress!, inputStride: sourceWidth,
                    output: output.baseAddress!, outputStride: destinationWidth,
                    rowCount: sourceHeight)
            }
        }

        var destination = [Float](repeating: 0, count: destinationHeight * destinationWidth)
        intermediate.withUnsafeBufferPointer { input in
            destination.withUnsafeMutableBufferPointer { output in
                vertical.applyDownColumns(
                    input: input.baseAddress!,
                    output: output.baseAddress!,
                    rowWidth: destinationWidth)
            }
        }
        return destination
    }
}

// MARK: - Weights

/// Per-output-index filter taps along one axis.
///
/// Stored as a dense `destinationSize × maxTaps` table with a start index and live tap
/// count per output. Rows near an edge use fewer taps than the interior; padding to a
/// fixed stride keeps the indexing uniform at the cost of a few unused slots.
private struct Weights: Sendable {
    let destinationSize: Int
    let maxTaps: Int
    /// First source index contributing to each output index.
    let starts: [Int]
    /// Number of live taps for each output index.
    let counts: [Int]
    /// `destinationSize * maxTaps` coefficients, normalized to sum to 1 per output.
    let values: [Float]

    init(sourceSize: Int, destinationSize: Int, antialias: Bool) {
        self.destinationSize = destinationSize
        let scale = Double(sourceSize) / Double(destinationSize)

        if antialias && scale > 1.0 {
            // Downsampling with a triangle filter whose support grows with the ratio.
            // Mirrors `aten/src/ATen/native/UpSample.h::_compute_weights_aa`.
            let support = scale
            let inverseScale = 1.0 / scale
            let taps = Int(support.rounded(.up)) * 2 + 1
            self.maxTaps = taps
            var starts = [Int](repeating: 0, count: destinationSize)
            var counts = [Int](repeating: 0, count: destinationSize)
            var values = [Float](repeating: 0, count: destinationSize * taps)
            for index in 0..<destinationSize {
                let center = scale * (Double(index) + 0.5)
                let start = max(0, Int(center - support + 0.5))
                let end = min(sourceSize, Int(center + support + 0.5))
                let count = max(0, end - start)
                starts[index] = start
                counts[index] = min(count, taps)
                var total = 0.0
                for tap in 0..<counts[index] {
                    let offset = (Double(tap + start) - center + 0.5) * inverseScale
                    let weight = max(0.0, 1.0 - abs(offset))
                    values[index * taps + tap] = Float(weight)
                    total += weight
                }
                if total != 0 {
                    for tap in 0..<counts[index] {
                        values[index * taps + tap] /= Float(total)
                    }
                }
            }
            self.starts = starts
            self.counts = counts
            self.values = values
            return
        }

        // Plain bilinear: two taps, matching `area_pixel_compute_source_index` with
        // `align_corners=false`. The clamp is on the position, not the index, so a negative
        // source position pins to the first pixel instead of inventing one past the edge.
        self.maxTaps = 2
        var starts = [Int](repeating: 0, count: destinationSize)
        var counts = [Int](repeating: 2, count: destinationSize)
        var values = [Float](repeating: 0, count: destinationSize * 2)
        for index in 0..<destinationSize {
            let position = max(0.0, scale * (Double(index) + 0.5) - 0.5)
            let low = min(Int(position), sourceSize - 1)
            let fraction = position - Double(low)
            let high = low < sourceSize - 1 ? low + 1 : low
            starts[index] = low
            counts[index] = high == low ? 1 : 2
            values[index * 2] = Float(1.0 - fraction)
            values[index * 2 + 1] = Float(fraction)
            if high == low {
                // Last row/column: torch reuses the same sample for both taps, which is
                // the same as folding both weights onto it.
                values[index * 2] = 1.0
                values[index * 2 + 1] = 0.0
            }
        }
        self.starts = starts
        self.counts = counts
        self.values = values
    }

    /// Resample each row independently. Weights vary per output column, so this is a
    /// gather-dot and stays scalar.
    func applyAcrossRows(
        input: UnsafePointer<Float>, inputStride: Int,
        output: UnsafeMutablePointer<Float>, outputStride: Int,
        rowCount: Int
    ) {
        starts.withUnsafeBufferPointer { starts in
            counts.withUnsafeBufferPointer { counts in
                values.withUnsafeBufferPointer { values in
                    for row in 0..<rowCount {
                        let sourceRow = input + row * inputStride
                        let destinationRow = output + row * outputStride
                        for index in 0..<destinationSize {
                            let start = starts[index]
                            let base = index * maxTaps
                            var sum: Float = 0
                            for tap in 0..<counts[index] {
                                sum += values[base + tap] * sourceRow[start + tap]
                            }
                            destinationRow[index] = sum
                        }
                    }
                }
            }
        }
    }

    /// Resample down the columns. Each tap is a whole contiguous row scaled by one
    /// coefficient, so this is a few `vDSP_vsma` calls per output row rather than a
    /// per-pixel loop.
    func applyDownColumns(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        rowWidth: Int
    ) {
        let length = vDSP_Length(rowWidth)
        for index in 0..<destinationSize {
            let start = starts[index]
            let base = index * maxTaps
            let destinationRow = output + index * rowWidth
            var first = values[base]
            vDSP_vsmul(input + start * rowWidth, 1, &first, destinationRow, 1, length)
            guard counts[index] > 1 else { continue }
            for tap in 1..<counts[index] {
                var weight = values[base + tap]
                if weight == 0 { continue }
                vDSP_vsma(
                    input + (start + tap) * rowWidth, 1, &weight,
                    destinationRow, 1, destinationRow, 1, length)
            }
        }
    }
}
