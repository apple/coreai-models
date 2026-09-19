// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Separable bilinear resampling of a planar `Float` image, matching
/// `torch.nn.functional.interpolate(mode: "bilinear", align_corners: false)`.
///
/// Construct once per (source, destination) pair and reuse it. Per-frame callers should take
/// `resample(_:into:scratch:)` and pass in their own storage.
///
/// `antialias` widens the filter when downsampling to match PIL and
/// `torchvision.transforms.Resize`. Upsampling ignores it.
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
        // Only the horizontal axis needs a control vector.
        self.horizontal = Weights(
            sourceSize: sourceWidth, destinationSize: destinationWidth, antialias: antialias,
            buildControlVector: true)
        self.vertical = Weights(
            sourceSize: sourceHeight, destinationSize: destinationHeight, antialias: antialias,
            buildControlVector: false)
    }

    /// True when the resampler would copy its input unchanged.
    public var isIdentity: Bool {
        sourceWidth == destinationWidth && sourceHeight == destinationHeight
    }

    /// True when the horizontal pass goes through `resampleTransposing` rather than
    /// `applyAcrossRows`. The common case.
    private var transposes: Bool { horizontal.maxTaps > 2 || !horizontal.controlIsExact }

    /// Elements of caller-owned storage `resample(_:into:scratch:)` needs.
    public var scratchCount: Int {
        if isIdentity { return 0 }
        if transposes {
            // Vertical output, its transpose, and the horizontal output before transposing back.
            return 2 * destinationHeight * sourceWidth + destinationWidth * destinationHeight
        }
        return sourceHeight * destinationWidth
    }

    /// Resample a row-major `sourceHeight × sourceWidth` buffer into caller-owned storage.
    ///
    /// `scratch` must hold at least `scratchCount` elements. Reuse it across calls and treat
    /// its contents as undefined.
    public func resample(
        _ source: UnsafeBufferPointer<Float>,
        into destination: UnsafeMutableBufferPointer<Float>,
        scratch: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(
            source.count >= sourceWidth * sourceHeight,
            "BilinearResampler expected \(sourceWidth * sourceHeight) samples, got \(source.count)")
        precondition(
            destination.count >= destinationWidth * destinationHeight,
            "BilinearResampler needs a \(destinationWidth * destinationHeight) element "
                + "destination, got \(destination.count)")
        let input = source.baseAddress!
        let output = destination.baseAddress!
        if isIdentity {
            output.update(from: input, count: sourceWidth * sourceHeight)
            return
        }
        precondition(
            scratch.count >= scratchCount,
            "BilinearResampler needs \(scratchCount) elements of scratch, got \(scratch.count)")
        let workspace = scratch.baseAddress!

        if transposes {
            resampleTransposing(input: input, output: output, workspace: workspace)
            return
        }

        // Horizontal first. The vertical pass then runs over the narrower rows when
        // downscaling.
        horizontal.applyAcrossRows(
            input: input, inputStride: sourceWidth,
            output: workspace, outputStride: destinationWidth,
            rowCount: sourceHeight)
        vertical.applyDownColumns(
            input: workspace, output: output, rowWidth: destinationWidth)
    }

    /// The horizontal pass for two kinds of resize `applyAcrossRows` cannot handle:
    /// antialiased downsamples needing more than two taps, and resizes whose control vector
    /// loses precision.
    ///
    /// Transposing turns the horizontal taps into contiguous rows, so both axes run through
    /// `applyDownColumns`.
    private func resampleTransposing(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        workspace: UnsafeMutablePointer<Float>
    ) {
        let rows = workspace
        let transposed = rows + destinationHeight * sourceWidth
        let columns = transposed + sourceWidth * destinationHeight

        vertical.applyDownColumns(input: input, output: rows, rowWidth: sourceWidth)
        // `vDSP_mtrans(A, _, C, _, M, N)` writes an M×N result from an N×M input.
        vDSP_mtrans(
            rows, 1, transposed, 1, vDSP_Length(sourceWidth), vDSP_Length(destinationHeight))
        horizontal.applyDownColumns(
            input: transposed, output: columns, rowWidth: destinationHeight)
        vDSP_mtrans(
            columns, 1, output, 1, vDSP_Length(destinationHeight),
            vDSP_Length(destinationWidth))
    }

    /// The array form of `resample(_:into:scratch:)`, No allocation.
    public func resample(
        _ source: [Float], into destination: inout [Float], scratch: inout [Float]
    ) {
        source.withUnsafeBufferPointer { input in
            destination.withUnsafeMutableBufferPointer { output in
                scratch.withUnsafeMutableBufferPointer { scratch in
                    resample(input, into: output, scratch: scratch)
                }
            }
        }
    }

    /// Resample a row-major `sourceHeight × sourceWidth` buffer.
    ///
    /// Allocates the result and the scratch on every call for one-shot use.
    public func resample(_ source: [Float]) -> [Float] {
        var destination = [Float](repeating: 0, count: destinationHeight * destinationWidth)
        var scratch = [Float](repeating: 0, count: scratchCount)
        resample(source, into: &destination, scratch: &scratch)
        return destination
    }
}

// MARK: - Weights

/// Per-output-index filter taps along one axis.
///
/// A dense `destinationSize × maxTaps` table with a start index and tap count per output.
/// Outputs near an edge use fewer taps and leave the spare slots unused.
private struct Weights: Sendable {
    let destinationSize: Int
    let maxTaps: Int
    /// First source index contributing to each output index.
    let starts: [Int]
    /// Number of live taps for each output index.
    let counts: [Int]
    /// `destinationSize * maxTaps` coefficients, normalized to sum to 1 per output.
    let values: [Float]
    /// Source coordinate per output. `vDSP_vlint` reads this as its control vector. Built
    /// only for the horizontal axis of a two-tap resize.
    let positions: [Float]
    /// First output whose second tap falls off the end of the row. See `applyAcrossRows`.
    let clampedFrom: Int
    /// True when `vDSP_vlint` reproduces the taps in `values` exactly.
    ///
    /// It recovers the interpolation fraction from a single `Float` coordinate. At video
    /// widths that loses enough precision to flip `FramePreprocessor`'s rounding. The error
    /// is zero for a binary-fraction resize ratio, so this is measured per resize.
    let controlIsExact: Bool

    init(sourceSize: Int, destinationSize: Int, antialias: Bool, buildControlVector: Bool) {
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
            // Multi-tap rows take the transposing path, which needs no control vector.
            self.positions = []
            self.clampedFrom = destinationSize
            self.controlIsExact = false
            return
        }

        // Plain bilinear: two taps, matching `area_pixel_compute_source_index` with
        // `align_corners=false`.
        self.maxTaps = 2
        var starts = [Int](repeating: 0, count: destinationSize)
        var counts = [Int](repeating: 2, count: destinationSize)
        var values = [Float](repeating: 0, count: destinationSize * 2)
        var positions = [Float](repeating: 0, count: buildControlVector ? destinationSize : 0)
        var exact = buildControlVector
        for index in 0..<destinationSize {
            let position = max(0.0, scale * (Double(index) + 0.5) - 0.5)
            let low = min(Int(position), sourceSize - 1)
            let fraction = position - Double(low)
            let high = low < sourceSize - 1 ? low + 1 : low
            starts[index] = low
            counts[index] = high == low ? 1 : 2
            values[index * 2] = Float(1.0 - fraction)
            values[index * 2 + 1] = Float(fraction)
            if buildControlVector {
                let control = Float(position)
                positions[index] = control
                // Only the interpolated outputs matter. The clamped tail bypasses
                // `vDSP_vlint`.
                if high != low {
                    let recovered = control - Float(low)
                    if Int(control) != low || recovered != Float(fraction) { exact = false }
                }
            }
            if high == low {
                // Last row/column: torch reuses the same sample for both taps, so fold
                // both weights onto it.
                values[index * 2] = 1.0
                values[index * 2 + 1] = 0.0
            }
        }
        self.starts = starts
        self.counts = counts
        self.values = values
        self.positions = positions
        self.controlIsExact = exact
        // `vDSP_vlint` reads `A[trunc(p) + 1]` unconditionally, so an output landing on the
        // last source sample would read past the row. Those outputs form a trailing run that
        // `applyAcrossRows` fills instead.
        self.clampedFrom =
            buildControlVector
            ? (positions.firstIndex { $0 >= Float(sourceSize - 1) } ?? destinationSize)
            : destinationSize
    }

    /// Resample each row independently with two Accelerate calls.
    ///
    /// Two-tap only. `vDSP_vlint` interpolates straight from the control vector, so `values`
    /// goes unused here.
    func applyAcrossRows(
        input: UnsafePointer<Float>, inputStride: Int,
        output: UnsafeMutablePointer<Float>, outputStride: Int,
        rowCount: Int
    ) {
        precondition(maxTaps == 2, "applyAcrossRows is the two-tap path")
        let interpolated = vDSP_Length(clampedFrom)
        let clamped = vDSP_Length(destinationSize - clampedFrom)
        let sourceLength = vDSP_Length(inputStride)
        positions.withUnsafeBufferPointer { positions in
            for row in 0..<rowCount {
                let sourceRow = input + row * inputStride
                let destinationRow = output + row * outputStride
                if interpolated > 0 {
                    vDSP_vlint(
                        sourceRow, positions.baseAddress!, 1, destinationRow, 1,
                        interpolated, sourceLength)
                }
                if clamped > 0 {
                    vDSP_vfill(
                        sourceRow + inputStride - 1, destinationRow + clampedFrom, 1, clamped)
                }
            }
        }
    }

    /// Resample down the columns. Each tap is a whole contiguous row scaled by one
    /// coefficient.
    func applyDownColumns(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        rowWidth: Int
    ) {
        let length = vDSP_Length(rowWidth)
        for index in 0..<destinationSize {
            let start = starts[index]
            let base = index * maxTaps
            let sourceRow = input + start * rowWidth
            let destinationRow = output + index * rowWidth

            if maxTaps == 2 {
                guard counts[index] > 1 else {
                    // Clamped edge: the table folds both weights onto one sample. The
                    // coefficient is 1 and this is a copy.
                    destinationRow.update(from: sourceRow, count: rowWidth)
                    continue
                }
                // `A + f * (B - A)` in one pass, where vsmul plus vsma would be two.
                var fraction = values[base + 1]
                vDSP_vintb(
                    sourceRow, 1, sourceRow + rowWidth, 1, &fraction, destinationRow, 1, length)
                continue
            }

            var first = values[base]
            vDSP_vsmul(sourceRow, 1, &first, destinationRow, 1, length)
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
