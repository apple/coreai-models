// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIShared

/// Every expectation here is a literal captured from
/// `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`, not from
/// this implementation. These resizes sit between graph calls in the tracker and the
/// resampled mask is what conditions every later frame, so "looks about right" is not
/// good enough.
@Suite("BilinearResampler")
struct BilinearResamplerTests {
    /// `torch.arange(16).reshape(4, 4)`.
    private let ramp4 = (0..<16).map(Float.init)
    /// `torch.arange(64).reshape(8, 8)`.
    private let ramp8 = (0..<64).map(Float.init)

    private func expectClose(
        _ actual: [Float], _ expected: [Float], tolerance: Float = 1e-5,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.count, sourceLocation: sourceLocation)
        guard actual.count == expected.count else { return }
        for index in actual.indices {
            #expect(
                abs(actual[index] - expected[index]) <= tolerance,
                "element \(index): \(actual[index]) vs \(expected[index])",
                sourceLocation: sourceLocation)
        }
    }

    @Test("4x4 to 8x8 matches torch bilinear")
    func upsample() {
        let resampler = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 8, destinationHeight: 8)
        expectClose(
            resampler.resample(ramp4),
            [
                0.00, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.00,
                1.00, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.00,
                3.00, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.00,
                5.00, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.00,
                7.00, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.00,
                9.00, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.00,
                11.00, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.00,
                12.00, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.00,
            ])
    }

    @Test("Antialiasing is a no-op when upsampling")
    func antialiasIgnoredWhenUpsampling() {
        // This is the fact the export relies on: three of its four resizes are upsamples,
        // so dropping `antialias=True`, which has no Core AI lowering, changes nothing.
        // Torch's antialiased path widens to three taps here, but the third weight is
        // exactly zero.
        let plain = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 8, destinationHeight: 8,
            antialias: false)
        let antialiased = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 8, destinationHeight: 8,
            antialias: true)
        expectClose(antialiased.resample(ramp4), plain.resample(ramp4), tolerance: 0)
    }

    @Test("4x4 to 2x2 without antialiasing matches torch")
    func downsamplePlain() {
        let resampler = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 2, destinationHeight: 2,
            antialias: false)
        expectClose(resampler.resample(ramp4), [2.5, 4.5, 10.5, 12.5])
    }

    @Test("4x4 to 2x2 with antialiasing matches torch, and differs from plain")
    func downsampleAntialiased() {
        let resampler = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 2, destinationHeight: 2,
            antialias: true)
        // Meaningfully different from the plain result above. This is the difference the
        // export moved to the host rather than approximating away.
        expectClose(
            resampler.resample(ramp4),
            [3.571429, 5.142858, 9.857143, 11.428572])
    }

    @Test("8x8 to 3x3 matches torch, both with and without antialiasing")
    func nonIntegerDownsample() {
        // A 2.667x ratio, so the antialiased support is fractional and the tap count is
        // the `ceil(support) * 2 + 1` that torch computes.
        let antialiased = BilinearResampler(
            sourceWidth: 8, sourceHeight: 8, destinationWidth: 3, destinationHeight: 3,
            antialias: true)
        expectClose(
            antialiased.resample(ramp8),
            [
                9.947369, 12.342107, 14.736842,
                29.105265, 31.500002, 33.894737,
                48.263161, 50.657898, 53.052635,
            ])

        let plain = BilinearResampler(
            sourceWidth: 8, sourceHeight: 8, destinationWidth: 3, destinationHeight: 3,
            antialias: false)
        expectClose(
            plain.resample(ramp8),
            [
                7.500000, 10.166667, 12.833335,
                28.833332, 31.500000, 34.166668,
                50.166672, 52.833336, 55.500008,
            ])
    }

    @Test("Non-square resizes use independent row and column tables")
    func nonSquare() {
        // 3x4 to 5x6 scales differently on each axis; a shared weight table would be
        // wrong on one of them and this ramp would show it.
        let resampler = BilinearResampler(
            sourceWidth: 4, sourceHeight: 3, destinationWidth: 6, destinationHeight: 5)
        expectClose(
            resampler.resample((0..<12).map(Float.init)),
            [
                0.000000, 0.500000, 1.166667, 1.833333, 2.500000, 3.000000,
                1.600000, 2.100000, 2.766667, 3.433333, 4.100000, 4.600000,
                4.000000, 4.500000, 5.166667, 5.833333, 6.500000, 7.000000,
                6.400001, 6.900001, 7.566667, 8.233334, 8.900001, 9.400001,
                8.000000, 8.500000, 9.166667, 9.833334, 10.500000, 11.000000,
            ])
    }

    @Test("A same-size resample is the identity")
    func identity() {
        let resampler = BilinearResampler(
            sourceWidth: 4, sourceHeight: 4, destinationWidth: 4, destinationHeight: 4)
        #expect(resampler.isIdentity)
        expectClose(resampler.resample(ramp4), ramp4, tolerance: 0)
    }

    @Test("A constant field stays constant, so the weights sum to one everywhere")
    func partitionOfUnity() {
        // Weights that do not sum to 1 would darken or brighten a flat region, which on a
        // mask logit field shifts the binarization threshold.
        for antialias in [false, true] {
            for (source, destination) in [(4, 9), (9, 4), (16, 5), (5, 16), (7, 7)] {
                let resampler = BilinearResampler(
                    sourceWidth: source, sourceHeight: source,
                    destinationWidth: destination, destinationHeight: destination,
                    antialias: antialias)
                let flat = [Float](repeating: 2.5, count: source * source)
                for value in resampler.resample(flat) {
                    #expect(
                        abs(value - 2.5) < 1e-5,
                        "antialias=\(antialias) \(source)->\(destination) gave \(value)")
                }
            }
        }
    }

    /// The two internal paths — `vDSP_vlint` when the `Float` control vector is exact, and
    /// the transposing one otherwise — have to agree, because which one a resize takes
    /// depends on whether its ratio happens to be a binary fraction, and nothing about the
    /// call site says so. Sizes here are wide enough for a `Float` coordinate to lose
    /// fractional precision, which the 4x4 cases above cannot see.
    @Test("Both internal paths agree on a wide source")
    func pathsAgree() {
        // A sawtooth, so neighbouring samples differ by the full range and any error in the
        // interpolation fraction shows up at full scale instead of being averaged away.
        func sawtooth(_ count: Int) -> [Float] { (0..<count).map { Float($0 % 37) } }

        for (sourceSide, destinationWidth, destinationHeight) in [
            (1920, 1008, 1008),  // FramePreprocessor, a non-binary ratio
            (252, 1920, 1080),  // MaskPostprocessor, also non-binary
            (252, 1008, 1008),  // exact ratio, so this one takes the vDSP_vlint path
        ] {
            let resampler = BilinearResampler(
                sourceWidth: sourceSide, sourceHeight: sourceSide,
                destinationWidth: destinationWidth, destinationHeight: destinationHeight)
            let source = sawtooth(sourceSide * sourceSide)
            let actual = resampler.resample(source)

            // Recomputed here in Double, the way the weight tables are, so this is a check
            // against the definition rather than against either path.
            let horizontalScale = Double(sourceSide) / Double(destinationWidth)
            let verticalScale = Double(sourceSide) / Double(destinationHeight)
            func tap(_ index: Int, _ scale: Double) -> (Int, Int, Double) {
                let position = max(0.0, scale * (Double(index) + 0.5) - 0.5)
                let low = min(Int(position), sourceSide - 1)
                let high = low < sourceSide - 1 ? low + 1 : low
                return (low, high, position - Double(low))
            }
            var worst: Float = 0
            for row in stride(from: 0, to: destinationHeight, by: 7) {
                let (top, bottom, rowFraction) = tap(row, verticalScale)
                for column in stride(from: 0, to: destinationWidth, by: 7) {
                    let (left, right, columnFraction) = tap(column, horizontalScale)
                    func sample(_ r: Int, _ c: Int) -> Double {
                        Double(source[r * sourceSide + c])
                    }
                    let upper =
                        sample(top, left) * (1 - columnFraction)
                        + sample(top, right) * columnFraction
                    let lower =
                        sample(bottom, left) * (1 - columnFraction)
                        + sample(bottom, right) * columnFraction
                    let expected = Float(upper * (1 - rowFraction) + lower * rowFraction)
                    worst = max(worst, abs(actual[row * destinationWidth + column] - expected))
                }
            }
            #expect(
                worst < 1e-4,
                "\(sourceSide)^2 -> \(destinationWidth)x\(destinationHeight) drifted by \(worst)")
        }
    }
}
