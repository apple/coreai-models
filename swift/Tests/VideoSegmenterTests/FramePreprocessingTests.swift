// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

/// Checks `FramePreprocessor` against literals captured from PyTorch.
///
/// Self-contained on purpose. An earlier version read `.npy` dumps from a path given by an
/// environment variable, which meant it silently skipped on every machine that didn't have
/// them, which is coverage in name only.
@Suite("Frame preprocessing")
struct FramePreprocessingTests {
    /// A 4-wide by 3-high interleaved RGB image. Values chosen so the resize lands on
    /// fractional samples and the rounding step actually matters.
    private let sourceRGB: [UInt8] = [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
        120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
        240, 250, 255, 1, 3, 7, 11, 13, 17, 19, 23, 29,
    ]

    /// `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)` on the
    /// above at 0-255, rounded back onto the integer grid the way torchvision's uint8 path
    /// does, then `(x / 255 - 0.5) / 0.5`. Planar `[3, 8, 8]`.
    private let expected: [Float] = [
        // channel 0
        -1.000000, -0.937255, -0.827451, -0.701961, -0.592157, -0.466667, -0.356863, -0.294118,
        -0.937255, -0.882353, -0.764706, -0.647059, -0.529412, -0.411765, -0.294118, -0.231373,
        -0.592157, -0.529412, -0.411765, -0.294118, -0.176471, -0.058824, 0.058824, 0.113726,
        -0.231373, -0.176471, -0.058824, 0.058824, 0.176471, 0.294118, 0.411765, 0.474510,
        0.113726, 0.074510, -0.003922, 0.011765, 0.113726, 0.215686, 0.317647, 0.364706,
        0.474510, 0.231373, -0.239216, -0.443137, -0.372549, -0.301961, -0.231373, -0.192157,
        0.819608, 0.388235, -0.482353, -0.898039, -0.850980, -0.811765, -0.772549, -0.756863,
        0.882353, 0.411765, -0.521569, -0.968627, -0.937255, -0.898039, -0.866667, -0.850980,
        // channel 1
        -0.921569, -0.858824, -0.749020, -0.623529, -0.513726, -0.388235, -0.278431, -0.215686,
        -0.858824, -0.803922, -0.686275, -0.568627, -0.450980, -0.333333, -0.215686, -0.152941,
        -0.513726, -0.450980, -0.333333, -0.215686, -0.098039, 0.019608, 0.137255, 0.192157,
        -0.152941, -0.098039, 0.019608, 0.137255, 0.254902, 0.372549, 0.490196, 0.552941,
        0.192157, 0.152941, 0.066667, 0.074510, 0.176471, 0.278431, 0.380392, 0.435294,
        0.552941, 0.301961, -0.192157, -0.403922, -0.325490, -0.254902, -0.184314, -0.145098,
        0.898039, 0.450980, -0.450980, -0.874510, -0.835294, -0.788235, -0.749020, -0.725490,
        0.960784, 0.474510, -0.490196, -0.952941, -0.921569, -0.874510, -0.843137, -0.819608,
        // channel 2
        -0.843137, -0.780392, -0.670588, -0.545098, -0.435294, -0.309804, -0.200000, -0.137255,
        -0.780392, -0.725490, -0.607843, -0.490196, -0.372549, -0.254902, -0.137255, -0.074510,
        -0.435294, -0.372549, -0.254902, -0.137255, -0.019608, 0.098039, 0.215686, 0.270588,
        -0.074510, -0.019608, 0.098039, 0.215686, 0.333333, 0.450980, 0.568627, 0.631373,
        0.270588, 0.223529, 0.137255, 0.145098, 0.247059, 0.349020, 0.458824, 0.505882,
        0.607843, 0.356863, -0.137255, -0.349020, -0.278431, -0.200000, -0.121569, -0.082353,
        0.945098, 0.490196, -0.411765, -0.843137, -0.796078, -0.749020, -0.701961, -0.670588,
        1.000000, 0.513726, -0.458824, -0.921569, -0.890196, -0.843137, -0.796078, -0.772549,
    ]

    private func preprocessor(targetSize: Int = 8) -> FramePreprocessor {
        FramePreprocessor(
            targetSize: targetSize, mean: (0.5, 0.5, 0.5), standardDeviation: (0.5, 0.5, 0.5))
    }

    @Test("Planar CHW output matches the PyTorch video processor")
    func matchesTorch() {
        let actual = preprocessor().preprocess(
            interleavedRGB: sourceRGB, width: 4, height: 3, channelStride: 3)
        #expect(actual.count == expected.count)
        guard actual.count == expected.count else { return }
        var worst: Float = 0
        for index in actual.indices {
            worst = max(worst, abs(actual[index] - expected[index]))
        }
        // One code value of 8-bit input spans 2/255 = 0.0078 on this range. The residual is
        // tie-breaking only: `vvnintf` rounds half away from zero, `torch.round` rounds half
        // to even, so they can disagree by exactly one code value on an exact .5.
        #expect(worst <= 0.0079, "worst |delta| \(worst)")
    }

    @Test("Rounding onto the 0-255 grid is what torchvision does")
    func roundsOntoCodeValueGrid() {
        // Every output must be reachable as (n/255 - 0.5)/0.5 for integer n, which is what
        // resizing a uint8 tensor and casting back produces.
        let actual = preprocessor().preprocess(
            interleavedRGB: sourceRGB, width: 4, height: 3, channelStride: 3)
        for value in actual {
            let code = (value * 0.5 + 0.5) * 255
            #expect(abs(code - code.rounded()) < 1e-3, "\(value) is off the grid (code \(code))")
        }
    }

    @Test("A flat image stays flat, so the weights sum to one")
    func flatImageStaysFlat() {
        // Weights that don't sum to 1 would shift a uniform region, which on a mask logit
        // field moves the binarization threshold.
        let grey = [UInt8](repeating: 128, count: 4 * 3 * 3)
        let actual = preprocessor(targetSize: 11).preprocess(
            interleavedRGB: grey, width: 4, height: 3, channelStride: 3)
        let want = (Float(128) / 255 - 0.5) / 0.5
        for value in actual {
            #expect(abs(value - want) < 1e-6)
        }
    }

    @Test("RGBA input skips the alpha byte")
    func honoursChannelStride() {
        // The `CGImage` path renders into a 4-byte-per-pixel context, so the preprocessor
        // has to stride past alpha. Same pixels, both layouts, same answer.
        var rgba: [UInt8] = []
        for index in 0..<(4 * 3) {
            rgba.append(contentsOf: sourceRGB[index * 3..<(index * 3 + 3)])
            rgba.append(255)
        }
        let viaRGB = preprocessor().preprocess(
            interleavedRGB: sourceRGB, width: 4, height: 3, channelStride: 3)
        let viaRGBA = preprocessor().preprocess(
            interleavedRGB: rgba, width: 4, height: 3, channelStride: 4)
        #expect(viaRGB == viaRGBA)
    }
}
