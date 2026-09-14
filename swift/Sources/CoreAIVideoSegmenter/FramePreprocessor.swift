// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import CoreGraphics
import Foundation

/// Turns a decoded frame into the planar CHW tensor `image_encode` expects.
///
/// Deliberately not `CoreAIShared.ImagePreprocessor`, which the image segmenter uses: that
/// resizes by drawing through a `CGContext` at `.high` interpolation, a Lanczos-family kernel
/// on 8-bit samples, while `Sam3VideoVideoProcessor` resizes with plain bilinear in float. For
/// a single image the difference is invisible; across a tracked video it compounds, because
/// every frame's input feeds the memory bank.
///
/// So: convert to float at native resolution first, resample in float, normalize last.
struct FramePreprocessor {
    let targetSize: Int
    let mean: (Float, Float, Float)
    let standardDeviation: (Float, Float, Float)

    /// Resamplers are keyed by source size and rebuilt only when it changes, which for a
    /// video is once.
    private final class Cache {
        var width = 0
        var height = 0
        var resampler: BilinearResampler?
    }
    private let cache = Cache()

    init(
        targetSize: Int,
        mean: (CGFloat, CGFloat, CGFloat),
        standardDeviation: (CGFloat, CGFloat, CGFloat)
    ) {
        self.targetSize = targetSize
        self.mean = (Float(mean.0), Float(mean.1), Float(mean.2))
        self.standardDeviation = (
            Float(standardDeviation.0), Float(standardDeviation.1), Float(standardDeviation.2)
        )
    }

    /// Preprocess a decoded frame. Returns flat `[3, targetSize, targetSize]`.
    func preprocess(_ image: CGImage) throws -> [Float] {
        let width = image.width
        let height = image.height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let base = context.data
        else {
            throw ImagePreprocessorError.renderFailed
        }
        // Drawn at native size: this is a format conversion, not a resize.
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let pixels = base.bindMemory(to: UInt8.self, capacity: width * height * 4)
        return preprocess(
            interleavedRGB: UnsafeBufferPointer(start: pixels, count: width * height * 4),
            width: width, height: height, channelStride: 4)
    }

    /// Preprocess interleaved 8-bit samples directly.
    ///
    /// `channelStride` is 4 for RGBA and 3 for packed RGB. The second form is what the
    /// preprocessing tests feed in, so they can hold the decoder constant and measure
    /// only this stage.
    func preprocess(
        interleavedRGB bytes: [UInt8], width: Int, height: Int, channelStride: Int
    ) -> [Float] {
        bytes.withUnsafeBufferPointer {
            preprocess(interleavedRGB: $0, width: width, height: height, channelStride: channelStride)
        }
    }

    private func preprocess(
        interleavedRGB bytes: UnsafeBufferPointer<UInt8>,
        width: Int, height: Int, channelStride: Int
    ) -> [Float] {
        let resampler = resampler(sourceWidth: width, sourceHeight: height)
        let sourcePixels = width * height
        let targetPixels = targetSize * targetSize

        var output = [Float](repeating: 0, count: 3 * targetPixels)
        var plane = [Float](repeating: 0, count: sourcePixels)
        var resized = [Float](repeating: 0, count: targetPixels)
        let means = [mean.0, mean.1, mean.2]
        let deviations = [standardDeviation.0, standardDeviation.1, standardDeviation.2]
        var elementCount = Int32(targetPixels)

        for channel in 0..<3 {
            // De-interleave, staying in 0-255 so the rounding below lands on the same
            // grid torchvision uses.
            vDSP_vfltu8(
                bytes.baseAddress! + channel, channelStride, &plane, 1, vDSP_Length(sourcePixels))

            resized = resampler.resample(plane)

            // torchvision resizes a uint8 tensor as uint8: it interpolates and then
            // rounds back to integers, so its `pixel_values` land exactly on the 0-255
            // grid (verified: every element is within 8e-6 of an integer). Skipping this
            // leaves a uniform ~0.25-code-value bias against the reference.
            vvnintf(&resized, resized, &elementCount)

            // Fold rescale and normalize into one affine pass: (x / 255 - m) / s.
            var slope = 1 / (255 * deviations[channel])
            var offset = -means[channel] / deviations[channel]
            output.withUnsafeMutableBufferPointer { out in
                vDSP_vsmsa(
                    resized, 1, &slope, &offset,
                    out.baseAddress! + channel * targetPixels, 1, vDSP_Length(targetPixels))
            }
        }
        return output
    }

    private func resampler(sourceWidth: Int, sourceHeight: Int) -> BilinearResampler {
        if let existing = cache.resampler, cache.width == sourceWidth, cache.height == sourceHeight {
            return existing
        }
        // `antialias: false` matches the video processor, and every real clip upscales to
        // 1008 anyway, where the flag makes no difference.
        let built = BilinearResampler(
            sourceWidth: sourceWidth, sourceHeight: sourceHeight,
            destinationWidth: targetSize, destinationHeight: targetSize,
            antialias: false)
        cache.width = sourceWidth
        cache.height = sourceHeight
        cache.resampler = built
        return built
    }
}
