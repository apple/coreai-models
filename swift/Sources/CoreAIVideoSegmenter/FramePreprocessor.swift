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
/// Separate from `CoreAIShared.ImagePreprocessor`. That one resizes through a `CGContext` on
/// 8-bit samples where `Sam3VideoVideoProcessor` resizes with plain bilinear in float. The
/// difference is invisible on one image but compounds over a tracked video, since every
/// frame's input feeds the memory bank.
final class FramePreprocessor {
    let targetSize: Int
    let mean: (Float, Float, Float)
    let standardDeviation: (Float, Float, Float)

    /// Resamplers are keyed by source size. For a video that means one build.
    private var cachedWidth = 0
    private var cachedHeight = 0
    private var cachedResampler: BilinearResampler?
    private var scratch: [Float] = []

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
        // Drawn at native size: a format conversion, with the resize left to the resampler.
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        // `base` belongs to the context, so keep it alive for as long as the pointer is read.
        return withExtendedLifetime(context) {
            let pixels = base.bindMemory(to: UInt8.self, capacity: width * height * 4)
            return preprocess(
                interleavedRGB: UnsafeBufferPointer(start: pixels, count: width * height * 4),
                width: width, height: height, channelStride: 4)
        }
    }

    /// Preprocess interleaved 8-bit samples directly.
    ///
    /// `channelStride` is 4 for RGBA and 3 for packed RGB. The preprocessing tests feed
    /// packed RGB to hold the decoder constant.
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
        precondition(width > 0 && height > 0, "FramePreprocessor needs a non-empty frame")
        precondition(channelStride >= 3, "FramePreprocessor needs at least three channels")
        // `vDSP_vfltu8` walks the last sample at `(pixels - 1) * stride + channel`, so the
        // buffer has to cover the third channel of the final pixel.
        precondition(
            bytes.count >= (width * height - 1) * channelStride + 3,
            "FramePreprocessor needs \((width * height - 1) * channelStride + 3) bytes for "
                + "\(width)x\(height) at stride \(channelStride), got \(bytes.count)")
        guard let base = bytes.baseAddress else { return [] }

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
            vDSP_vfltu8(base + channel, channelStride, &plane, 1, vDSP_Length(sourcePixels))

            resampler.resample(plane, into: &resized, scratch: &scratch)

            // torchvision resizes a uint8 tensor as uint8: it interpolates and then rounds
            // back to integers. Matching that removes a uniform ~0.25-code-value bias against
            // the reference.
            resized.withUnsafeMutableBufferPointer { buffer in
                if let address = buffer.baseAddress {
                    vvnintf(address, address, &elementCount)
                }
            }

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
        if let existing = cachedResampler, cachedWidth == sourceWidth, cachedHeight == sourceHeight {
            return existing
        }
        // `antialias: false` matches the video processor. Every real clip upscales to 1008,
        // where the flag has no effect.
        let built = BilinearResampler(
            sourceWidth: sourceWidth, sourceHeight: sourceHeight,
            destinationWidth: targetSize, destinationHeight: targetSize,
            antialias: false)
        cachedWidth = sourceWidth
        cachedHeight = sourceHeight
        cachedResampler = built
        scratch = [Float](repeating: 0, count: built.scratchCount)
        return built
    }
}
