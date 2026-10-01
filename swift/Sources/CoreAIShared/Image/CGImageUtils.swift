// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreGraphics
import Foundation
import ImageIO

/// Shared CGImage conversion utilities for diffusion and other vision pipelines.
public enum CGImageUtils {
    /// Load and decode the first image in a file into a `CGImage`. Expands a leading `~`.
    ///
    /// Throws `ImagePreprocessorError.loadFailed` when the file cannot be opened or decoded.
    public static func load(from path: String) throws -> CGImage {
        try load(from: URL(fileURLWithPath: NSString(string: path).expandingTildeInPath))
    }

    /// Load and decode the first image at `url` into a `CGImage`.
    ///
    /// Throws `ImagePreprocessorError.loadFailed` when the file cannot be opened or decoded.
    public static func load(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImagePreprocessorError.loadFailed(url)
        }
        return image
    }

    /// Resize a CGImage to `side × side` using CGContext with high-quality interpolation.
    public static func resize(_ image: CGImage, to side: Int) -> CGImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: side, height: side,
                bitsPerComponent: 8, bytesPerRow: 4 * side,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return ctx.makeImage()
    }

    /// Convert a CGImage to planar RGB floats in [-1, 1] with layout [3, H, W] flattened.
    ///
    /// Renders into an sRGB RGBA8 context, then vectorizes UInt8→Float32 and applies
    /// (pixel / 127.5) − 1.0 for all three channels in a single vDSP pass.
    public static func toNormalizedPlanarRGB(_ image: CGImage) throws -> [Float] {
        let width = image.width
        let height = image.height

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: 4 * width,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let ptr = context.data?.bindMemory(to: UInt8.self, capacity: width * height * 4)
        else {
            throw ImagePreprocessorError.renderFailed
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let pixelCount = height * width
        var result = [Float](repeating: 0, count: 3 * pixelCount)
        result.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            vDSP_vfltu8(ptr, 4, base, 1, vDSP_Length(pixelCount))
            vDSP_vfltu8(ptr + 1, 4, base + pixelCount, 1, vDSP_Length(pixelCount))
            vDSP_vfltu8(ptr + 2, 4, base + 2 * pixelCount, 1, vDSP_Length(pixelCount))
            var scale: Float = 1.0 / 127.5
            var bias: Float = -1.0
            vDSP_vsmsa(base, 1, &scale, &bias, base, 1, vDSP_Length(3 * pixelCount))
        }
        return result
    }
}
