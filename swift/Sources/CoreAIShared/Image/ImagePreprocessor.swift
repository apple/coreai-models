// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import ImageIO

// MARK: - Image Strategy

public enum ImageStrategy: String, Codable, Sendable {
    case stretch
    case centerCrop = "center_crop"
    case pad
}

// MARK: - ImagePreprocessor

/// Resizes an image to a target size and applies per-channel normalization
/// (`(pixel * rescale - mean) / std`).
///
/// The pipeline draws the source image into an sRGB RGBA8 CGContext at the
/// target size (high-interpolation, matches PIL BICUBIC closely), then
/// converts to Float32 with the configured rescale/mean/std. Output is
/// `[H, W, 4]` Float32 (NHWC, alpha unused). The caller transposes to
/// `[1, 3, H, W]` (NCHW) before binding to a vision encoder input.
///
/// ## Usage
/// ```swift
/// let preprocessor = ImagePreprocessor.gemma3
/// let (data, width, height) = try preprocessor.preprocess(imageURL: imageURL)
/// // data: Float32 RGBA, width x height
/// ```
public struct ImagePreprocessor: Sendable {
    public let targetSize: CGSize
    public let mean: (CGFloat, CGFloat, CGFloat)
    public let std: (CGFloat, CGFloat, CGFloat)
    public let rescaleFactor: CGFloat

    // Thread safe according to API docs
    private static let ciContext = CIContext()

    public init(
        targetSize: CGSize,
        mean: (CGFloat, CGFloat, CGFloat),
        std: (CGFloat, CGFloat, CGFloat),
        rescaleFactor: CGFloat
    ) {
        self.targetSize = targetSize
        self.mean = mean
        self.std = std
        self.rescaleFactor = rescaleFactor
    }

    /// Gemma 3 / SigLIP preset (ImageNet normalization, 896x896).
    public static let gemma3 = ImagePreprocessor(
        targetSize: CGSize(width: 896, height: 896),
        mean: (0.485, 0.456, 0.406),
        std: (0.229, 0.224, 0.225),
        rescaleFactor: 1.0
    )

    /// CLIP preset (336x336).
    public static let clip = ImagePreprocessor(
        targetSize: CGSize(width: 336, height: 336),
        mean: (0.48145466, 0.4578275, 0.40821073),
        std: (0.26862954, 0.26130258, 0.27577711),
        rescaleFactor: 1.0
    )

    /// Preprocess an image from a file URL.
    ///
    /// - Parameter imageURL: URL to an image file (JPEG, PNG, HEIC, etc.)
    /// - Returns: Tuple of (Float32 RGBA pixel data, width, height)
    /// - Throws: ``ImagePreprocessorError/loadFailed(_:)`` if the image cannot be loaded
    public func preprocess(imageURL: URL) throws -> (Data, Int, Int) {
        guard let ciImage = CIImage(contentsOf: imageURL) else {
            throw ImagePreprocessorError.loadFailed(imageURL)
        }
        return try preprocess(image: ciImage)
    }

    /// Preprocess a CIImage.
    ///
    /// - Parameter image: Source CIImage
    /// - Returns: Tuple of (Float32 RGBA pixel data, width, height)
    /// - Throws: ``ImagePreprocessorError/renderFailed`` if rendering fails
    public func preprocess(image: CIImage) throws -> (Data, Int, Int) {
        guard let cgImage = Self.ciContext.createCGImage(image, from: image.extent) else {
            throw ImagePreprocessorError.renderFailed
        }
        return try preprocess(cgImage: cgImage)
    }

    /// Preprocess a CGImage. Draws into an sRGB RGBA8 CGContext (resize),
    /// then per-pixel rescale/normalize into a Float32 RGBA buffer.
    public func preprocess(cgImage: CGImage) throws -> (Data, Int, Int) {
        let w = Int(targetSize.width)
        let h = Int(targetSize.height)
        let resized = try renderToContext(
            cgImage: cgImage, canvasWidth: w, canvasHeight: h,
            drawRect: CGRect(x: 0, y: 0, width: w, height: h))
        return try normalize(pixels: resized, width: w, height: h)
    }

    /// Preprocess a CGImage and return a flat CHW `[C, H, W]` Float32 array.
    ///
    /// Convenience wrapper over ``preprocess(cgImage:)`` that transposes the
    /// NHWC RGBA output into the planar `[3, H, W]` layout expected by most
    /// vision encoder inputs.
    public func preprocessCHW(cgImage: CGImage) throws -> [Float] {
        let (data, w, h) = try preprocess(cgImage: cgImage)
        let pixelCount = w * h
        var chw = [Float](repeating: 0, count: 3 * pixelCount)
        data.withUnsafeBytes { rawSrc in
            let src = rawSrc.bindMemory(to: Float.self)
            for c in 0..<3 {
                for i in 0..<pixelCount {
                    chw[c * pixelCount + i] = src[i * 4 + c]
                }
            }
        }
        return chw
    }

    /// Preprocess with center-crop strategy: resize shortest edge to target,
    /// then center-crop to square. Returns flat CHW `[3, H, W]` Float32.
    public func preprocessCHWCenterCrop(cgImage: CGImage) throws -> [Float] {
        let targetW = Int(targetSize.width)
        let targetH = Int(targetSize.height)
        let srcW = cgImage.width
        let srcH = cgImage.height

        let scale =
            srcW < srcH
            ? CGFloat(targetW) / CGFloat(srcW)
            : CGFloat(targetH) / CGFloat(srcH)
        let resizedW = Int(round(CGFloat(srcW) * scale))
        let resizedH = Int(round(CGFloat(srcH) * scale))

        let resized = try renderToContext(
            cgImage: cgImage, canvasWidth: resizedW, canvasHeight: resizedH,
            drawRect: CGRect(x: 0, y: 0, width: resizedW, height: resizedH))

        let cropX = (resizedW - targetW) / 2
        let cropY = (resizedH - targetH) / 2
        guard let cropped = resized.cropping(to: CGRect(x: cropX, y: cropY, width: targetW, height: targetH)) else {
            throw ImagePreprocessorError.renderFailed
        }

        return try preprocessCHW(cgImage: cropped)
    }

    /// Preprocess with pad strategy: resize longest edge to target, zero-pad
    /// the shorter dimension. Returns flat CHW `[3, H, W]` Float32.
    public func preprocessCHWPad(cgImage: CGImage) throws -> [Float] {
        let targetW = Int(targetSize.width)
        let targetH = Int(targetSize.height)
        let srcW = cgImage.width
        let srcH = cgImage.height

        let scale =
            srcW > srcH
            ? CGFloat(targetW) / CGFloat(srcW)
            : CGFloat(targetH) / CGFloat(srcH)
        let resizedW = Int(round(CGFloat(srcW) * scale))
        let resizedH = Int(round(CGFloat(srcH) * scale))

        let offsetX = (targetW - resizedW) / 2
        let offsetY = (targetH - resizedH) / 2

        let padded = try renderToContext(
            cgImage: cgImage, canvasWidth: targetW, canvasHeight: targetH,
            drawRect: CGRect(x: offsetX, y: offsetY, width: resizedW, height: resizedH))

        return try preprocessCHW(cgImage: padded)
    }

    /// Dispatch preprocessing based on strategy. Returns flat CHW `[3, H, W]`.
    public func preprocessCHW(cgImage: CGImage, strategy: ImageStrategy) throws -> [Float] {
        switch strategy {
        case .stretch: return try preprocessCHW(cgImage: cgImage)
        case .centerCrop: return try preprocessCHWCenterCrop(cgImage: cgImage)
        case .pad: return try preprocessCHWPad(cgImage: cgImage)
        }
    }

    // MARK: - Private Helpers

    private func renderToContext(
        cgImage: CGImage, canvasWidth: Int, canvasHeight: Int,
        drawRect: CGRect
    ) throws -> CGImage {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw ImagePreprocessorError.renderFailed
        }
        guard
            let ctx = CGContext(
                data: nil,
                width: canvasWidth,
                height: canvasHeight,
                bitsPerComponent: 8,
                bytesPerRow: canvasWidth * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )
        else {
            throw ImagePreprocessorError.renderFailed
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: drawRect)
        guard let result = ctx.makeImage() else {
            throw ImagePreprocessorError.renderFailed
        }
        return result
    }

    private func normalize(pixels cgImage: CGImage, width w: Int, height h: Int) throws -> (Data, Int, Int) {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw ImagePreprocessorError.renderFailed
        }
        guard
            let ctx = CGContext(
                data: nil,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: w * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )
        else {
            throw ImagePreprocessorError.renderFailed
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        guard let pixelData = ctx.data else {
            throw ImagePreprocessorError.renderFailed
        }
        let rawPixels = pixelData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        let pixelCount = w * h
        let bytesPerPixel = 4 * MemoryLayout<Float>.size
        var data = Data(count: pixelCount * bytesPerPixel)
        data.withUnsafeMutableBytes { dstPtr in
            guard let dstBase = dstPtr.bindMemory(to: Float.self).baseAddress else { return }

            let scale = Float(rescaleFactor) / 255.0
            let means: [Float] = [Float(mean.0), Float(mean.1), Float(mean.2)]
            let stds: [Float] = [Float(std.0), Float(std.1), Float(std.2)]

            var channel = [Float](repeating: 0, count: pixelCount)
            let n = vDSP_Length(pixelCount)
            for c in 0..<3 {
                var a = scale / stds[c]
                var b = -means[c] / stds[c]
                vDSP_vfltu8(rawPixels.advanced(by: c), 4, &channel, 1, n)
                vDSP_vsmsa(channel, 1, &a, &b, dstBase.advanced(by: c), 4, n)
            }

            var zero: Float = 0
            vDSP_vfill(&zero, dstBase.advanced(by: 3), 4, n)
        }

        return (data, w, h)
    }
}

// MARK: - Errors

public enum ImagePreprocessorError: Error, LocalizedError {
    case loadFailed(URL)
    case renderFailed

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let url):
            return "Failed to load image from: \(url.path)"
        case .renderFailed:
            return "Failed to render preprocessed image"
        }
    }
}
