// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import CoreImage
import Foundation
import Testing

@testable import CoreAIShared

@Suite("ImagePreprocessor")
struct ImagePreprocessorTests {
    /// Build a synthetic CGImage with a known constant color so we can
    /// verify the rescale + normalize math without depending on a file.
    ///
    /// Uses raw bytes (not `CGColor`+`setFillColor`) so the byte values
    /// hit the preprocessor exactly as written — `CGColor(red:green:blue:alpha:)`
    /// is in device RGB, which can shift values when drawn into an sRGB
    /// context on CI machines with different colour profiles.
    private static func makeSolidImage(
        width: Int, height: Int,
        r: UInt8, g: UInt8, b: UInt8
    ) -> CGImage {
        let pixelCount = width * height
        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
        for i in 0..<pixelCount {
            rgba[i * 4 + 0] = r
            rgba[i * 4 + 1] = g
            rgba[i * 4 + 2] = b
            rgba[i * 4 + 3] = 255
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    @Test("Output buffer has the expected size and shape")
    func outputShape() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 32, height: 32),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let img = Self.makeSolidImage(width: 64, height: 64, r: 0, g: 0, b: 0)

        let (data, w, h) = try pre.preprocess(cgImage: img)

        #expect(w == 32)
        #expect(h == 32)
        #expect(data.count == 32 * 32 * 4 * MemoryLayout<Float>.size)
    }

    @Test("Normalization matches (pixel/255 * rescale - mean) / std")
    func normalizationMath() throws {
        // Mean=0, std=1, rescale=1 → output value = pixel / 255
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 1, height: 1),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let img = Self.makeSolidImage(width: 1, height: 1, r: 255, g: 128, b: 0)

        let (data, _, _) = try pre.preprocess(cgImage: img)
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        // RGBA, single pixel → 4 floats
        #expect(floats.count == 4)
        #expect(abs(floats[0] - 1.0) < 1e-5)  // R: 255/255 = 1.0
        #expect(abs(floats[1] - 128.0 / 255.0) < 1e-5)  // G
        #expect(abs(floats[2] - 0.0) < 1e-5)  // B
        #expect(floats[3] == 0)  // alpha unused
    }

    @Test("ImageNet normalization (gemma3 preset)")
    func imageNetNormalization() throws {
        // Solid mid-gray (128, 128, 128) at 1x1, ImageNet mean/std
        let img = Self.makeSolidImage(width: 1, height: 1, r: 128, g: 128, b: 128)
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 1, height: 1),
            mean: (0.485, 0.456, 0.406),
            std: (0.229, 0.224, 0.225),
            rescaleFactor: 1
        )
        let (data, _, _) = try pre.preprocess(cgImage: img)
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        let pixel = Float(128.0 / 255.0)
        let expectedR = (pixel - 0.485) / 0.229
        let expectedG = (pixel - 0.456) / 0.224
        let expectedB = (pixel - 0.406) / 0.225

        #expect(abs(floats[0] - expectedR) < 1e-4)
        #expect(abs(floats[1] - expectedG) < 1e-4)
        #expect(abs(floats[2] - expectedB) < 1e-4)
    }

    @Test("loadFailed thrown for missing file")
    func loadFailedForMissingFile() {
        let pre = ImagePreprocessor.gemma3
        let missing = URL(fileURLWithPath: "/nonexistent/image.png")
        #expect(throws: ImagePreprocessorError.self) {
            _ = try pre.preprocess(imageURL: missing)
        }
    }

    @Test("Preset constants are stable")
    func presetConstants() {
        #expect(ImagePreprocessor.gemma3.targetSize == CGSize(width: 896, height: 896))
        #expect(ImagePreprocessor.clip.targetSize == CGSize(width: 336, height: 336))
        #expect(ImagePreprocessor.gemma3.mean.0 == 0.485)
        #expect(ImagePreprocessor.clip.mean.0 == 0.48145466)
    }

    @Test("Multi-pixel stride: per-pixel and per-channel values are independent")
    func multiPixelStride() throws {
        // 4x4 image with distinct RGBA per pixel — exercises the vectorized
        // (stride 4) read/write path. A stride bug or channel mix-up would
        // produce wrong values for at least one pixel.
        let w = 4
        let h = 4
        let pixelCount = w * h

        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
        for i in 0..<pixelCount {
            rgba[i * 4 + 0] = UInt8(i * 4)  // R: 0, 4, 8, ..., 60
            rgba[i * 4 + 1] = UInt8(i * 8)  // G: 0, 8, ..., 120
            rgba[i * 4 + 2] = UInt8(min(255, i * 12))  // B: 0, 12, ..., 180
            rgba[i * 4 + 3] = 255
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let cgImage = CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )!

        // Identity preprocessor — same size, mean=0, std=1 → output = pixel/255
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: w, height: h),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )

        let (data, ow, oh) = try pre.preprocess(cgImage: cgImage)
        #expect(ow == w)
        #expect(oh == h)

        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(floats.count == pixelCount * 4)

        for i in 0..<pixelCount {
            let expR = Float(rgba[i * 4 + 0]) / 255.0
            let expG = Float(rgba[i * 4 + 1]) / 255.0
            let expB = Float(rgba[i * 4 + 2]) / 255.0
            #expect(abs(floats[i * 4 + 0] - expR) < 1e-4)
            #expect(abs(floats[i * 4 + 1] - expG) < 1e-4)
            #expect(abs(floats[i * 4 + 2] - expB) < 1e-4)
            #expect(floats[i * 4 + 3] == 0)
        }
    }

    @Test("Upscale: small input image to larger targetSize")
    func upscaleSmallerInput() throws {
        // 2x2 solid red source → 8x8 target. Verifies the resize path
        // produces the right output size and that all pixels see the
        // (resized) color, not zeros from any stride bug.
        let src = Self.makeSolidImage(width: 2, height: 2, r: 200, g: 100, b: 50)
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 8, height: 8),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )

        let (data, w, h) = try pre.preprocess(cgImage: src)
        #expect(w == 8)
        #expect(h == 8)

        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(floats.count == 8 * 8 * 4)

        let expR = Float(200) / 255.0
        let expG = Float(100) / 255.0
        let expB = Float(50) / 255.0

        // All 64 pixels should see (approximately) the source color.
        // Tolerance of 0.1 (~25/255) covers sRGB→linear→sRGB round-trip
        // gamma artifacts during the resize while still catching real bugs:
        // a channel mix-up or stride error would produce deltas of 0.5+.
        for i in 0..<(8 * 8) {
            #expect(abs(floats[i * 4 + 0] - expR) < 0.1)
            #expect(abs(floats[i * 4 + 1] - expG) < 0.1)
            #expect(abs(floats[i * 4 + 2] - expB) < 0.1)
            #expect(floats[i * 4 + 3] == 0)
        }
    }

    // MARK: - Image Strategy Tests

    @Test("center_crop: landscape image crops horizontally")
    func centerCropLandscape() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 32, height: 32),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let landscape = Self.makeSolidImage(width: 64, height: 32, r: 100, g: 150, b: 200)
        let chw = try pre.preprocessCHWCenterCrop(cgImage: landscape)
        #expect(chw.count == 3 * 32 * 32)
    }

    @Test("center_crop: portrait image crops vertically")
    func centerCropPortrait() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 32, height: 32),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let portrait = Self.makeSolidImage(width: 32, height: 64, r: 100, g: 150, b: 200)
        let chw = try pre.preprocessCHWCenterCrop(cgImage: portrait)
        #expect(chw.count == 3 * 32 * 32)
    }

    @Test("pad: landscape image has correct output size")
    func padLandscape() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 32, height: 32),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let landscape = Self.makeSolidImage(width: 64, height: 32, r: 100, g: 150, b: 200)
        let chw = try pre.preprocessCHWPad(cgImage: landscape)
        #expect(chw.count == 3 * 32 * 32)
    }

    @Test("pad: portrait image has correct output size")
    func padPortrait() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 32, height: 32),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let portrait = Self.makeSolidImage(width: 32, height: 64, r: 100, g: 150, b: 200)
        let chw = try pre.preprocessCHWPad(cgImage: portrait)
        #expect(chw.count == 3 * 32 * 32)
    }

    @Test("strategy dispatch: all three strategies produce correct output size")
    func strategyDispatch() throws {
        let pre = ImagePreprocessor(
            targetSize: CGSize(width: 16, height: 16),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1
        )
        let img = Self.makeSolidImage(width: 32, height: 16, r: 128, g: 128, b: 128)
        let expected = 3 * 16 * 16

        #expect(try pre.preprocessCHW(cgImage: img, strategy: .stretch).count == expected)
        #expect(try pre.preprocessCHW(cgImage: img, strategy: .centerCrop).count == expected)
        #expect(try pre.preprocessCHW(cgImage: img, strategy: .pad).count == expected)
    }
}
