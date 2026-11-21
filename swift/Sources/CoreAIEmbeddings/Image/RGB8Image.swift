// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreGraphics
import Foundation

/// An 8-bit RGB image, rows top to bottom, pixels interleaved `(r, g, b)` (HWC).
struct RGB8Image: Sendable, Equatable {
    var width: Int
    var height: Int
    /// `height * width * 3` bytes.
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(width > 0 && height > 0 && pixels.count == width * height * 3, "RGB8Image size mismatch")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Decodes `image` as `PIL.Image.convert("RGB")` does: the stored samples without color
    /// management (RGB, gray and palette images are relabeled sRGB and gray, so decoding converts
    /// nothing), alpha dropped without compositing (premultiplied alpha is undone first), gray
    /// replicated, 16-bit samples cut to their high byte, and 16-bit gray without alpha clipped to
    /// 255. Other color spaces are converted to sRGB. JPEG and CMYK inputs can differ from PIL by a
    /// few levels, because ImageIO and PIL decode them differently.
    // TODO: ImageIO's JPEG decode differs from HF's (PIL / libjpeg-turbo): chroma upsampling, a luma
    // filter below quality 90, and IDCT rounding. Image embeddings land at cos 0.989–0.9997 vs HF for
    // JPEG inputs. Add an exact libjpeg-turbo-equivalent decode if parity is needed.
    init(cgImage image: CGImage) throws {
        let (width, height) = (image.width, image.height)
        guard width > 0, height > 0 else { throw EmbeddingError.imageConversionFailed }
        if let gray = Self.clipped16BitGray(image) {
            self.init(width: width, height: height, pixels: gray)
            return
        }
        let model = image.colorSpace?.model
        let gray = model == .monochrome
        let space = CGColorSpace(name: gray ? CGColorSpace.genericGrayGamma2_2 : CGColorSpace.sRGB)!
        var label: CGColorSpace? = model == .rgb || gray ? space : nil
        if model == .indexed, image.colorSpace?.baseColorSpace?.model == .rgb, let table = image.colorSpace?.colorTable
        {
            label = CGColorSpace(indexedBaseSpace: space, last: table.count / 3 - 1, colorTable: table)
        }
        let source = label.flatMap { image.copy(colorSpace: $0) } ?? image
        // vImage unpremultiplies gray with different rounding than RGB, so gray is unpremultiplied here.
        let premultiplied =
            gray && image.bitsPerComponent == 8
            && [.premultipliedLast, .premultipliedFirst].contains(image.alphaInfo)
        // 16-bit samples are read as 16-bit and cut to their high byte (little-endian: the second).
        let (bytes, channels) = (image.bitsPerComponent > 8 ? 2 : 1, gray ? 2 : 4)
        let alpha: CGImageAlphaInfo = premultiplied ? .premultipliedLast : .last
        guard
            let format = vImage_CGImageFormat(
                bitsPerComponent: 8 * bytes, bitsPerPixel: 8 * bytes * channels, colorSpace: space,
                bitmapInfo: [CGBitmapInfo(rawValue: alpha.rawValue), bytes == 2 ? .byteOrder16Little : []]),
            var buffer = try? vImage_Buffer(cgImage: source, format: format)
        else { throw EmbeddingError.imageConversionFailed }
        defer { buffer.free() }
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            let row = (buffer.data + y * buffer.rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width * 3 {
                var v = Int(row[(x / 3 * channels + (gray ? 0 : x % 3)) * bytes + bytes - 1])
                if premultiplied {
                    let a = Int(row[x / 3 * 2 + 1])
                    v = a == 0 ? 0 : min(255, (v * 255 + a / 2) / a)
                }
                pixels[y * width * 3 + x] = UInt8(v)
            }
        }
        self.init(width: width, height: height, pixels: pixels)
    }

    /// PIL reads 16-bit gray without alpha as `I;16`, which `convert("RGB")` clips to 255.
    private static func clipped16BitGray(_ image: CGImage) -> [UInt8]? {
        guard image.colorSpace?.model == .monochrome, image.bitsPerPixel == 16, image.bitsPerComponent == 16,
            !image.bitmapInfo.contains(.floatComponents), let data = image.dataProvider?.data as Data?
        else { return nil }
        let little = image.byteOrderInfo == .order16Little
        return (0..<image.height).flatMap { y in
            (0..<image.width).flatMap { x in
                let i = y * image.bytesPerRow + x * 2
                let (hi, lo) = little ? (data[i + 1], data[i]) : (data[i], data[i + 1])
                return [UInt8](repeating: UInt8(min(Int(hi) << 8 | Int(lo), 255)), count: 3)
            }
        }
    }
}
