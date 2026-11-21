// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIEmbeddings

@Suite struct ImageTests {
    /// FNV-1a hashes of torch's `interpolate(uint8, mode="bicubic", antialias=True)` on a wrapping
    /// gradient: downscale, upscale, and one axis only.
    @Test func resizeMatchesTorch() {
        let cases: [(width: Int, height: Int, toWidth: Int, toHeight: Int, hash: UInt64)] = [
            (640, 480, 298, 224, 0x3564_6b67_93f0_0a45), (333, 200, 372, 224, 0xf3c8_9b92_453f_9193),
            (31, 17, 96, 48, 0xb321_8e9f_302d_7685), (300, 200, 300, 96, 0xecaf_abcd_05d0_8d8f),
            (300, 200, 113, 200, 0xdb7a_6b36_a522_55d8),
        ]
        for c in cases {
            let pixels = (0..<c.height).flatMap { y in
                (0..<c.width * 3).map { i in UInt8((i / 3 * 7 + y * 13 + i % 3 * 85 + i / 3 * y / 5) & 255) }
            }
            let image = RGB8Image(width: c.width, height: c.height, pixels: pixels)
            let resized = AntialiasedBicubicResize.resize(image, toWidth: c.toWidth, toHeight: c.toHeight)
            let hash = resized.pixels.reduce(0xcbf2_9ce4_8422_2325 as UInt64) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 }
            #expect(hash == c.hash, "\(c.width)x\(c.height) -> \(c.toWidth)x\(c.toHeight)")
        }
        #expect(ImagePreparer.shortestEdgeSize(width: 333, height: 200, to: 224) == (372, 224))
    }

    /// Samples come through as stored: no color conversion (also for a palette over Display P3),
    /// alpha dropped unblended (premultiplied gray undone first), gray replicated, 16-bit RGB(A) cut to its high byte, and 16-bit gray clipped to 255
    /// without alpha (PIL's `I;16`) but cut to its high byte with alpha.
    @Test func decodeKeepsStoredSamples() throws {
        let p3 = try decode(
            [255, 0, 0, 255, 10, 200, 30, 128, 1, 2, 3, 0], bits: 8, channels: 4,
            CGColorSpace(name: CGColorSpace.displayP3)!)
        #expect(p3.pixels == [255, 0, 0, 10, 200, 30, 1, 2, 3])
        let table: [UInt8] = [255, 0, 0, 10, 200, 30, 128, 64, 250]
        let palette = CGColorSpace(
            indexedBaseSpace: CGColorSpace(name: CGColorSpace.displayP3)!, last: 2, colorTable: table)
        #expect(try decode([0, 1, 2], bits: 8, channels: 1, palette!).pixels == table)
        let premultiplied = try decode(
            [100, 200, 50, 128, 10, 0, 3, 7], bits: 8, channels: 2, alpha: .premultipliedLast)
        #expect(premultiplied.pixels == [128, 100, 0, 109].flatMap { [UInt8](repeating: $0, count: 3) })
        let gray: [UInt16] = [0x0000, 0x00ff, 0x0100, 0x1234]
        let clipped = try decode(gray.flatMap { [UInt8($0 >> 8), UInt8($0 & 255)] }, bits: 16, channels: 1)
        #expect(clipped.pixels == [0, 255, 255, 255].flatMap { [UInt8](repeating: $0, count: 3) })
        let withAlpha = try decode(gray.flatMap { [UInt8($0 >> 8), UInt8($0 & 255), 255, 255] }, bits: 16, channels: 2)
        #expect(withAlpha.pixels == [0, 0, 1, 0x12].flatMap { [UInt8](repeating: $0, count: 3) })
        #expect(try decode([0, 77, 255], bits: 8, channels: 1).pixels == [0, 0, 0, 77, 77, 77, 255, 255, 255])
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let rgb16: [UInt16] = [0x1234, 0x00ff, 0xff01, 0x8000, 0x0100, 0xfffe]
        let big = rgb16.flatMap { [UInt8($0 >> 8), UInt8($0 & 255)] }
        #expect(try decode(big, bits: 16, channels: 3, srgb).pixels == [0x12, 0x00, 0xff, 0x80, 0x01, 0xff])
        let rgba16 = [big[0..<6], [0x80, 0x00], big[6..<12], [0xff, 0xff]].flatMap { $0 }
        #expect(try decode(rgba16, bits: 16, channels: 4, srgb).pixels == [0x12, 0x00, 0xff, 0x80, 0x01, 0xff])
    }

    /// One row of big-endian samples, alpha last (by default unpremultiplied) when `channels` is 2 or 4.
    private func decode(
        _ bytes: [UInt8], bits: Int, channels: Int,
        _ space: CGColorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!, alpha: CGImageAlphaInfo? = nil
    ) throws -> RGB8Image {
        let alpha = alpha ?? (channels % 2 == 0 ? .last : .none)
        let image = try #require(
            CGImage(
                width: bytes.count * 8 / (bits * channels), height: 1, bitsPerComponent: bits,
                bitsPerPixel: bits * channels, bytesPerRow: bytes.count, space: space,
                bitmapInfo: [CGBitmapInfo(rawValue: alpha.rawValue), bits == 16 ? .byteOrder16Big : []],
                provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent))
        return try RGB8Image(cgImage: image)
    }
}
