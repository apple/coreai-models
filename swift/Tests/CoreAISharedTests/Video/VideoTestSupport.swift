// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation

/// Frame fixtures and pixel sampling shared by the video reader and writer suites.
enum VideoTestSupport {
    /// A solid-colour frame, so codec loss can't be confused with a frame-ordering bug.
    static func solidFrame(width: Int, height: Int, grey: UInt8) -> CGImage {
        let context = makeContext(width: width, height: height)
        let level = CGFloat(grey) / 255
        context.setFillColor(red: level, green: level, blue: level, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    /// Black on the left, `grey` on the right. The vertical edge is the point: a wrong row
    /// stride in the writer shears each row sideways and turns that edge into a diagonal.
    static func splitFrame(width: Int, height: Int, grey: UInt8) -> CGImage {
        let context = makeContext(width: width, height: height)
        context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let level = CGFloat(grey) / 255
        context.setFillColor(red: level, green: level, blue: level, alpha: 1)
        context.fill(CGRect(x: width / 2, y: 0, width: width - width / 2, height: height))
        return context.makeImage()!
    }

    /// Mean of the RGB channels over a region. Omitted ranges cover the whole image.
    static func meanGrey(
        of image: CGImage, columns: Range<Int>? = nil, rows: Range<Int>? = nil
    ) -> Int {
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes { raw in
            let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        let xs = columns ?? 0..<width
        let ys = rows ?? 0..<height
        var total = 0
        for y in ys {
            for x in xs {
                let offset = y * width * 4 + x * 4
                total += Int(pixels[offset]) + Int(pixels[offset + 1]) + Int(pixels[offset + 2])
            }
        }
        return total / (ys.count * xs.count * 3)
    }

    /// Write a clip carrying a rotation flag in its track.
    ///
    /// `StreamingVideoWriter` deliberately has no transform knob — it writes frames the
    /// reader has already straightened — so a rotated fixture has to be built by hand.
    static func writeRotatedFixture(
        to url: URL, frames: [CGImage], width: Int, height: Int, frameRate: Int,
        transform: CGAffineTransform
    ) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
            ]
        )
        input.transform = transform
        let receiver = writer.inputPixelBufferReceiver(
            for: input,
            pixelBufferAttributes: CVPixelBufferCreationAttributes(
                pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32ARGB),
                size: CVImageSize(width: width, height: height)
            )
        )
        try writer.start()
        writer.startSession(atSourceTime: .zero)

        for (index, frame) in frames.enumerated() {
            var buffer = try receiver.pixelBufferPool!.makeMutablePixelBuffer()
            buffer.accessUnsafeMutableRawPlaneBytes { planes in
                let context = CGContext(
                    data: planes[0].bytes.baseAddress,
                    width: width, height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: planes[0].properties.bytesPerRow,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
                context.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
            let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(frameRate))
            try await receiver.append(CVReadOnlyPixelBuffer(buffer), with: time)
        }
        receiver.finish()
        await writer.finishWriting()
    }

    static func temporaryURL(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "coreai-\(label)-\(UUID().uuidString).mp4")
    }

    private static func makeContext(width: Int, height: Int) -> CGContext {
        CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    }
}
