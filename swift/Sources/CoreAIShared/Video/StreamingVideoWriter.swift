// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation

/// Encodes frames to an MP4 one at a time, without holding the whole video in memory.
///
/// Frames must be appended in order. `finish()` must be called; deallocating without it
/// leaves a truncated file.
public actor StreamingVideoWriter {
    private let writer: AVAssetWriter
    private let receiver: AVAssetWriterInput.PixelBufferReceiver
    private let width: Int
    private let height: Int
    private let frameRate: Int
    private var nextFrameIndex = 0
    private var finished = false

    /// - Parameters:
    ///   - url: Destination. An existing file at this path is replaced.
    ///   - width: Frame width. Rounded up to even, which H.264 requires.
    ///   - height: Frame height. Rounded up to even.
    ///   - frameRate: Frames per second of the output timeline.
    ///   - codec: Defaults to H.264 for the widest player support.
    public init(
        url: URL,
        width: Int,
        height: Int,
        frameRate: Int,
        codec: AVVideoCodecType = .h264
    ) throws {
        guard width > 0, height > 0 else {
            throw VideoWriterError.invalidDimensions(width: width, height: height)
        }
        guard frameRate > 0 else {
            // Reachable from a CLI: `videodiffusion-runner --fps 0` is unvalidated upstream,
            // and a zero timescale would otherwise produce frames with an invalid `CMTime`.
            throw VideoWriterError.invalidFrameRate(frameRate)
        }
        // H.264 macroblocks are 16x16 and the encoder rejects odd dimensions outright.
        let evenWidth = width + (width % 2)
        let evenHeight = height + (height % 2)

        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: codec,
                AVVideoWidthKey: evenWidth,
                AVVideoHeightKey: evenHeight,
            ]
        )
        guard writer.canAdd(input) else {
            throw VideoWriterError.cannotCreateDestination
        }
        // Registers the input and hands back the surface we append through.
        let receiver = writer.inputPixelBufferReceiver(
            for: input,
            pixelBufferAttributes: CVPixelBufferCreationAttributes(
                pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32ARGB),
                size: CVImageSize(width: evenWidth, height: evenHeight)
            )
        )
        do {
            try writer.start()
        } catch {
            throw VideoWriterError.writeFailed(error)
        }
        writer.startSession(atSourceTime: .zero)

        self.writer = writer
        self.receiver = receiver
        self.width = evenWidth
        self.height = evenHeight
        self.frameRate = frameRate
    }

    /// Append the next frame. Scaled to the writer's dimensions if it doesn't match.
    public func append(_ image: CGImage) async throws {
        guard !finished else {
            throw VideoWriterError.writeAfterFinish
        }
        guard let pool = receiver.pixelBufferPool else {
            throw VideoWriterError.writeFailed(writer.error)
        }
        guard var buffer = try? pool.makeMutablePixelBuffer() else {
            throw VideoWriterError.cannotCreateDestination
        }

        // 32ARGB is interleaved, so there is one plane. Use the stride it reports rather
        // than `width * 4` — the pool pads rows, and 66px lands on 320 bytes, not 264.
        try buffer.accessUnsafeMutableRawPlaneBytes { planes in
            guard let plane = planes.first,
                let context = CGContext(
                    data: plane.bytes.baseAddress,
                    width: width, height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: plane.properties.bytesPerRow,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                )
            else {
                throw VideoWriterError.cannotCreateDestination
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        // Awaiting the append is the back-pressure: it suspends until the encoder has room.
        let time = CMTime(value: CMTimeValue(nextFrameIndex), timescale: CMTimeScale(frameRate))
        do {
            try await receiver.append(CVReadOnlyPixelBuffer(buffer), with: time)
        } catch {
            throw VideoWriterError.writeFailed(error)
        }
        nextFrameIndex += 1
    }

    /// Flush the encoder and close the file. Safe to call more than once.
    @discardableResult
    public func finish() async throws -> Int {
        guard !finished else { return nextFrameIndex }
        finished = true
        receiver.finish()
        await writer.finishWriting()
        if writer.status == .failed {
            throw VideoWriterError.writeFailed(writer.error)
        }
        return nextFrameIndex
    }

    /// Number of frames appended so far.
    public var frameCount: Int { nextFrameIndex }
}
