// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation

/// Decodes every frame of a video in order.
///
/// Useful for video segmentation models that perform tracking across the whole sequence.
public struct SequentialVideoReader: Sendable {
    /// What the container says about the track, read before any frame is decoded.
    public struct Metadata: Sendable {
        public let width: Int
        public let height: Int
        public let nominalFrameRate: Double
        /// Frame count implied by duration × frame rate. Approximate: containers round,
        /// and variable-frame-rate media has no single answer. Use it for progress
        /// reporting and preallocation, not for indexing.
        public let estimatedFrameCount: Int
        public let duration: Double
    }

    /// Read the track's dimensions and frame rate without decoding.
    public static func metadata(of url: URL) async throws -> Metadata {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw VideoInputError.fileNotFound(url)
        }
        let asset = AVURLAsset(url: url)
        let duration = try await CMTimeGetSeconds(asset.load(.duration))
        guard duration > 0, duration.isFinite else {
            throw VideoInputError.invalidVideo("Video has zero or invalid duration")
        }
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoInputError.noVideoTrack
        }
        let (naturalSize, transform, frameRate) = try await track.load(
            .naturalSize, .preferredTransform, .nominalFrameRate)
        let displayed = naturalSize.applying(transform)
        let width = Int(abs(displayed.width).rounded())
        let height = Int(abs(displayed.height).rounded())
        guard width > 0, height > 0 else {
            throw VideoInputError.invalidVideo("Video track has zero dimensions")
        }
        let fps = Double(frameRate)
        guard fps > 0 else {
            throw VideoInputError.invalidVideo("Video has invalid frame rate")
        }
        return Metadata(
            width: width,
            height: height,
            nominalFrameRate: fps,
            estimatedFrameCount: max(1, Int((duration * fps).rounded())),
            duration: duration
        )
    }

    /// Stream frames in presentation order, stopping after `maxFrames` if given.
    ///
    /// Each `VideoFrame` owns its pixels: the decoded buffer is copied into a standalone
    /// `CGImage` inside the scope that lends it, so the caller can hold frames past the
    /// point where the decoder recycles it.
    public static func frames(
        of url: URL,
        maxFrames: Int? = nil
    ) -> VideoFrameSequence {
        VideoFrameSequence(
            stream: AsyncThrowingStream<VideoFrame, Error> { continuation in
                let task = Task {
                    do {
                        try await decode(url: url, maxFrames: maxFrames, into: continuation)
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            })
    }

    private static func decode(
        url: URL,
        maxFrames: Int?,
        into continuation: AsyncThrowingStream<VideoFrame, Error>.Continuation
    ) async throws {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoInputError.noVideoTrack
        }
        let transform = try await track.load(.preferredTransform)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        let provider = reader.outputProvider(for: output)
        try reader.start()
        defer { reader.cancelReading() }

        var index = 0
        while let sample = try await provider.next() {
            if Task.isCancelled { return }
            if let maxFrames, index >= maxFrames { return }
            guard case .pixelBuffer(let readOnly) = sample.content else { continue }
            let decoded = readOnly.withUnsafeBuffer { makeCGImage(from: $0) }
            guard let decoded else {
                throw VideoInputError.frameExtractionFailed(
                    underlying: VideoInputError.invalidVideo("frame \(index) failed to convert"))
            }
            // The track stores pixels in recording orientation and carries the rotation
            // as a transform.
            let image = transform.isIdentity ? decoded : applying(transform, to: decoded)
            guard let image else {
                throw VideoInputError.frameExtractionFailed(
                    underlying: VideoInputError.invalidVideo("frame \(index) failed to rotate"))
            }
            continuation.yield(VideoFrame(image: image, index: index))
            index += 1
        }
    }

    /// Redraw `image` through `transform`, sized to the transformed bounds.
    private static func applying(_ transform: CGAffineTransform, to image: CGImage) -> CGImage? {
        let source = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let target = source.applying(transform)
        let width = Int(abs(target.width).rounded())
        let height = Int(abs(target.height).rounded())
        guard width > 0, height > 0,
            let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        // `applying` can move the rect off-origin (a 90° rotation puts it in a negative
        // quadrant); translate it back before drawing.
        context.translateBy(x: -target.origin.x, y: -target.origin.y)
        context.concatenate(transform)
        context.draw(image, in: source)
        return context.makeImage()
    }

    /// Copy a BGRA pixel buffer into a standalone `CGImage`.
    private static func makeCGImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let data = Data(bytes: base, count: bytesPerRow * height)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        // BGRA in memory is byteOrder32Little with the alpha byte first.
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
