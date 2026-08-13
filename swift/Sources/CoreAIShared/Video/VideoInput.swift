// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// A frame produced by video extraction: the decoded image and its position.
public struct VideoFrame: Sendable {
    public let image: CGImage
    /// Frame index (0-based ordinal within the extraction sequence).
    public let index: Int
}

/// Concrete, Sendable async sequence of video frames.
///
/// Wraps an `AsyncThrowingStream` so that `VideoInput` can conform to `Sendable`
/// in Swift 6 strict concurrency (existential `any AsyncSequence` cannot).
public struct VideoFrameSequence: AsyncSequence, Sendable {
    public typealias Element = VideoFrame

    private let stream: AsyncThrowingStream<VideoFrame, Error>

    init(stream: AsyncThrowingStream<VideoFrame, Error>) {
        self.stream = stream
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(base: stream.makeAsyncIterator())
    }

    public struct Iterator: AsyncIteratorProtocol {
        var base: AsyncThrowingStream<VideoFrame, Error>.AsyncIterator

        public mutating func next() async throws -> VideoFrame? {
            try await base.next()
        }
    }
}

/// Default number of frames sampled from a video when no explicit count is provided.
public let defaultVideoFrameCount = 8

/// Extracted video frames ready for vision encoding.
///
/// Frames are delivered lazily via `VideoFrameSequence`. The engine processes
/// and releases each frame incrementally, keeping peak memory at 1-2
/// decoded frames plus accumulated embeddings.
public struct VideoInput: Sendable {
    /// Number of frames, if known ahead of time (nil for live streams).
    public let frameCount: Int?
    /// Video duration in seconds, if known (nil for live streams).
    public let duration: Double?
    /// Lazy frame sequence. Throws on extraction errors unless `skipErrors` was set.
    public let frames: VideoFrameSequence

    public init(
        frameCount: Int?,
        duration: Double?,
        frames: VideoFrameSequence
    ) {
        self.frameCount = frameCount
        self.duration = duration
        self.frames = frames
    }

    /// Extract frames from a local video file.
    ///
    /// - Parameters:
    ///   - url: File URL to a video (MP4, MOV, etc.).
    ///   - sampling: Frame sampling strategy (default: 8 uniform frames).
    ///   - skipErrors: When true, frames that fail to decode are skipped
    ///     instead of throwing. Default is false.
    /// - Throws: ``VideoInputError`` if the file cannot be read or has no video track.
    public static func fromURL(
        _ url: URL,
        sampling: FrameSamplingStrategy = .uniform(count: defaultVideoFrameCount),
        skipErrors: Bool = false
    ) async throws -> VideoInput {
        let (count, duration, frames) = try await VideoFrameExtractor.extractFrames(
            from: url, sampling: sampling, skipErrors: skipErrors)
        return VideoInput(frameCount: count, duration: duration, frames: frames)
    }

    /// Wrap pre-extracted frames (e.g. from camera capture).
    public static func fromFrames(
        _ images: [CGImage]
    ) -> VideoInput {
        let count = images.count
        let stream = AsyncThrowingStream<VideoFrame, Error> { continuation in
            for (i, image) in images.enumerated() {
                continuation.yield(VideoFrame(image: image, index: i))
            }
            continuation.finish()
        }
        return VideoInput(frameCount: count, duration: nil, frames: VideoFrameSequence(stream: stream))
    }
}

// MARK: - Errors

public enum VideoInputError: Error, LocalizedError {
    case invalidVideo(String)
    case noVideoTrack
    case fileNotFound(URL)
    case frameExtractionFailed(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .invalidVideo(let reason):
            return "Invalid video: \(reason)"
        case .noVideoTrack:
            return "File has no video track (may be audio-only)"
        case .fileNotFound(let url):
            return "Video file not found: \(url.path)"
        case .frameExtractionFailed(let underlying):
            return "Frame extraction failed: \(underlying.localizedDescription)"
        }
    }
}
