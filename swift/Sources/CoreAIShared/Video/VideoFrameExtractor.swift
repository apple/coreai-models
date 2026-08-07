// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import CoreMedia

/// Extracts frames from video files using AVAssetImageGenerator.
struct VideoFrameExtractor {
    /// Extract frames from a video file according to the given sampling strategy.
    ///
    /// - Parameters:
    ///   - url: Local file URL to the video.
    ///   - sampling: How to sample frames from the video.
    ///   - skipErrors: When true, frames that fail to decode are skipped.
    ///     When false (default), the first failure throws.
    /// - Returns: Tuple of (frame count, duration in seconds, lazy frame sequence).
    /// - Throws: ``VideoInputError`` if the video cannot be read.
    static func extractFrames(
        from url: URL,
        sampling: FrameSamplingStrategy,
        skipErrors: Bool = false
    ) async throws -> (count: Int, duration: Double, frames: VideoFrameSequence) {
        let asset = AVURLAsset(url: url)
        let duration = try await CMTimeGetSeconds(asset.load(.duration))

        guard duration > 0, duration.isFinite else {
            throw VideoInputError.invalidVideo("Video has zero or invalid duration")
        }

        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoInputError.noVideoTrack
        }

        let frameRate = try await Double(videoTrack.load(.nominalFrameRate))
        guard frameRate > 0 else {
            throw VideoInputError.invalidVideo("Video has invalid frame rate")
        }

        let sampleTimes = sampling.sampleTimes(forDuration: duration, videoFrameRate: frameRate)
        guard !sampleTimes.isEmpty else {
            throw VideoInputError.invalidVideo("No frames to extract")
        }

        let count = sampleTimes.count
        let cmTimes = sampleTimes.map {
            CMTime(seconds: $0, preferredTimescale: 600)
        }

        let seq = VideoFrameSequence(
            stream: AsyncThrowingStream<VideoFrame, Error> { continuation in
                Task {
                    let asset = AVURLAsset(url: url)
                    let generator = AVAssetImageGenerator(asset: asset)
                    generator.appliesPreferredTrackTransform = true
                    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
                    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

                    var frameIndex = 0
                    for await result in generator.images(for: cmTimes) {
                        do {
                            let image = try result.image
                            continuation.yield(VideoFrame(image: image, index: frameIndex))
                            frameIndex += 1
                        } catch {
                            if skipErrors {
                                continue
                            }
                            continuation.finish(
                                throwing: VideoInputError.frameExtractionFailed(underlying: error))
                            return
                        }
                    }
                    continuation.finish()
                }
            })

        return (count: count, duration: duration, frames: seq)
    }
}
