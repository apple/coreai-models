// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics

/// Result of a video generation pipeline run, containing the decoded frames and target FPS.
public struct VideoGenerationResult: Sendable {
    public let frames: [CGImage]
    public let fps: Int
    public init(frames: [CGImage], fps: Int) {
        self.frames = frames
        self.fps = fps
    }
}

/// Progress update emitted during video generation.
public struct VideoProgress: Sendable {
    public let step: Int
    public let totalSteps: Int
    public let phase: Phase

    public enum Phase: String, Sendable {
        case encoding
        case denoising
        case decoding
        case assembling
    }

    public init(step: Int, totalSteps: Int, phase: Phase = .denoising) {
        self.step = step
        self.totalSteps = totalSteps
        self.phase = phase
    }
}

/// A video generation pipeline that produces frames from a text prompt.
public protocol VideoPipeline {
    /// Default output resolution for this pipeline's model.
    var defaultVideoSize: (width: Int, height: Int) { get }

    /// Default number of frames the pipeline produces.
    var defaultFrameCount: Int { get }

    /// Generate video frames from the given configuration.
    ///
    /// The progress handler is called periodically and may return `false` to cancel.
    func generateVideo(
        configuration: VideoConfiguration,
        progressHandler: @Sendable (VideoProgress) -> Bool
    ) async throws -> VideoGenerationResult
}
