// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// How to extract frames from a video for vision encoding.
public enum FrameSamplingStrategy: Sendable {
    /// N evenly-spaced frames across the video duration.
    case uniform(count: Int)
    /// One frame every `rate` seconds, capped at `maxFrames`.
    case fps(rate: Double, maxFrames: Int)

    /// Return the same strategy with the frame count overridden.
    public func withFrameCount(_ count: Int) -> FrameSamplingStrategy {
        switch self {
        case .uniform: .uniform(count: count)
        case .fps(let rate, _): .fps(rate: rate, maxFrames: count)
        }
    }

    /// Compute sample times (in seconds) for a video of the given duration.
    /// Each time targets the midpoint of its sampling interval.
    ///
    /// - Parameters:
    ///   - duration: Video duration in seconds.
    ///   - videoFrameRate: Native frame rate of the video (e.g. 30.0).
    /// - Returns: Array of sample times in seconds.
    public func sampleTimes(forDuration duration: Double, videoFrameRate: Double) -> [Double] {
        guard duration > 0, videoFrameRate > 0 else { return [] }

        let totalFrames = Int(duration * videoFrameRate)
        guard totalFrames > 0 else { return [] }

        switch self {
        case .uniform(let count):
            let n = max(1, min(count, totalFrames))
            if n == 1 {
                return [duration / 2.0]
            }
            let step = duration / Double(n)
            return (0..<n).map { step * Double($0) + step / 2.0 }

        case .fps(let rate, let maxFrames):
            guard rate > 0, maxFrames > 0 else { return [] }
            let interval = 1.0 / rate
            var times: [Double] = []
            var t = interval / 2.0
            while t < duration && times.count < maxFrames {
                times.append(t)
                t += interval
            }
            if times.isEmpty {
                times.append(duration / 2.0)
            }
            return times
        }
    }
}
