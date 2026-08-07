// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIShared

@Suite("Frame Sampling Strategy")
struct FrameSamplingStrategyTests {
    @Test("Uniform sampling with 8 frames from 30fps 10s video")
    func uniformSamplingBasic() {
        let timestamps = FrameSamplingStrategy.uniform(count: 8)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 8)
        // Frames should be evenly spaced
        for i in 1..<timestamps.count {
            #expect(timestamps[i] > timestamps[i - 1])
        }
        // All within duration
        for t in timestamps {
            #expect(t >= 0 && t < 10.0)
        }
    }

    @Test("Uniform sampling with single frame returns midpoint")
    func uniformSamplingSingleFrame() {
        let timestamps = FrameSamplingStrategy.uniform(count: 1)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 1)
        #expect(timestamps[0] == 5.0)
    }

    @Test("Uniform sampling clamps to available frames")
    func uniformSamplingMoreThanAvailable() {
        // 5 frames total at 1fps for 5 seconds
        let timestamps = FrameSamplingStrategy.uniform(count: 16)
            .sampleTimes(forDuration: 5.0, videoFrameRate: 1.0)
        #expect(timestamps.count == 5)
    }

    @Test("Uniform sampling with zero count gives 1 frame")
    func uniformSamplingZeroCount() {
        let timestamps = FrameSamplingStrategy.uniform(count: 0)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 1)
    }

    @Test("Uniform sampling with zero duration returns empty")
    func uniformSamplingZeroDuration() {
        let timestamps = FrameSamplingStrategy.uniform(count: 8)
            .sampleTimes(forDuration: 0.0, videoFrameRate: 30.0)
        #expect(timestamps.isEmpty)
    }

    @Test("Uniform sampling with negative duration returns empty")
    func uniformSamplingNegativeDuration() {
        let timestamps = FrameSamplingStrategy.uniform(count: 8)
            .sampleTimes(forDuration: -5.0, videoFrameRate: 30.0)
        #expect(timestamps.isEmpty)
    }

    @Test("FPS-based sampling at 1fps from 10s video")
    func fpsSampling() {
        let timestamps = FrameSamplingStrategy.fps(rate: 1.0, maxFrames: 20)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 10)
        for t in timestamps {
            #expect(t >= 0 && t < 10.0)
        }
    }

    @Test("FPS-based sampling caps at maxFrames")
    func fpsSamplingCapped() {
        let timestamps = FrameSamplingStrategy.fps(rate: 1.0, maxFrames: 5)
            .sampleTimes(forDuration: 60.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 5)
    }

    @Test("FPS-based sampling at 2fps")
    func fpsSamplingHighRate() {
        let timestamps = FrameSamplingStrategy.fps(rate: 2.0, maxFrames: 100)
            .sampleTimes(forDuration: 5.0, videoFrameRate: 30.0)
        #expect(timestamps.count == 10)
    }

    @Test("FPS-based sampling with zero rate returns empty")
    func fpsSamplingZeroRate() {
        let timestamps = FrameSamplingStrategy.fps(rate: 0.0, maxFrames: 10)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.isEmpty)
    }

    @Test("FPS-based sampling with zero maxFrames returns empty")
    func fpsSamplingZeroMaxFrames() {
        let timestamps = FrameSamplingStrategy.fps(rate: 1.0, maxFrames: 0)
            .sampleTimes(forDuration: 10.0, videoFrameRate: 30.0)
        #expect(timestamps.isEmpty)
    }

    @Test("FPS-based sampling short video returns at least 1 frame")
    func fpsSamplingShortVideo() {
        let timestamps = FrameSamplingStrategy.fps(rate: 1.0, maxFrames: 10)
            .sampleTimes(forDuration: 0.5, videoFrameRate: 30.0)
        // 0.5s video at 1fps: interval center at 0.5 is outside, fallback gives 1 frame
        #expect(timestamps.count == 1)
    }
}

@Suite("VideoInput")
struct VideoInputTests {
    @Test("fromFrames wraps CGImages correctly")
    func fromFrames() async throws {
        let images = (0..<3).map { _ in makeSolidImage(width: 64, height: 64) }
        let input = VideoInput.fromFrames(images)
        #expect(input.frameCount == 3)
        #expect(input.duration == nil)

        var count = 0
        for try await _ in input.frames {
            count += 1
        }
        #expect(count == 3)
    }

    @Test("fromFrames with empty array gives zero frames")
    func fromFramesEmpty() async throws {
        let input = VideoInput.fromFrames([])
        #expect(input.frameCount == 0)

        var count = 0
        for try await _ in input.frames {
            count += 1
        }
        #expect(count == 0)
    }
}

// MARK: - Test Helpers

private func makeSolidImage(width: Int, height: Int) -> CGImage {
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    )!
    ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return ctx.makeImage()!
}
