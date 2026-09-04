// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation
import Testing

@testable import CoreAIShared

/// Covers ``SequentialVideoReader`` on its own terms.
///
/// The writer suite already round-trips frames through it, but only as the checking arm of a
/// writer test — so the reader's own edges (rotation, metadata arithmetic, refusing bad
/// input) were never exercised.
@Suite("SequentialVideoReader")
struct SequentialVideoReaderTests {
    // MARK: - Metadata

    @Test("Metadata reports the frame rate, duration, and implied frame count")
    func metadataArithmetic() async throws {
        let url = VideoTestSupport.temporaryURL("reader-meta")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 48, height: 32, frameRate: 10)
        for index in 0..<20 {
            try await writer.append(
                VideoTestSupport.solidFrame(width: 48, height: 32, grey: UInt8(10 + index * 8)))
        }
        try await writer.finish()

        let metadata = try await SequentialVideoReader.metadata(of: url)
        #expect(metadata.width == 48)
        #expect(metadata.height == 32)
        #expect(abs(metadata.nominalFrameRate - 10) < 0.5)
        // 20 frames at 10 fps. Containers round, so this is deliberately loose.
        #expect(abs(metadata.duration - 2.0) < 0.2, "duration was \(metadata.duration)")
        #expect(abs(metadata.estimatedFrameCount - 20) <= 2)
    }

    @Test("estimatedFrameCount is close to what the reader actually yields")
    func estimateTracksReality() async throws {
        // The estimate feeds progress reporting and preallocation. It is allowed to be off,
        // but not by so much that a caller sizing a buffer from it gets it badly wrong.
        let url = VideoTestSupport.temporaryURL("reader-estimate")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 15)
        for index in 0..<30 {
            try await writer.append(
                VideoTestSupport.solidFrame(width: 32, height: 32, grey: UInt8(index * 8)))
        }
        try await writer.finish()

        let metadata = try await SequentialVideoReader.metadata(of: url)
        var actual = 0
        for try await _ in SequentialVideoReader.frames(of: url) { actual += 1 }
        #expect(actual == 30)
        #expect(
            abs(metadata.estimatedFrameCount - actual) <= 2,
            "estimated \(metadata.estimatedFrameCount), decoded \(actual)")
    }

    // MARK: - Rotation

    @Test("A rotation flag is applied, so frames match the metadata dimensions")
    func appliesPreferredTransform() async throws {
        // The track stores 64x32 landscape pixels plus a 90 degree flag. Both `metadata` and
        // the decode path have to compose that transform, and independently — they read it
        // in different functions. If they disagree, a caller that sizes an output writer
        // from `metadata` gets frames that don't fit.
        let url = VideoTestSupport.temporaryURL("reader-rotated")
        defer { try? FileManager.default.removeItem(at: url) }

        let frames = (0..<4).map {
            VideoTestSupport.splitFrame(width: 64, height: 32, grey: UInt8(200 - $0 * 10))
        }
        try await VideoTestSupport.writeRotatedFixture(
            to: url, frames: frames, width: 64, height: 32, frameRate: 10,
            transform: CGAffineTransform(rotationAngle: .pi / 2))

        let metadata = try await SequentialVideoReader.metadata(of: url)
        #expect(metadata.width == 32, "rotated width was \(metadata.width)")
        #expect(metadata.height == 64, "rotated height was \(metadata.height)")

        var seen = 0
        for try await frame in SequentialVideoReader.frames(of: url) {
            seen += 1
            #expect(frame.image.width == metadata.width)
            #expect(frame.image.height == metadata.height)

            // The source splits left/right. After a quarter turn that edge is horizontal,
            // so the halves must differ top-to-bottom and agree left-to-right. This is what
            // separates a real rotation from a plain resize.
            let top = VideoTestSupport.meanGrey(of: frame.image, rows: 4..<28)
            let bottom = VideoTestSupport.meanGrey(of: frame.image, rows: 36..<60)
            let left = VideoTestSupport.meanGrey(of: frame.image, columns: 2..<14)
            let right = VideoTestSupport.meanGrey(of: frame.image, columns: 18..<30)
            #expect(abs(top - bottom) > 100, "expected a horizontal edge, got \(top) vs \(bottom)")
            #expect(abs(left - right) < 40, "expected no vertical edge, got \(left) vs \(right)")
        }
        #expect(seen == frames.count)
    }

    @Test("An identity transform leaves the frame alone")
    func identityTransformIsUntouched() async throws {
        // The decode path skips `applying(_:to:)` entirely when the transform is identity.
        // This pins that the skip and the rotation branch agree on the common case.
        let url = VideoTestSupport.temporaryURL("reader-identity")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 64, height: 32, frameRate: 10)
        try await writer.append(VideoTestSupport.splitFrame(width: 64, height: 32, grey: 200))
        try await writer.finish()

        var seen = 0
        for try await frame in SequentialVideoReader.frames(of: url) {
            seen += 1
            #expect(frame.image.width == 64)
            #expect(frame.image.height == 32)
            let left = VideoTestSupport.meanGrey(of: frame.image, columns: 4..<28)
            let right = VideoTestSupport.meanGrey(of: frame.image, columns: 36..<60)
            #expect(left < 16, "left half should be black, got \(left)")
            #expect(right > 150, "right half should be bright, got \(right)")
        }
        #expect(seen == 1)
    }

    // MARK: - Frame limits and cancellation

    @Test("maxFrames of zero yields nothing rather than everything")
    func maxFramesZero() async throws {
        // `nil` means unlimited, `0` means none. An `if let maxFrames` guard that tested
        // truthiness instead of nil-ness would quietly decode the whole video here.
        let url = VideoTestSupport.temporaryURL("reader-zero")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 10)
        for index in 0..<5 {
            try await writer.append(
                VideoTestSupport.solidFrame(width: 32, height: 32, grey: UInt8(20 + index * 40)))
        }
        try await writer.finish()

        var count = 0
        for try await _ in SequentialVideoReader.frames(of: url, maxFrames: 0) { count += 1 }
        #expect(count == 0)
    }

    @Test("Breaking out of the stream early terminates instead of hanging")
    func earlyBreakTerminates() async throws {
        // Leaving the loop cancels the decode task through `onTermination`. If that wiring
        // is wrong the reader keeps decoding into a dead continuation.
        let url = VideoTestSupport.temporaryURL("reader-break")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 10)
        for index in 0..<40 {
            try await writer.append(
                VideoTestSupport.solidFrame(width: 32, height: 32, grey: UInt8(index * 6)))
        }
        try await writer.finish()

        var seen: [Int] = []
        for try await frame in SequentialVideoReader.frames(of: url) {
            seen.append(frame.index)
            if seen.count == 3 { break }
        }
        #expect(seen == [0, 1, 2])
    }

    @Test("Frame indices are the reader's own counter, starting at zero")
    func indicesAreSequential() async throws {
        let url = VideoTestSupport.temporaryURL("reader-index")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 10)
        for index in 0..<6 {
            try await writer.append(
                VideoTestSupport.solidFrame(width: 32, height: 32, grey: UInt8(index * 40)))
        }
        try await writer.finish()

        var indices: [Int] = []
        for try await frame in SequentialVideoReader.frames(of: url, maxFrames: 4) {
            indices.append(frame.index)
        }
        #expect(indices == [0, 1, 2, 3])
    }

    // MARK: - Rejecting bad input

    @Test("A missing file is reported as missing, not as a decode failure")
    func missingFile() async throws {
        let url = URL(fileURLWithPath: "/nonexistent/coreai-does-not-exist.mp4")
        await #expect(throws: VideoInputError.self) {
            _ = try await SequentialVideoReader.metadata(of: url)
        }
        do {
            _ = try await SequentialVideoReader.metadata(of: url)
        } catch let error as VideoInputError {
            guard case .fileNotFound = error else {
                Issue.record("wrong case: \(error)")
                return
            }
        }
    }

    @Test("A file that isn't a video throws rather than yielding zero frames")
    func notAVideo() async throws {
        // Silently returning an empty sequence here would look identical to a legitimately
        // empty video, and the segmenter would report "0 frames" instead of a bad input.
        let url = VideoTestSupport.temporaryURL("reader-garbage")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0xAB, count: 8192).write(to: url)

        await #expect(throws: (any Error).self) {
            _ = try await SequentialVideoReader.metadata(of: url)
        }
        await #expect(throws: (any Error).self) {
            for try await _ in SequentialVideoReader.frames(of: url) {}
        }
    }

    @Test("frames(of:) surfaces an error for a missing file too")
    func missingFileThroughStream() async throws {
        // `metadata` guards on `fileExists` up front; the decode path does not, and leans on
        // AVFoundation to complain. Either way the caller must see a thrown error.
        let url = URL(fileURLWithPath: "/nonexistent/coreai-also-missing.mp4")
        await #expect(throws: (any Error).self) {
            for try await _ in SequentialVideoReader.frames(of: url) {}
        }
    }
}
