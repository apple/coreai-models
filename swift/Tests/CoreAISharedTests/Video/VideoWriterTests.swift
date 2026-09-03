// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation
import Testing

@testable import CoreAIShared

/// Covers `VideoWriter.writeMP4`'s batch API, which `videodiffusion-runner` calls as
/// `VideoWriter.write(frames:fps:to:)`.
///
/// `writeMP4` was reimplemented on top of ``StreamingVideoWriter`` when `VideoWriter` moved
/// into CoreAIShared, so this pins the contract the diffusion runner depends on: N frames
/// in, N frames out, at the dimensions it asked for.
@Suite("VideoWriter batch API")
struct VideoWriterTests {
    private func frame(width: Int, height: Int, grey: UInt8) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let level = CGFloat(grey) / 255
        context.setFillColor(red: level, green: level, blue: level, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    private func temporaryURL(_ suffix: String) -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "coreai-videowriter-\(UUID().uuidString).\(suffix)")
    }

    @Test("Wan's default 832x480 round-trips at full frame count")
    func diffusionDefaultSize() async throws {
        // The size `VideoConfiguration` defaults to, so this is the shape the diffusion
        // runner actually writes.
        let url = temporaryURL("mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let frames = (0..<12).map { frame(width: 832, height: 480, grey: UInt8(20 + $0 * 18)) }
        try await VideoWriter.write(frames: frames, fps: 16, to: url)

        let asset = AVURLAsset(url: url)
        let track = try await asset.loadTracks(withMediaType: .video).first
        #expect(track != nil)
        let size = try await track!.load(.naturalSize)
        #expect(Int(size.width) == 832)
        #expect(Int(size.height) == 480)

        var decoded = 0
        for try await _ in SequentialVideoReader.frames(of: url) { decoded += 1 }
        #expect(decoded == frames.count)
    }

    @Test("An empty frame list writes nothing and does not throw")
    func emptyInput() async throws {
        // Pre-existing behaviour: `writeMP4` returns early on an empty array rather than
        // creating a zero-length file. The runner relies on not having to guard.
        let url = temporaryURL("mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try await VideoWriter.write(frames: [], fps: 16, to: url)
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
    }

    @Test("An existing file at the destination is replaced, not appended to")
    func replacesExistingFile() async throws {
        let url = temporaryURL("mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0xAB, count: 4096).write(to: url)

        let frames = (0..<4).map { frame(width: 64, height: 64, grey: UInt8(40 + $0 * 40)) }
        try await VideoWriter.write(frames: frames, fps: 10, to: url)

        var decoded = 0
        for try await _ in SequentialVideoReader.frames(of: url) { decoded += 1 }
        #expect(decoded == frames.count)
    }

    @Test("Extension dispatch still routes mp4 and mov to the H.264 path")
    func extensionDispatch() async throws {
        for suffix in ["mp4", "mov"] {
            let url = temporaryURL(suffix)
            defer { try? FileManager.default.removeItem(at: url) }
            let frames = (0..<3).map { frame(width: 32, height: 32, grey: UInt8(50 + $0 * 50)) }
            try await VideoWriter.write(frames: frames, fps: 10, to: url)
            #expect(FileManager.default.fileExists(atPath: url.path), "\(suffix) was not written")
        }
    }

    @Test("A zero frame rate is rejected, and the message says so")
    func rejectsZeroFrameRate() async throws {
        // Reachable: `videodiffusion-runner --fps 0` is unvalidated upstream, and Wan only
        // checks numFrames, stepCount, and the width/height multiples. The old writer
        // passed fps straight into `CMTimeScale`, so 0 silently produced frames with an
        // invalid `CMTime`. The message must name the frame rate — an earlier version of
        // this guard reported the dimensions, which sent you looking in the wrong place.
        let url = temporaryURL("mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        let frames = [frame(width: 32, height: 32, grey: 128)]
        do {
            try await VideoWriter.write(frames: frames, fps: 0, to: url)
            Issue.record("expected a thrown error")
        } catch let error as VideoWriterError {
            guard case .invalidFrameRate(let rate) = error else {
                Issue.record("wrong case: \(error)")
                return
            }
            #expect(rate == 0)
            #expect(error.errorDescription?.contains("frame rate") == true)
        }
    }

    @Test("Non-positive dimensions are reported as dimensions")
    func rejectsZeroDimensions() async throws {
        let url = temporaryURL("mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            _ = try StreamingVideoWriter(url: url, width: 0, height: 48, frameRate: 10)
            Issue.record("expected a thrown error")
        } catch let error as VideoWriterError {
            guard case .invalidDimensions = error else {
                Issue.record("wrong case: \(error)")
                return
            }
        }
    }
}

/// Round-trips frames through ``StreamingVideoWriter`` and ``SequentialVideoReader``.
///
/// Replaces a test that compared AVFoundation's decode against PyAV's, which needed both a
/// video fixture and a PyTorch reference and so could never run unattended. This covers
/// what is actually ours: that we write every frame, read every frame back in order, and
/// preserve dimensions.
@Suite("Video round trip")
struct VideoRoundTripTests {
    /// A solid-colour frame, so codec loss can't be confused with a frame-ordering bug.
    private func frame(width: Int, height: Int, grey: UInt8) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let level = CGFloat(grey) / 255
        context.setFillColor(red: level, green: level, blue: level, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    /// Black on the left, `grey` on the right. The vertical edge is the point: a wrong row
    /// stride in the writer shears each row sideways and turns that edge into a diagonal.
    private func splitFrame(width: Int, height: Int, grey: UInt8) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let level = CGFloat(grey) / 255
        context.setFillColor(red: level, green: level, blue: level, alpha: 1)
        context.fill(CGRect(x: width / 2, y: 0, width: width - width / 2, height: height))
        return context.makeImage()!
    }

    /// Mean of the RGB channels over `columns`, across every row.
    private func meanGrey(of image: CGImage, columns: Range<Int>) -> Int {
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
        var total = 0
        for y in 0..<height {
            for x in columns {
                let offset = y * width * 4 + x * 4
                total += Int(pixels[offset]) + Int(pixels[offset + 1]) + Int(pixels[offset + 2])
            }
        }
        return total / (height * columns.count * 3)
    }

    @Test("Every frame written comes back, in order, at the same size")
    func roundTrip() async throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coreai-video-roundtrip-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        // Ten levels, spaced widely enough to survive H.264. Width 66 is deliberate: the
        // pool pads rows to 320 bytes there, so a `width * 4` stride bug shows up. At 64
        // the two agree and it would slip through.
        let levels: [UInt8] = (0..<10).map { UInt8(20 + $0 * 22) }
        let writer = try StreamingVideoWriter(url: url, width: 66, height: 48, frameRate: 10)
        for level in levels {
            try await writer.append(splitFrame(width: 66, height: 48, grey: level))
        }
        #expect(try await writer.finish() == levels.count)

        let metadata = try await SequentialVideoReader.metadata(of: url)
        #expect(metadata.width == 66)
        #expect(metadata.height == 48)

        var decoded: [Int] = []
        for try await frame in SequentialVideoReader.frames(of: url) {
            decoded.append(frame.index)
            // Sample away from the edge so the codec's ringing around it doesn't count.
            let left = meanGrey(of: frame.image, columns: 4..<29)
            let right = meanGrey(of: frame.image, columns: 37..<62)
            let expected = Int(levels[frame.index])
            #expect(left < 16, "frame \(frame.index) left half should be black, got \(left)")
            #expect(
                abs(right - expected) <= 12,
                "frame \(frame.index) right half should be ~\(expected), got \(right)")
        }
        #expect(decoded == Array(0..<levels.count), "frames must arrive in order, none dropped")
    }

    @Test("maxFrames stops the reader early")
    func honoursMaxFrames() async throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coreai-video-max-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 10)
        for index in 0..<8 {
            try await writer.append(frame(width: 32, height: 32, grey: UInt8(30 + index * 20)))
        }
        try await writer.finish()

        var count = 0
        for try await _ in SequentialVideoReader.frames(of: url, maxFrames: 3) { count += 1 }
        #expect(count == 3)
    }

    @Test("Odd dimensions are rounded up, because H.264 rejects them")
    func evenDimensions() async throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coreai-video-odd-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 33, height: 21, frameRate: 10)
        try await writer.append(frame(width: 33, height: 21, grey: 128))
        try await writer.finish()

        let metadata = try await SequentialVideoReader.metadata(of: url)
        #expect(metadata.width == 34)
        #expect(metadata.height == 22)
    }

    @Test("Appending after finish is an error, not a silent no-op")
    func writeAfterFinish() async throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coreai-video-after-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingVideoWriter(url: url, width: 32, height: 32, frameRate: 10)
        try await writer.append(frame(width: 32, height: 32, grey: 100))
        try await writer.finish()
        await #expect(throws: VideoWriterError.self) {
            try await writer.append(self.frame(width: 32, height: 32, grey: 100))
        }
    }
}
