// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Writes video frames to various output formats (MP4, GIF, APNG, WebP).
public struct VideoWriter {
    public enum OutputFormat: String, Sendable, CaseIterable {
        case mp4
        case gif
        case apng
        case webp
    }

    /// Write frames to an MP4 file using AVFoundation.
    ///
    /// Convenience wrapper over ``StreamingVideoWriter`` for callers that already hold
    /// every frame. Anything producing frames incrementally should use that directly and
    /// avoid materializing the array.
    public static func writeMP4(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) async throws {
        guard let firstFrame = frames.first else { return }
        let writer = try StreamingVideoWriter(
            url: outputURL,
            width: firstFrame.width,
            height: firstFrame.height,
            frameRate: fps
        )
        for frame in frames {
            try await writer.append(frame)
        }
        try await writer.finish()
    }

    /// Write frames as an animated GIF.
    public static func writeGIF(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) throws {
        let frameDelay = 1.0 / Double(fps)
        let fileProperties: [String: Any] = [
            kCGImagePropertyGIFDictionary as String: [
                kCGImagePropertyGIFLoopCount as String: 0
            ]
        ]
        let frameProperties: [String: Any] = [
            kCGImagePropertyGIFDictionary as String: [
                kCGImagePropertyGIFUnclampedDelayTime as String: frameDelay
            ]
        ]

        guard
            let dest = CGImageDestinationCreateWithURL(
                outputURL as CFURL,
                UTType.gif.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.cannotCreateDestination
        }

        CGImageDestinationSetProperties(dest, fileProperties as CFDictionary)
        for frame in frames {
            CGImageDestinationAddImage(dest, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(dest) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    /// Write frames as an animated PNG (APNG).
    public static func writeAPNG(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) throws {
        let frameDelay = 1.0 / Double(fps)
        let fileProperties: [String: Any] = [
            kCGImagePropertyPNGDictionary as String: [
                kCGImagePropertyAPNGLoopCount as String: 0
            ]
        ]
        let frameProperties: [String: Any] = [
            kCGImagePropertyPNGDictionary as String: [
                kCGImagePropertyAPNGUnclampedDelayTime as String: frameDelay
            ]
        ]

        guard
            let dest = CGImageDestinationCreateWithURL(
                outputURL as CFURL,
                UTType.png.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.cannotCreateDestination
        }

        CGImageDestinationSetProperties(dest, fileProperties as CFDictionary)
        for frame in frames {
            CGImageDestinationAddImage(dest, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(dest) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    /// Write frames as an animated WebP.
    public static func writeWebP(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) throws {
        let frameDelay = 1.0 / Double(fps)
        let frameProperties: [String: Any] = [
            kCGImagePropertyWebPDictionary as String: [
                kCGImagePropertyWebPUnclampedDelayTime as String: frameDelay
            ]
        ]

        guard
            let dest = CGImageDestinationCreateWithURL(
                outputURL as CFURL,
                UTType.webP.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.cannotCreateDestination
        }

        for frame in frames {
            CGImageDestinationAddImage(dest, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(dest) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    /// Auto-detect format from file extension and write.
    public static func write(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) async throws {
        let ext = outputURL.pathExtension.lowercased()
        switch ext {
        case "mp4", "mov":
            try await writeMP4(frames: frames, fps: fps, to: outputURL)
        case "gif":
            try writeGIF(frames: frames, fps: fps, to: outputURL)
        case "apng", "png":
            try writeAPNG(frames: frames, fps: fps, to: outputURL)
        case "webp":
            try writeWebP(frames: frames, fps: fps, to: outputURL)
        default:
            try await writeMP4(frames: frames, fps: fps, to: outputURL)
        }
    }
}

public enum VideoWriterError: Error, LocalizedError {
    case cannotCreateDestination
    case finalizationFailed
    case invalidDimensions(width: Int, height: Int)
    case invalidFrameRate(Int)
    case writeAfterFinish
    case writeFailed(Error?)

    public var errorDescription: String? {
        switch self {
        case .cannotCreateDestination:
            return "Failed to create image destination for video output"
        case .finalizationFailed:
            return "Failed to finalize video output"
        case .invalidDimensions(let width, let height):
            return "Video dimensions must be positive; got \(width)×\(height)"
        case .invalidFrameRate(let frameRate):
            return "Video frame rate must be positive; got \(frameRate)"
        case .writeAfterFinish:
            return "Cannot append frames after finish() has been called"
        case .writeFailed(let underlying):
            return "Video encoding failed: \(underlying?.localizedDescription ?? "unknown error")"
        }
    }
}
