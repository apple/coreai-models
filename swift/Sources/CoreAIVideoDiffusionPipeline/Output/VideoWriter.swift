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
    public static func writeMP4(
        frames: [CGImage],
        fps: Int,
        to outputURL: URL
    ) async throws {
        guard let firstFrame = frames.first else { return }
        let width = firstFrame.width
        let height = firstFrame.height

        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )

        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

        for (i, frame) in frames.enumerated() {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(10))
            }
            guard let pool = adaptor.pixelBufferPool else { continue }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
            guard let buffer = pixelBuffer else { continue }

            CVPixelBufferLockBaseAddress(buffer, [])
            let ctx = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width, height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
            )
            ctx?.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))
            CVPixelBufferUnlockBaseAddress(buffer, [])

            let presentationTime = CMTime(value: CMTimeValue(i), timescale: frameDuration.timescale)
            adaptor.append(buffer, withPresentationTime: presentationTime)
        }

        input.markAsFinished()
        await writer.finishWriting()
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

    public var errorDescription: String? {
        switch self {
        case .cannotCreateDestination:
            return "Failed to create image destination for video output"
        case .finalizationFailed:
            return "Failed to finalize video output"
        }
    }
}
