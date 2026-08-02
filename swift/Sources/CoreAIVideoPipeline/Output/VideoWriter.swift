// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Utilities for writing video frames to MP4, GIF, or numbered PNG files.
public enum VideoWriter {
    // MARK: - MP4 (HEVC)

    /// Write frames as an HEVC-encoded MP4 to the given file URL.
    public static func writeMP4(frames: [CGImage], fps: Int, to url: URL) async throws {
        guard let first = frames.first else {
            throw VideoWriterError.noFrames
        }

        let width = first.width
        let height = first.height

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )

        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

        for (index, cgImage) in frames.enumerated() {
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(10))
            }

            guard let pixelBuffer = pixelBuffer(from: cgImage, width: width, height: height) else {
                throw VideoWriterError.pixelBufferCreationFailed
            }

            let presentationTime = CMTime(
                value: CMTimeValue(index), timescale: CMTimeScale(fps))
            adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
        }

        // Pad the last frame by one frame duration so the video has the expected length.
        _ = frameDuration

        writerInput.markAsFinished()
        await writer.finishWriting()

        if let error = writer.error {
            throw error
        }
    }

    // MARK: - GIF

    /// Write frames as an animated GIF to the given file URL.
    public static func writeGIF(frames: [CGImage], fps: Int, to url: URL) throws {
        guard !frames.isEmpty else {
            throw VideoWriterError.noFrames
        }

        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL,
                UTType.gif.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.destinationCreationFailed
        }

        let gifProperties: [String: Any] = [
            kCGImagePropertyGIFDictionary as String: [
                kCGImagePropertyGIFLoopCount as String: 0
            ]
        ]
        CGImageDestinationSetProperties(destination, gifProperties as CFDictionary)

        let frameDelay = 1.0 / Double(fps)
        let frameProperties: [String: Any] = [
            kCGImagePropertyGIFDictionary as String: [
                kCGImagePropertyGIFDelayTime as String: frameDelay
            ]
        ]

        for frame in frames {
            CGImageDestinationAddImage(destination, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(destination) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    // MARK: - APNG (Animated PNG)

    /// Write frames as an animated PNG file (full color, lossless).
    public static func writeAPNG(frames: [CGImage], fps: Int, to url: URL) throws {
        guard !frames.isEmpty else {
            throw VideoWriterError.noFrames
        }

        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL,
                UTType.png.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.destinationCreationFailed
        }

        let pngProperties: [String: Any] = [
            kCGImagePropertyPNGDictionary as String: [
                kCGImagePropertyAPNGLoopCount as String: 0
            ]
        ]
        CGImageDestinationSetProperties(destination, pngProperties as CFDictionary)

        let frameDelay = 1.0 / Double(fps)
        let frameProperties: [String: Any] = [
            kCGImagePropertyPNGDictionary as String: [
                kCGImagePropertyAPNGDelayTime as String: frameDelay
            ]
        ]

        for frame in frames {
            CGImageDestinationAddImage(destination, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(destination) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    // MARK: - WebP (Animated)

    /// Write frames as an animated WebP file (lossy, small file size).
    @available(macOS 14.0, iOS 17.0, *)
    public static func writeWebP(frames: [CGImage], fps: Int, to url: URL) throws {
        guard !frames.isEmpty else {
            throw VideoWriterError.noFrames
        }

        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL,
                UTType.webP.identifier as CFString,
                frames.count,
                nil
            )
        else {
            throw VideoWriterError.destinationCreationFailed
        }

        let webPProperties: [String: Any] = [
            kCGImagePropertyWebPDictionary as String: [
                kCGImagePropertyWebPLoopCount as String: 0
            ]
        ]
        CGImageDestinationSetProperties(destination, webPProperties as CFDictionary)

        let frameDelay = 1.0 / Double(fps)
        let frameProperties: [String: Any] = [
            kCGImagePropertyWebPDictionary as String: [
                kCGImagePropertyWebPDelayTime as String: frameDelay
            ]
        ]

        for frame in frames {
            CGImageDestinationAddImage(destination, frame, frameProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(destination) else {
            throw VideoWriterError.finalizationFailed
        }
    }

    // MARK: - PNG Frames

    /// Write each frame as a numbered PNG file in the given directory.
    public static func writeFrames(frames: [CGImage], to directory: URL) throws {
        guard !frames.isEmpty else {
            throw VideoWriterError.noFrames
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let digitCount = String(frames.count - 1).count
        for (index, frame) in frames.enumerated() {
            let name = String(format: "frame_%0\(digitCount)d.png", index)
            let fileURL = directory.appendingPathComponent(name)

            guard
                let dest = CGImageDestinationCreateWithURL(
                    fileURL as CFURL, UTType.png.identifier as CFString, 1, nil)
            else {
                throw VideoWriterError.destinationCreationFailed
            }
            CGImageDestinationAddImage(dest, frame, nil)
            guard CGImageDestinationFinalize(dest) else {
                throw VideoWriterError.finalizationFailed
            }
        }
    }

    // MARK: - Helpers

    private static func pixelBuffer(from image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32ARGB,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
            )
        else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

/// Errors specific to video output writing.
public enum VideoWriterError: Error, CustomStringConvertible {
    case noFrames
    case pixelBufferCreationFailed
    case destinationCreationFailed
    case finalizationFailed

    public var description: String {
        switch self {
        case .noFrames: "No frames provided for video output"
        case .pixelBufferCreationFailed: "Failed to create pixel buffer from CGImage"
        case .destinationCreationFailed: "Failed to create image destination"
        case .finalizationFailed: "Failed to finalize image destination"
        }
    }
}
