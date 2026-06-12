// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIObjectDetector
import CoreGraphics
import CoreText
import Foundation
import ImageIO

@main
struct ObjectDetectorCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "object-detector",
        abstract: "Run object detection on an image using a CoreAI .aimodel model."
    )

    // MARK: - Options

    @Option(name: .long, help: "Path to the .aimodel directory.")
    var model: String

    @Option(
        name: .long,
        help:
            "Path to an input image. Pass --image multiple times to run detection on a batch of images (one --image per source file)."
    )
    var image: [String] = []

    @Option(name: .long, help: "Confidence threshold (0–1).")
    var threshold: Float = 0.3

    @Option(name: .long, help: "Maximum number of detections to return.")
    var maxDetections: Int = 100

    @Option(
        name: .long,
        help:
            "Override model input height (only used for dynamic models). Defaults to DetectionParameters.inputHeight if not set."
    )
    var inputHeight: Int?

    @Option(
        name: .long,
        help:
            "Override model input width (only used for dynamic models). Defaults to DetectionParameters.inputWidth if not set."
    )
    var inputWidth: Int?

    @Flag(name: .long, help: "Run a warmup pass before timed inference.")
    var warmup: Bool = false

    @Option(
        name: .long,
        help:
            "Render detections onto the input image(s). For a single --image, this is a file path. For multiple --image inputs, this is a directory; the CLI writes <source-stem>_detections.png into it."
    )
    var outputImage: String?

    @Option(
        name: .long,
        help:
            "Write JSON results to this path instead of stdout. With one --image, the JSON is an array of detections; with multiple, it's an array of {image, detections} objects."
    )
    var outputJson: String?

    @Flag(name: .long, help: "Print verbose progress information.")
    var verbose: Bool = false

    // MARK: - Validation

    func validate() throws {
        if image.isEmpty {
            throw ValidationError("At least one --image must be provided.")
        }
    }

    // MARK: - Run

    func run() async throws {
        if verbose { print("Loading model from \(model)...") }
        var params = DetectionParameters(
            threshold: threshold,
            maxDetections: maxDetections
        )
        if let h = inputHeight { params.inputHeight = h }
        if let w = inputWidth { params.inputWidth = w }
        let detector = try await ObjectDetector(resourcesAt: model)

        let loaded: [(path: String, image: CGImage)] = try image.map { path in
            let cgImage = try loadCGImage(from: path)
            if verbose { print("Loaded image \(path): \(cgImage.width)×\(cgImage.height)") }
            return (path, cgImage)
        }

        if warmup {
            if verbose { print("Running warmup...") }
            try await detector.warmup(imageCount: loaded.count, parameters: params)
        }

        if verbose { print("Running detection on \(loaded.count) image(s)...") }
        let start = SuspendingClock().now
        let allDetections = try await detector.detect(
            images: loaded.map { $0.image },
            parameters: params
        )
        let elapsed = SuspendingClock().now - start

        if verbose {
            print("Inference time (total): \(elapsed)")
        }

        // Build per-image JSON entries
        let entries: [JSONImageResult] = zip(loaded, allDetections).map { (entry, detections) in
            JSONImageResult(
                image: entry.path,
                detections: detections.map { d in
                    JSONDetection(
                        label: d.label,
                        labelIndex: d.labelIndex,
                        score: d.confidence,
                        box: JSONDetection.Box(
                            x: d.boundingBox.origin.x,
                            y: d.boundingBox.origin.y,
                            width: d.boundingBox.size.width,
                            height: d.boundingBox.size.height
                        )
                    )
                }
            )
        }

        try writeJsonOutput(entries: entries, multiImage: loaded.count > 1)

        // Stdout summary (suppressed when --output-json is set, matching prior behavior)
        if outputJson == nil {
            for (i, entry) in entries.enumerated() {
                if loaded.count > 1 {
                    print("\nImage \(i + 1)/\(loaded.count): \(entry.image)")
                }
                print("Detections (\(entry.detections.count)):")
                for (idx, d) in entry.detections.enumerated() {
                    print(
                        "  [\(idx)] \(d.label) score=\(String(format: "%.3f", d.score))"
                            + "  box=(\(Int(d.box.x)),\(Int(d.box.y)),\(Int(d.box.width))×\(Int(d.box.height)))"
                    )
                }
            }
        }

        // Render output image(s)
        if let imagePath = outputImage {
            try renderOutputImages(
                imagePath: imagePath,
                loaded: loaded,
                allDetections: allDetections
            )
        }
    }

    // MARK: - Output helpers

    private func writeJsonOutput(entries: [JSONImageResult], multiImage: Bool) throws {
        guard let jsonPath = outputJson else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: NSString(string: jsonPath).expandingTildeInPath)
        let data: Data
        if multiImage {
            data = try encoder.encode(entries)
        } else {
            // Single-image: keep the prior schema (top-level array of detections).
            data = try encoder.encode(entries.first?.detections ?? [])
        }
        try data.write(to: url)
        print("Results written to \(jsonPath)")
    }

    private func renderOutputImages(
        imagePath: String,
        loaded: [(path: String, image: CGImage)],
        allDetections: [[DetectedObject]]
    ) throws {
        let expanded = NSString(string: imagePath).expandingTildeInPath
        if loaded.count == 1 {
            let outputURL = URL(fileURLWithPath: expanded)
            try renderDetections(onto: loaded[0].image, detections: allDetections[0], saveTo: outputURL)
            print("Output image written to \(imagePath)")
            return
        }

        // Multi-image: treat --output-image as a directory.
        let dirURL = URL(fileURLWithPath: expanded)
        try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        for (entry, detections) in zip(loaded, allDetections) {
            let stem = (entry.path as NSString).lastPathComponent
            let stemNoExt = (stem as NSString).deletingPathExtension
            let outURL = dirURL.appendingPathComponent("\(stemNoExt)_detections.png")
            try renderDetections(onto: entry.image, detections: detections, saveTo: outURL)
            print("Output image written to \(outURL.path)")
        }
    }
}

// MARK: - Image Loading

private func loadCGImage(from path: String) throws -> CGImage {
    let expanded = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw ValidationError("Cannot open image at \(path)")
    }
    guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ValidationError("Cannot decode image at \(path)")
    }
    return cgImage
}

// MARK: - Rendering

private let boxColors: [(UInt8, UInt8, UInt8)] = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

private func renderDetections(onto cgImage: CGImage, detections: [DetectedObject], saveTo url: URL) throws {
    let width = cgImage.width
    let height = cgImage.height

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
        let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    else {
        throw ValidationError("Cannot create drawing context")
    }

    // Draw original image. CGContext origin is bottom-left; draw without
    // extra transforms — CGContext.draw() maps image top-left to rect bottom-left,
    // which produces a correct final image from makeImage().
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Draw detections
    for detection in detections {
        let color = boxColors[detection.labelIndex % boxColors.count]
        let cgColor = CGColor(
            colorSpace: colorSpace,
            components: [
                CGFloat(color.0) / 255.0,
                CGFloat(color.1) / 255.0,
                CGFloat(color.2) / 255.0,
                1.0,
            ]
        )!

        // Postprocessor returns top-left origin; flip y for CGContext (bottom-left origin).
        let box = detection.boundingBox
        let rect = CGRect(
            x: box.origin.x,
            y: Double(height) - box.origin.y - box.size.height,
            width: box.size.width,
            height: box.size.height
        )

        ctx.setStrokeColor(cgColor)
        ctx.setLineWidth(3.0)
        ctx.stroke(rect)

        // Build label text and measure its width precisely before sizing the background.
        let text = "\(detection.label) \(String(format: "%.2f", detection.confidence))"
        let fontSize: CGFloat = 14
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(
                "Helvetica" as CFString, fontSize, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
                colorSpace: colorSpace, components: [1, 1, 1, 1])!,
        ]
        let attributedString = CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary)!
        let line = CTLineCreateWithAttributedString(attributedString)

        var ascent: CGFloat = 0
        var descent: CGFloat = 0
        var leading: CGFloat = 0
        let measuredWidth = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
        let textWidth = measuredWidth + 8
        let textHeight = fontSize + 4

        // Label background sits above the box (rect.maxY is box top in CGContext space).
        let labelRect = CGRect(
            x: rect.origin.x,
            y: rect.maxY,
            width: textWidth,
            height: textHeight
        )
        ctx.setFillColor(cgColor)
        ctx.fill(labelRect)

        // Label text
        ctx.saveGState()
        ctx.textPosition = CGPoint(x: labelRect.origin.x + 4, y: labelRect.origin.y + 2)
        CTLineDraw(line, ctx)
        ctx.restoreGState()
    }

    guard let outputImage = ctx.makeImage() else {
        throw ValidationError("Failed to render output image")
    }

    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        throw ValidationError("Cannot create image destination at \(url.path)")
    }
    CGImageDestinationAddImage(dest, outputImage, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw ValidationError("Failed to write image to \(url.path)")
    }
}

// MARK: - JSON Types

private struct JSONDetection: Codable {
    let label: String
    let labelIndex: Int
    let score: Float
    let box: Box

    struct Box: Codable {
        let x, y, width, height: Double
    }
}

private struct JSONImageResult: Codable {
    let image: String
    let detections: [JSONDetection]
}
