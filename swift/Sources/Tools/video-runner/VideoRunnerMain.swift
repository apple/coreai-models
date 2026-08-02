// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIShared
import CoreAIVideoPipeline
import Foundation

@main
struct VideoRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video-runner",
        abstract: "Generate videos using text-to-video diffusion models"
    )

    @Option(help: "Path to model directory containing .aimodel components")
    var model: String

    @Option(help: "Text prompt for video generation")
    var prompt: String

    @Option(help: "Number of output frames (default: model default)")
    var numFrames: Int?

    @Option(help: "Output FPS")
    var fps: Int = 24

    @Option(help: "Output resolution as WxH, e.g. 512x320")
    var resolution: String?

    @Option(help: "Number of denoising steps")
    var steps: Int = 50

    @Option(name: .customLong("guidance-scale"), help: "Classifier-free guidance scale")
    var guidanceScale: Float = 7.5

    @Option(help: "Random seed")
    var seed: UInt32 = 42

    @Option(help: "Output file path")
    var output: String = "output.mp4"

    @Option(
        name: .customLong("output-format"),
        help: "Output format: mp4, gif, apng, webp, or frames")
    var outputFormat: String = "mp4"

    @Flag(help: "Enable verbose logging")
    var verbose: Bool = false

    func run() async throws {
        let parsedWidth: Int?
        let parsedHeight: Int?
        if let resolution {
            let parts = resolution.split(separator: "x")
            guard parts.count == 2,
                let w = Int(parts[0]),
                let h = Int(parts[1])
            else {
                print("Error: --resolution must be in WxH format (e.g. 512x320)")
                throw ExitCode.failure
            }
            parsedWidth = w
            parsedHeight = h
        } else {
            parsedWidth = nil
            parsedHeight = nil
        }

        print("Video Generation Configuration")
        print("  Model:          \(model)")
        print("  Prompt:         \(prompt)")
        print("  Frames:         \(numFrames.map(String.init) ?? "model default")")
        print("  FPS:            \(fps)")
        print(
            "  Resolution:     \(parsedWidth.map { "\($0)x\(parsedHeight!)" } ?? "model default")")
        print("  Steps:          \(steps)")
        print("  Guidance Scale: \(guidanceScale)")
        print("  Seed:           \(seed)")
        print("  Output:         \(output)")
        print("  Format:         \(outputFormat)")
        print()
        print("Video pipeline not yet connected \u{2014} model loading coming soon")
    }
}
