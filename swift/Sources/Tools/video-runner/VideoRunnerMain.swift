// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreAIVideoPipeline
import CoreGraphics
import Foundation

@main
struct VideoRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video-runner",
        abstract: "Generate video using Wan 2.1 text-to-video pipeline"
    )

    @Option(help: "Path to exported model directory (containing Transformer.aimodel, etc.)")
    var model: String

    @Option(help: "Text prompt for video generation")
    var prompt: String = "A cat walking on grass"

    @Option(help: "Negative prompt")
    var negativePrompt: String = ""

    @Option(help: "Number of denoising steps (default: 50)")
    var steps: Int?

    @Option(help: "Guidance scale (default: 5.0)")
    var guidanceScale: Float?

    @Option(help: "Random seed (default: 42)")
    var seed: UInt32 = 42

    @Option(help: "Number of output frames (default: 81)")
    var numFrames: Int?

    @Option(help: "Output FPS (default: 16)")
    var fps: Int?

    @Option(help: "Video width (default: 832)")
    var width: Int?

    @Option(help: "Video height (default: 480)")
    var height: Int?

    @Option(help: "Output file path (default: output.mp4)")
    var output: String = "output.mp4"

    @Option(name: .customLong("dump-dir"), help: "Directory to dump intermediate latents as .npy files")
    var dumpDirectory: String?

    @Option(name: .customLong("load-noise"), help: "Path to .npy file containing initial noise (bypass RNG)")
    var loadNoisePath: String?

    @Flag(
        inversion: .prefixedNo,
        help: "Load models on demand and unload after each stage (default: on)"
    )
    var lazyModelLoading: Bool = true

    func run() async throws {
        let modelURL = URL(fileURLWithPath: model)

        print("Loading Wan 2.1 pipeline from: \(model)")
        let pipeline = try await WanPipeline(from: modelURL, lazyModelLoading: lazyModelLoading)

        let effectiveSteps = steps ?? pipeline.defaultSteps
        let effectiveGuidance = guidanceScale ?? pipeline.defaultGuidanceScale
        let effectiveFrames = numFrames ?? pipeline.defaultFrameCount
        let effectiveFPS = fps ?? 16
        let effectiveWidth = width ?? pipeline.defaultVideoSize.width
        let effectiveHeight = height ?? pipeline.defaultVideoSize.height

        let config = VideoConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: effectiveSteps,
            guidanceScale: effectiveGuidance,
            numFrames: effectiveFrames,
            fps: effectiveFPS,
            width: effectiveWidth,
            height: effectiveHeight,
            dumpDirectory: dumpDirectory,
            loadNoisePath: loadNoisePath
        )

        print("Generating video: \"\(prompt)\"")
        print("  Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
        print("  Frames: \(effectiveFrames), FPS: \(effectiveFPS)")
        print("  Size: \(effectiveWidth)x\(effectiveHeight)")

        let start = ContinuousClock.now

        let result = try await pipeline.generateVideo(configuration: config) { progress in
            switch progress.phase {
            case .encoding:
                print("  Encoding text...")
            case .denoising:
                print("  Denoising step \(progress.step + 1)/\(progress.totalSteps)")
            case .decoding:
                print("  Decoding frames...")
            case .assembling:
                print("  Assembling video...")
            }
            return true
        }

        let elapsed = ContinuousClock.now - start
        print("Generated \(result.frames.count) frames in \(String(format: "%.2f", elapsed.inSeconds))s")

        // Write output
        let outputURL = URL(fileURLWithPath: output)
        try await VideoWriter.write(frames: result.frames, fps: result.fps, to: outputURL)
        print("Saved: \(output)")
    }
}
