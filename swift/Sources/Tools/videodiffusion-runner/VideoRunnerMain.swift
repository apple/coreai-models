// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreAIVideoDiffusionPipeline
import CoreGraphics
import Foundation

extension QualityPreset: ExpressibleByArgument {}

final class PhaseTiming: @unchecked Sendable {
    // Safe: only mutated from the progress handler which is called synchronously
    // from the pipeline's generateVideo loop (single-threaded access).
    var phaseStart = SuspendingClock().now
    var lastStepStart = SuspendingClock().now
    var encodeTime: Duration = .zero
    var denoiseTime: Duration = .zero
    var decodeTime: Duration = .zero
    var stepTimes: [Double] = []
}

@main
struct VideoDiffusionRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "videodiffusion-runner",
        abstract: "Generate video from text using an exported diffusion model"
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

    @Option(help: "Number of output frames (default: 81, max 81)")
    var numFrames: Int?

    @Option(help: "Video duration in seconds (alternative to --num-frames, max 5s)")
    var duration: Double?

    @Option(help: "Quality preset: fast (12 steps, ~2s), balanced (25 steps, 5s), best (50 steps, 5s)")
    var quality: QualityPreset?

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

    @Option(
        name: .customLong("tile-size"),
        help:
            "VAE spatial tile size in latent space (e.g. 32 = 256x256 pixel tiles). Enables tiled decoding to reduce peak memory."
    )
    var tileSize: Int?

    @Option(
        name: .customLong("tile-overlap"),
        help: "Overlap between tiles in latent space units (default: 4 = 32 pixels)"
    )
    var tileOverlap: Int = 4

    @Option(
        name: .customLong("cfg-cutoff"),
        help: "Skip unconditional pass for this fraction of final steps (e.g. 0.3 = last 30%)"
    )
    var cfgCutoff: Float?

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    func validate() throws {
        if let numFrames, numFrames <= 0 {
            throw ValidationError("--num-frames must be > 0")
        }
        if let duration, duration <= 0 {
            throw ValidationError("--duration must be > 0")
        }
        if numFrames != nil && duration != nil {
            throw ValidationError("--num-frames and --duration are mutually exclusive")
        }
        if let cfgCutoff, cfgCutoff < 0 || cfgCutoff > 1 {
            throw ValidationError("--cfg-cutoff must be between 0.0 and 1.0")
        }
        if let tileSize, tileSize <= 0 {
            throw ValidationError("--tile-size must be > 0")
        }
        if !FileManager.default.fileExists(atPath: model) {
            throw ValidationError("Model path not found: \(model)")
        }
    }

    func run() async throws {
        let modelURL = URL(fileURLWithPath: model)

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: modelURL)
            print("Cleared specialization cache for \(modelURL.lastPathComponent) (\(cleared.count) component(s))")
        }

        print("Loading pipeline from: \(model)")
        let loadStart = SuspendingClock().now
        let pipeline: WanPipeline
        do {
            pipeline = try await WanPipeline(from: modelURL, lazyModelLoading: lazyModelLoading)
        } catch {
            print("Error: \(error.localizedDescription)")
            throw ExitCode.failure
        }
        let loadElapsed = SuspendingClock().now - loadStart
        print("  done in \(String(format: "%.2f", loadElapsed.inSeconds))s")

        // Resolve effective parameters: explicit flags > quality preset > pipeline defaults
        let preset = quality
        let effectiveSteps = steps ?? preset?.steps ?? pipeline.defaultSteps
        let effectiveGuidance = guidanceScale ?? pipeline.defaultGuidanceScale
        let effectiveFPS = fps ?? 16
        let effectiveWidth = width ?? pipeline.defaultVideoSize.width
        let effectiveHeight = height ?? pipeline.defaultVideoSize.height

        // Frame count: --num-frames > --duration > preset > default
        let effectiveFrames: Int
        if let numFrames {
            effectiveFrames = numFrames
        } else if let duration {
            let raw = Int(duration * Double(effectiveFPS))
            effectiveFrames = min(81, max(5, raw - (raw - 1) % 4))
        } else {
            effectiveFrames = preset?.numFrames ?? pipeline.defaultFrameCount
        }

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
            loadNoisePath: loadNoisePath,
            vaeTileSize: tileSize,
            vaeTileOverlap: tileOverlap,
            cfgCutoffFraction: cfgCutoff
        )

        print("Generating video: \"\(prompt)\"")
        if let preset { print("  Quality: \(preset.rawValue)") }
        print("  Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
        let videoDuration = String(format: "%.1f", Double(effectiveFrames) / Double(effectiveFPS))
        print("  Frames: \(effectiveFrames), FPS: \(effectiveFPS), Duration: \(videoDuration)s")
        print("  Size: \(effectiveWidth)x\(effectiveHeight)")
        if let cutoff = cfgCutoff {
            let skipped = Int(Float(effectiveSteps) * cutoff)
            print("  CFG cutoff: last \(skipped) steps skip unconditional pass")
        }
        if let ts = tileSize {
            let pixelTile = ts * 8
            print("  Tiled VAE: \(ts)x\(ts) latent (\(pixelTile)x\(pixelTile) px), overlap: \(tileOverlap)")
        }

        let start = SuspendingClock().now
        let timing = PhaseTiming()

        let result: VideoGenerationResult
        do {
            result = try await pipeline.generateVideo(configuration: config) { progress in
                let now = SuspendingClock().now
                switch progress.phase {
                case .encoding:
                    if progress.step == 0 {
                        timing.phaseStart = now
                        timing.lastStepStart = now
                        print("  Encoding text...")
                    }
                case .denoising:
                    if progress.step == 0 {
                        timing.encodeTime = now - timing.phaseStart
                        timing.phaseStart = now
                        timing.lastStepStart = now
                        print("  [encode] \(String(format: "%.2f", timing.encodeTime.inSeconds))s")
                    } else {
                        let stepDt = (now - timing.lastStepStart).inSeconds
                        timing.stepTimes.append(stepDt)
                        timing.lastStepStart = now
                    }
                    let remaining = progress.totalSteps - progress.step - 1
                    if !timing.stepTimes.isEmpty {
                        let avg = timing.stepTimes.reduce(0, +) / Double(timing.stepTimes.count)
                        let eta = avg * Double(remaining)
                        print(
                            "  Denoising step \(progress.step + 1)/\(progress.totalSteps)"
                                + " [ETA: \(String(format: "%.0f", eta))s]")
                    } else {
                        print("  Denoising step \(progress.step + 1)/\(progress.totalSteps)")
                    }
                case .decoding:
                    let stepDt = (now - timing.lastStepStart).inSeconds
                    timing.stepTimes.append(stepDt)
                    timing.denoiseTime = now - timing.phaseStart
                    timing.phaseStart = now
                    let avgStep =
                        timing.stepTimes.isEmpty
                        ? 0 : timing.stepTimes.reduce(0, +) / Double(timing.stepTimes.count)
                    print(
                        "  [denoise] \(String(format: "%.2f", timing.denoiseTime.inSeconds))s (\(String(format: "%.2f", avgStep))s/step)"
                    )
                    print("  Decoding frames...")
                case .assembling:
                    timing.decodeTime = now - timing.phaseStart
                    print("  [decode] \(String(format: "%.2f", timing.decodeTime.inSeconds))s")
                    print("  Assembling video...")
                }
                return true
            }
        } catch let error as WanError {
            print("Error: \(error.localizedDescription)")
            throw ExitCode.failure
        }

        let elapsed = SuspendingClock().now - start
        let totalInference =
            timing.encodeTime.inSeconds + timing.denoiseTime.inSeconds + timing.decodeTime.inSeconds
        print(
            "Generated \(result.frames.count) frames in \(String(format: "%.2f", elapsed.inSeconds))s"
        )
        print(
            "  encode=\(String(format: "%.2f", timing.encodeTime.inSeconds))s denoise=\(String(format: "%.2f", timing.denoiseTime.inSeconds))s decode=\(String(format: "%.2f", timing.decodeTime.inSeconds))s"
        )
        print("  inference_total=\(String(format: "%.2f", totalInference))s")

        // Write output
        let outputURL = URL(fileURLWithPath: output)
        try await VideoWriter.write(frames: result.frames, fps: result.fps, to: outputURL)
        print("Saved: \(output)")
    }
}
