// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreGraphics
import Foundation
import ImageIO

extension DecodeResolution: ExpressibleByArgument {}
extension ReferenceGrid: ExpressibleByArgument {}
extension GuidanceMode: ExpressibleByArgument {}

@main
struct DiffusionRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diffusion-runner",
        abstract: "Generate images using Core AI diffusion models"
    )

    @Option(help: "Path to model directory containing .aimodel components")
    var model: String?

    @Option(help: "Text prompt for image generation")
    var prompt: String = "a photo of a cat"

    @Option(help: "Negative prompt")
    var negativePrompt: String = ""

    @Option(help: "Number of denoising steps (default: pipeline default, else 20)")
    var steps: Int?

    @Option(help: "Guidance scale (default: pipeline default, else 7.5)")
    var guidanceScale: Float?

    @Option(help: "Random seed (default: 42)")
    var seed: UInt32 = 42

    @Option(help: "Output image path (default: output.png)")
    var output: String = "output.png"

    @Option(help: "Path to input image for image-to-image generation")
    var inputImage: String?

    @Option(
        help:
            "Denoising strength for image-to-image, 0.0–1.0 (default: 0.85). Use 0.8–0.9 for semantic edits, 0.5–0.75 for style/texture changes."
    )
    var strength: Float = 0.85

    @Flag(
        inversion: .prefixedNo,
        help:
            "Load models on demand and unload after each stage to reduce peak memory; disable to exercise full memory pressure (default: on)"
    )
    var lazyModelLoading: Bool = true

    @Option(help: "VAE decode resolution: full, half, or tiled (default: full)")
    var decodeResolution: DecodeResolution = .full

    @Option(
        help:
            "img2img reference-token grid, relative to the output grid: full (1:1, ~4096 tokens @1024 / 1024 @512), half (1/4 the tokens), quarter (1/16 the tokens). Only applies with --input-image. Default: full"
    )
    var referenceGrid: ReferenceGrid?

    @Option(
        help:
            "Whether the pipeline performs classifier-free guidance itself: distilled (one forward pass, no CFG — the model is guidance-distilled and ignores its guidance input) or manual (two passes interpolated by the pipeline using --guidance-scale — stronger text adherence in img2img, ~2× compute per step). --guidance-scale only has an effect with manual. Default: distilled"
    )
    var guidanceMode: GuidanceMode?

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    @Option(
        name: .customLong("tune-preview"),
        help:
            "Phase 1 (collect): record per-step latents and decoded image into <dir> for later fitting. Run once per prompt, then use --tune-fit on the parent directory to fit jointly."
    )
    var tunePreviewDir: String?

    @Option(
        name: .customLong("tune-fit"),
        help:
            "Phase 2 (fit): read all collected latent/image pairs from subdirectories of <dir>, fit a single [C, 3] projection jointly, and print the resulting coefficients. No model loading required."
    )
    var tuneFitDir: String?

    @Option(
        name: .customLong("parity-test"),
        help: ArgumentHelp("Path to parity data directory (numpy .npy files)", visibility: .hidden)
    )
    var parityTestDir: String?

    func run() async throws {
        // --tune-fit: fit coefficients from collected pairs, no model needed
        if let fitDir = tuneFitDir {
            PreviewTuneHelper.runFit(dir: URL(fileURLWithPath: fitDir))
            return
        }

        guard let model else {
            print("Error: --model is required for generation")
            throw ExitCode.failure
        }

        let bundleURL = URL(fileURLWithPath: model)

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundleURL)
            print("🗑️  Cleared specialization cache for \(bundleURL.lastPathComponent) (\(cleared.count) component(s))")
        }

        if let parityDir = parityTestDir {
            try await runParityTest(modelURL: bundleURL, dataDir: URL(fileURLWithPath: parityDir))
            return
        }

        print("Loading pipeline from: \(model)")

        // Determine pipeline type and dispatch
        let bundle = try DiffusionBundle(at: bundleURL)
        let diffusion = bundle.config
        let isFlowTransformer = diffusion.type == .flux2 || diffusion.type == .sanaSprint

        guard isFlowTransformer else {
            if let type = diffusion.type {
                print("Error: unsupported pipeline type '\(type.rawValue)'")
            } else {
                print(
                    "Error: could not determine the pipeline type for this bundle. "
                        + "Re-export with metadata.json (via `coreai.diffusion.export`) so the runner can detect it.")
            }
            throw ExitCode.failure
        }

        let schedulerType: SchedulerType = .discreteFlow
        let effectiveSteps = steps ?? diffusion.defaultSteps ?? DiffusionDefaults.Runner.steps
        let effectiveGuidance =
            guidanceScale ?? diffusion.defaultGuidanceScale ?? DiffusionDefaults.Runner.guidanceScale

        var startingCGImage: CGImage? = nil
        if let imagePath = inputImage {
            do {
                startingCGImage = try CGImageUtils.load(from: imagePath)
            } catch {
                print("Error: could not load input image at \(imagePath)")
                throw ExitCode.failure
            }
        }

        // --reference-grid only affects the img2img reference-token path; it is
        // ignored for txt2img. Warn if it was set without an input image.
        if referenceGrid != nil && startingCGImage == nil {
            FileHandle.standardError.write(
                Data("Warning: --reference-grid is ignored without --input-image (txt2img).\n".utf8))
        }

        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: effectiveSteps,
            guidanceScale: effectiveGuidance,
            schedulerType: schedulerType,
            startingImage: startingCGImage,
            strength: strength,
            referenceGrid: referenceGrid ?? .full,
            guidanceMode: guidanceMode ?? .distilled,
            encoderScaleFactor: diffusion.encoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
            decoderScaleFactor: diffusion.decoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
            decoderShiftFactor: diffusion.decoderShiftFactor ?? DiffusionDefaults.Image.decoderShiftFactor,
            decodeResolution: decodeResolution,
            lazyModelLoading: lazyModelLoading
        )

        if isFlowTransformer {
            let pipeline = try await FlowTransformerPipeline(
                from: bundleURL, mode: decodeResolution)

            let family = diffusion.type == .sanaSprint ? "Sana Sprint" : "FLUX.2"
            print("Generating (\(family)): \"\(prompt)\"")
            print("Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
            print("Image size: \(pipeline.defaultImageSize.width)x\(pipeline.defaultImageSize.height)")

            let tuneHelper = tunePreviewDir.map { PreviewTuneHelper(outputDir: URL(fileURLWithPath: $0)) }
            let start = ContinuousClock.now

            let result = try await pipeline.generateImages(configuration: config) { progress in
                if let tuneHelper {
                    return tuneHelper.progressHandler(progress)
                }
                print("  Step \(progress.step)/\(progress.totalSteps)")
                return true
            }

            let elapsed = ContinuousClock.now - start
            print("Generated in \(String(format: "%.2f", elapsed.inSeconds))s")

            guard let image = result.images.first else {
                print("Error: No image generated")
                throw ExitCode.failure
            }

            if let tuneHelper { tuneHelper.finish(image: image) }

            let outputURL = URL(fileURLWithPath: output)
            try saveImage(image, to: outputURL)
            print("Saved: \(output)")
        }
    }

    private func saveImage(_ image: CGImage, to url: URL) throws {
        guard
            let dest = CGImageDestinationCreateWithURL(
                url as CFURL, "public.png" as CFString, 1, nil)
        else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw CocoaError(.fileWriteUnknown)
        }
    }
}
