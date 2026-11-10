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
extension ReferenceSize: ExpressibleByArgument {}
extension GuidanceMode: ExpressibleByArgument {}

/// `--image-size`: a preset or a square size in pixels.
enum ImageSizeOption: Equatable, ExpressibleByArgument {
    case auto, full, half
    case pixels(Int)

    init?(argument: String) {
        switch argument {
        case "auto": self = .auto
        case "full": self = .full
        case "half": self = .half
        default:
            guard let pixels = Int(argument), pixels > 0 else { return nil }
            self = .pixels(pixels)
        }
    }
}

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

    @Option(
        help:
            "Path to an input image for image-to-image generation. Repeat for multiple FLUX.2 reference images (dynamic bundles only), referred to in the prompt as image 1, image 2, ... in order."
    )
    var inputImage: [String] = []

    @Option(
        help: ArgumentHelp(
            "Output size: auto (the best the bundle has), full, half, or a square pixel size such as 768. Dynamic bundles take any multiple of 16 in their range; static bundles only the sizes they were exported for. Default: auto",
            valueName: "auto|full|half|px")
    )
    var imageSize: ImageSizeOption?

    @Flag(
        help:
            "Decode the image in half-size tiles. Lowers peak memory on static bundles, which then load the half-size decoder. Needs a full-size output (at least 512 px on a dynamic bundle)."
    )
    var tiledDecode = false

    /// Deprecated spelling of `--image-size auto|full|half` and `--tiled-decode`.
    @Option(help: .hidden)
    var decodeResolution: DecodeResolution?

    @Option(
        help: ArgumentHelp(
            "img2img reference size: full, half, or quarter of the output size (half = 1/4 the tokens), or a pixel size such as 384 (dynamic bundles only; a multiple of 16, at least 128). Only applies with --input-image. Default: full",
            valueName: "full|half|quarter|px")
    )
    var referenceSize: ReferenceSize?

    /// Deprecated spelling of `--reference-size full|half|quarter`.
    @Option(help: .hidden)
    var referenceGrid: ReferenceGrid?

    @Option(
        help:
            "How much noise to add to a starting image, 0.0–1.0 (default: 0.85). FLUX.2 ignores it: its input images are references, not a starting point."
    )
    var strength: Float = 0.85

    @Flag(
        inversion: .prefixedNo,
        help:
            "Load models on demand and unload after each stage to reduce peak memory; disable to exercise full memory pressure (default: on)"
    )
    var lazyModelLoading: Bool = true

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

    func validate() throws {
        if referenceSize != nil && referenceGrid != nil {
            throw ValidationError("--reference-grid is a deprecated alias of --reference-size; pass only one.")
        }
        if decodeResolution != nil && (imageSize != nil || tiledDecode) {
            throw ValidationError(
                "--decode-resolution is a deprecated alias of --image-size and --tiled-decode; pass only one.")
        }
    }

    /// The pipeline's load mode plus an explicit per-run size, from `--image-size`,
    /// `--tiled-decode` and the deprecated `--decode-resolution`. A static bundle picks its
    /// assets by mode, so a pixel size there must be its full or half size.
    private func resolveOutputSize(_ diffusion: DiffusionConfig) throws -> (DecodeResolution, Int?) {
        if let decodeResolution { return (decodeResolution, nil) }
        var mode: DecodeResolution
        var pixels: Int?
        switch imageSize ?? .auto {
        case .auto: mode = .auto
        case .full: mode = .full
        case .half: mode = .half
        case .pixels(let requested):
            let full = diffusion.imageSize ?? DiffusionDefaults.Image.imageSize
            let isFlux2 = diffusion.type == .flux2
            if diffusion.dynamicShapes == true {
                (mode, pixels) = (.full, requested)
            } else if requested == full {
                mode = .full
            } else if isFlux2 && requested == full / 2 {
                mode = .half
            } else {
                let sizes = isFlux2 ? "\(full) (full) or \(full / 2) (half)" : "\(full)"
                let remedy = isFlux2 ? " Export with --dynamic for other sizes." : ""
                print("Error: this bundle only produces \(sizes); got --image-size \(requested).\(remedy)")
                throw ExitCode.failure
            }
        }
        if tiledDecode {
            // Checked after mapping, since a static bundle's pixel size can name half too.
            guard mode != .half else {
                print("Error: --tiled-decode splits a full-size output into tiles; it can't take a half-size output.")
                throw ExitCode.failure
            }
            mode = .tiled
        }
        return (mode, pixels)
    }

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

        var inputCGImages: [CGImage] = []
        for imagePath in inputImage {
            do {
                inputCGImages.append(try CGImageUtils.load(from: imagePath))
            } catch {
                print("Error: could not load input image at \(imagePath)")
                throw ExitCode.failure
            }
        }

        let (decodeMode, requestedSize) = try resolveOutputSize(diffusion)
        let effectiveReferenceSize = referenceSize ?? referenceGrid.map { .relative($0) }

        // The reference size only affects the img2img reference-token path; it is
        // ignored for txt2img. Warn if it was set without an input image.
        if inputCGImages.isEmpty && effectiveReferenceSize != nil {
            FileHandle.standardError.write(
                Data("Warning: --reference-size is ignored without --input-image (txt2img).\n".utf8))
        }

        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: effectiveSteps,
            guidanceScale: effectiveGuidance,
            schedulerType: schedulerType,
            // FLUX.2 conditions on every input image; a starting-image model starts from the first.
            startingImage: diffusion.type == .flux2 ? nil : inputCGImages.first,
            referenceImages: diffusion.type == .flux2 ? inputCGImages : Array(inputCGImages.dropFirst()),
            strength: strength,
            referenceSize: effectiveReferenceSize ?? .full,
            guidanceMode: guidanceMode ?? .distilled,
            encoderScaleFactor: diffusion.encoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
            decoderScaleFactor: diffusion.decoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
            decoderShiftFactor: diffusion.decoderShiftFactor ?? DiffusionDefaults.Image.decoderShiftFactor,
            decodeResolution: decodeMode,
            imageSize: requestedSize,
            lazyModelLoading: lazyModelLoading
        )

        if isFlowTransformer {
            let pipeline = try await FlowTransformerPipeline(
                from: bundleURL, mode: decodeMode)

            let family = diffusion.type == .sanaSprint ? "Sana Sprint" : "FLUX.2"
            print("Generating (\(family)): \"\(prompt)\"")
            print("Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
            let outputSize = requestedSize ?? pipeline.defaultImageSize.width
            print("Image size: \(outputSize)x\(outputSize)\(pipeline.mode == .tiled ? " (tiled decode)" : "")")
            if !inputCGImages.isEmpty, diffusion.type == .flux2 {
                let referencePixels = config.referenceSize.pixelSize(outputSize: outputSize)
                print("Reference images: \(inputCGImages.count) at \(referencePixels)x\(referencePixels)")
            }

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
