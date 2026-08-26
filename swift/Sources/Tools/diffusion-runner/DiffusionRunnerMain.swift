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

@main
struct DiffusionRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diffusion-runner",
        abstract: "Generate images using Stable Diffusion models"
    )

    @Option(help: "Path to model directory containing .aimodel components (or pipeline.json)")
    var model: String

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

    @Option(help: "Scheduler: pndm or dpmpp (default: dpmpp)")
    var scheduler: String = "dpmpp"

    @Option(help: "Output image path (default: output.png)")
    var output: String = "output.png"

    @Option(name: .customLong("config"), help: "Path to pipeline.json (auto-detected if not specified)")
    var configPath: String?

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

    @Flag(help: "Use manual CFG (two forward passes) for stronger text guidance in img2img")
    var manualCfg: Bool = false

    @Option(name: .customLong("parity-test"), help: "Path to parity data directory (numpy .npy files)")
    var parityTestDir: String?

    @Option(
        name: .customLong("trace-inputs"),
        help: "Path to pipeline trace dir — use Python's noise + embeddings instead of generating")
    var traceInputsDir: String?

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    func run() async throws {
        let bundleURL = URL(fileURLWithPath: model)

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundleURL)
            print("🗑️  Cleared specialization cache for \(bundleURL.lastPathComponent) (\(cleared.count) component(s))")
        }

        if let parityDir = parityTestDir {
            try await runParityTest(modelURL: bundleURL, dataDir: URL(fileURLWithPath: parityDir))
            return
        }

        if let traceDir = traceInputsDir {
            try await runWithTraceInputs(modelURL: bundleURL, traceDir: URL(fileURLWithPath: traceDir))
            return
        }

        print("Loading pipeline from: \(model)")

        let configSource: PipelineDescriptor.ConfigSource
        if let configPath {
            configSource = .file(URL(fileURLWithPath: configPath))
        } else {
            configSource = .auto
        }

        // Determine pipeline type and dispatch
        let resolvedDescriptor = try PipelineDescriptor.resolve(at: bundleURL, config: configSource)
        let isFlux2 = resolvedDescriptor.type == .flux2
        let isSd3 = resolvedDescriptor.type == .stableDiffusion3

        let schedulerType: SchedulerType =
            (isFlux2 || isSd3) ? .discreteFlow : (scheduler == "pndm" ? .pndm : .dpmSolverMultistep)
        let effectiveSteps = steps ?? resolvedDescriptor.defaultSteps ?? 20
        let effectiveGuidance = guidanceScale ?? resolvedDescriptor.defaultGuidanceScale ?? 7.5

        var startingCGImage: CGImage? = nil
        if let imagePath = inputImage {
            guard let img = loadCGImage(from: URL(fileURLWithPath: imagePath)) else {
                print("Error: could not load input image at \(imagePath)")
                throw ExitCode.failure
            }
            startingCGImage = img
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
            manualCFG: manualCfg,
            encoderScaleFactor: resolvedDescriptor.encoderScaleFactor ?? 0.18215,
            decoderScaleFactor: resolvedDescriptor.decoderScaleFactor ?? 0.18215,
            decoderShiftFactor: resolvedDescriptor.decoderShiftFactor ?? 0.0,
            decodeResolution: decodeResolution,
            lazyModelLoading: lazyModelLoading
        )

        if isFlux2 {
            let pipeline = try await Flux2Pipeline(from: bundleURL, config: configSource, mode: decodeResolution)

            print("Generating (FLUX.2): \"\(prompt)\"")
            print("Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
            print("Image size: \(pipeline.defaultImageSize.width)x\(pipeline.defaultImageSize.height)")

            let start = ContinuousClock.now

            let result = try await pipeline.generateImages(configuration: config) { progress in
                print("  Step \(progress.step)/\(progress.totalSteps)")
                return true
            }

            let elapsed = ContinuousClock.now - start
            print("Generated in \(String(format: "%.2f", elapsed.inSeconds))s")

            guard let image = result.images.first else {
                print("Error: No image generated")
                throw ExitCode.failure
            }

            let outputURL = URL(fileURLWithPath: output)
            try saveImage(image, to: outputURL)
            print("Saved: \(output)")
        } else if isSd3 {
            let pipeline = try await SD3Pipeline(from: bundleURL, config: configSource)

            print("Generating (SD 3.x): \"\(prompt)\"")
            print("Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
            print("Image size: \(pipeline.defaultImageSize.width)x\(pipeline.defaultImageSize.height)")

            let start = ContinuousClock.now

            let result = try await pipeline.generateImages(configuration: config) { progress in
                print("  Step \(progress.step)/\(progress.totalSteps)")
                return true
            }

            let elapsed = ContinuousClock.now - start
            print("Generated in \(String(format: "%.2f", elapsed.inSeconds))s")

            guard let image = result.images.first else {
                print("Error: No image generated")
                throw ExitCode.failure
            }

            let outputURL = URL(fileURLWithPath: output)
            try saveImage(image, to: outputURL)
            print("Saved: \(output)")
        } else {
            let pipeline = try await StableDiffusionPipeline.load(from: bundleURL, config: configSource)

            print("Generating: \"\(prompt)\"")
            print("Steps: \(effectiveSteps), Guidance: \(effectiveGuidance), Seed: \(seed)")
            print("Image size: \(pipeline.defaultImageSize.width)x\(pipeline.defaultImageSize.height)")

            let start = ContinuousClock.now

            let result = try await pipeline.generateImages(configuration: config) { progress in
                print("  Step \(progress.step + 1)/\(progress.totalSteps)")
                return true
            }

            let elapsed = ContinuousClock.now - start
            print("Generated in \(String(format: "%.2f", elapsed.inSeconds))s")

            guard let image = result.images.first else {
                print("Error: No image generated")
                throw ExitCode.failure
            }

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

    private func loadCGImage(from url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        return image
    }

    // MARK: - Parity Test

    private func runParityTest(modelURL: URL, dataDir: URL) async throws {
        print("Running parity test with data from: \(dataDir.path)")

        // Check if this is a pipeline trace (has initial_latents.npy) or component test
        let isFullTrace = FileManager.default.fileExists(
            atPath: dataDir.appendingPathComponent("initial_latents.npy").path)
        let isSD3 = FileManager.default.fileExists(
            atPath: dataDir.appendingPathComponent("mmdit_hidden_states.npy").path)
        let isFlux2 = FileManager.default.fileExists(
            atPath: dataDir.appendingPathComponent("text_encoder_attention_mask.npy").path)

        if isFullTrace {
            try await runFullPipelineTrace(modelURL: modelURL, dataDir: dataDir)
        } else if isSD3 {
            try await runSD3ComponentParity(modelURL: modelURL, dataDir: dataDir)
        } else if isFlux2 {
            try await runFlux2ComponentParity(modelURL: modelURL, dataDir: dataDir)
        } else {
            try await runComponentParity(modelURL: modelURL, dataDir: dataDir)
        }
    }

    // MARK: - Full Pipeline Trace (step-by-step scheduler verification)

    private func runFullPipelineTrace(modelURL: URL, dataDir: URL) async throws {
        print("\n=== Full Pipeline Trace ===")

        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        var descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)
        _ = try await descriptor.loadComponents(from: modelURL)

        // Load trace data
        let initialLatents = try loadNpy(dataDir.appendingPathComponent("initial_latents.npy"))
        let textEmbeddings = try loadNpy(dataDir.appendingPathComponent("text_embeddings.npy"))
        let negEmbeddings = try loadNpy(dataDir.appendingPathComponent("neg_embeddings.npy"))
        let timestepsData = try loadNpy(dataDir.appendingPathComponent("timesteps.npy"))

        let timesteps = timestepsData.data.map { Int($0) }
        print("  Timesteps: \(timesteps)")
        print(
            "  Initial latents: shape=\(initialLatents.shape) mean=\(mean(initialLatents.data)) absmax=\(absmax(initialLatents.data))"
        )

        // Create scheduler with matching step count
        let predictionType = descriptor.predictionType ?? .epsilon
        let defaults = descriptor.scheduler ?? PipelineDescriptor.SchedulerDefaults()
        let scheduler = DPMSolverMultistepScheduler(
            stepCount: timesteps.count,
            trainStepCount: defaults.trainingSteps,
            betaSchedule: .scaledLinear,
            betaStart: defaults.betaStart,
            betaEnd: defaults.betaEnd,
            predictionType: predictionType
        )
        print("  Our timesteps: \(scheduler.timeSteps)")

        // Prepare embeddings as NDArrays
        _ = floatsToNDArray(textEmbeddings.data, asInt32: false, shape: textEmbeddings.shape)
        _ = floatsToNDArray(negEmbeddings.data, asInt32: false, shape: negEmbeddings.shape)

        // Run step-by-step with traced inputs
        var latents = initialLatents.data

        for (i, t) in timesteps.enumerated() {
            // Load reference noise prediction and expected output
            let refNoisePred = try loadNpy(dataDir.appendingPathComponent("noise_pred_step\(i).npy"))
            let refLatents = try loadNpy(dataDir.appendingPathComponent("latents_step\(i).npy"))

            print(
                "  Step \(i) (t=\(t)): noise_pred=\(refNoisePred.shape) (\(refNoisePred.data.count) floats) latents=\(latents.count)"
            )
            print(
                "    alphaT[\(t)]=\(scheduler.alphaT[t]) sigmaT[\(t)]=\(scheduler.sigmaT[t]) lambdaT[\(t)]=\(scheduler.lambdaT[t])"
            )

            // Manual step-through for diagnostics
            print("    convertModelOutput...")
            let converted = scheduler.convertModelOutput(modelOutput: refNoisePred.data, timestep: t, sample: latents)
            print("    converted: count=\(converted.count) absmax=\(absmax(converted))")
            print("    firstOrderUpdate...")
            let schedulerResult = scheduler.step(output: refNoisePred.data, timeStep: t, sample: latents)
            print("    step done: count=\(schedulerResult.count)")
            let schedulerCos = cosineSimilarity(schedulerResult, refLatents.data)
            let schedulerAbsmax = absmax(schedulerResult)
            let refAbsmax = absmax(refLatents.data)

            // Test 2: Skip full pipeline for now — just report scheduler parity
            print("    Scheduler-only: cos=\(schedulerCos) absmax=\(schedulerAbsmax) (ref=\(refAbsmax))")

            // Use reference latents for next step (so errors don't accumulate)
            latents = refLatents.data
        }

        print("\nDone.")
    }

    // MARK: - Run with Python trace inputs (bypass tokenizer + RNG)

    private func runWithTraceInputs(modelURL: URL, traceDir: URL) async throws {
        print("Running pipeline with Python trace inputs from: \(traceDir.path)")

        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        var descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)
        let components = try await descriptor.loadComponents(from: modelURL)

        let initialLatents = try loadNpy(traceDir.appendingPathComponent("initial_latents.npy"))
        let textEmbeddings = try loadNpy(traceDir.appendingPathComponent("text_embeddings.npy"))
        let negEmbeddings = try loadNpy(traceDir.appendingPathComponent("neg_embeddings.npy"))
        let timestepsData = try loadNpy(traceDir.appendingPathComponent("timesteps.npy"))

        let timesteps = timestepsData.data.map { Int($0) }
        let latentShape = initialLatents.shape
        var latents = initialLatents.data

        let predictionType = descriptor.predictionType ?? .epsilon
        let defaults = descriptor.scheduler ?? PipelineDescriptor.SchedulerDefaults()
        let scheduler = DPMSolverMultistepScheduler(
            stepCount: timesteps.count,
            trainStepCount: defaults.trainingSteps,
            betaSchedule: .scaledLinear,
            betaStart: defaults.betaStart,
            betaEnd: defaults.betaEnd,
            predictionType: predictionType
        )

        print("  Steps: \(timesteps.count), Timesteps: \(timesteps)")

        // Prepare batched embeddings [2, 77, dim]
        let embDim = textEmbeddings.data.count / 77
        let batchedEmbeddings = negEmbeddings.data + textEmbeddings.data
        let batchedEmbShape = [2, 77, embDim]

        for (i, t) in timesteps.enumerated() {
            // Batch latents [2, C, H, W]
            let batchedLatents = latents + latents
            let batchedLatentShape = [2] + latentShape[1...]

            // UNet with [Float] API
            let batchSize = 2
            let timestepData = [Float](repeating: Float(t), count: batchSize)
            let unetOutput = try await components.denoiser.function.run(floatInputs: [
                (batchedLatents, Array(batchedLatentShape)),
                (timestepData, [batchSize]),
                (batchedEmbeddings, batchedEmbShape),
            ])

            // CFG (use the explicit CLI value if present, else fall back to
            // the SD-1.x default that this parity test was written against).
            let cfgScale = guidanceScale ?? 7.5
            let half = unetOutput.count / 2
            var guided = [Float](repeating: 0, count: half)
            for j in 0..<half {
                guided[j] = unetOutput[j] + cfgScale * (unetOutput[half + j] - unetOutput[j])
            }

            // Compare with Python reference
            let refNoisePath = traceDir.appendingPathComponent("noise_pred_step\(i).npy")
            if let refNoise = try? loadNpy(refNoisePath) {
                let cos = cosineSimilarity(guided, refNoise.data)
                print("  Step \(i) (t=\(t)): guided_cos=\(cos) absmax=\(absmax(guided)) ref=\(absmax(refNoise.data))")
            }

            latents = scheduler.step(output: guided, timeStep: t, sample: latents)
        }

        // Decode
        let scaleFactor = descriptor.decoderScaleFactor ?? 0.18215
        let scaledLatents = latents.map { $0 / scaleFactor }
        let pixels = try await components.decoder.function.run(floatInputs: [(scaledLatents, latentShape)])

        // Save image
        let size = descriptor.imageSize ?? 512
        let image = try DiffusionUtilities.pixelsToCGImage(pixels, height: size, width: size)
        let outputURL = URL(fileURLWithPath: output)
        try saveImage(image, to: outputURL)
        print("  Saved: \(output)")
    }

    private func runComponentParity(modelURL: URL, dataDir: URL) async throws {
        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        let descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)

        guard let teAsset = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }
        guard let unetAsset = descriptor.components.unet else {
            throw PipelineLoadError.missingComponent("unet")
        }
        guard let vaeAsset = descriptor.components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }

        let textEncoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(teAsset))
        let unet = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(unetAsset))
        let vaeDecoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(vaeAsset))

        // --- Text Encoder ---
        print("\n=== Text Encoder ===")
        let inputIds = try loadNpy(dataDir.appendingPathComponent("text_encoder_input_ids.npy"))
        let expectedTE = try loadNpy(dataDir.appendingPathComponent("text_encoder_output.npy"))

        let inputArray = floatsToNDArray(inputIds.data, asInt32: true, shape: inputIds.shape)
        let teOutputs = try await textEncoder.predictAutoNamed(inputs: [inputArray])
        if let actual = teOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedTE.data)
            print("  Cosine similarity: \(cosine)")
        }

        // --- UNet ---
        print("\n=== UNet ===")
        let unetSample = try loadNpy(dataDir.appendingPathComponent("unet_sample.npy"))
        let unetTimestep = try loadNpy(dataDir.appendingPathComponent("unet_timestep.npy"))
        let unetHidden = try loadNpy(dataDir.appendingPathComponent("unet_hidden_states.npy"))
        let expectedUnet = try loadNpy(dataDir.appendingPathComponent("unet_output.npy"))

        try await unet.loadResources()
        let unetInputDescs = try await unet.inputDescriptors

        let sampleType = unetInputDescs["sample"]?.scalarType ?? .float32
        let tsType = unetInputDescs["timestep"]?.scalarType ?? .float32
        let hiddenType = unetInputDescs["encoder_hidden_states"]?.scalarType ?? .float32

        let unetOutputs = try await unet.predict(inputs: [
            "sample": floatsToNDArray(unetSample.data, asInt32: false, shape: unetSample.shape, scalarType: sampleType),
            "timestep": floatsToNDArray(
                unetTimestep.data, asInt32: false, shape: unetTimestep.shape, scalarType: tsType),
            "encoder_hidden_states": floatsToNDArray(
                unetHidden.data, asInt32: false, shape: unetHidden.shape, scalarType: hiddenType),
        ])
        if let actual = unetOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedUnet.data)
            let lInf = zip(actual, expectedUnet.data).map { abs($0 - $1) }.max() ?? 0
            print("  Cosine similarity: \(cosine)")
            print("  L∞: \(lInf)")
        }

        // --- VAE Decoder ---
        print("\n=== VAE Decoder ===")
        let vaeInput = try loadNpy(dataDir.appendingPathComponent("vae_decoder_input.npy"))
        let expectedVAE = try loadNpy(dataDir.appendingPathComponent("vae_decoder_output.npy"))

        try await vaeDecoder.loadResources()
        let vaeInputDescs = try await vaeDecoder.inputDescriptors
        let vaeType = vaeInputDescs.values.first?.scalarType ?? .float32
        let vaeND = floatsToNDArray(vaeInput.data, asInt32: false, shape: vaeInput.shape, scalarType: vaeType)
        let vaeOutputs = try await vaeDecoder.predictAutoNamed(inputs: [vaeND])
        if let actual = vaeOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedVAE.data)
            let lInf = zip(actual, expectedVAE.data).map { abs($0 - $1) }.max() ?? 0
            print("  Cosine similarity: \(cosine)")
            print("  L∞: \(lInf)")
        }

        print("\nDone.")
    }

    // MARK: - SD3 Component Parity (MMDiT + dual CLIP)

    private func runSD3ComponentParity(modelURL: URL, dataDir: URL) async throws {
        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        let descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)

        guard let teAsset = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }
        guard let te2Asset = descriptor.components.textEncoder2 else {
            throw PipelineLoadError.missingComponent("text_encoder_2")
        }
        guard let mmditAsset = descriptor.components.unet else {
            throw PipelineLoadError.missingComponent("transformer/mmdit")
        }
        guard let vaeAsset = descriptor.components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }

        let textEncoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(teAsset))
        let textEncoder2 = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(te2Asset))
        let transformer = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(mmditAsset))
        let vaeDecoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(vaeAsset))

        // --- Text Encoder (CLIP-L) ---
        print("\n=== Text Encoder (CLIP-L) ===")
        let teInputIds = try loadNpy(dataDir.appendingPathComponent("text_encoder_input_ids.npy"))
        let expectedTEHidden = try loadNpy(dataDir.appendingPathComponent("text_encoder_hidden.npy"))

        let teInputND = floatsToNDArray(teInputIds.data, asInt32: true, shape: teInputIds.shape)
        let teOutputs = try await textEncoder.predictAllOutputs(inputs: ["input_ids": teInputND])
        if let hiddenKey = teOutputs.keys.first(where: { $0.contains("hidden") || $0.contains("last") }),
            let actual = teOutputs[hiddenKey]
        {
            let cosine = cosineSimilarity(actual, expectedTEHidden.data)
            print("  Cosine similarity: \(cosine)")
        } else if let actual = teOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedTEHidden.data)
            print("  Cosine similarity (first output): \(cosine)")
        }

        // --- Text Encoder 2 (CLIP-G) ---
        print("\n=== Text Encoder 2 (CLIP-G) ===")
        let te2InputIds = try loadNpy(dataDir.appendingPathComponent("text_encoder_2_input_ids.npy"))
        let expectedTE2Hidden = try loadNpy(dataDir.appendingPathComponent("text_encoder_2_hidden.npy"))

        let te2InputND = floatsToNDArray(te2InputIds.data, asInt32: true, shape: te2InputIds.shape)
        let te2Outputs = try await textEncoder2.predictAllOutputs(inputs: ["input_ids": te2InputND])
        if let hiddenKey = te2Outputs.keys.first(where: { $0.contains("hidden") || $0.contains("last") }),
            let actual = te2Outputs[hiddenKey]
        {
            let cosine = cosineSimilarity(actual, expectedTE2Hidden.data)
            print("  Cosine similarity: \(cosine)")
        } else if let actual = te2Outputs.values.first {
            let cosine = cosineSimilarity(actual, expectedTE2Hidden.data)
            print("  Cosine similarity (first output): \(cosine)")
        }

        // --- MMDiT (Transformer) ---
        print("\n=== MMDiT (Transformer) ===")
        let mmditHidden = try loadNpy(dataDir.appendingPathComponent("mmdit_hidden_states.npy"))
        let mmditTimestep = try loadNpy(dataDir.appendingPathComponent("mmdit_timestep.npy"))
        let mmditEnc = try loadNpy(dataDir.appendingPathComponent("mmdit_encoder_hidden_states.npy"))
        let mmditPooled = try loadNpy(dataDir.appendingPathComponent("mmdit_pooled_projections.npy"))
        let expectedMMDiT = try loadNpy(dataDir.appendingPathComponent("mmdit_output.npy"))

        try await transformer.loadResources()
        let mmditInputDescs = try await transformer.inputDescriptors
        print("  Model inputs: \(mmditInputDescs.keys.sorted())")

        // Map reference tensor names to model input names by position
        let mmditInputNames = mmditInputDescs.keys.sorted()
        let referenceData: [(String, NpyData, Bool)] = [
            ("sample", mmditHidden, false),
            ("timestep", mmditTimestep, false),
            ("encoder_hidden_states", mmditEnc, false),
            ("pooled_projections", mmditPooled, false),
        ]

        var mmditInputs: [String: NDArray] = [:]
        for (refName, npy, asInt) in referenceData {
            if mmditInputNames.contains(refName) {
                let expectedType = mmditInputDescs[refName]?.scalarType ?? .float32
                mmditInputs[refName] = floatsToNDArray(
                    npy.data, asInt32: asInt, shape: npy.shape, scalarType: expectedType)
            }
        }

        // If no exact matches, map by position (export may rename inputs)
        if mmditInputs.isEmpty {
            print("  No exact name matches — mapping by input order:")
            for (i, (refName, npy, asInt)) in referenceData.enumerated() where i < mmditInputNames.count {
                let modelName = mmditInputNames[i]
                let expectedType = mmditInputDescs[modelName]?.scalarType ?? .float32
                print("    \(refName) → \(modelName)")
                mmditInputs[modelName] = floatsToNDArray(
                    npy.data, asInt32: asInt, shape: npy.shape, scalarType: expectedType)
            }
        }

        let mmditOutputs = try await transformer.predict(inputs: mmditInputs)
        if let actual = mmditOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedMMDiT.data)
            let lInf = zip(actual, expectedMMDiT.data).map { abs($0 - $1) }.max() ?? 0
            print("  Cosine similarity: \(cosine)")
            print("  L∞: \(lInf)")
        }

        // --- VAE Decoder ---
        print("\n=== VAE Decoder ===")
        let vaeInput = try loadNpy(dataDir.appendingPathComponent("vae_decoder_input.npy"))
        let expectedVAE = try loadNpy(dataDir.appendingPathComponent("vae_decoder_output.npy"))

        try await vaeDecoder.loadResources()
        let vaeInputDescs = try await vaeDecoder.inputDescriptors
        let vaeScalarType = vaeInputDescs.values.first?.scalarType ?? .float32
        let vaeND = floatsToNDArray(vaeInput.data, asInt32: false, shape: vaeInput.shape, scalarType: vaeScalarType)
        let vaeOutputs = try await vaeDecoder.predictAutoNamed(inputs: [vaeND])
        if let actual = vaeOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedVAE.data)
            let lInf = zip(actual, expectedVAE.data).map { abs($0 - $1) }.max() ?? 0
            print("  Cosine similarity: \(cosine)")
            print("  L∞: \(lInf)")
        }

        print("\nDone.")
    }

    // MARK: - Flux2 Component Parity

    private func runFlux2ComponentParity(modelURL: URL, dataDir: URL) async throws {
        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        let descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)

        guard let teAsset = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }
        guard let vaeAsset = descriptor.components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }

        let textEncoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(teAsset))
        let vaeDecoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(vaeAsset))

        // --- Text Encoder ---
        print("\n=== Text Encoder ===")
        let teInputIds = try loadNpy(dataDir.appendingPathComponent("text_encoder_input_ids.npy"))
        let teAttMask = try loadNpy(dataDir.appendingPathComponent("text_encoder_attention_mask.npy"))
        let expectedTE = try loadNpy(dataDir.appendingPathComponent("text_encoder_output.npy"))

        try await textEncoder.loadResources()
        let teInputDescs = try await textEncoder.inputDescriptors
        let idsType = teInputDescs["input_ids"]?.scalarType ?? .int32
        let maskType = teInputDescs["attention_mask"]?.scalarType ?? .int32

        let teOutputs = try await textEncoder.predictAllOutputs(inputs: [
            "input_ids": floatsToNDArray(teInputIds.data, asInt32: true, shape: teInputIds.shape, scalarType: idsType),
            "attention_mask": floatsToNDArray(
                teAttMask.data, asInt32: true, shape: teAttMask.shape, scalarType: maskType),
        ])
        if let hiddenKey = teOutputs.keys.first(where: { $0.contains("hidden") }),
            let actual = teOutputs[hiddenKey]
        {
            let cosine = cosineSimilarity(actual, expectedTE.data)
            print("  Cosine similarity: \(cosine)")
        } else if let actual = teOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedTE.data)
            print("  Cosine similarity (first output): \(cosine)")
        }

        // --- VAE Decoder ---
        print("\n=== VAE Decoder ===")
        let vaeInput = try loadNpy(dataDir.appendingPathComponent("vae_decoder_input.npy"))
        let expectedVAE = try loadNpy(dataDir.appendingPathComponent("vae_decoder_output.npy"))

        try await vaeDecoder.loadResources()
        let vaeInputDescs = try await vaeDecoder.inputDescriptors
        let vaeType = vaeInputDescs.values.first?.scalarType ?? .float32
        let vaeND = floatsToNDArray(vaeInput.data, asInt32: false, shape: vaeInput.shape, scalarType: vaeType)
        let vaeOutputs = try await vaeDecoder.predictAutoNamed(inputs: [vaeND])
        if let actual = vaeOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedVAE.data)
            print("  Cosine similarity: \(cosine)")
        }

        print("\nDone.")
    }

    // MARK: - Numpy Loader

    struct NpyData {
        let shape: [Int]
        let data: [Float]
    }

    private func loadNpy(_ url: URL) throws -> NpyData {
        let raw = try Data(contentsOf: url)
        // Minimal .npy parser: magic, version, header_len, then FORTRAN header, then data
        guard raw.count > 10, raw[0] == 0x93, raw[1] == 0x4E else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let version = raw[6]
        let headerLen: Int
        let headerStart: Int
        if version == 1 {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8) | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let header = String(data: raw[headerStart..<dataStart], encoding: .ascii) ?? ""

        // Parse shape from header: 'shape': (2, 4, 64, 64)
        let shape = parseShape(from: header)

        // Parse dtype
        let isFloat32 = header.contains("f4") || header.contains("float32")
        let isInt64 = header.contains("i8") || header.contains("int64")
        let isInt32 = header.contains("i4") || header.contains("int32")

        let elementCount = shape.reduce(1, *)
        var floats = [Float](repeating: 0, count: elementCount)

        raw.withUnsafeBytes { ptr in
            let dataPtr = ptr.baseAddress! + dataStart
            if isFloat32 {
                let src = dataPtr.assumingMemoryBound(to: Float.self)
                for i in 0..<elementCount { floats[i] = src[i] }
            } else if isInt64 {
                let src = dataPtr.assumingMemoryBound(to: Int64.self)
                for i in 0..<elementCount { floats[i] = Float(src[i]) }
            } else if isInt32 {
                let src = dataPtr.assumingMemoryBound(to: Int32.self)
                for i in 0..<elementCount { floats[i] = Float(src[i]) }
            }
        }

        return NpyData(shape: shape, data: floats)
    }

    private func parseShape(from header: String) -> [Int] {
        guard let start = header.range(of: "("),
            let end = header.range(of: ")", range: start.upperBound..<header.endIndex)
        else { return [] }
        let shapeStr = header[start.upperBound..<end.lowerBound]
        return shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    // MARK: - Helpers

    private func floatsToNDArray(_ floats: [Float], asInt32: Bool, shape: [Int], scalarType: NDArray.ScalarType? = nil)
        -> NDArray
    {
        if asInt32 {
            var array = NDArray(shape: shape, scalarType: .int32)
            let view = array.mutableView(as: Int32.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<floats.count { ptr[i] = Int32(floats[i]) }
            }
            return array
        } else if scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            var array = NDArray(shape: shape, scalarType: .float16)
            let view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<floats.count { ptr[i] = Float16(floats[i]) }
            }
            return array
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            var array = NDArray(shape: shape, scalarType: scalarType ?? .float32)
            let view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<floats.count { ptr[i] = floats[i] }
            }
            return array
        }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func mean(_ a: [Float]) -> Float {
        a.reduce(0, +) / Float(a.count)
    }

    private func absmax(_ a: [Float]) -> Float {
        a.map(abs).max() ?? 0
    }
}
