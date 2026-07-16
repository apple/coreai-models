// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// Stable Diffusion pipeline (SD 1.5 / 2.0 / 2.1).
///
/// Orchestrates: text encode → denoise loop → VAE decode.
/// All intermediate computation in [Float]. NDArray only at model I/O boundary.
public struct StableDiffusionPipeline: DiffusionPipeline {
    public let descriptor: PipelineDescriptor
    private let components: CoreAIDiffusionComponents

    public var defaultImageSize: (width: Int, height: Int) {
        let size = descriptor.imageSize ?? 512
        return (size, size)
    }

    public var supportedSchedulers: [SchedulerType] {
        [.pndm, .dpmSolverMultistep]
    }

    public var supportsImageToImage: Bool {
        components.encoder != nil
    }

    // MARK: - Loading

    /// Load a Stable Diffusion pipeline from a model directory.
    ///
    /// ```swift
    /// let pipeline = try await StableDiffusionPipeline.load(from: directory)
    ///
    /// let config = PipelineConfiguration(
    ///     prompt: "a photograph of an astronaut riding a horse",
    ///     stepCount: 20,
    ///     guidanceScale: 7.5,
    ///     schedulerType: .dpmSolverMultistep
    /// )
    ///
    /// let result = try await pipeline.generateImages(configuration: config) { progress in
    ///     print("Step \(progress.step)/\(progress.totalSteps)")
    ///     return true  // return false to cancel
    /// }
    /// // result.images: [CGImage]
    /// ```
    public static func load(
        from url: URL,
        config: PipelineDescriptor.ConfigSource = .auto
    ) async throws -> StableDiffusionPipeline {
        var descriptor = try PipelineDescriptor.resolve(at: url, config: config)
        let components = try await descriptor.loadComponents(from: url)
        return StableDiffusionPipeline(descriptor: descriptor, components: components)
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        try await components.textEncoder.function.loadResources()
        try await components.denoiser.function.loadResources()
        try await components.decoder.function.loadResources()
        if let encoder = components.encoder {
            try await encoder.function.loadResources()
        }
    }

    public func unloadResources() async {
        await components.textEncoder.function.unloadResources()
        await components.denoiser.function.unloadResources()
        await components.decoder.function.unloadResources()
        if let encoder = components.encoder {
            await encoder.function.unloadResources()
        }
    }

    // MARK: - Generation

    public func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: (PipelineProgress) -> Bool
    ) async throws -> GenerationResult {
        let scaleFactor = descriptor.decoderScaleFactor ?? 0.18215
        let predictionType = descriptor.predictionType ?? .epsilon

        // 1. Encode text (returns [Float])
        let textEmbeddings = try await encodeText(configuration.prompt)
        let negativeEmbeddings = try await encodeText(configuration.negativePrompt)
        if configuration.lazyModelLoading {
            await components.textEncoder.function.unloadResources()
        }

        // 2. Create schedule
        let schedule = try createSchedule(
            type: configuration.schedulerType,
            stepCount: configuration.stepCount,
            predictionType: predictionType
        )

        // 3. Prepare initial latents
        let size = defaultImageSize
        let latentHeight = size.height / 8
        let latentWidth = size.width / 8
        let latentShape = [1, 4, latentHeight, latentWidth]
        let latentCount = latentShape.reduce(1, *)
        var latents = generateNoise(count: latentCount, seed: configuration.seed)

        // 4. Denoise loop — pure [Float] math
        let batchedEmbeddings = negativeEmbeddings + textEmbeddings  // [2, 77, dim]
        let batchedEmbShape = [2, 77, textEmbeddings.count / 77]

        for (step, timeStep) in schedule.timeSteps.enumerated() {
            let progress = PipelineProgress(step: step, totalSteps: schedule.timeSteps.count, currentLatent: nil)
            if !progressHandler(progress) { break }

            // CFG: batch latents [2, 4, H, W]
            let batchedLatents = latents + latents
            let batchedLatentShape = [2] + latentShape[1...]

            // UNet forward
            let unetOutput = try await runDenoiser(
                latents: batchedLatents, latentShape: Array(batchedLatentShape),
                timestep: timeStep,
                embeddings: batchedEmbeddings, embeddingShape: batchedEmbShape
            )

            // Split + guidance
            let half = unetOutput.count / 2
            var guided = [Float](repeating: 0, count: half)
            let scale = configuration.guidanceScale
            for i in 0..<half {
                guided[i] = unetOutput[i] + scale * (unetOutput[half + i] - unetOutput[i])
            }

            // Scheduler step
            latents = schedule.step(guided, timeStep, latents)
        }

        if configuration.lazyModelLoading {
            await components.denoiser.function.unloadResources()
        }

        // 5. Decode latents → pixels → CGImage
        var scaledLatents = [Float](repeating: 0, count: latentCount)
        let invScale = 1.0 / scaleFactor
        for i in 0..<latentCount {
            scaledLatents[i] = latents[i] * invScale
        }

        let pixels = try await components.decoder.function.run(
            floatInputs: [(scaledLatents, latentShape)])
        if configuration.lazyModelLoading {
            await components.decoder.function.unloadResources()
        }
        let image = try DiffusionUtilities.pixelsToCGImage(pixels, height: size.height, width: size.width)

        // Wrap latents back to NDArray for GenerationResult
        var latentsND = NDArray(shape: latentShape, scalarType: .float32)
        var latentsView = latentsND.mutableView(as: Float.self)
        latentsView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<latents.count { ptr[i] = latents[i] }
        }

        return GenerationResult(images: [image], latents: [latentsND])
    }

    // MARK: - Private Helpers

    private func encodeText(_ text: String) async throws -> [Float] {
        let output = try await components.textEncoder.encode(text)
        let shape = output.hiddenStates.shape
        let count = shape.reduce(1, *)
        return readNDArray(output.hiddenStates, as: Float.self, count: count)
    }

    private func runDenoiser(
        latents: [Float], latentShape: [Int],
        timestep: Int,
        embeddings: [Float], embeddingShape: [Int]
    ) async throws -> [Float] {
        let batchSize = latentShape[0]
        let timestepData = [Float](repeating: Float(timestep), count: batchSize)
        return try await components.denoiser.function.run(
            floatInputs: [
                (latents, latentShape),
                (timestepData, [batchSize]),
                (embeddings, embeddingShape),
            ])
    }

    private struct Schedule {
        let timeSteps: [Int]
        let step: ([Float], Int, [Float]) -> [Float]
    }

    private func createSchedule(
        type: SchedulerType, stepCount: Int, predictionType: PredictionType
    ) throws -> Schedule {
        let defaults = descriptor.scheduler ?? PipelineDescriptor.SchedulerDefaults()
        switch type {
        case .pndm:
            let s = PNDMScheduler(
                stepCount: stepCount,
                trainStepCount: defaults.trainingSteps,
                betaSchedule: .scaledLinear,
                betaStart: defaults.betaStart,
                betaEnd: defaults.betaEnd,
                predictionType: predictionType
            )
            return Schedule(timeSteps: s.timeSteps, step: s.step)
        case .dpmSolverMultistep:
            let s = DPMSolverMultistepScheduler(
                stepCount: stepCount,
                trainStepCount: defaults.trainingSteps,
                betaSchedule: .scaledLinear,
                betaStart: defaults.betaStart,
                betaEnd: defaults.betaEnd,
                predictionType: predictionType
            )
            return Schedule(timeSteps: s.timeSteps, step: s.step)
        case .discreteFlow:
            throw CoreAIComponentError.invalidShape(
                "discreteFlow is not supported by StableDiffusionPipeline — use SD3Pipeline or Flux2Pipeline")
        }
    }

    private func generateNoise(count: Int, seed: UInt32) -> [Float] {
        var rng = NumPyRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    }
}
