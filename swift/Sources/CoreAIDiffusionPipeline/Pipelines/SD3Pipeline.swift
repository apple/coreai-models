// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// Stable Diffusion 3.x (MMDiT) pipeline using Core AI backend.
///
/// Orchestrates: tokenize × 2 → encode CLIP-L + CLIP-G → compose encoder
/// hidden states and pooled projections → noise → denoise loop (flow-match
/// Euler) with classifier-free guidance → VAE decode → image.
///
/// T5-less path: the T5 portion of `encoder_hidden_states` is zero-padded
/// (matches diffusers' `text_encoder_3=None` behaviour).
public struct SD3Pipeline: DiffusionPipeline {
    public let descriptor: PipelineDescriptor

    public let transformer: CoreAIDiffusionModelFunction  // MMDiT
    public let textEncoder: CoreAIDiffusionModelFunction  // CLIP-L
    public let textEncoder2: CoreAIDiffusionModelFunction  // CLIP-G
    public let decoder: CoreAIDiffusionModelFunction  // VAE decoder
    public let tokenizer: BPETokenizer  // CLIP-L
    public let tokenizer2: BPETokenizer  // CLIP-G

    public var defaultImageSize: (width: Int, height: Int) {
        let size = descriptor.imageSize ?? 1024
        return (size, size)
    }

    public var supportedSchedulers: [SchedulerType] {
        [.discreteFlow]
    }

    public var supportsImageToImage: Bool { false }

    public init(
        descriptor: PipelineDescriptor,
        transformer: CoreAIDiffusionModelFunction,
        textEncoder: CoreAIDiffusionModelFunction,
        textEncoder2: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        tokenizer: BPETokenizer,
        tokenizer2: BPETokenizer
    ) {
        self.descriptor = descriptor
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.textEncoder2 = textEncoder2
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        try await transformer.loadResources()
        try await textEncoder.loadResources()
        try await textEncoder2.loadResources()
        try await decoder.loadResources()
    }

    public func unloadResources() async {
        await transformer.unloadResources()
        await textEncoder.unloadResources()
        await textEncoder2.unloadResources()
        await decoder.unloadResources()
    }

    // MARK: - Generation

    public func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: (PipelineProgress) -> Bool
    ) async throws -> GenerationResult {
        if !configuration.lazyModelLoading { try await loadResources() }

        let steps = configuration.stepCount
        let cfgScale = configuration.guidanceScale
        let negativePrompt = configuration.negativePrompt

        // Cache dims from the first CLIP-L text encode (also gives us the
        // hidden/pooled dims we need for zero-padding the other encoder).
        let imageSize = defaultImageSize.width
        let sampleSize = imageSize / 8  // VAE downscale factor
        let inChannels = 16  // SD 3.5 VAE latent channels
        let latentShape = [1, inChannels, sampleSize, sampleSize]
        let batchedLatentShape = [2, inChannels, sampleSize, sampleSize]

        // 1. Text encode (both prompts through both encoders)
        let uncond = try await encodeDualText(negativePrompt)
        let cond = try await encodeDualText(configuration.prompt)
        if configuration.lazyModelLoading {
            await textEncoder.unloadResources()
            await textEncoder2.unloadResources()
        }

        // 2. Compose batched (uncond, cond) inputs for CFG
        let encoderHiddenStates = uncond.composedHidden + cond.composedHidden
        let pooledProjections = uncond.composedPooled + cond.composedPooled
        let encoderHiddenShape = [2, Self.jointSeqLen, Self.jointAttentionDim]
        let pooledShape = [2, Self.pooledProjectionDim]

        // 3. Initial noise
        var latents = generateNoise(
            count: latentShape.reduce(1, *),
            seed: configuration.seed)

        // 4. Scheduler (SD3 flow matching; plain shift, no dynamic mu)
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps,
            trainStepCount: 1000,
            timeStepShift: 3.0)

        // 5. Denoise loop
        for (step, t) in scheduler.timeSteps.enumerated() {
            let batchedLatents = latents + latents
            let timestepValue = Float(t)
            let batchedTimesteps: [Float] = [timestepValue, timestepValue]

            let output = try await transformer.run(floatInputs: [
                (batchedLatents, batchedLatentShape),
                (batchedTimesteps, [2]),
                (encoderHiddenStates, encoderHiddenShape),
                (pooledProjections, pooledShape),
            ])

            // CFG: guided = uncond + scale * (cond - uncond)
            let half = output.count / 2
            var guided = [Float](repeating: 0, count: half)
            for i in 0..<half {
                let uncondV = output[i]
                let condV = output[half + i]
                guided[i] = uncondV + cfgScale * (condV - uncondV)
            }

            latents = scheduler.step(output: guided, timeStep: t, sample: latents)

            let progress = PipelineProgress(step: step + 1, totalSteps: steps, currentLatent: nil)
            if !progressHandler(progress) { break }
        }

        if configuration.lazyModelLoading { await transformer.unloadResources() }

        // 6. VAE decode
        // SD3 convention: z_vae = (z / scaling_factor) + shift_factor
        let scaleFactor = descriptor.decoderScaleFactor ?? 1.5305
        let shiftFactor = descriptor.decoderShiftFactor ?? 0.0609
        let scaledLatents = latents.map { $0 / scaleFactor + shiftFactor }
        let pixels = try await decoder.run(floatInputs: [(scaledLatents, latentShape)])
        if configuration.lazyModelLoading { await decoder.unloadResources() }

        // 7. To CGImage. SD3 VAE outputs in [-1, 1] like SD 1.5/2.x.
        let image = try DiffusionUtilities.pixelsToCGImage(
            pixels, height: imageSize, width: imageSize)

        var latentsND = NDArray(shape: latentShape, scalarType: .float32)
        var latentsView = latentsND.mutableView(as: Float.self)
        latentsView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<latents.count { ptr[i] = latents[i] }
        }

        return GenerationResult(images: [image], latents: [latentsND])
    }

    // MARK: - Text Encoding

    private struct DualTextOutput {
        let composedHidden: [Float]  // [1, 154, 4096] flattened
        let composedPooled: [Float]  // [1, 2048] flattened
    }

    private func encodeDualText(_ text: String) async throws -> DualTextOutput {
        async let clipL = encode(
            text, via: textEncoder, tokenizer: tokenizer,
            hiddenDim: Self.clipLHiddenDim)
        async let clipG = encode(
            text, via: textEncoder2, tokenizer: tokenizer2,
            hiddenDim: Self.clipGHiddenDim)
        let (l, g) = try await (clipL, clipG)

        // Zero-pad each encoder's hidden state along channel dim to jointAttentionDim (4096)
        // then concat along seq dim. Matches generate_parity_data.py.
        let lPadded = Self.padChannels(
            l.hidden, seqLen: Self.clipSeqLen,
            fromDim: Self.clipLHiddenDim, toDim: Self.jointAttentionDim)
        let gPadded = Self.padChannels(
            g.hidden, seqLen: Self.clipSeqLen,
            fromDim: Self.clipGHiddenDim, toDim: Self.jointAttentionDim)
        let composedHidden = lPadded + gPadded  // [1, 154, 4096]

        // Pooled projections: concat CLIP-L pooled + CLIP-G pooled → [1, 2048]
        let composedPooled = l.pooled + g.pooled

        return DualTextOutput(
            composedHidden: composedHidden,
            composedPooled: composedPooled)
    }

    private struct TextEncoderResult {
        let hidden: [Float]  // [1, seq, hiddenDim]
        let pooled: [Float]  // [1, pooledDim]
    }

    private func encode(
        _ text: String,
        via function: CoreAIDiffusionModelFunction,
        tokenizer: BPETokenizer,
        hiddenDim: Int
    ) async throws -> TextEncoderResult {
        let (_, intIds) = tokenizer.tokenize(input: text, minCount: Self.clipSeqLen)
        var ids = intIds.prefix(Self.clipSeqLen).map { Int32($0) }
        if ids.count < Self.clipSeqLen {
            ids += [Int32](repeating: 0, count: Self.clipSeqLen - ids.count)
        }

        let inputDescs = try await function.inputDescriptors
        guard inputDescs.count == 1, let inputName = inputDescs.keys.first else {
            throw CoreAIComponentError.invalidShape(
                "Text encoder expects 1 input; got \(inputDescs.count)")
        }

        var inputArray = NDArray(shape: [1, Self.clipSeqLen], scalarType: .int32)
        var view = inputArray.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<ids.count { ptr[i] = ids[i] }
        }

        let outputs = try await function.predictAllOutputs(inputs: [inputName: inputArray])
        let outputDescs = try await function.outputDescriptors

        var hidden: [Float]?
        var pooled: [Float]?
        for (name, floats) in outputs {
            let rank = outputDescs[name]?.shape.count ?? 0
            if rank == 3 {
                hidden = floats
            } else if rank == 2 {
                pooled = floats
            }
        }

        guard let hidden else {
            throw CoreAIComponentError.invalidShape(
                "SD3 text encoder returned no rank-3 hidden state")
        }
        guard let pooled else {
            throw CoreAIComponentError.invalidShape(
                "SD3 text encoder returned no rank-2 pooled output")
        }
        guard hidden.count == Self.clipSeqLen * hiddenDim else {
            throw CoreAIComponentError.invalidShape(
                "Hidden state has \(hidden.count) elts; expected \(Self.clipSeqLen * hiddenDim)")
        }
        return TextEncoderResult(hidden: hidden, pooled: pooled)
    }

    // MARK: - Helpers

    /// Zero-pad every `fromDim`-channel token to `toDim` channels.
    /// Input: [seqLen * fromDim]; output: [seqLen * toDim].
    private static func padChannels(
        _ input: [Float], seqLen: Int, fromDim: Int, toDim: Int
    ) -> [Float] {
        precondition(toDim >= fromDim)
        if fromDim == toDim { return input }
        var result = [Float](repeating: 0, count: seqLen * toDim)
        for t in 0..<seqLen {
            let srcBase = t * fromDim
            let dstBase = t * toDim
            for c in 0..<fromDim {
                result[dstBase + c] = input[srcBase + c]
            }
        }
        return result
    }

    private func generateNoise(count: Int, seed: UInt32) -> [Float] {
        var rng = NumPyRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    }

    // MARK: - SD 3.5 Medium constants

    private static let clipSeqLen = 77
    private static let clipLHiddenDim = 768
    private static let clipGHiddenDim = 1280
    private static let jointSeqLen = 154  // 77 × 2
    private static let jointAttentionDim = 4096  // MMDiT channel dim
    private static let pooledProjectionDim = 2048  // 768 + 1280
}
