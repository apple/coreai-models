// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Tokenizers

/// A traced img2img graph: which asset holds it, and under which entrypoint.
public struct Img2ImgRoute: Sendable {
    public let function: CoreAIDiffusionModelFunction
    public let entrypoint: String

    public init(function: CoreAIDiffusionModelFunction, entrypoint: String) {
        self.function = function
        self.entrypoint = entrypoint
    }
}

/// Flow-matching diffusion-transformer pipeline using Core AI backend.
///
/// Orchestrates: tokenize → text encode → noise → denoise loop (flow-match Euler or
/// stochastic) → VAE decode. The model family (`descriptor.type`) supplies the text
/// conditioning, transformer inputs, sigma schedule, and latent → VAE conversion:
///
/// - `.flux2` (FLUX.2 Klein): Qwen3 encoder, packed tokens with RoPE ids, BN + unpatchify.
///   See `FlowTransformerPipeline+Flux2.swift`.
/// - `.sanaSprint`: Gemma2 encoder, NCHW latents, SCM schedule with stochastic steps.
///   See `FlowTransformerPipeline+SanaSprint.swift`.
public struct FlowTransformerPipeline: DiffusionPipeline {
    public let descriptor: PipelineDescriptor
    public let mode: DecodeResolution

    public let transformer: CoreAIDiffusionModelFunction
    /// How each reference grid reaches a traced graph, resolved at load time.
    ///
    /// img2img arrives two ways and a bundle can contain both, because export directories
    /// accumulate assets across runs. Resolving to (asset, entrypoint) pairs up front
    /// keeps the choice in one place:
    ///
    /// - multi-function: an `img2img_*` entrypoint on `transformer` — preferred, since it
    ///   reuses the already-loaded asset instead of a second ~2 GB weight set
    /// - single-function: a `Transformer[_512]_img2img_<grid>` asset, entrypoint `main`
    ///
    /// A grid absent from both is simply not supported by the bundle.
    public let img2imgRoutes: [ReferenceGrid: Img2ImgRoute]
    public let textEncoder: CoreAIDiffusionModelFunction
    public let decoder: CoreAIDiffusionModelFunction
    public let encoder: CoreAIDiffusionModelFunction?
    public let transformerFunctionName: String
    public let tokenizer: any Tokenizer

    public let batchNormMean: [Float]?
    public let batchNormVar: [Float]?
    public let batchNormEps: Float

    /// Image size is determined by the mode selected at init.
    public var defaultImageSize: (width: Int, height: Int) {
        let full = descriptor.imageSize ?? 1024
        let size = (mode == .half) ? full / 2 : full
        return (size, size)
    }

    public var supportedSchedulers: [SchedulerType] {
        [.discreteFlow]
    }

    public var supportsImageToImage: Bool {
        encoder != nil
    }

    public init(
        descriptor: PipelineDescriptor,
        mode: DecodeResolution = .full,
        transformer: CoreAIDiffusionModelFunction,
        img2imgRoutes: [ReferenceGrid: Img2ImgRoute] = [:],
        textEncoder: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        encoder: CoreAIDiffusionModelFunction?,
        transformerFunctionName: String = "main",
        tokenizer: any Tokenizer,
        batchNormMean: [Float]?,
        batchNormVar: [Float]?,
        batchNormEps: Float
    ) {
        self.descriptor = descriptor
        self.mode = mode
        self.transformer = transformer
        self.img2imgRoutes = img2imgRoutes
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.encoder = encoder
        self.transformerFunctionName = transformerFunctionName
        self.tokenizer = tokenizer
        self.batchNormMean = batchNormMean
        self.batchNormVar = batchNormVar
        self.batchNormEps = batchNormEps

        if descriptor.type != .sanaSprint && tokenizer.convertTokenToId("<|endoftext|>") == nil {
            CLILogger.log(
                "⚠️ FlowTransformerPipeline: tokenizer has no <|endoftext|> token, using Qwen3 fallback pad ID",
                component: "Diffusion")
        }
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        try await transformer.loadResources()
        try await textEncoder.loadResources()
        try await decoder.loadResources()
        if let encoder { try await encoder.loadResources() }
        // The img2img transformers are deliberately *not* loaded here. Each is a full
        // weight set (~2 GB resident on GPU) that a txt2img run never touches, and
        // `CoreAIDiffusionModelFunction` loads itself on first use anyway. They are still
        // unloaded below, so a run that did use one releases it.
    }

    public func unloadResources() async {
        await transformer.unloadResources()
        // Distinct assets only: multi-function routes point back at `transformer`.
        for route in img2imgRoutes.values where route.function !== transformer {
            await route.function.unloadResources()
        }
        await textEncoder.unloadResources()
        await decoder.unloadResources()
        if let encoder { await encoder.unloadResources() }
    }

    // MARK: - Generation

    /// What a model family hands the shared denoise loop and decoder.
    struct DenoisingPlan {
        var latents: [Float]
        let scheduler: DiscreteFlowScheduler
        /// Fresh noise per step for stochastic sampling; nil selects the Euler step.
        let renoise: (() -> [Float])?
        /// Velocity for the current latents at `(step, sigma)`.
        let predict: (_ latents: [Float], _ step: Int, _ sigma: Float) async throws -> [Float]
        /// Denoiser latents → VAE decoder input of shape `vaeShape`.
        let vaeLatents: (_ latents: [Float]) -> [Float]
        let vaeShape: [Int]
        /// The asset `predict` runs, released after the loop under lazy loading.
        let denoiser: CoreAIDiffusionModelFunction
        let initialNoise: [Float]
        let noiseShape: [Int]
    }

    public func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: ((PipelineProgress) -> Bool)?
    ) async throws -> GenerationResult {
        var plan =
            descriptor.type == .sanaSprint
            ? try await makeSanaSprintPlan(configuration)
            : try await makeFlux2Plan(configuration)
        let scheduler = plan.scheduler
        let steps = scheduler.inferenceStepCount

        for step in 0..<steps {
            // Capture the denoising state BEFORE the scheduler advances so the
            // preview below can form the x0 estimate.
            let sigma = scheduler.currentSigma
            let sampleBeforeStep = plan.latents
            let output = try await plan.predict(plan.latents, step, sigma)

            if let renoise = plan.renoise {
                plan.latents = scheduler.stepStochastic(output: output, sample: plan.latents, noise: renoise())
            } else {
                plan.latents = scheduler.step(
                    output: output, timeStep: scheduler.timeSteps[step], sample: plan.latents)
            }
            try checkLatentsAreFinite(plan.latents, step: step)

            if let progressHandler {
                // Preview the DENOISED estimate, not the raw post-step sample. The
                // sample after the Euler step is still mostly noise until the last
                // step or two, so on a few-step model (e.g. FLUX.2 Klein at 4 steps)
                // the early previews look like static. Flow-matching gives the
                // estimate for one multiply-add: with x_t = (1-σ)·x0 + σ·ε and the
                // model predicting v = ε - x0, x0 = x_t - σ·v. Blurry on step one,
                // but it shows the composition and converges to the final image.
                var previewX0 = sampleBeforeStep
                if sigma > 0 {
                    var negSigma = -sigma
                    vDSP_vsma(
                        output, 1, &negSigma, sampleBeforeStep, 1, &previewX0, 1,
                        vDSP_Length(output.count))
                }
                // Array copies into the VAE's latent layout, no model call.
                let vae = plan.vaeLatents(previewX0)
                var previewLatents = NDArray(shape: plan.vaeShape, scalarType: .float32)
                previewLatents.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
                    for i in 0..<vae.count { ptr[i] = vae[i] }
                }
                let progress = PipelineProgress(step: step + 1, totalSteps: steps, currentLatent: previewLatents)
                if !progressHandler(progress) { break }
            }
        }

        if configuration.lazyModelLoading {
            // Release whichever asset ran
            await plan.denoiser.unloadResources()
        }

        // VAE decode
        // Note: self.decoder is mode-appropriate (loaded at init):
        //   .full → VAEDecoder, .half/.tiled → VAEDecoder_half
        let vaeLatents = plan.vaeLatents(plan.latents)
        let vaeShape = plan.vaeShape
        let imageSize = defaultImageSize.width
        let pixels: [Float]

        switch mode {
        case .full, .half:
            pixels = try await decoder.run(floatInputs: [(vaeLatents, vaeShape)])

        case .tiled:
            pixels = try await decodeTiled(
                latents: vaeLatents, channels: vaeShape[1], height: vaeShape[2], width: vaeShape[3],
                decoder: decoder, outputScale: imageSize / vaeShape[2])

        case .auto:
            preconditionFailure("auto resolved at init")
        }

        if configuration.lazyModelLoading { await decoder.unloadResources() }

        let image = try DiffusionUtilities.pixelsToCGImage(pixels, height: imageSize, width: imageSize)

        var latentsND = NDArray(shape: plan.noiseShape, scalarType: .float32)
        let latentsView = latentsND.mutableView(as: Float.self)
        latentsView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<plan.initialNoise.count { ptr[i] = plan.initialNoise[i] }
        }

        return GenerationResult(images: [image], latents: [latentsND])
    }

    // MARK: - Classifier-Free Guidance

    /// `uncond + g*(cond - uncond)`, written into `destination` rather than returned.
    ///
    /// The caller reuses one buffer across denoising steps; at 1024×1024 each result is
    /// ~2 MB, so returning a fresh array would allocate one per step.
    static func applyClassifierFreeGuidance(
        cond: ArraySlice<Float>, uncond: ArraySlice<Float>,
        guidanceScale: Float, into destination: inout [Float]
    ) {
        // Reusing the buffer means a short input would leave the previous step's values
        // in the tail rather than merely producing a short array, so require an exact fit.
        precondition(
            cond.count == destination.count && uncond.count == destination.count,
            "CFG expected \(destination.count) noise values, got "
                + "cond=\(cond.count) uncond=\(uncond.count)")
        for (offset, (u, c)) in zip(uncond, cond).enumerated() {
            destination[offset] = u + guidanceScale * (c - u)
        }
    }

    // MARK: - Half/Tiled Decode Helpers

    /// Area-average downsample BCHW latents by an integer factor using vDSP.
    static func downsampleLatents(
        _ input: [Float], channels: Int, height: Int, width: Int, factor: Int
    ) -> [Float] {
        let outH = height / factor
        let outW = width / factor
        let scale = 1.0 / Float(factor * factor)
        var output = [Float](repeating: 0, count: channels * outH * outW)
        for c in 0..<channels {
            let chIn = c * height * width
            let chOut = c * outH * outW
            for oh in 0..<outH {
                for ow in 0..<outW {
                    var sum: Float = 0
                    for dy in 0..<factor {
                        let rowStart = chIn + (oh * factor + dy) * width + ow * factor
                        for dx in 0..<factor {
                            sum += input[rowStart + dx]
                        }
                    }
                    output[chOut + oh * outW + ow] = sum * scale
                }
            }
        }
        return output
    }

    /// Bicubic 2× upsample planar [C, H, W] image.
    static func bicubicUpsample2x(
        _ input: [Float], channels: Int, height: Int, width: Int
    ) -> [Float] {
        let outH = height * 2
        let outW = width * 2
        var output = [Float](repeating: 0, count: channels * outH * outW)

        for c in 0..<channels {
            let chOffset = c * height * width
            let outChOffset = c * outH * outW
            for oy in 0..<outH {
                let srcY = Float(oy) / 2.0 - 0.25
                for ox in 0..<outW {
                    let srcX = Float(ox) / 2.0 - 0.25
                    output[outChOffset + oy * outW + ox] = bicubicSample(
                        input, offset: chOffset, height: height, width: width, y: srcY, x: srcX)
                }
            }
        }
        return output
    }

    private static func bicubicSample(
        _ data: [Float], offset: Int, height: Int, width: Int, y: Float, x: Float
    ) -> Float {
        let iy = Int(floor(y))
        let ix = Int(floor(x))
        let fy = y - Float(iy)
        let fx = x - Float(ix)

        var result: Float = 0
        for j in -1...2 {
            let wy = cubicWeight(Float(j) - fy)
            for i in -1...2 {
                let wx = cubicWeight(Float(i) - fx)
                let sy = min(max(iy + j, 0), height - 1)
                let sx = min(max(ix + i, 0), width - 1)
                result += wy * wx * data[offset + sy * width + sx]
            }
        }
        return result
    }

    private static func cubicWeight(_ t: Float) -> Float {
        let a: Float = -0.5
        let at = abs(t)
        if at <= 1 {
            return (a + 2) * at * at * at - (a + 3) * at * at + 1
        } else if at < 2 {
            return a * at * at * at - 5 * a * at * at + 8 * a * at - 4 * a
        }
        return 0
    }

    /// Tiled VAE decode: split latents into a grid of tiles, decode each with the half-res VAE, blend overlaps.
    private func decodeTiled(
        latents: [Float], channels: Int, height: Int, width: Int,
        decoder: CoreAIDiffusionModelFunction, outputScale: Int
    ) async throws -> [Float] {
        let tileSize = height / 2
        let overlap = 4
        let stride = tileSize - overlap

        let outTileSize = tileSize * outputScale
        let outOverlap = overlap * outputScale
        let outH = height * outputScale
        let outW = width * outputScale
        let outChannels = 3

        var output = [Float](repeating: 0, count: outChannels * outH * outW)
        var weights = [Float](repeating: 0, count: outH * outW)

        let startsY = tileStarts(length: height, tileSize: tileSize, stride: stride)
        let startsX = tileStarts(length: width, tileSize: tileSize, stride: stride)

        for startY in startsY {
            for startX in startsX {
                let tile = extractTile(
                    from: latents, channels: channels, height: height, width: width,
                    startY: startY, startX: startX, tileSize: tileSize)

                let tileShape = [1, channels, tileSize, tileSize]
                let decodedTile = try await decoder.run(floatInputs: [(tile, tileShape)])

                blendTile(
                    decodedTile, into: &output, weights: &weights,
                    outChannels: outChannels, outH: outH, outW: outW,
                    outTileSize: outTileSize, outOverlap: outOverlap,
                    outStartY: startY * outputScale, outStartX: startX * outputScale)
            }
        }

        normalizeByWeights(&output, weights: weights, channels: outChannels, size: outH * outW)
        return output
    }

    private func extractTile(
        from latents: [Float], channels: Int, height: Int, width: Int,
        startY: Int, startX: Int, tileSize: Int
    ) -> [Float] {
        var tile = [Float](repeating: 0, count: channels * tileSize * tileSize)
        for c in 0..<channels {
            for y in 0..<tileSize {
                for x in 0..<tileSize {
                    let srcY = min(startY + y, height - 1)
                    let srcX = min(startX + x, width - 1)
                    tile[c * tileSize * tileSize + y * tileSize + x] =
                        latents[c * height * width + srcY * width + srcX]
                }
            }
        }
        return tile
    }

    private func blendTile(
        _ decodedTile: [Float], into output: inout [Float], weights: inout [Float],
        outChannels: Int, outH: Int, outW: Int,
        outTileSize: Int, outOverlap: Int,
        outStartY: Int, outStartX: Int
    ) {
        for c in 0..<outChannels {
            for y in 0..<outTileSize {
                let outY = outStartY + y
                guard outY < outH else { continue }
                let wy = blendWeight(y, outTileSize, outOverlap)
                for x in 0..<outTileSize {
                    let outX = outStartX + x
                    guard outX < outW else { continue }
                    let w = wy * blendWeight(x, outTileSize, outOverlap)
                    output[c * outH * outW + outY * outW + outX] +=
                        w * decodedTile[c * outTileSize * outTileSize + y * outTileSize + x]
                    if c == 0 { weights[outY * outW + outX] += w }
                }
            }
        }
    }

    private func normalizeByWeights(
        _ output: inout [Float], weights: [Float], channels: Int, size: Int
    ) {
        for c in 0..<channels {
            let offset = c * size
            for i in 0..<size where weights[i] > 0 {
                output[offset + i] /= weights[i]
            }
        }
    }

    /// Generate tile start positions that cover [0, length) with given tile size and stride.
    private func tileStarts(length: Int, tileSize: Int, stride: Int) -> [Int] {
        var starts: [Int] = []
        var pos = 0
        while pos + tileSize <= length {
            starts.append(pos)
            pos += stride
        }
        if starts.isEmpty || starts.last! + tileSize < length {
            starts.append(length - tileSize)
        }
        return starts
    }

    private func blendWeight(_ pos: Int, _ size: Int, _ overlap: Int) -> Float {
        if pos < overlap {
            return Float(pos) / Float(overlap)
        } else if pos >= size - overlap {
            return Float(size - 1 - pos) / Float(overlap)
        }
        return 1.0
    }
}
