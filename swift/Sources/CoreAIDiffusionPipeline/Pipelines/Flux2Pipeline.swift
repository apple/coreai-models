// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Tokenizers

/// FLUX.2 Klein pipeline using Core AI backend.
///
/// Orchestrates: tokenize → text encode → RoPE compute → noise → pack →
/// denoise loop (flow-match Euler) → unpack → BN denorm → unpatchify → VAE decode.
///
/// Key design: RoPE embeddings are pre-computed in Swift and passed as model inputs
/// (not computed in-graph) to avoid graph optimizer issues on monolithic
/// 25-block transformers.
public struct Flux2Pipeline: DiffusionPipeline {
    public let descriptor: PipelineDescriptor
    public let mode: DecodeResolution

    public let transformer: CoreAIDiffusionModelFunction
    public let textEncoder: CoreAIDiffusionModelFunction
    public let decoder: CoreAIDiffusionModelFunction
    public let encoder: CoreAIDiffusionModelFunction?
    public let tokenizer: any Tokenizer

    public let batchNormMean: [Float]?
    public let batchNormVar: [Float]?
    public let batchNormEps: Float

    // MARK: - Architecture Constants

    private static let patchSize = 16
    private static let latentChannels = 128
    private static let textSeqLen = 512
    private static let defaultRopeTheta: Float = 2000.0

    /// FLUX.2 flow-matching timestep shift.
    ///
    /// Mirrors diffusers `compute_empirical_mu` (diffusers 0.37.1):
    ///   pipelines/flux2/pipeline_flux2_klein.py:63-78
    ///   (copied from pipelines/flux2/pipeline_flux2.py)
    /// Call site — pipeline_flux2_klein.py:810-811:
    ///   image_seq_len = latents.shape[1]
    ///   mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    ///
    /// Reference implementation:
    ///   def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    ///       a1, b1 = 8.73809524e-05, 1.89833333
    ///       a2, b2 = 0.00016927, 0.45666666
    ///       if image_seq_len > 4300:
    ///           mu = a2 * image_seq_len + b2
    ///           return float(mu)
    ///       m_200 = a2 * image_seq_len + b2
    ///       m_10 = a1 * image_seq_len + b1
    ///       a = (m_200 - m_10) / 190.0
    ///       b = m_200 - 200.0 * a
    ///       mu = a * num_steps + b
    ///       return float(mu)
    private static func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05
        let b1: Float = 1.89833333
        let a2: Float = 0.00016927
        let b2: Float = 0.45666666
        let seq = Float(imageSeqLen)
        if imageSeqLen > 4300 {
            return a2 * seq + b2
        }
        let m200 = a2 * seq + b2
        let m10 = a1 * seq + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(numSteps) + b
    }

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
        textEncoder: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        encoder: CoreAIDiffusionModelFunction?,
        tokenizer: any Tokenizer,
        batchNormMean: [Float]?,
        batchNormVar: [Float]?,
        batchNormEps: Float
    ) {
        self.descriptor = descriptor
        self.mode = mode
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.batchNormMean = batchNormMean
        self.batchNormVar = batchNormVar
        self.batchNormEps = batchNormEps
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        try await transformer.loadResources()
        try await textEncoder.loadResources()
        try await decoder.loadResources()
        if let encoder { try await encoder.loadResources() }
    }

    public func unloadResources() async {
        await transformer.unloadResources()
        await textEncoder.unloadResources()
        await decoder.unloadResources()
        if let encoder { await encoder.unloadResources() }
    }

    // MARK: - Generation

    public func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: (PipelineProgress) -> Bool
    ) async throws -> GenerationResult {
        let steps = configuration.stepCount
        let guidanceScale = configuration.guidanceScale

        // 1. Encode text
        let textEmbeddings = try await encodeText(configuration.prompt)
        if configuration.lazyModelLoading { await textEncoder.unloadResources() }
        let textSeqLen = textEmbeddings.count / hiddenDim(textEmbeddings)

        // 2. Determine latent dimensions from image size
        let imageSize = defaultImageSize.width
        let spatialSide = imageSize / Self.patchSize
        let inChannels = Self.latentChannels
        let seqLen = spatialSide * spatialSide

        // 3. Setup scheduler.
        // For img2img: the schedule covers [strength → 0] using all steps, so every requested step
        // contributes to denoising and the noising sigma matches the first scheduled step exactly.
        // For txt2img: the schedule covers [1.0 → 0] as usual.
        let mu = Self.computeEmpiricalMu(imageSeqLen: seqLen, numSteps: steps)
        let isActuallyImg2Img = configuration.isImageToImage && encoder != nil && configuration.startingImage != nil
        let sigmaMax: Float = isActuallyImg2Img ? configuration.strength : 1.0
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps,
            trainStepCount: 1000,
            timeStepShift: 1.0,
            mu: mu,
            sigmaMax: sigmaMax
        )

        // 4. Generate noise [1, inChannels, spatialSide, spatialSide]
        let latentShape = [1, inChannels, spatialSide, spatialSide]
        let latentCount = latentShape.reduce(1, *)
        let noise = generateNoise(count: latentCount, seed: configuration.seed)
        let noisePacked = packLatentsSpatialFlatten(
            noise, channels: inChannels, height: spatialSide, width: spatialSide)

        // 5. Initialize packed latents (txt2img: pure noise; img2img: encoded image + noise blend)
        var packedLatents: [Float]

        if isActuallyImg2Img,
            let enc = encoder,
            let srcImage = configuration.startingImage
        {
            packedLatents = try await prepareImg2ImgLatents(
                encoder: enc,
                srcImage: srcImage,
                imageSize: imageSize,
                spatialSide: spatialSide,
                inChannels: inChannels,
                noisePacked: noisePacked,
                scheduler: scheduler
            )
            if configuration.lazyModelLoading { await enc.unloadResources() }
        } else {
            packedLatents = noisePacked
        }

        // 6. Pre-compute RoPE embeddings (cos, sin)
        let axesDims = descriptor.ropeAxesDims ?? [32, 32, 32, 32]
        let theta = descriptor.ropeTheta ?? Self.defaultRopeTheta
        let (rotaryCos, rotarySin) = computeRotaryEmbeddings(
            imgHeight: spatialSide, imgWidth: spatialSide,
            textSeqLen: textSeqLen, axesDims: axesDims, theta: theta
        )

        // 7. Denoising loop
        let totalDim = axesDims.reduce(0, +)
        let totalSeqLen = textSeqLen + seqLen
        let ropeShape = [totalSeqLen, totalDim]

        for (step, t) in scheduler.timeSteps.enumerated() {
            let timestepValue = Float(t) / 1000.0

            let output = try await transformer.run(floatInputs: [
                (packedLatents, [1, seqLen, inChannels]),
                (textEmbeddings, [1, textSeqLen, hiddenDim(textEmbeddings)]),
                ([timestepValue], [1]),
                ([guidanceScale], [1]),
                (rotaryCos, ropeShape),
                (rotarySin, ropeShape),
            ])

            packedLatents = scheduler.step(output: output, timeStep: t, sample: packedLatents)

            let progress = PipelineProgress(step: step + 1, totalSteps: steps, currentLatent: nil)
            if !progressHandler(progress) { break }
        }

        if configuration.lazyModelLoading { await transformer.unloadResources() }

        // 8. Unpack: (B, H*W, C) → (B, C, H, W)
        var spatialLatents = unpackLatentsSpatialFlatten(
            packedLatents, channels: inChannels, height: spatialSide, width: spatialSide
        )

        // 9. Batch norm denormalization
        spatialLatents = applyBatchNormDenorm(
            spatialLatents, channels: inChannels, height: spatialSide, width: spatialSide)

        // 10. Unpatchify: (B, 128, 64, 64) → (B, 32, 128, 128)
        let vaeChannels = inChannels / 4
        let vaeHeight = spatialSide * 2
        let vaeWidth = spatialSide * 2
        let unpatchified = Self.unpatchifyLatents(
            spatialLatents, channels: inChannels, height: spatialSide, width: spatialSide)

        // 11. VAE decode
        // Note: self.decoder is mode-appropriate (loaded at init):
        //   .full → VAEDecoder (128×128 input), .half/.tiled → VAEDecoder_half (64×64 input)
        let vaeShape = [1, vaeChannels, vaeHeight, vaeWidth]
        let pixels: [Float]
        let outputHeight: Int
        let outputWidth: Int

        switch mode {
        case .full, .half:
            pixels = try await decoder.run(floatInputs: [(unpatchified, vaeShape)])
            outputHeight = imageSize
            outputWidth = imageSize

        case .tiled:
            pixels = try await decodeTiled(
                latents: unpatchified, channels: vaeChannels, height: vaeHeight, width: vaeWidth,
                decoder: decoder, outputScale: 8)
            outputHeight = imageSize
            outputWidth = imageSize

        case .auto:
            preconditionFailure("auto resolved at init")
        }

        if configuration.lazyModelLoading { await decoder.unloadResources() }

        // 12. Convert to image
        let image = try DiffusionUtilities.pixelsToCGImage(pixels, height: outputHeight, width: outputWidth)

        var latentsND = NDArray(shape: latentShape, scalarType: .float32)
        var latentsView = latentsND.mutableView(as: Float.self)
        latentsView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<noise.count { ptr[i] = noise[i] }
        }

        return GenerationResult(images: [image], latents: [latentsND])
    }

    // MARK: - Img2Img Latent Preparation

    private func prepareImg2ImgLatents(
        encoder: CoreAIDiffusionModelFunction,
        srcImage: CGImage,
        imageSize: Int,
        spatialSide: Int,
        inChannels: Int,
        noisePacked: [Float],
        scheduler: DiscreteFlowScheduler
    ) async throws -> [Float] {
        let resized = CGImageUtils.resize(srcImage, to: imageSize) ?? srcImage
        let encoderScaleFactor = descriptor.encoderScaleFactor ?? 0.18215

        let imagePixels = try CGImageUtils.toNormalizedPlanarRGB(resized)
        let encodedFloats = try await encoder.run(floatInputs: [(imagePixels, [1, 3, imageSize, imageSize])])

        let scaledEncoded = encodedFloats.map { $0 * encoderScaleFactor }
        let patchified = Self.patchifyLatents(
            scaledEncoded, inChannels: inChannels, height: spatialSide, width: spatialSide)
        let normalized = applyBatchNormNorm(
            patchified, channels: inChannels, height: spatialSide, width: spatialSide)
        let cleanPacked = packLatentsSpatialFlatten(
            normalized, channels: inChannels, height: spatialSide, width: spatialSide)

        return scheduler.addNoise(to: cleanPacked, noise: noisePacked, at: scheduler.startSigma)
    }

    // MARK: - Text Encoding

    private func encodeText(_ text: String) async throws -> [Float] {
        let seqLen = Self.textSeqLen

        // Tokenize using Qwen3 chat template.
        //
        // Must match diffusers `_get_qwen3_prompt_embeds`
        // (diffusers 0.37.1, pipelines/flux2/pipeline_flux2_klein.py), which builds the
        // input as:
        //     messages = [{"role": "user", "content": single_prompt}]
        //     text = tokenizer.apply_chat_template(
        //         messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        //
        // `enable_thinking=False` is significant for the Qwen3 template: it appends an
        // empty `<think>\n\n</think>\n\n` block after the assistant prompt. Leaving it
        // undefined omits that block, changing the trailing conditioning tokens and
        // hurting prompt adherence. Pass it via additionalContext to match the reference.
        var ids: [Int]
        let messages: [[String: String]] = [["role": "user", "content": text]]
        do {
            ids = try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil,
                addGenerationPrompt: true, truncation: true, maxLength: seqLen, tools: nil,
                additionalContext: ["enable_thinking": false]
            )
        } catch {
            let tokens = tokenizer.tokenize(text: text)
            ids = tokens.compactMap { tokenizer.convertTokenToId($0) }
        }

        if ids.count > seqLen {
            ids = Array(ids.prefix(seqLen))
        }

        let realTokenCount = ids.count
        // diffusers pads with the tokenizer's pad_token, not the eos_token. For
        // FLUX.2 klein's Qwen tokenizer these differ: pad_token is <|endoftext|>
        // (151643) while eos_token is <|im_end|> (151645). The reference builds
        // input_ids via `tokenizer(text, padding="max_length", max_length=512)`
        // (diffusers 0.37.1, pipeline_flux2_klein.py `_get_qwen3_prompt_embeds`),
        // which uses pad_token. These ~490 padding tokens are fed to the DiT
        // UNMASKED, so the id must match the reference exactly.
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? 151643

        while ids.count < seqLen {
            ids.append(padTokenId)
        }

        // input_ids: Int32, attention_mask: Int32
        let int32Ids = ids.map { Int32($0) }
        var maskValues = [Int32](repeating: 0, count: seqLen)
        for i in 0..<realTokenCount { maskValues[i] = 1 }

        let hiddenStates = try await textEncoder.run(intInputs: [
            (int32Ids, [1, seqLen]),
            (maskValues, [1, seqLen]),
        ])

        return hiddenStates
    }

    private func hiddenDim(_ embeddings: [Float]) -> Int {
        embeddings.count / Self.textSeqLen
    }

    // MARK: - RoPE Pre-computation

    private func computeRotaryEmbeddings(
        imgHeight: Int, imgWidth: Int,
        textSeqLen: Int, axesDims: [Int], theta: Float
    ) -> ([Float], [Float]) {
        let imgSeqLen = imgHeight * imgWidth
        let totalSeqLen = textSeqLen + imgSeqLen
        let totalDim = axesDims.reduce(0, +)

        var cosScalars = [Float](repeating: 0, count: totalSeqLen * totalDim)
        var sinScalars = [Float](repeating: 0, count: totalSeqLen * totalDim)

        var axisOffset = 0
        for (axisIdx, axisDim) in axesDims.enumerated() {
            let halfDim = axisDim / 2

            var invFreq = [Double](repeating: 0, count: halfDim)
            for k in 0..<halfDim {
                let exponent = Double(2 * k) / Double(axisDim)
                invFreq[k] = 1.0 / pow(Double(theta), exponent)
            }

            // Text tokens (first in sequence): axis 3 = sequential, others = 0
            for s in 0..<textSeqLen {
                let pos: Double = axisIdx == 3 ? Double(s) : 0.0
                let outBase = s * totalDim + axisOffset
                for k in 0..<halfDim {
                    let angle = pos * invFreq[k]
                    let c = Float(cos(angle))
                    let sn = Float(sin(angle))
                    cosScalars[outBase + 2 * k] = c
                    cosScalars[outBase + 2 * k + 1] = c
                    sinScalars[outBase + 2 * k] = sn
                    sinScalars[outBase + 2 * k + 1] = sn
                }
            }

            // Image tokens (after text): axis 1 = h, axis 2 = w, others = 0
            for h in 0..<imgHeight {
                for w in 0..<imgWidth {
                    let s = textSeqLen + h * imgWidth + w
                    let pos: Double
                    switch axisIdx {
                    case 1: pos = Double(h)
                    case 2: pos = Double(w)
                    default: pos = 0.0
                    }
                    let outBase = s * totalDim + axisOffset
                    for k in 0..<halfDim {
                        let angle = pos * invFreq[k]
                        let c = Float(cos(angle))
                        let sn = Float(sin(angle))
                        cosScalars[outBase + 2 * k] = c
                        cosScalars[outBase + 2 * k + 1] = c
                        sinScalars[outBase + 2 * k] = sn
                        sinScalars[outBase + 2 * k + 1] = sn
                    }
                }
            }

            axisOffset += axisDim
        }

        return (cosScalars, sinScalars)
    }

    // MARK: - Latent Packing/Unpacking

    /// (B, C, H, W) → (B, H*W, C) — spatial flatten for patch_size=1
    private func packLatentsSpatialFlatten(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let seqLen = height * width
        var packed = [Float](repeating: 0, count: seqLen * channels)
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let srcIdx = c * height * width + h * width + w
                    let token = h * width + w
                    let dstIdx = token * channels + c
                    packed[dstIdx] = latents[srcIdx]
                }
            }
        }
        return packed
    }

    /// (B, H*W, C) → (B, C, H, W) — inverse spatial flatten
    private func unpackLatentsSpatialFlatten(_ packed: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        var unpacked = [Float](repeating: 0, count: channels * height * width)
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let token = h * width + w
                    let srcIdx = token * channels + c
                    let dstIdx = c * height * width + h * width + w
                    unpacked[dstIdx] = packed[srcIdx]
                }
            }
        }
        return unpacked
    }

    // MARK: - Batch Norm Denormalization

    /// latents = latents * sqrt(var + eps) + mean (per-channel in BCHW format)
    func applyBatchNormDenorm(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        guard let bnMean = batchNormMean, let bnVar = batchNormVar,
            bnMean.count == channels, bnVar.count == channels
        else {
            return latents
        }

        let spatialSize = height * width
        let std = bnVar.map { sqrtf($0 + batchNormEps) }

        var result = [Float](repeating: 0, count: latents.count)
        for c in 0..<channels {
            let offset = c * spatialSize
            for i in 0..<spatialSize {
                result[offset + i] = latents[offset + i] * std[c] + bnMean[c]
            }
        }
        return result
    }

    // MARK: - Unpatchify

    /// (B, C*4, H, W) → (B, C, H*2, W*2) — reverses 2×2 patchification
    static func unpatchifyLatents(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let outChannels = channels / 4
        let outH = height * 2
        let outW = width * 2

        var result = [Float](repeating: 0, count: outChannels * outH * outW)
        for c in 0..<outChannels {
            for i in 0..<height {
                for j in 0..<width {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let srcC = c * 4 + dy * 2 + dx
                            let srcIdx = srcC * height * width + i * width + j
                            let dstIdx = c * outH * outW + (i * 2 + dy) * outW + (j * 2 + dx)
                            result[dstIdx] = latents[srcIdx]
                        }
                    }
                }
            }
        }
        return result
    }

    // MARK: - Patchify / BN Normalize (img2img forward path)

    /// (B, C, H*2, W*2) → (B, C*4, H, W) — forward 2×2 patchification (inverse of unpatchifyLatents).
    static func patchifyLatents(_ latents: [Float], inChannels: Int, height: Int, width: Int) -> [Float] {
        let inCh = inChannels / 4  // vaeChannels (32)
        let inH = height * 2
        let inW = width * 2

        var result = [Float](repeating: 0, count: inChannels * height * width)
        for c in 0..<inCh {
            for i in 0..<height {
                for j in 0..<width {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let dstC = c * 4 + dy * 2 + dx
                            let srcIdx = c * inH * inW + (i * 2 + dy) * inW + (j * 2 + dx)
                            let dstIdx = dstC * height * width + i * width + j
                            result[dstIdx] = latents[srcIdx]
                        }
                    }
                }
            }
        }
        return result
    }

    /// Inverse of applyBatchNormDenorm: x_norm = (x − mean) / sqrt(var + eps) per channel.
    func applyBatchNormNorm(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        guard let bnMean = batchNormMean, let bnVar = batchNormVar,
            bnMean.count == channels, bnVar.count == channels
        else {
            return latents
        }

        let spatialSize = height * width
        let std = bnVar.map { sqrtf($0 + batchNormEps) }

        var result = [Float](repeating: 0, count: latents.count)
        for c in 0..<channels {
            let offset = c * spatialSize
            for i in 0..<spatialSize {
                result[offset + i] = (latents[offset + i] - bnMean[c]) / std[c]
            }
        }
        return result
    }

    // MARK: - Noise Generation

    private func generateNoise(count: Int, seed: UInt32) -> [Float] {
        var rng = NumPyRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    }

    // MARK: - Image Conversion

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
