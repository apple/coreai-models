// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Tokenizers

/// FLUX.2 Klein: tokenize → text encode → noise → pack → denoise loop
/// (flow-match Euler) → unpack → BN denorm → unpatchify → VAE decode.
///
/// RoPE is computed inside the transformer graph; this pipeline only supplies
/// position IDs, which depend on grid geometry alone.
extension FlowTransformerPipeline {
    // MARK: - Architecture Constants

    private static let patchSize = 16
    private static let latentChannels = 128
    private static let textSeqLen = 512
    private static let qwen3PadTokenId = 151643

    /// Reference tokens sit at T=10 on RoPE axis 0, separating them from the noise
    /// grid (T=0) where H/W would otherwise collide. Matches the export-time dummies.
    private static let referenceTokenTimeOffset: Float = 10

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

    // MARK: - Denoising Plan

    func makeFlux2Plan(_ configuration: PipelineConfiguration) async throws -> DenoisingPlan {
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
        // Reference-token img2img uses the full schedule (1.0 → 0): structure comes
        // from the concatenated reference tokens, not from noise blending. txt2img
        // also uses [1.0 → 0]. So sigmaMax is 1.0 in both cases.
        let mu = Self.computeEmpiricalMu(imageSeqLen: seqLen, numSteps: steps)
        let isActuallyImg2Img = configuration.isImageToImage && encoder != nil && configuration.startingImage != nil

        // Reference-token img2img is incompatible with tiled decode: tiled uses the
        // half-resolution VAE encoder (traced for 512×512), but the reference is
        // encoded at the full image size — feeding it a 1024 image crashes on a
        // shape mismatch. Fail early with a clear message instead.
        if isActuallyImg2Img && mode == .tiled {
            throw PipelineLoadError.unsupportedConfiguration(
                "img2img is not supported with tiled decode. Use --decode-resolution full or half.")
        }

        let sigmaMax: Float = 1.0
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
        let noise = generateNoise(count: latentCount, seed: configuration.seed, sourceType: .torch)
        let noisePacked = packLatentsSpatialFlatten(
            noise, channels: inChannels, height: spatialSide, width: spatialSide)

        // 5. Initialize reference tokens. Latents start from pure noise either way.
        var referenceTokens: [Float]?
        var refSide: Int = 0

        if isActuallyImg2Img,
            let enc = encoder,
            let srcImage = configuration.startingImage
        {
            // Reference grid size based on the requested reference grid
            switch configuration.referenceGrid {
            case .full: refSide = spatialSide  // 64×64 = 4096 tokens
            case .half: refSide = spatialSide / 2  // 32×32 = 1024 tokens
            case .quarter: refSide = spatialSide / 4  // 16×16 = 256 tokens
            }

            // Encode reference at full resolution, then subsample tokens if needed
            let fullRefPacked = try await encodeReferenceImage(
                encoder: enc, srcImage: srcImage, imageSize: imageSize,
                spatialSide: spatialSide, inChannels: inChannels
            )
            let refPacked: [Float]
            if refSide == spatialSide {
                refPacked = fullRefPacked
            } else {
                refPacked = Self.subsampleTokens(
                    fullRefPacked, fromSide: spatialSide, toSide: refSide, channels: inChannels)
            }
            referenceTokens = refPacked
            if configuration.lazyModelLoading { await enc.unloadResources() }

            // FLUX.2 reference-token img2img: noise latents start from PURE NOISE.
            // The reference tokens concatenated at each step provide structural
            // guidance via cross-attention. The text prompt steers content.
            // (This differs from SD-style img2img which blends noise with the encoded image.)
        }

        // 6. Build RoPE position IDs — the transformer computes the frequencies in-graph
        let axesDims = descriptor.ropeAxesDims ?? [32, 32, 32, 32]
        let axisCount = axesDims.count
        // Image ids put H/W on axes 1/2; text ids put the seq index on the last axis.
        guard axisCount >= 3 else {
            throw PipelineLoadError.missingConfig(
                "rope_axes_dims has \(axisCount) axes; FLUX.2 RoPE needs at least 3")
        }
        let refSeqLen = refSide * refSide
        // img2img appends reference tokens after the noise tokens, marked T=10 on axis 0
        // so the transformer's in-graph RoPE separates them from the noise grid.
        let imageIds: [Float]
        if referenceTokens != nil {
            imageIds = buildImageIdsWithReference(
                noiseSide: spatialSide, refSide: refSide, axisCount: axisCount)
        } else {
            imageIds = buildImageIds(side: spatialSide, axisCount: axisCount)
        }
        let textIds = buildTextIds(textSeqLen: textSeqLen, axisCount: axisCount)

        // 7. Pick the asset + entrypoint that serves this pass. img2img arrives one of two
        // ways: as a named entrypoint on the multi-function transformer, or as its own
        // single-function asset. Which one is decided at load time by whether that asset
        // exists on disk.
        let denoiser: CoreAIDiffusionModelFunction
        let fnName: String
        if referenceTokens != nil {
            // Name what the bundle does have, not just what it lacks: re-exporting is the
            // only remedy, so the available set is the actionable part.
            guard let route = img2imgRoutes[configuration.referenceGrid] else {
                let available = img2imgRoutes.keys.map(\.rawValue).sorted()
                throw PipelineLoadError.unsupportedConfiguration(
                    available.isEmpty
                        ? "this bundle has no img2img transformer. Export the img2img "
                            + "components, or export without --single-function to get every "
                            + "grid from a single asset."
                        : "this bundle has no img2img transformer for the "
                            + "\(configuration.referenceGrid) reference grid. Available: "
                            + "\(available.joined(separator: ", ")).")
            }
            denoiser = route.function
            fnName = route.entrypoint
        } else {
            denoiser = transformer
            fnName = transformerFunctionName
        }

        // Manual CFG needs an unconditional pass, so encode an empty prompt.
        //
        // At exactly 1.0 the interpolation reduces to the conditional pass (the
        // unconditional term's coefficient is zero) so the second forward pass is
        // wasted. Below 1.0 the blend runs *toward* the unconditional prediction, i.e. 2x
        // the compute to follow the prompt less, so this gate refuses that too.
        //
        // That second half is a deliberate divergence from diffusers, whose
        // `ClassifierFreeGuidance` guider gates on `not isclose(scale, 1.0)` and so honours
        // sub-1.0 scales. Widen this to match if a caller ever has a real use for them.
        let emptyEmbeddings: [Float]?
        if configuration.guidanceMode == .manual && guidanceScale > 1.0 {
            emptyEmbeddings = try await encodeText("")
        } else {
            if configuration.guidanceMode == .manual {
                CLILogger.log(
                    "⚠️ FlowTransformerPipeline: --guidance-mode manual needs a guidance scale above 1.0 "
                        + "(got \(guidanceScale)); falling back to distilled, which applies no "
                        + "guidance at all. Raise the scale to get the two-pass path.",
                    component: "Diffusion")
            }
            emptyEmbeddings = nil
        }

        // Reused across steps: manual CFG would otherwise allocate a fresh
        // seqLen*inChannels array on every one.
        var cfgBuffer = [Float](repeating: 0, count: seqLen * inChannels)

        let predict: (_ latents: [Float], _ step: Int, _ sigma: Float) async throws -> [Float] = {
            packedLatents, step, _ in
            let timestepValue = Float(scheduler.timeSteps[step]) / 1000.0

            // For img2img: concatenate noise + reference tokens at each step
            let inputTokens: [Float]
            let inputSeqLen: Int
            if let ref = referenceTokens {
                inputTokens = packedLatents + ref
                inputSeqLen = seqLen + refSeqLen
            } else {
                inputTokens = packedLatents
                inputSeqLen = seqLen
            }

            if let emptyEmb = emptyEmbeddings {
                // Manual CFG: two forward passes. The guidance input is 0 only because the
                // traced signature requires a value. The Flux2 model sets
                // `guidance_embeds: false`, so `guidance_embedder` is nil and
                // `Flux2TimestepGuidanceEmbeddings` drops the input entirely. Any value
                // behaves identically; all the guidance here comes from the interpolation
                // below, not from the model.
                let condOutput = try await denoiser.run(
                    floatInputs: [
                        (inputTokens, [1, inputSeqLen, inChannels]),
                        (textEmbeddings, [1, textSeqLen, hiddenDim(textEmbeddings)]),
                        ([timestepValue], [1]),
                        ([Float(0)], [1]),
                        (imageIds, [1, inputSeqLen, axisCount]),
                        (textIds, [1, textSeqLen, axisCount]),
                    ], functionName: fnName)

                let uncondOutput = try await denoiser.run(
                    floatInputs: [
                        (inputTokens, [1, inputSeqLen, inChannels]),
                        (emptyEmb, [1, textSeqLen, hiddenDim(emptyEmb)]),
                        ([timestepValue], [1]),
                        ([Float(0)], [1]),
                        (imageIds, [1, inputSeqLen, axisCount]),
                        (textIds, [1, textSeqLen, axisCount]),
                    ], functionName: fnName)

                let condSlice: ArraySlice<Float>
                let uncondSlice: ArraySlice<Float>
                if referenceTokens != nil {
                    condSlice = condOutput[0..<(seqLen * inChannels)]
                    uncondSlice = uncondOutput[0..<(seqLen * inChannels)]
                } else {
                    condSlice = condOutput[0..<condOutput.count]
                    uncondSlice = uncondOutput[0..<uncondOutput.count]
                }
                Self.applyClassifierFreeGuidance(
                    cond: condSlice, uncond: uncondSlice,
                    guidanceScale: guidanceScale, into: &cfgBuffer)
                return cfgBuffer
            }

            // Distilled: one pass, no CFG. `guidanceScale` is passed to satisfy the
            // traced signature but this checkpoint discards it (see above).
            let fullOutput = try await denoiser.run(
                floatInputs: [
                    (inputTokens, [1, inputSeqLen, inChannels]),
                    (textEmbeddings, [1, textSeqLen, hiddenDim(textEmbeddings)]),
                    ([timestepValue], [1]),
                    ([guidanceScale], [1]),
                    (imageIds, [1, inputSeqLen, axisCount]),
                    (textIds, [1, textSeqLen, axisCount]),
                ], functionName: fnName)

            if referenceTokens != nil {
                return Array(fullOutput[0..<(seqLen * inChannels)])
            }
            return fullOutput
        }

        // Unpack → BN denorm → unpatchify: (1, 64*64, 128) → (1, 32, 128, 128)
        let vaeLatents: (_ latents: [Float]) -> [Float] = { packed in
            let spatial = unpackLatentsSpatialFlatten(
                packed, channels: inChannels, height: spatialSide, width: spatialSide)
            let denormed = applyBatchNormDenorm(
                spatial, channels: inChannels, height: spatialSide, width: spatialSide)
            return Self.unpatchifyLatents(
                denormed, channels: inChannels, height: spatialSide, width: spatialSide)
        }

        return DenoisingPlan(
            latents: noisePacked,
            scheduler: scheduler,
            renoise: nil,
            predict: predict,
            vaeLatents: vaeLatents,
            vaeShape: [1, inChannels / 4, spatialSide * 2, spatialSide * 2],
            denoiser: denoiser,
            initialNoise: noise,
            noiseShape: latentShape)
    }

    // MARK: - Img2Img

    /// Encode a reference image into packed latent tokens (no noise blending).
    private func encodeReferenceImage(
        encoder: CoreAIDiffusionModelFunction,
        srcImage: CGImage,
        imageSize: Int,
        spatialSide: Int,
        inChannels: Int
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
        return packLatentsSpatialFlatten(
            normalized, channels: inChannels, height: spatialSide, width: spatialSide)
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
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? Self.qwen3PadTokenId

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

    // MARK: - RoPE Position IDs

    /// `img_ids` for in-graph RoPE: `[1, side*side, axisCount]` flattened row-major,
    /// one row per image token as [T, H, W, L].
    private func buildImageIds(side: Int, axisCount: Int) -> [Float] {
        var ids = [Float](repeating: 0, count: side * side * axisCount)
        for h in 0..<side {
            for w in 0..<side {
                let idx = h * side + w
                ids[idx * axisCount + 1] = Float(h)
                ids[idx * axisCount + 2] = Float(w)
            }
        }
        return ids
    }

    /// `img_ids` for img2img: noise tokens followed by reference tokens.
    /// Reference rows carry T=10 on axis 0 so in-graph RoPE keeps them positionally
    /// distinct from the noise grid even where H/W coincide.
    private func buildImageIdsWithReference(
        noiseSide: Int, refSide: Int, axisCount: Int
    ) -> [Float] {
        let noiseSeq = noiseSide * noiseSide
        let refSeq = refSide * refSide
        var ids = [Float](repeating: 0, count: (noiseSeq + refSeq) * axisCount)

        for h in 0..<noiseSide {
            for w in 0..<noiseSide {
                let idx = h * noiseSide + w
                ids[idx * axisCount + 1] = Float(h)
                ids[idx * axisCount + 2] = Float(w)
            }
        }

        for h in 0..<refSide {
            for w in 0..<refSide {
                let idx = noiseSeq + h * refSide + w
                ids[idx * axisCount + 0] = Self.referenceTokenTimeOffset
                ids[idx * axisCount + 1] = Float(h)
                ids[idx * axisCount + 2] = Float(w)
            }
        }

        return ids
    }

    /// `txt_ids` for in-graph RoPE: `[1, textSeqLen, axisCount]` flattened row-major.
    /// Text tokens are [0, 0, 0, s] — sequence index on the last axis, spatial unused.
    private func buildTextIds(textSeqLen: Int, axisCount: Int) -> [Float] {
        var ids = [Float](repeating: 0, count: textSeqLen * axisCount)
        for s in 0..<textSeqLen {
            ids[s * axisCount + (axisCount - 1)] = Float(s)
        }
        return ids
    }

    /// Spatially downsample packed tokens to a smaller grid, area-averaging each
    /// `stride`×`stride` block channel-wise.
    ///
    /// Point sampling would keep only 1/stride² of the encoded reference and throw
    /// the rest away; the block mean retains all of it, so structure survives at the
    /// half/quarter grids. Channel index encodes intra-patch position, so averaging
    /// per channel keeps corresponding sub-positions aligned.
    /// Input: [fromSide*fromSide, channels], Output: [toSide*toSide, channels]
    static func subsampleTokens(
        _ tokens: [Float], fromSide: Int, toSide: Int, channels: Int
    ) -> [Float] {
        let stride = fromSide / toSide
        let scale = 1.0 / Float(stride * stride)
        var result = [Float](repeating: 0, count: toSide * toSide * channels)
        for h in 0..<toSide {
            for w in 0..<toSide {
                let dstIdx = (h * toSide + w) * channels
                for bh in 0..<stride {
                    let srcRow = h * stride + bh
                    for bw in 0..<stride {
                        let srcIdx = (srcRow * fromSide + w * stride + bw) * channels
                        for c in 0..<channels {
                            result[dstIdx + c] += tokens[srcIdx + c]
                        }
                    }
                }
                for c in 0..<channels {
                    result[dstIdx + c] *= scale
                }
            }
        }
        return result
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
}
