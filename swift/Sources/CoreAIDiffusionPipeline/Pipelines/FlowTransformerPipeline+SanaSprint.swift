// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

/// Sana Sprint: instruction-prefixed prompt → Gemma2 → denoise loop (stochastic) → DC-AE.
///
/// Its TrigFlow/SCM sampler is flow matching in other coordinates: with σ = sin t/(sin t + cos t)
/// the transformer takes the flow latent at timestep σ and returns the flow velocity, and each
/// SCM step is x0 = x − σ·v renoised to the next σ with fresh noise. So the latents here are
/// plain NCHW flow latents and no TrigFlow math appears at runtime.
extension FlowTransformerPipeline {
    static let sanaLatentChannels = 32
    static let sanaSpatialCompression = 32

    /// Sana Sprint's TrigFlow angles mapped to flow sigmas, matching diffusers `SCMScheduler`:
    /// `[max, intermediate]` for 2 steps, else `linspace(max, 0, steps + 1)` minus the final 0.
    static func sanaSprintSigmas(steps: Int, maxTimestep: Float, intermediateTimestep: Float?) -> [Float] {
        let angles: [Float]
        if steps == 2, let intermediateTimestep {
            angles = [maxTimestep, intermediateTimestep]
        } else {
            angles = (0..<steps).map { maxTimestep * (1 - Float($0) / Float(steps)) }
        }
        return angles.map { sin($0) / (sin($0) + cos($0)) }
    }

    /// The text encoder keeps BOS plus the last `outputLength − 1` positions (diffusers'
    /// `select_index`), so the transformer mask takes the same entries.
    static func sanaSelectMask(_ mask: [Float], outputLength: Int) -> [Float] {
        [mask[0]] + mask.suffix(outputLength - 1)
    }

    func makeSanaSprintPlan(_ configuration: PipelineConfiguration) async throws -> DenoisingPlan {
        if configuration.isImageToImage && configuration.startingImage != nil {
            throw PipelineLoadError.unsupportedConfiguration("Sana Sprint supports text-to-image only.")
        }
        if configuration.guidanceMode == .manual {
            throw PipelineLoadError.unsupportedConfiguration(
                "Sana Sprint embeds guidance in the transformer; use --guidance-mode distilled.")
        }

        let (textEmbeddings, textMask) = try await encodeSanaPrompt(configuration.prompt)
        if configuration.lazyModelLoading { await textEncoder.unloadResources() }
        let textSeqLen = textMask.count
        let textShape = [1, textSeqLen, textEmbeddings.count / textSeqLen]

        let side = defaultImageSize.width / Self.sanaSpatialCompression
        let latentShape = [1, Self.sanaLatentChannels, side, side]
        let latentCount = latentShape.reduce(1, *)

        // diffusers draws the initial latents and every step's renoise from one generator.
        var rng = TorchRandomSource(seed: configuration.seed)
        let noise = rng.normalArray([latentCount])

        let scheduler = DiscreteFlowScheduler(
            sigmas: Self.sanaSprintSigmas(
                steps: configuration.stepCount,
                maxTimestep: descriptor.maxTimesteps ?? 1.5708,
                intermediateTimestep: descriptor.intermediateTimesteps))

        let guidanceScale = configuration.guidanceScale
        let transformer = transformer
        let predict: (_ latents: [Float], _ step: Int, _ sigma: Float) async throws -> [Float] = {
            latents, _, sigma in
            try await transformer.run(floatInputs: [
                (latents, latentShape),
                (textEmbeddings, textShape),
                (textMask, [1, textSeqLen]),
                ([sigma], [1]),
                ([guidanceScale], [1]),
            ])
        }

        let scaleFactor = descriptor.decoderScaleFactor ?? 1.0
        return DenoisingPlan(
            latents: noise,
            scheduler: scheduler,
            renoise: { rng.normalArray([latentCount]) },
            predict: predict,
            vaeLatents: { $0.map { $0 / scaleFactor } },
            vaeShape: latentShape,
            denoiser: transformer,
            initialNoise: noise,
            noiseShape: latentShape)
    }

    /// Returns the text encoder output `[1, textSequenceLength, D]` and its float mask.
    ///
    /// Mirrors diffusers `SanaSprintPipeline._get_gemma_prompt_embeds`: lowercase and strip
    /// the prompt, prepend the instruction, add BOS, and right-pad to the traced length.
    func encodeSanaPrompt(_ prompt: String) async throws -> (embeddings: [Float], mask: [Float]) {
        guard let inputLength = descriptor.textInputLength,
            let outputLength = descriptor.textSequenceLength
        else {
            throw PipelineLoadError.missingConfig("text_input_length / text_sequence_length")
        }

        let cleaned = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        var ids = tokenizer.encode(text: (descriptor.promptPrefix ?? "") + cleaned)
        if ids.count > inputLength { ids = Array(ids.prefix(inputLength)) }
        let realTokenCount = ids.count
        let padTokenId = tokenizer.convertTokenToId("<pad>") ?? 0
        ids += [Int](repeating: padTokenId, count: inputLength - ids.count)

        var mask = [Float](repeating: 0, count: inputLength)
        for i in 0..<realTokenCount { mask[i] = 1 }

        let embeddings = try await textEncoder.run(intInputs: [
            (ids.map { Int32($0) }, [1, inputLength]),
            (mask.map { Int32($0) }, [1, inputLength]),
        ])
        return (embeddings, Self.sanaSelectMask(mask, outputLength: outputLength))
    }
}
