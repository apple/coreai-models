// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The `diffusion` block of a `metadata.json` (schema 0.2).
///
/// Shared config for image and video diffusion; every field is optional, so a
/// bundle carries only the keys its pipeline needs.
///
/// ```swift
/// let scale = config.decoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor  // FLUX.2 / Sana
/// let frames = config.defaultNumFrames ?? DiffusionDefaults.Video.frameCount    // Wan
/// ```
public struct DiffusionConfig: Codable, Sendable, Equatable {
    /// Fine-grained pipeline type for the image path (`flux2`, `sana-sprint`).
    /// `nil` for video (Wan exports `"wan2.1"`, which is not an image `PipelineType`
    /// and dispatches via the bundle kind instead) or any unrecognized value.
    public let type: PipelineType?
    public let predictionType: PredictionType?
    public let imageSize: Int?

    public let encoderScaleFactor: Float?
    public let decoderScaleFactor: Float?
    public let decoderShiftFactor: Float?

    // MARK: FLUX.2

    public let batchNormEps: Float?
    public let ropeAxesDims: [Int]?
    public let defaultGuidanceScale: Float?
    public let defaultSteps: Int?

    // MARK: Sana Sprint

    /// TrigFlow angles for the SCM schedule; the pipeline maps them to flow sigmas.
    public let maxTimesteps: Float?
    public let intermediateTimesteps: Float?
    /// Instruction prepended to every prompt before tokenizing.
    public let promptPrefix: String?
    /// Token count the text encoder was traced at (prefix + prompt, padded).
    public let textInputLength: Int?
    /// Token count of the text encoder output fed to the transformer.
    public let textSequenceLength: Int?

    // MARK: Video (Wan)

    /// Text-encoder embedding dimension (Wan `text_dim`).
    public let textDim: Int?
    /// Latent channel count (Wan `z_dim`).
    public let latentChannels: Int?
    /// Flow-matching scheduler shift (Wan `default_shift`).
    public let schedulerShift: Float?
    /// Default frame count for a generated clip (Wan `default_num_frames`).
    public let defaultNumFrames: Int?

    public init(
        type: PipelineType? = nil,
        predictionType: PredictionType? = nil,
        imageSize: Int? = nil,
        encoderScaleFactor: Float? = nil,
        decoderScaleFactor: Float? = nil,
        decoderShiftFactor: Float? = nil,
        batchNormEps: Float? = nil,
        ropeAxesDims: [Int]? = nil,
        defaultGuidanceScale: Float? = nil,
        defaultSteps: Int? = nil,
        maxTimesteps: Float? = nil,
        intermediateTimesteps: Float? = nil,
        promptPrefix: String? = nil,
        textInputLength: Int? = nil,
        textSequenceLength: Int? = nil,
        textDim: Int? = nil,
        latentChannels: Int? = nil,
        schedulerShift: Float? = nil,
        defaultNumFrames: Int? = nil
    ) {
        self.type = type
        self.predictionType = predictionType
        self.imageSize = imageSize
        self.encoderScaleFactor = encoderScaleFactor
        self.decoderScaleFactor = decoderScaleFactor
        self.decoderShiftFactor = decoderShiftFactor
        self.batchNormEps = batchNormEps
        self.ropeAxesDims = ropeAxesDims
        self.defaultGuidanceScale = defaultGuidanceScale
        self.defaultSteps = defaultSteps
        self.maxTimesteps = maxTimesteps
        self.intermediateTimesteps = intermediateTimesteps
        self.promptPrefix = promptPrefix
        self.textInputLength = textInputLength
        self.textSequenceLength = textSequenceLength
        self.textDim = textDim
        self.latentChannels = latentChannels
        self.schedulerShift = schedulerShift
        self.defaultNumFrames = defaultNumFrames
    }

    enum CodingKeys: String, CodingKey {
        case type
        case predictionType = "prediction_type"
        case imageSize = "image_size"
        case encoderScaleFactor = "encoder_scale_factor"
        case decoderScaleFactor = "decoder_scale_factor"
        case decoderShiftFactor = "decoder_shift_factor"
        case batchNormEps = "batch_norm_eps"
        case ropeAxesDims = "rope_axes_dims"
        case defaultGuidanceScale = "default_guidance_scale"
        case defaultSteps = "default_steps"
        case maxTimesteps = "max_timesteps"
        case intermediateTimesteps = "intermediate_timesteps"
        case promptPrefix = "prompt_prefix"
        case textInputLength = "text_input_length"
        case textSequenceLength = "text_sequence_length"
        case textDim = "text_dim"
        case latentChannels = "z_dim"
        case schedulerShift = "default_shift"
        case defaultNumFrames = "default_num_frames"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Enum fields decode leniently: an unrecognized raw value (e.g. Wan's
        // "wan2.1" type) becomes nil rather than throwing, so the shared config
        // tolerates every kind and any value a future exporter adds.
        self.type = (try c.decodeIfPresent(String.self, forKey: .type)).flatMap(PipelineType.init(rawValue:))
        self.predictionType = (try c.decodeIfPresent(String.self, forKey: .predictionType))
            .flatMap(PredictionType.init(rawValue:))
        self.imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize)
        self.encoderScaleFactor = try c.decodeIfPresent(Float.self, forKey: .encoderScaleFactor)
        self.decoderScaleFactor = try c.decodeIfPresent(Float.self, forKey: .decoderScaleFactor)
        self.decoderShiftFactor = try c.decodeIfPresent(Float.self, forKey: .decoderShiftFactor)
        self.batchNormEps = try c.decodeIfPresent(Float.self, forKey: .batchNormEps)
        self.ropeAxesDims = try c.decodeIfPresent([Int].self, forKey: .ropeAxesDims)
        self.defaultGuidanceScale = try c.decodeIfPresent(Float.self, forKey: .defaultGuidanceScale)
        self.defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps)
        self.maxTimesteps = try c.decodeIfPresent(Float.self, forKey: .maxTimesteps)
        self.intermediateTimesteps = try c.decodeIfPresent(Float.self, forKey: .intermediateTimesteps)
        self.promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix)
        self.textInputLength = try c.decodeIfPresent(Int.self, forKey: .textInputLength)
        self.textSequenceLength = try c.decodeIfPresent(Int.self, forKey: .textSequenceLength)
        self.textDim = try c.decodeIfPresent(Int.self, forKey: .textDim)
        self.latentChannels = try c.decodeIfPresent(Int.self, forKey: .latentChannels)
        self.schedulerShift = try c.decodeIfPresent(Float.self, forKey: .schedulerShift)
        self.defaultNumFrames = try c.decodeIfPresent(Int.self, forKey: .defaultNumFrames)
    }

    /// Fine-grained pipeline type for the image diffusion path.
    public enum PipelineType: String, Codable, Sendable {
        case flux2 = "flux2"
        case sanaSprint = "sana-sprint"
    }
}
