// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics

/// Controls which model components are loaded and how decoding is performed.
public enum DecodeResolution: String, Hashable, Sendable, CaseIterable {
    /// Auto-detect: picks the highest quality mode available in the model directory.
    case auto
    /// Full-resolution: Transformer + VAEDecoder → 1024×1024.
    case full
    /// Half-resolution: Transformer_512 + VAEDecoder_half → 512×512 (4× faster).
    case half
    /// Tiled: Transformer + VAEDecoder_half in tiles → 1024×1024, low memory.
    case tiled
}

extension DecodeResolution: CustomStringConvertible {
    public var description: String { rawValue }
}

/// Size of the img2img reference-token grid, relative to the output/noise grid
/// (FLUX.2 reference-token concatenation). A larger grid gives stronger structural
/// fidelity to the reference at the cost of more compute per step. Token counts are
/// resolution-dependent: at 1024 the grid side is 64, at 512 it is 32.
public enum ReferenceGrid: String, Hashable, Sendable, CaseIterable {
    /// Full grid — matches the output grid 1:1 (64×64=4096 tokens @1024, 32×32=1024 @512). ~2× compute.
    case full
    /// Half the linear side — a quarter of the tokens (1024 @1024, 256 @512). Good balance.
    case half
    /// Quarter the linear side — 1/16 of the tokens (256 @1024, 64 @512). Fast, coarse guidance.
    case quarter
}

/// User-facing configuration for image generation.
public struct PipelineConfiguration: Hashable, Sendable {
    public var prompt: String
    public var negativePrompt: String
    public var seed: UInt32
    public var stepCount: Int
    public var guidanceScale: Float
    public var schedulerType: SchedulerType

    // Image-to-image
    public var startingImage: CGImage?
    public var strength: Float
    public var referenceGrid: ReferenceGrid
    public var manualCFG: Bool

    // VAE scale factors (from pipeline.json)
    public var encoderScaleFactor: Float
    public var decoderScaleFactor: Float
    public var decoderShiftFactor: Float

    // Decode resolution
    public var decodeResolution: DecodeResolution

    // SDXL geometry conditioning
    public var originalSize: Float
    public var targetSize: Float

    /// Load model components on demand and unload after each pipeline stage to reduce peak memory.
    /// Disable to keep all models resident and exercise full memory pressure (e.g. profiling peak footprint).
    public var lazyModelLoading: Bool

    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 0,
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        schedulerType: SchedulerType = .dpmSolverMultistep,
        startingImage: CGImage? = nil,
        strength: Float = 1.0,
        referenceGrid: ReferenceGrid = .full,
        manualCFG: Bool = false,
        encoderScaleFactor: Float = 0.18215,
        decoderScaleFactor: Float = 0.18215,
        decoderShiftFactor: Float = 0.0,
        decodeResolution: DecodeResolution = .full,
        originalSize: Float = 1024,
        targetSize: Float = 1024,
        lazyModelLoading: Bool = true
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.schedulerType = schedulerType
        self.startingImage = startingImage
        self.strength = strength
        self.referenceGrid = referenceGrid
        self.manualCFG = manualCFG
        self.encoderScaleFactor = encoderScaleFactor
        self.decoderScaleFactor = decoderScaleFactor
        self.decoderShiftFactor = decoderShiftFactor
        self.decodeResolution = decodeResolution
        self.originalSize = originalSize
        self.targetSize = targetSize
        self.lazyModelLoading = lazyModelLoading
    }

    public var isImageToImage: Bool { startingImage != nil }
}

/// Hashable conformance — CGImage excluded (not Hashable).
extension PipelineConfiguration {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(prompt)
        hasher.combine(negativePrompt)
        hasher.combine(seed)
        hasher.combine(stepCount)
        hasher.combine(guidanceScale)
        hasher.combine(schedulerType)
        hasher.combine(strength)
        hasher.combine(referenceGrid)
        hasher.combine(encoderScaleFactor)
        hasher.combine(decoderScaleFactor)
        hasher.combine(decoderShiftFactor)
        hasher.combine(decodeResolution)
        hasher.combine(originalSize)
        hasher.combine(targetSize)
        hasher.combine(lazyModelLoading)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.prompt == rhs.prompt
            && lhs.negativePrompt == rhs.negativePrompt
            && lhs.seed == rhs.seed
            && lhs.stepCount == rhs.stepCount
            && lhs.guidanceScale == rhs.guidanceScale
            && lhs.schedulerType == rhs.schedulerType
            && lhs.strength == rhs.strength
            && lhs.referenceGrid == rhs.referenceGrid
            && lhs.encoderScaleFactor == rhs.encoderScaleFactor
            && lhs.decoderScaleFactor == rhs.decoderScaleFactor
            && lhs.decoderShiftFactor == rhs.decoderShiftFactor
            && lhs.decodeResolution == rhs.decodeResolution
            && lhs.originalSize == rhs.originalSize
            && lhs.targetSize == rhs.targetSize
            && lhs.lazyModelLoading == rhs.lazyModelLoading
    }
}
