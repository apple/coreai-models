// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics

/// Controls which model components are loaded and how decoding is performed.
///
/// A dynamically-shaped bundle loads the same assets in every mode: `.half` makes half the
/// bundle's size the default output size, and `.tiled` decodes in half-size tiles.
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

    /// Reference token-grid side for an output grid `outputSide` tokens across.
    func side(outputSide: Int) -> Int {
        switch self {
        case .full: outputSide
        case .half: outputSide / 2
        case .quarter: outputSide / 4
        }
    }
}

/// Size each img2img reference image is encoded at: `.full`, `.half` or `.quarter` of
/// the output, or `.pixels(384)`.
///
/// Round-trips through `LosslessStringConvertible` as `full`/`half`/`quarter`/`384` for
/// command-line parsing.
public enum ReferenceSize: Hashable, Sendable {
    /// A preset fraction of the output size. Works on static and dynamic bundles.
    case relative(ReferenceGrid)
    /// An absolute square size in pixels, a multiple of 16 within the bundle's encoder
    /// range. Dynamic bundles only, unless it equals one of a static bundle's presets.
    case pixels(Int)

    public static let full = ReferenceSize.relative(.full)
    public static let half = ReferenceSize.relative(.half)
    public static let quarter = ReferenceSize.relative(.quarter)

    /// The square size, in pixels, each reference is encoded at for an output of
    /// `outputSize` pixels. A preset rounds down to whole 16-pixel tokens; a pixel size is
    /// returned as given.
    public func pixelSize(outputSize: Int) -> Int {
        switch self {
        case .relative(let grid): grid.side(outputSide: outputSize / 16) * 16
        case .pixels(let pixels): pixels
        }
    }

    /// Reference token-grid side for an output grid `outputSide` tokens across, where one
    /// token covers `patchSize` pixels.
    func side(outputSide: Int, patchSize: Int) -> Int {
        switch self {
        case .relative(let grid): grid.side(outputSide: outputSide)
        case .pixels(let pixels): pixels / patchSize
        }
    }

    /// The preset this size names at the given output grid. Static bundles are traced
    /// per preset, so this is the only form they can serve.
    func preset(outputSide: Int, patchSize: Int) -> ReferenceGrid? {
        switch self {
        case .relative(let grid):
            return grid
        case .pixels(let pixels):
            guard pixels % patchSize == 0 else { return nil }
            return ReferenceGrid.allCases.first { $0.side(outputSide: outputSide) == pixels / patchSize }
        }
    }
}

extension ReferenceSize: LosslessStringConvertible {
    public init?(_ description: String) {
        if let grid = ReferenceGrid(rawValue: description) {
            self = .relative(grid)
        } else if let pixels = Int(description), pixels > 0 {
            self = .pixels(pixels)
        } else {
            return nil
        }
    }

    public var description: String {
        switch self {
        case .relative(let grid): grid.rawValue
        case .pixels(let pixels): String(pixels)
        }
    }
}

/// Whether the pipeline performs classifier-free guidance (CFG) itself.
///
/// FLUX.2 Klein 4B is guidance-distilled and its transformer config sets
/// `guidance_embeds: false`, so the traced graph's `guidance` input is unused. The model
/// discards it and produces a usable image from a single unguided pass. Real CFG is
/// therefore something the pipeline adds on top, not something the model applies.
public enum GuidanceMode: String, Hashable, Sendable, CaseIterable {
    /// One forward pass, no classifier-free guidance. The distillation is what makes this
    /// work without it. `guidanceScale` is unused in this mode.
    case distilled
    /// Two forward passes — conditional and unconditional — interpolated by the pipeline
    /// with `guidanceScale` as the weight. Stronger text adherence in img2img, at ~2× the
    /// compute per step. This is the only mode in which `guidanceScale` has any effect.
    case manual
}

extension GuidanceMode: CustomStringConvertible {
    public var description: String { rawValue }
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
    /// The image to start denoising from, noised according to `strength` (classic img2img).
    /// FLUX.2 has no starting latent, so it treats this as its first reference image, for
    /// callers written before `referenceImages`.
    public var startingImage: CGImage?
    /// Images the model conditions on, in order: FLUX.2 reference tokens, referred to in
    /// the prompt as image 1, image 2, ... More than one requires a dynamic bundle.
    public var referenceImages: [CGImage]
    /// How much noise to add to `startingImage`, 0...1. FLUX.2 ignores it: its images are
    /// conditioning, not a starting point.
    public var strength: Float
    /// Size every reference image is encoded at.
    public var referenceSize: ReferenceSize
    public var guidanceMode: GuidanceMode

    // VAE scale factors (from the metadata.json diffusion block)
    public var encoderScaleFactor: Float
    public var decoderScaleFactor: Float
    public var decoderShiftFactor: Float

    // Decode resolution
    public var decodeResolution: DecodeResolution

    /// Square output size in pixels. `nil` uses the size the pipeline was loaded for
    /// (its `DecodeResolution`). A dynamically-shaped bundle accepts any supported size;
    /// a static one only the size it was loaded for.
    public var imageSize: Int?

    /// Load model components on demand and unload after each pipeline stage to reduce peak memory.
    /// Disable to keep all models resident and exercise full memory pressure (e.g. profiling peak footprint).
    public var lazyModelLoading: Bool

    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 0,
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        schedulerType: SchedulerType = .discreteFlow,
        startingImage: CGImage? = nil,
        referenceImages: [CGImage] = [],
        strength: Float = 1.0,
        referenceSize: ReferenceSize = .full,
        guidanceMode: GuidanceMode = .distilled,
        encoderScaleFactor: Float = 0.18215,
        decoderScaleFactor: Float = 0.18215,
        decoderShiftFactor: Float = 0.0,
        decodeResolution: DecodeResolution = .full,
        imageSize: Int? = nil,
        lazyModelLoading: Bool = true
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.schedulerType = schedulerType
        self.startingImage = startingImage
        self.referenceImages = referenceImages
        self.strength = strength
        self.referenceSize = referenceSize
        self.guidanceMode = guidanceMode
        self.encoderScaleFactor = encoderScaleFactor
        self.decoderScaleFactor = decoderScaleFactor
        self.decoderShiftFactor = decoderShiftFactor
        self.decodeResolution = decodeResolution
        self.imageSize = imageSize
        self.lazyModelLoading = lazyModelLoading
    }

    /// The images FLUX.2 conditions on: `startingImage` (if set) as image 1, then
    /// `referenceImages`.
    public var allReferenceImages: [CGImage] {
        (startingImage.map { [$0] } ?? []) + referenceImages
    }

    public var isImageToImage: Bool { !allReferenceImages.isEmpty }
}

// MARK: - Pre-`referenceSize` API

extension PipelineConfiguration {
    /// The reference preset. A `.pixels` size has no preset without knowing the output
    /// size, so it reads as `.full`.
    @available(*, deprecated, renamed: "referenceSize")
    public var referenceGrid: ReferenceGrid {
        get {
            if case .relative(let grid) = referenceSize { return grid }
            return .full
        }
        set { referenceSize = .relative(newValue) }
    }

    @available(*, deprecated, message: "Use init(prompt:...referenceSize:...) instead.")
    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 0,
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        schedulerType: SchedulerType = .discreteFlow,
        startingImage: CGImage? = nil,
        strength: Float = 1.0,
        referenceGrid: ReferenceGrid,
        guidanceMode: GuidanceMode = .distilled,
        encoderScaleFactor: Float = 0.18215,
        decoderScaleFactor: Float = 0.18215,
        decoderShiftFactor: Float = 0.0,
        decodeResolution: DecodeResolution = .full,
        lazyModelLoading: Bool = true
    ) {
        self.init(
            prompt: prompt, negativePrompt: negativePrompt, seed: seed, stepCount: stepCount,
            guidanceScale: guidanceScale, schedulerType: schedulerType,
            startingImage: startingImage, strength: strength,
            referenceSize: .relative(referenceGrid), guidanceMode: guidanceMode,
            encoderScaleFactor: encoderScaleFactor, decoderScaleFactor: decoderScaleFactor,
            decoderShiftFactor: decoderShiftFactor, decodeResolution: decodeResolution,
            lazyModelLoading: lazyModelLoading)
    }
}

/// Hashable conformance — CGImage excluded (not Hashable); the reference count is kept.
extension PipelineConfiguration {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(prompt)
        hasher.combine(negativePrompt)
        hasher.combine(seed)
        hasher.combine(stepCount)
        hasher.combine(guidanceScale)
        hasher.combine(schedulerType)
        hasher.combine(allReferenceImages.count)
        hasher.combine(strength)
        hasher.combine(referenceSize)
        hasher.combine(guidanceMode)
        hasher.combine(encoderScaleFactor)
        hasher.combine(decoderScaleFactor)
        hasher.combine(decoderShiftFactor)
        hasher.combine(decodeResolution)
        hasher.combine(imageSize)
        hasher.combine(lazyModelLoading)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.prompt == rhs.prompt
            && lhs.negativePrompt == rhs.negativePrompt
            && lhs.seed == rhs.seed
            && lhs.stepCount == rhs.stepCount
            && lhs.guidanceScale == rhs.guidanceScale
            && lhs.schedulerType == rhs.schedulerType
            && lhs.allReferenceImages.count == rhs.allReferenceImages.count
            && lhs.strength == rhs.strength
            && lhs.referenceSize == rhs.referenceSize
            && lhs.guidanceMode == rhs.guidanceMode
            && lhs.encoderScaleFactor == rhs.encoderScaleFactor
            && lhs.decoderScaleFactor == rhs.decoderScaleFactor
            && lhs.decoderShiftFactor == rhs.decoderShiftFactor
            && lhs.decodeResolution == rhs.decodeResolution
            && lhs.imageSize == rhs.imageSize
            && lhs.lazyModelLoading == rhs.lazyModelLoading
    }
}
