// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Quality presets that trade off speed vs output quality.
public enum QualityPreset: String, CaseIterable, Sendable {
    case fast
    case balanced
    case best

    public var steps: Int {
        switch self {
        case .fast: 12
        case .balanced: 30
        case .best: 50
        }
    }

    public var numFrames: Int {
        switch self {
        case .fast: 33
        case .balanced: 81
        case .best: 81
        }
    }

    public var cfgCutoff: Float? {
        switch self {
        case .fast: 0.5
        case .balanced: 0.5
        case .best: nil
        }
    }
}

/// Configuration for video generation.
public struct VideoConfiguration: Sendable {
    public var prompt: String
    public var negativePrompt: String
    public var seed: UInt32
    public var stepCount: Int
    public var guidanceScale: Float
    public var numFrames: Int
    public var fps: Int
    public var width: Int
    public var height: Int
    public var dumpDirectory: String?
    public var loadNoisePath: String?

    /// Tiled VAE decoding configuration.
    /// When `vaeTileSize` is non-nil, the VAE decodes spatial tiles independently
    /// and blends overlapping regions, reducing peak memory for high-resolution outputs.
    public var vaeTileSize: Int?

    /// Overlap between adjacent VAE tiles in latent space units.
    /// Default: 4 (= 32 pixels at 8x spatial compression).
    public var vaeTileOverlap: Int

    /// Fraction of final steps to skip the unconditional (negative) pass.
    /// E.g., 0.3 means the last 30% of steps use only the conditional prediction.
    /// nil = disabled (full CFG for all steps). Saves one transformer call per skipped step.
    public var cfgCutoffFraction: Float?

    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 42,
        stepCount: Int = 50,
        guidanceScale: Float = 5.0,
        numFrames: Int = 81,
        fps: Int = 16,
        width: Int = 832,
        height: Int = 480,
        dumpDirectory: String? = nil,
        loadNoisePath: String? = nil,
        vaeTileSize: Int? = nil,
        vaeTileOverlap: Int = 4,
        cfgCutoffFraction: Float? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.numFrames = numFrames
        self.fps = fps
        self.width = width
        self.height = height
        self.dumpDirectory = dumpDirectory
        self.loadNoisePath = loadNoisePath
        self.vaeTileSize = vaeTileSize
        self.vaeTileOverlap = vaeTileOverlap
        self.cfgCutoffFraction = cfgCutoffFraction
    }

    /// Convenience: create from a quality preset with optional overrides.
    public static func from(
        preset: QualityPreset,
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 42,
        stepCount: Int? = nil,
        guidanceScale: Float = 5.0,
        numFrames: Int? = nil,
        fps: Int = 16,
        width: Int = 832,
        height: Int = 480,
        vaeTileSize: Int? = nil,
        vaeTileOverlap: Int = 4,
        cfgCutoffFraction: Float? = nil
    ) -> VideoConfiguration {
        VideoConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: stepCount ?? preset.steps,
            guidanceScale: guidanceScale,
            numFrames: numFrames ?? preset.numFrames,
            fps: fps,
            width: width,
            height: height,
            vaeTileSize: vaeTileSize,
            vaeTileOverlap: vaeTileOverlap,
            cfgCutoffFraction: cfgCutoffFraction ?? preset.cfgCutoff
        )
    }
}
