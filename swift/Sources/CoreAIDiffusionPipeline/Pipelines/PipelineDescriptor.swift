// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Describes the model bundle layout and configuration for a diffusion pipeline.
///
/// This is the serializable representation of `pipeline.json`. It describes what
/// components exist, their file paths, and model-level settings (prediction type,
/// image size, scale factors).
///
/// Fields are optional when auto-detecting — they get filled in during asset loading
/// by inspecting the actual model descriptors. If a `pipeline.json` provides explicit
/// values, they are validated against the loaded model.
///
/// Separate from `PipelineConfiguration` which is per-generation (prompt, seed, steps).
public struct PipelineDescriptor: Codable, Sendable {
    public var type: PipelineType?
    public var version: String?
    public var predictionType: PredictionType?
    public var imageSize: Int?
    public var components: ComponentPaths
    public var scheduler: SchedulerDefaults?
    public var encoderScaleFactor: Float?
    public var decoderScaleFactor: Float?
    public var decoderShiftFactor: Float?

    // FLUX.2-specific fields
    public var batchNormEps: Float?
    public var guidanceEmbeds: Bool?
    public var ropeAxesDims: [Int]?
    public var ropeTheta: Float?
    public var defaultGuidanceScale: Float?
    public var defaultSteps: Int?

    public init(
        type: PipelineType? = nil,
        version: String? = nil,
        predictionType: PredictionType? = nil,
        imageSize: Int? = nil,
        components: ComponentPaths = ComponentPaths(),
        scheduler: SchedulerDefaults? = nil,
        encoderScaleFactor: Float? = nil,
        decoderScaleFactor: Float? = nil,
        decoderShiftFactor: Float? = nil,
        batchNormEps: Float? = nil,
        guidanceEmbeds: Bool? = nil,
        ropeAxesDims: [Int]? = nil,
        ropeTheta: Float? = nil,
        defaultGuidanceScale: Float? = nil,
        defaultSteps: Int? = nil
    ) {
        self.type = type
        self.version = version
        self.predictionType = predictionType
        self.imageSize = imageSize
        self.components = components
        self.scheduler = scheduler
        self.encoderScaleFactor = encoderScaleFactor
        self.decoderScaleFactor = decoderScaleFactor
        self.decoderShiftFactor = decoderShiftFactor
        self.batchNormEps = batchNormEps
        self.guidanceEmbeds = guidanceEmbeds
        self.ropeAxesDims = ropeAxesDims
        self.ropeTheta = ropeTheta
        self.defaultGuidanceScale = defaultGuidanceScale
        self.defaultSteps = defaultSteps
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(PipelineType.self, forKey: .type)
        self.version = try container.decodeIfPresent(String.self, forKey: .version)
        self.predictionType = try container.decodeIfPresent(PredictionType.self, forKey: .predictionType)
        self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize)
        self.components = (try container.decodeIfPresent(ComponentPaths.self, forKey: .components)) ?? ComponentPaths()
        self.scheduler = try container.decodeIfPresent(SchedulerDefaults.self, forKey: .scheduler)
        self.encoderScaleFactor = try container.decodeIfPresent(Float.self, forKey: .encoderScaleFactor)
        self.decoderScaleFactor = try container.decodeIfPresent(Float.self, forKey: .decoderScaleFactor)
        self.decoderShiftFactor = try container.decodeIfPresent(Float.self, forKey: .decoderShiftFactor)
        self.batchNormEps = try container.decodeIfPresent(Float.self, forKey: .batchNormEps)
        self.guidanceEmbeds = try container.decodeIfPresent(Bool.self, forKey: .guidanceEmbeds)
        self.ropeAxesDims = try container.decodeIfPresent([Int].self, forKey: .ropeAxesDims)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        self.defaultGuidanceScale = try container.decodeIfPresent(Float.self, forKey: .defaultGuidanceScale)
        self.defaultSteps = try container.decodeIfPresent(Int.self, forKey: .defaultSteps)
    }

    private enum CodingKeys: String, CodingKey {
        case type, version, predictionType, imageSize, components, scheduler
        case encoderScaleFactor, decoderScaleFactor, decoderShiftFactor
        case batchNormEps, guidanceEmbeds, ropeAxesDims, ropeTheta
        case defaultGuidanceScale, defaultSteps
    }

    // MARK: - Loading

    public enum ConfigSource {
        case auto
        case file(URL)
        case explicit(PipelineDescriptor)
    }

    /// Load or detect a pipeline descriptor from a model directory.
    ///
    /// Priority:
    /// 1. `metadata.json` (v0.2 schema with `kind: "diffusion"`)
    /// 2. `pipeline.json` (deprecated — prints migration warning)
    /// 3. Directory scan for known component filenames
    ///
    /// Fields left nil by auto-detection are filled in later during `loadComponents(from:)`
    /// by inspecting the actual model descriptors.
    public static func resolve(at url: URL, config: ConfigSource = .auto) throws -> PipelineDescriptor {
        switch config {
        case .auto:
            let metadataURL = url.appendingPathComponent("metadata.json")
            if FileManager.default.fileExists(atPath: metadataURL.path) {
                return try loadFromMetadata(at: metadataURL)
            }
            let pipelineURL = url.appendingPathComponent("pipeline.json")
            if FileManager.default.fileExists(atPath: pipelineURL.path) {
                throw PipelineLoadError.deprecatedFormat(
                    "This bundle uses the legacy pipeline.json format which is no longer supported.\n"
                        + "Please re-export with `coreai.diffusion.export` to produce metadata.json.\n"
                        + "See: https://github.com/apple/coreai-models/issues/TBD"
                )
            }
            return detect(at: url)
        case .file(let configURL):
            return try load(from: configURL)
        case .explicit(let descriptor):
            return descriptor
        }
    }

    /// Parse a metadata.json file (v0.2 schema) and extract the diffusion config.
    public static func loadFromMetadata(at url: URL) throws -> PipelineDescriptor {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]

        guard let diffusion = json["diffusion"] as? [String: Any] else {
            throw PipelineLoadError.missingConfig("metadata.json has no 'diffusion' block")
        }
        guard let assets = json["assets"] as? [String: String] else {
            throw PipelineLoadError.missingConfig("metadata.json has no 'assets' map")
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let diffusionData = try JSONSerialization.data(withJSONObject: diffusion)
        var descriptor = try decoder.decode(PipelineDescriptor.self, from: diffusionData)

        // Map assets to component paths
        descriptor.components.textEncoder = assets["text_encoder"]
        descriptor.components.textEncoder2 = assets["text_encoder_2"]
        descriptor.components.unet = assets["transformer"] ?? assets["unet"]
        descriptor.components.vaeDecoder = assets["vae_decoder"]
        descriptor.components.vaeEncoder = assets["vae_encoder"]

        return descriptor
    }

    /// Parse a pipeline.json file.
    /// Supports both the new format (with `components`) and the legacy format
    /// (where component paths are inferred from the directory).
    public static func load(from url: URL) throws -> PipelineDescriptor {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        var descriptor = try decoder.decode(PipelineDescriptor.self, from: data)

        // If no components were specified, detect from the same directory
        if descriptor.components.textEncoder == nil && descriptor.components.unet == nil {
            let dir = url.deletingLastPathComponent()
            let detected = detect(at: dir)
            descriptor.components = detected.components
        }

        return descriptor
    }

    /// Scan a directory for known component filenames.
    /// Only fills in component paths — model metadata (type, version, prediction,
    /// image size) is left nil and inferred later from loaded assets.
    public static func detect(at url: URL) -> PipelineDescriptor {
        var descriptor = PipelineDescriptor()

        let contents = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []

        for file in contents {
            let lower = file.lowercased()
            if lower.contains("textencoder2") || lower.contains("text_encoder_2") {
                descriptor.components.textEncoder2 = file
            } else if lower.contains("textencoder") || lower.contains("text_encoder") {
                descriptor.components.textEncoder = file
            } else if lower.contains("unet") || lower.contains("transformer") || lower.contains("mmdit") {
                descriptor.components.unet = file
            } else if (lower.contains("vaedecoder") || lower.contains("vae_decoder"))
                && !lower.contains("half")
            {
                descriptor.components.vaeDecoder = file
            } else if (lower.contains("vaeencoder") || lower.contains("vae_encoder"))
                && !lower.contains("half")
            {
                descriptor.components.vaeEncoder = file
            }
        }

        return descriptor
    }

    // MARK: - Nested Types

    public enum PipelineType: String, Codable, Sendable {
        case stableDiffusion = "stable-diffusion"
        case stableDiffusionXL = "stable-diffusion-xl"
        case stableDiffusion3 = "stable-diffusion-3"
        case flux2 = "flux2"
    }

    public struct ComponentPaths: Codable, Sendable {
        public var textEncoder: String?
        public var textEncoder2: String?
        public var unet: String?
        public var vaeDecoder: String?
        public var vaeEncoder: String?

        public init(
            textEncoder: String? = nil,
            textEncoder2: String? = nil,
            unet: String? = nil,
            vaeDecoder: String? = nil,
            vaeEncoder: String? = nil
        ) {
            self.textEncoder = textEncoder
            self.textEncoder2 = textEncoder2
            self.unet = unet
            self.vaeDecoder = vaeDecoder
            self.vaeEncoder = vaeEncoder
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.textEncoder = try container.decodeIfPresent(String.self, forKey: .textEncoder)
            self.textEncoder2 = try container.decodeIfPresent(String.self, forKey: .textEncoder2)
            self.unet = try container.decodeIfPresent(String.self, forKey: .unet)
            self.vaeDecoder = try container.decodeIfPresent(String.self, forKey: .vaeDecoder)
            self.vaeEncoder = try container.decodeIfPresent(String.self, forKey: .vaeEncoder)
        }
    }

    public struct SchedulerDefaults: Codable, Sendable {
        public var trainingSteps: Int
        public var betaStart: Float
        public var betaEnd: Float
        public var betaSchedule: String

        public init(
            trainingSteps: Int = 1000,
            betaStart: Float = 0.00085,
            betaEnd: Float = 0.012,
            betaSchedule: String = "scaled_linear"
        ) {
            self.trainingSteps = trainingSteps
            self.betaStart = betaStart
            self.betaEnd = betaEnd
            self.betaSchedule = betaSchedule
        }
    }
}
