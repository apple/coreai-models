// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Loaded diffusion pipeline components backed by Core AI model functions.
public struct CoreAIDiffusionComponents: Sendable {
    public let textEncoder: CoreAITextEncoder
    public let denoiser: CoreAIDenoiser
    public let decoder: CoreAILatentDecoder
    public let encoder: CoreAILatentEncoder?
}

/// Errors during pipeline loading.
public enum PipelineLoadError: Error, LocalizedError {
    case missingComponent(String)
    case missingConfig(String)
    case deprecatedFormat(String)
    case configMismatch(field: String, expected: String, actual: String)
    case unsupportedConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let name):
            return "Required component '\(name)' not found in model directory"
        case .missingConfig(let detail):
            return "Invalid bundle configuration: \(detail)"
        case .deprecatedFormat(let message):
            return message
        case .configMismatch(let field, let expected, let actual):
            return "Config mismatch for '\(field)': config says \(expected), model says \(actual)"
        case .unsupportedConfiguration(let detail):
            return "Unsupported configuration: \(detail)"
        }
    }
}

/// Resolves `path` against `directory` and verifies the asset exists on disk, throwing
/// `PipelineLoadError.missingComponent` with the attempted filename if not — e.g. when a
/// pipeline descriptor still names a source `.aimodel` that's since been compiled to
/// `.aimodelc` without updating the descriptor.
func resolveExistingPipelineAsset(_ path: String, in directory: URL, component: String) throws -> URL {
    let url = directory.appendingPathComponent(path)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw PipelineLoadError.missingComponent("\(component) (expected \(url.lastPathComponent))")
    }
    return url
}

extension PipelineDescriptor {
    /// Load Core AI components from a model directory based on this descriptor.
    ///
    /// - Infers nil metadata fields from loaded model descriptors
    /// - Validates explicit config values against actual model shapes
    /// - Returns ready-to-use components
    public mutating func loadComponents(from baseURL: URL) async throws -> CoreAIDiffusionComponents {
        guard let unetPath = components.unet else {
            throw PipelineLoadError.missingComponent("unet")
        }
        guard let decoderPath = components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }
        guard let textEncoderPath = components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }

        let unetURL = try resolveExistingPipelineAsset(unetPath, in: baseURL, component: "unet")
        let decoderURL = try resolveExistingPipelineAsset(decoderPath, in: baseURL, component: "vae_decoder")
        let textEncoderURL = try resolveExistingPipelineAsset(
            textEncoderPath, in: baseURL, component: "text_encoder")
        let encoderURL = try components.vaeEncoder.map {
            try resolveExistingPipelineAsset($0, in: baseURL, component: "vae_encoder")
        }

        // Create model functions
        let unetFunction = CoreAIDiffusionModelFunction(modelURL: unetURL)
        let decoderFunction = CoreAIDiffusionModelFunction(modelURL: decoderURL)
        let textEncoderFunction = CoreAIDiffusionModelFunction(modelURL: textEncoderURL)
        let encoderFunction = encoderURL.map { CoreAIDiffusionModelFunction(modelURL: $0) }

        // Load UNet to inspect its descriptors
        try await unetFunction.loadResources()

        // Infer/validate metadata from UNet input shape
        let unetInputs = try await unetFunction.inputDescriptors
        if let sampleDesc = unetInputs["sample"] ?? unetInputs["latent_model_input"] {
            let shape = sampleDesc.shape
            // shape is [1, C, H, W] — image_size = H * 8 (latent space is 8x downsampled)
            if shape.count == 4 && shape[2] > 0 {
                let inferredSize = shape[2] * 8
                if let configSize = imageSize {
                    if configSize != inferredSize {
                        throw PipelineLoadError.configMismatch(
                            field: "imageSize",
                            expected: "\(configSize)",
                            actual: "\(inferredSize)")
                    }
                } else {
                    imageSize = inferredSize
                }
            }
        }

        // Default metadata if still nil after inference
        if type == nil { type = .stableDiffusion }
        if predictionType == nil { predictionType = .epsilon }
        if decoderScaleFactor == nil { decoderScaleFactor = 0.18215 }

        // Load tokenizer
        let vocabURL = baseURL.appendingPathComponent("vocab.json")
        let mergesURL = baseURL.appendingPathComponent("merges.txt")
        let tokenizer: BPETokenizer
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            tokenizer = try BPETokenizer(mergesAt: mergesURL, vocabularyAt: vocabURL)
        } else {
            let tokenizerDir = baseURL.appendingPathComponent("tokenizer")
            tokenizer = try BPETokenizer(
                mergesAt: tokenizerDir.appendingPathComponent("merges.txt"),
                vocabularyAt: tokenizerDir.appendingPathComponent("vocab.json"))
        }

        let textEncoderSeqLength = try await textEncoderFunction.inferSequenceLength() ?? 77

        let textEncoder = CoreAITextEncoder(
            function: textEncoderFunction,
            tokenize: { text in
                let (_, ids) = tokenizer.tokenize(input: text, minCount: textEncoderSeqLength)
                return ids.map(Int32.init)
            },
            maxLength: textEncoderSeqLength
        )

        let denoiser = CoreAIDenoiser(function: unetFunction)
        let decoder = CoreAILatentDecoder(function: decoderFunction)
        let encoder = encoderFunction.map { CoreAILatentEncoder(function: $0) }

        return CoreAIDiffusionComponents(
            textEncoder: textEncoder,
            denoiser: denoiser,
            decoder: decoder,
            encoder: encoder
        )
    }
}
