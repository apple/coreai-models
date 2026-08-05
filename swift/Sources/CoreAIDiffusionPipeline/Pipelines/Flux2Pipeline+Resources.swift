// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

extension Flux2Pipeline {
    /// Load a FLUX.2 pipeline from a directory containing .aimodel files, tokenizer/, and pipeline.json.
    ///
    /// The `mode` parameter selects which components are loaded:
    /// - `.full`: Transformer + VAEDecoder (1024×1024)
    /// - `.half`: Transformer_512 + VAEDecoder_half (512×512, 4× faster)
    /// - `.tiled`: Transformer + VAEDecoder_half (1024×1024 via tiled decode)
    public init(
        from url: URL,
        config: PipelineDescriptor.ConfigSource = .auto,
        mode: DecodeResolution = .auto
    ) async throws {
        let descriptor = try PipelineDescriptor.resolve(at: url, config: config)

        // Resolve .auto → best available mode
        let resolvedMode: DecodeResolution
        if mode == .auto {
            resolvedMode = try Self.bestAvailableMode(at: url, descriptor: descriptor)
        } else {
            resolvedMode = mode
        }

        guard let textEncoderPath = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }

        // Select transformer by mode (explicit name, not auto-detect)
        let transformerName: String
        switch resolvedMode {
        case .full, .tiled:
            guard let path = Self.resolveAsset(at: url, name: "Transformer") else {
                throw PipelineLoadError.missingComponent("Transformer")
            }
            transformerName = path
        case .half:
            guard let path = Self.resolveAsset(at: url, name: "Transformer_512") else {
                throw PipelineLoadError.missingComponent("Transformer_512")
            }
            transformerName = path
        case .auto:
            preconditionFailure("auto resolved above")
        }

        // Select decoder by mode (explicit name)
        let decoderName: String
        switch resolvedMode {
        case .full:
            guard let path = Self.resolveAsset(at: url, name: "VAEDecoder") else {
                throw PipelineLoadError.missingComponent("VAEDecoder")
            }
            decoderName = path
        case .half, .tiled:
            guard let path = Self.resolveAsset(at: url, name: "VAEDecoder_half") else {
                throw PipelineLoadError.missingComponent("VAEDecoder_half")
            }
            decoderName = path
        case .auto:
            preconditionFailure("auto resolved above")
        }

        let transformer = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(transformerName))
        let textEncoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(textEncoderPath))
        let decoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(decoderName))

        // Encoder for img2img (optional)
        let encoderName: String?
        switch resolvedMode {
        case .full:
            encoderName = descriptor.components.vaeEncoder
        case .half, .tiled:
            encoderName = Self.resolveAsset(at: url, name: "VAEEncoder_half")
        case .auto:
            preconditionFailure("auto resolved above")
        }
        let encoder: CoreAIDiffusionModelFunction?
        if let name = encoderName {
            encoder = CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent(name))
        } else {
            encoder = nil
        }

        // Load Qwen3 tokenizer
        let tokenizerDir = url.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load VAE batch norm statistics
        let bnMean = Flux2Pipeline.loadNpyFloat32(url.appendingPathComponent("vae_bn_mean.npy"))
        let bnVar = Flux2Pipeline.loadNpyFloat32(url.appendingPathComponent("vae_bn_var.npy"))
        let bnEps = descriptor.batchNormEps ?? 1e-5

        self.descriptor = descriptor
        self.mode = resolvedMode
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.batchNormMean = bnMean
        self.batchNormVar = bnVar
        self.batchNormEps = bnEps
    }

    /// Resolve an asset name to a filename, checking for .aimodel or .aimodelc.
    private static func resolveAsset(at url: URL, name: String) -> String? {
        let resolved = ModelBundle.resolveAssetURL("\(name).aimodel", in: url)
        guard FileManager.default.fileExists(atPath: resolved.path) else { return nil }
        return resolved.lastPathComponent
    }

    /// Probe available assets and pick the highest quality mode.
    /// Priority: .full > .tiled > .half. Throws if no valid combination exists.
    private static func bestAvailableMode(
        at url: URL, descriptor: PipelineDescriptor
    ) throws -> DecodeResolution {
        let hasFullTransformer = descriptor.components.unet != nil
        let hasFullDecoder = descriptor.components.vaeDecoder != nil
        let hasHalfDecoder = resolveAsset(at: url, name: "VAEDecoder_half") != nil
        let hasHalfTransformer = resolveAsset(at: url, name: "Transformer_512") != nil

        if hasFullTransformer && hasFullDecoder { return .full }
        if hasFullTransformer && hasHalfDecoder { return .tiled }
        if hasHalfTransformer && hasHalfDecoder { return .half }
        throw PipelineLoadError.missingComponent(
            "No valid component combination found. Need Transformer+VAEDecoder, "
                + "Transformer+VAEDecoder_half, or Transformer_512+VAEDecoder_half.")
    }

    // MARK: - Npy Reader

    private static func loadNpyFloat32(_ url: URL) -> [Float]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard data.count > 10,
            data[0] == 0x93, data[1] == 0x4E, data[2] == 0x55,
            data[3] == 0x4D, data[4] == 0x50, data[5] == 0x59
        else {
            return nil
        }
        let majorVersion = data[6]
        let headerLen: Int
        let headerStart: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let rawData = data[dataStart...]
        return rawData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float32.self))
        }
    }
}
