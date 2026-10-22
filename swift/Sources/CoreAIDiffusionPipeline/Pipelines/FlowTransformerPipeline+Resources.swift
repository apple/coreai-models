// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

extension FlowTransformerPipeline {
    /// Load a FLUX.2 or Sana Sprint pipeline from a directory containing .aimodel files,
    /// tokenizer/, and metadata.json.
    ///
    /// For FLUX.2,
    /// the `mode` parameter selects which components are loaded:
    /// - `.full`: Transformer + VAEDecoder (1024×1024)
    /// - `.half`: Transformer_512 + VAEDecoder_half (512×512, 4× faster)
    /// - `.tiled`: Transformer + VAEDecoder_half (1024×1024 via tiled decode)
    public init(
        from url: URL,
        mode: DecodeResolution = .auto
    ) async throws {
        let bundle = try DiffusionBundle(at: url)
        let config = bundle.config
        if config.type == .sanaSprint {
            self = try await Self.loadSanaSprint(from: url, bundle: bundle, mode: mode)
            return
        }

        // Resolve .auto → best available mode
        let resolvedMode: DecodeResolution
        if mode == .auto {
            resolvedMode = try Self.bestAvailableMode(bundle: bundle)
        } else {
            resolvedMode = mode
        }

        let textEncoderURL = try resolveDeclaredPipelineAsset(AssetKey.textEncoder, in: bundle)

        // Select transformer by mode.
        // Prefer the multi-function transformer. Only fall back to the single-function
        // transformer_512 when the multi-function model is absent.
        let transformer: CoreAIDiffusionModelFunction
        let transformerFnName: String
        switch resolvedMode {
        case .full, .tiled:
            transformer = CoreAIDiffusionModelFunction(
                modelURL: try resolveDeclaredPipelineAsset(AssetKey.transformer, in: bundle))
            transformerFnName = "main"
        case .half:
            if let transformerURL = try Self.optionalAsset(AssetKey.transformer, in: bundle) {
                let candidate = CoreAIDiffusionModelFunction(modelURL: transformerURL)
                if try await candidate.hasFunction(named: "half") {
                    // Multi-function model: use its "half" function (img2img uses
                    // img2img_512_* from the same asset).
                    transformer = candidate
                    transformerFnName = "half"
                } else if let url512 = try Self.optionalAsset(AssetKey.transformer512, in: bundle) {
                    transformer = CoreAIDiffusionModelFunction(modelURL: url512)
                    transformerFnName = "main"
                } else {
                    // Full-only Transformer without a "half" function and no
                    // Transformer_512 — best effort with "main".
                    transformer = candidate
                    transformerFnName = "main"
                }
            } else if let url512 = try Self.optionalAsset(AssetKey.transformer512, in: bundle) {
                transformer = CoreAIDiffusionModelFunction(modelURL: url512)
                transformerFnName = "main"
            } else {
                throw PipelineLoadError.missingComponent(
                    "\(AssetKey.transformer) or \(AssetKey.transformer512) (not declared in metadata.json)")
            }
        case .auto:
            preconditionFailure("auto resolved above")
        }

        // Resolve how each reference grid reaches a traced graph.
        //
        // Prefer the multi-function transformer when available.
        let entrypointPrefix = resolvedMode == .half ? "img2img_512_" : "img2img_"
        let keyPrefix = resolvedMode == .half ? "transformer_512_img2img" : "transformer_img2img"
        var img2imgRoutes: [ReferenceGrid: Img2ImgRoute] = [:]
        for grid in ReferenceGrid.allCases {
            let entrypoint = "\(entrypointPrefix)\(grid.rawValue)"
            if try await transformer.hasFunction(named: entrypoint) {
                img2imgRoutes[grid] = Img2ImgRoute(function: transformer, entrypoint: entrypoint)
            } else if let assetURL = try Self.optionalAsset("\(keyPrefix)_\(grid.rawValue)", in: bundle) {
                // Single-function: its own asset, traced at one concatenated sequence
                // length, always entered through "main".
                img2imgRoutes[grid] = Img2ImgRoute(
                    function: CoreAIDiffusionModelFunction(modelURL: assetURL), entrypoint: "main")
            }
        }

        // Select decoder by mode
        let decoderURL: URL
        switch resolvedMode {
        case .full:
            decoderURL = try resolveDeclaredPipelineAsset(AssetKey.vaeDecoder, in: bundle)
        case .half, .tiled:
            decoderURL = try resolveDeclaredPipelineAsset(AssetKey.vaeDecoderHalf, in: bundle)
        case .auto:
            preconditionFailure("auto resolved above")
        }

        let textEncoder = CoreAIDiffusionModelFunction(modelURL: textEncoderURL)
        let decoder = CoreAIDiffusionModelFunction(modelURL: decoderURL)

        // Encoder for img2img (optional): nil when absent so supportsImageToImage is accurate.
        let encoderURL: URL?
        switch resolvedMode {
        case .full:
            encoderURL = try Self.optionalAsset(AssetKey.vaeEncoder, in: bundle)
        case .half, .tiled:
            encoderURL = try Self.optionalAsset(AssetKey.vaeEncoderHalf, in: bundle)
        case .auto:
            preconditionFailure("auto resolved above")
        }
        let encoder = encoderURL.map { CoreAIDiffusionModelFunction(modelURL: $0) }

        // Load Qwen3 tokenizer
        let tokenizerDir = url.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load VAE batch norm statistics
        let bnMean = Self.loadNpyFloat32(url.appendingPathComponent("vae_bn_mean.npy"))
        let bnVar = Self.loadNpyFloat32(url.appendingPathComponent("vae_bn_var.npy"))
        let bnEps = config.batchNormEps ?? DiffusionDefaults.Image.batchNormEps

        self.config = config
        self.mode = resolvedMode
        self.transformer = transformer
        self.img2imgRoutes = img2imgRoutes
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.encoder = encoder
        self.transformerFunctionName = transformerFnName
        self.tokenizer = tokenizer
        self.batchNormMean = bnMean
        self.batchNormVar = bnVar
        self.batchNormEps = bnEps
    }

    /// Sana Sprint bundles carry one transformer and one VAE decoder, both at full resolution.
    private static func loadSanaSprint(
        from url: URL, bundle: DiffusionBundle, mode: DecodeResolution
    ) async throws -> Self {
        guard mode == .auto || mode == .full else {
            throw PipelineLoadError.unsupportedConfiguration(
                "Sana Sprint bundles decode at full resolution only; got \(mode).")
        }
        func component(_ name: String, _ path: String?) throws -> CoreAIDiffusionModelFunction {
            guard let path else { throw PipelineLoadError.missingComponent(name) }
            let assetURL = try resolveExistingPipelineAsset(path, in: url, component: name)
            return CoreAIDiffusionModelFunction(modelURL: assetURL)
        }

        return Self(
            config: bundle.config,
            mode: .full,
            transformer: try component(DiffusionComponentKey.transformer, bundle.transformerFilename),
            textEncoder: try component(DiffusionComponentKey.textEncoder, bundle.textEncoderFilename),
            decoder: try component(DiffusionComponentKey.vaeDecoder, bundle.vaeDecoderFilename),
            encoder: nil,
            tokenizer: try await AutoTokenizer.from(modelFolder: url.appendingPathComponent("tokenizer")),
            batchNormMean: nil,
            batchNormVar: nil,
            batchNormEps: 0)
    }

    /// `metadata.json` asset keys for FLUX.2. The manifest is the only source of asset
    /// filenames; an undeclared key means the asset isn't in the bundle.
    private enum AssetKey {
        static let textEncoder = DiffusionComponentKey.textEncoder
        static let transformer = DiffusionComponentKey.transformer
        static let transformer512 = "transformer_512"
        static let vaeDecoder = DiffusionComponentKey.vaeDecoder
        static let vaeDecoderHalf = "vae_decoder_half"
        static let vaeEncoder = DiffusionComponentKey.vaeEncoder
        static let vaeEncoderHalf = "vae_encoder_half"
    }

    /// The URL of an optional asset, or nil when `metadata.json` doesn't declare it. A declared
    /// asset whose file is missing throws, like a required one.
    private static func optionalAsset(_ key: String, in bundle: DiffusionBundle) throws -> URL? {
        guard bundle.assetFilename(for: key) != nil else { return nil }
        return try resolveDeclaredPipelineAsset(key, in: bundle)
    }

    /// Probe available assets and pick the highest quality mode.
    /// Priority: .full > .tiled > .half. Throws if no valid combination exists.
    private static func bestAvailableMode(bundle: DiffusionBundle) throws -> DecodeResolution {
        let hasFullTransformer = try optionalAsset(AssetKey.transformer, in: bundle) != nil
        let hasFullDecoder = try optionalAsset(AssetKey.vaeDecoder, in: bundle) != nil
        let hasHalfDecoder = try optionalAsset(AssetKey.vaeDecoderHalf, in: bundle) != nil
        let hasHalfTransformer = try optionalAsset(AssetKey.transformer512, in: bundle) != nil

        if hasFullTransformer && hasFullDecoder { return .full }
        if hasFullTransformer && hasHalfDecoder { return .tiled }
        if hasHalfTransformer && hasHalfDecoder { return .half }
        throw PipelineLoadError.missingComponent(
            "No valid component combination declared in metadata.json. Need transformer+vae_decoder, "
                + "transformer+vae_decoder_half, or transformer_512+vae_decoder_half.")
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
