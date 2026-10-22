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

        guard let textEncoderPath = try Self.resolveAsset(.textEncoder, in: bundle) else {
            throw Self.missing(.textEncoder, in: bundle)
        }

        // Select transformer by mode.
        // Prefer the multi-function Transformer.aimodel. Only fall back to the
        // single-function Transformer_512.aimodel when the multi-function model is
        // absent.
        let transformer: CoreAIDiffusionModelFunction
        let transformerFnName: String
        switch resolvedMode {
        case .full, .tiled:
            guard let path = try Self.resolveAsset(.transformer, in: bundle) else {
                throw Self.missing(.transformer, in: bundle)
            }
            transformer = CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent(path))
            transformerFnName = "main"
        case .half:
            if let path = try Self.resolveAsset(.transformer, in: bundle) {
                let candidate = CoreAIDiffusionModelFunction(
                    modelURL: url.appendingPathComponent(path))
                if try await candidate.hasFunction(named: "half") {
                    // Multi-function model: use its "half" function (img2img uses
                    // img2img_512_* from the same asset).
                    transformer = candidate
                    transformerFnName = "half"
                } else if let path512 = try Self.resolveAsset(.transformer512, in: bundle) {
                    transformer = CoreAIDiffusionModelFunction(
                        modelURL: url.appendingPathComponent(path512))
                    transformerFnName = "main"
                } else {
                    // Full-only Transformer without a "half" function and no
                    // Transformer_512 — best effort with "main".
                    transformer = candidate
                    transformerFnName = "main"
                }
            } else if let path512 = try Self.resolveAsset(.transformer512, in: bundle) {
                transformer = CoreAIDiffusionModelFunction(
                    modelURL: url.appendingPathComponent(path512))
                transformerFnName = "main"
            } else {
                throw PipelineLoadError.missingComponent("\(Asset.transformer.stem) or \(Asset.transformer512.stem)")
            }
        case .auto:
            preconditionFailure("auto resolved above")
        }

        // Resolve how each reference grid reaches a traced graph.
        //
        // Prefer the multi-function transformer when available.
        let entrypointPrefix = resolvedMode == .half ? "img2img_512_" : "img2img_"
        let assetPrefix = resolvedMode == .half ? "Transformer_512_img2img" : "Transformer_img2img"
        var img2imgRoutes: [ReferenceGrid: Img2ImgRoute] = [:]
        for grid in ReferenceGrid.allCases {
            let entrypoint = "\(entrypointPrefix)\(grid.rawValue)"
            if try await transformer.hasFunction(named: entrypoint) {
                img2imgRoutes[grid] = Img2ImgRoute(function: transformer, entrypoint: entrypoint)
            } else if let path = try Self.resolveAsset(
                Asset(stem: "\(assetPrefix)_\(grid.rawValue)"), in: bundle)
            {
                // Single-function: its own asset, traced at one concatenated sequence
                // length, always entered through "main".
                img2imgRoutes[grid] = Img2ImgRoute(
                    function: CoreAIDiffusionModelFunction(
                        modelURL: url.appendingPathComponent(path)),
                    entrypoint: "main")
            }
        }

        // Select decoder by mode (explicit name)
        let decoderName: String
        switch resolvedMode {
        case .full:
            guard let path = try Self.resolveAsset(.vaeDecoder, in: bundle) else {
                throw Self.missing(.vaeDecoder, in: bundle)
            }
            decoderName = path
        case .half, .tiled:
            guard let path = try Self.resolveAsset(.vaeDecoderHalf, in: bundle) else {
                throw Self.missing(.vaeDecoderHalf, in: bundle)
            }
            decoderName = path
        case .auto:
            preconditionFailure("auto resolved above")
        }

        // Resolve compiled assets; report a missing text encoder up front.
        let textEncoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(textEncoderPath))
        let decoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(decoderName))

        // Encoder for img2img (optional)
        let encoderName: String?
        switch resolvedMode {
        case .full:
            encoderName = try Self.resolveAsset(.vaeEncoder, in: bundle)
        case .half, .tiled:
            encoderName = try Self.resolveAsset(.vaeEncoderHalf, in: bundle)
        case .auto:
            preconditionFailure("auto resolved above")
        }
        let encoder: CoreAIDiffusionModelFunction?
        if let name = encoderName {
            // Optional component: nil when absent so supportsImageToImage is accurate.
            let encoderURL = url.appendingPathComponent(name)
            encoder =
                FileManager.default.fileExists(atPath: encoderURL.path)
                ? CoreAIDiffusionModelFunction(modelURL: encoderURL)
                : nil
        } else {
            encoder = nil
        }

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

    /// A FLUX.2 asset: its `metadata.json` key and the `<stem>.aimodel` filename the
    /// exporter writes when the bundle doesn't name it. The key defaults to the lowercased stem.
    private struct Asset {
        let key: String
        let stem: String

        init(key: String? = nil, stem: String) {
            self.key = key ?? stem.lowercased()
            self.stem = stem
        }

        static let textEncoder = Asset(key: DiffusionComponentKey.textEncoder, stem: "TextEncoder")
        static let transformer = Asset(key: DiffusionComponentKey.transformer, stem: "Transformer")
        static let transformer512 = Asset(stem: "Transformer_512")
        static let vaeDecoder = Asset(key: DiffusionComponentKey.vaeDecoder, stem: "VAEDecoder")
        static let vaeDecoderHalf = Asset(key: "vae_decoder_half", stem: "VAEDecoder_half")
        static let vaeEncoder = Asset(key: DiffusionComponentKey.vaeEncoder, stem: "VAEEncoder")
        static let vaeEncoderHalf = Asset(key: "vae_encoder_half", stem: "VAEEncoder_half")
    }

    /// The asset's filename when it is on disk, else nil (an absent optional variant).
    ///
    /// A filename declared in `metadata.json` must exist: a missing one throws rather than
    /// falling back to `<stem>.aimodel`. The `<stem>.aimodel` default applies only when the key
    /// isn't declared, as in bundles exported before the manifest listed every component.
    private static func resolveAsset(_ asset: Asset, in bundle: DiffusionBundle) throws -> String? {
        let filename = expectedFilename(asset, in: bundle)
        if FileManager.default.fileExists(atPath: bundle.bundlePath.appendingPathComponent(filename).path) {
            return filename
        }
        if bundle.assetFilename(for: asset.key) != nil { throw missing(asset, in: bundle) }
        return nil
    }

    private static func expectedFilename(_ asset: Asset, in bundle: DiffusionBundle) -> String {
        bundle.assetFilename(for: asset.key, fallback: "\(asset.stem).aimodel")
    }

    /// The error for an asset that isn't on disk, naming the file looked for.
    private static func missing(_ asset: Asset, in bundle: DiffusionBundle) -> PipelineLoadError {
        .missingComponent("\(asset.key) (expected \(expectedFilename(asset, in: bundle)))")
    }

    /// Probe available assets and pick the highest quality mode.
    /// Priority: .full > .tiled > .half. Throws if no valid combination exists.
    private static func bestAvailableMode(bundle: DiffusionBundle) throws -> DecodeResolution {
        let hasFullTransformer = try resolveAsset(.transformer, in: bundle) != nil
        let hasFullDecoder = try resolveAsset(.vaeDecoder, in: bundle) != nil
        let hasHalfDecoder = try resolveAsset(.vaeDecoderHalf, in: bundle) != nil
        let hasHalfTransformer = try resolveAsset(.transformer512, in: bundle) != nil

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
