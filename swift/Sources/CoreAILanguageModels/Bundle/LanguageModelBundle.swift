// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// LLM-specific bundle wrapper around `ModelBundle`.
///
/// Combines the common `ModelBundle` fields with `LanguageConfig` (tokenizer,
/// vocab, context, optional `function_map`) and the model asset path.
///
/// Two strict-load constructors:
/// - `LanguageModelBundle(at: url)` — read metadata.json, throws if not LLM
/// - `LanguageModelBundle(bundle: existing)` — upgrade an inspected `ModelBundle`
///
/// For lossy peeks see `extension ModelBundle { var language: LanguageModelBundle? }`.
public struct LanguageModelBundle: Sendable {
    public let modelBundle: ModelBundle
    public let modelAssetPath: String
    public let language: LanguageConfig
    public let visionConfig: VisionConfig?
    /// `speculative.drafter_kind`: how the `drafter` component proposes tokens, nil when the bundle
    /// declares none.
    package let drafterKind: String?

    public init(from path: String) throws {
        let expanded = (path as NSString).expandingTildeInPath
        try self.init(at: URL(fileURLWithPath: expanded, isDirectory: true))
    }

    public init(at url: URL) throws {
        try self.init(bundle: try ModelBundle(at: url))
    }

    public init(bundle: ModelBundle) throws {
        guard bundle.kind == .llm || bundle.kind == .vlm else {
            throw ModelBundle.BundleError.kindMismatch(expected: .llm, got: bundle.kind)
        }
        self.modelBundle = bundle
        let payload = try JSONDecoder().decode(LanguagePayload.self, from: bundle.raw)
        guard let main = payload.assets.main else {
            throw ModelBundle.BundleError.missingField("assets.main")
        }
        guard let language = payload.language else {
            throw ModelBundle.BundleError.missingField("language")
        }
        self.modelAssetPath = main
        self.language = language
        self.visionConfig = payload.vision
        self.drafterKind = payload.speculative?.drafterKind

        if bundle.kind == .vlm && self.visionConfig == nil {
            throw ModelBundle.BundleError.missingField("vision")
        }
    }

    // MARK: - Convenience accessors

    public var name: String { modelBundle.name }
    public var bundlePath: URL { modelBundle.bundlePath }
    public var tokenizer: String { language.tokenizer }
    public var vocabSize: Int { language.vocabSize }
    public var maxContextLength: Int { language.maxContextLength }
    /// Largest batch the graph accepts. 1 for single-sequence graphs.
    package var maxBatchSize: Int { language.maxBatchSize }

    /// Model-specific runtime settings (`language.overrides`); nil for most models.
    public var overrides: LanguageOverrides? { language.overrides }

    /// Tensor data the engine reads alongside the model, keyed by `auxiliary_assets`
    /// role. Only roles the bundle declares are present.
    public var tensorData: [String: URL] {
        var resolved: [String: URL] = [:]
        let key = EngineOptions.TensorDataKey.perLayerEmbeddings
        if let url = modelBundle.auxiliaryAssetURL(for: key) {
            resolved[key] = url
        }
        return resolved
    }

    /// `auxiliary_assets` roles of a draft head's sidecars.
    package enum DrafterAssetKey {
        package static let config = "drafter_config"
        package static let scan = "drafter_scan"
        package static let scanIndex = "drafter_scan_index"
    }

    /// The `speculative.drafter_kind` of a draft head that runs off the target's final hidden state
    /// and shared K/V.
    package static let mtpDrafterKind = "mtp"

    /// The bundle's draft head, when it ships one: the program under `assets.drafter` and its
    /// sidecars under `auxiliary_assets` (``DrafterAssetKey``). Throws when the program is listed
    /// without its sidecars, or when the drafter is of a kind other than ``mtpDrafterKind``, since
    /// other drafters declare the same asset key with different semantics.
    package var drafter: DrafterComponent? {
        get throws {
            guard let step = modelBundle.modelURL(for: ModelBundle.ComponentKey.drafter) else { return nil }
            guard drafterKind == Self.mtpDrafterKind else {
                throw InferenceRuntimeError.invalidState(
                    "the bundle's drafter is of kind \(drafterKind.map { "'\($0)'" } ?? "unspecified"), "
                        + "not '\(Self.mtpDrafterKind)'; this runner reads hidden-state draft heads only")
            }
            func sidecar(_ key: String) throws -> URL {
                guard let url = modelBundle.auxiliaryAssetURL(for: key) else {
                    throw ModelBundle.BundleError.missingField("auxiliary_assets.\(key)")
                }
                return url
            }
            return DrafterComponent(
                step: step, config: try sidecar(DrafterAssetKey.config), scan: try sidecar(DrafterAssetKey.scan),
                scanIndex: try sidecar(DrafterAssetKey.scanIndex))
        }
    }

    /// Whether the decode graphs emit the hidden states a draft head seeds from.
    public var emitsHiddenStates: Bool { overrides?.emitsHiddenStates ?? false }

    /// Raw metadata bytes for passing to engine config parsers.
    public var rawMetadata: Data { modelBundle.raw }

    // MARK: - Tokenizer

    /// Path to the embedded tokenizer directory, if present.
    public var tokenizerPath: URL? {
        guard language.embeddedTokenizer else { return nil }
        let dir = bundlePath.appending(path: "tokenizer")
        let json = dir.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: json.path) else { return nil }
        return dir
    }

    public var hasEmbeddedTokenizer: Bool { tokenizerPath != nil }

    /// Load tokenizer from bundle (embedded preferred, HuggingFace fallback).
    public func loadTokenizer() async throws -> any Tokenizer {
        if let path = tokenizerPath {
            return try await AutoTokenizer.from(modelFolder: path)
        }
        return try await AutoTokenizer.from(pretrained: language.tokenizer)
    }
}

// MARK: - 0.2 payload shape

extension LanguageModelBundle {
    fileprivate struct LanguagePayload: Decodable {
        let assets: Assets
        let language: LanguageConfig?
        let vision: VisionConfig?
        let speculative: Speculative?

        struct Assets: Decodable {
            let main: String?
        }

        struct Speculative: Decodable {
            let drafterKind: String?

            enum CodingKeys: String, CodingKey { case drafterKind = "drafter_kind" }
        }
    }
}
