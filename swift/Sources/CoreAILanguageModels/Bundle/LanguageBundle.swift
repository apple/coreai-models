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
/// - `LanguageBundle(at: url)` — read metadata.json, throws if not LLM
/// - `LanguageBundle(bundle: existing)` — upgrade an inspected `ModelBundle`
///
/// For lossy peeks see `extension ModelBundle { var language: LanguageBundle? }`.
public struct LanguageBundle: Sendable {
    public let bundle: ModelBundle
    public let modelAssetPath: String
    public let language: LanguageConfig
    public let visionConfig: VisionConfig?

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
        self.bundle = bundle
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
    }

    // MARK: - Convenience accessors

    public var name: String { bundle.name }
    public var bundlePath: URL { bundle.bundlePath }
    public var tokenizer: String { language.tokenizer }
    public var vocabSize: Int { language.vocabSize }
    public var maxContextLength: Int { language.maxContextLength }

    /// Raw metadata bytes for passing to engine config parsers.
    public var rawMetadata: Data { bundle.raw }

    // MARK: - Component resolution (forwarded from ModelBundle)

    public var componentKeys: [String] { bundle.componentKeys }

    public func modelURL(for key: String) -> URL? {
        bundle.modelURL(for: key)
    }

    public func requireModelURL(for key: String) throws -> URL {
        try bundle.requireModelURL(for: key)
    }

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

extension LanguageBundle {
    fileprivate struct LanguagePayload: Decodable {
        let assets: Assets
        let language: LanguageConfig?
        let vision: VisionConfig?

        struct Assets: Decodable {
            let main: String?
        }
    }
}
