// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import FoundationModels
import Tokenizers

/// Unified Core AI runner that creates FM API-compatible LanguageModel instances.
///
/// ## Usage
/// ```swift
/// let url = URL(fileURLWithPath: "/path/to/model")
/// let runner = try CoreAIRunner(contentsOf: url)
/// let engine = try await runner.makeInferenceEngine()
/// ```
public struct CoreAIRunner {
    // MARK: - Properties

    private let bundle: LanguageModelBundle
    private let engineVariant: String?
    private let kvCacheStrategy: KVCacheStrategy
    private let prefillChunkSizeOverride: Int?
    private let prefillChunkThresholdOverride: Int?

    // MARK: - Initialization

    /// Creates a runner by loading a model bundle from a URL.
    public init(
        contentsOf url: URL,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil
    ) throws {
        self.init(
            bundle: try LanguageModelBundle(at: url),
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            prefillChunkSize: prefillChunkSize,
            prefillChunkThreshold: prefillChunkThreshold
        )
    }

    /// Creates a runner from a LanguageModelBundle.
    public init(
        bundle: LanguageModelBundle,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil
    ) {
        self.bundle = bundle
        self.engineVariant = variant
        self.kvCacheStrategy = kvCacheStrategy
        self.prefillChunkSizeOverride = prefillChunkSize
        self.prefillChunkThresholdOverride = prefillChunkThreshold
    }

    // MARK: - Engine Creation

    /// Creates an inference engine using auto-detection.
    public func makeInferenceEngine() async throws -> any InferenceEngine {
        try bundle.modelBundle.validateModelAssets()

        let config = makeConfig()

        // A VLM whose decoder gathers its own token embeddings (`vision.token_merge` is
        // `decoder_gather`, e.g. Gemma 4): the factory builds the multimodal StaticShapeEngine so
        // `encodeImage` + soft-token scatter work. VLMs with a separate `embedding` asset are
        // constructed elsewhere (`CoreAIVisionLanguageModel`), not through this runner.
        if bundle.modelBundle.kind == .vlm, bundle.visionConfig?.tokenMerge == .decoderGather {
            return try await EngineFactory.createEngine(
                bundle: bundle,
                options: EngineOptions(
                    variant: engineVariant,
                    kvCacheStrategy: kvCacheStrategy,
                    prefillChunkSize: prefillChunkSizeOverride ?? bundle.language.prefillChunkSize,
                    prefillChunkThreshold: prefillChunkThresholdOverride
                        ?? bundle.language.prefillChunkThreshold,
                    tensorData: bundle.tensorData))
        }

        let configData = try JSONEncoder().encode(config)

        let resolvedChunkSize = prefillChunkSizeOverride ?? bundle.language.prefillChunkSize
        let resolvedThreshold = prefillChunkThresholdOverride ?? bundle.language.prefillChunkThreshold

        let options = EngineOptions(
            variant: engineVariant,
            kvCacheStrategy: kvCacheStrategy,
            prefillChunkSize: resolvedChunkSize,
            prefillChunkThreshold: resolvedThreshold,
            tensorData: bundle.tensorData
        )

        return try await EngineFactory.createEngine(
            config: configData,
            modelURL: try bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main),
            options: options
        )
    }

    // MARK: - Private Helpers

    private func makeConfig() -> ModelConfig {
        ModelConfig(
            bundle: bundle,
            source: ModelSource(hfModelId: bundle.tokenizer, modelDefinition: .pyTorch))
    }
}
