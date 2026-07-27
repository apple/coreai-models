// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// A factory that creates inference engines from model configurations and assets.
///
/// Use `EngineFactory` to instantiate an `InferenceEngine` from a model bundle on disk. The
/// factory inspects the model's structure and selects a compatible engine variant — sequential,
/// pipelined, static-shape, or vision-language.
///
/// Call ``createEngine(config:modelURL:options:)`` to create the engine, and pass an
/// ``EngineOptions`` value to override the variant or customize the KV cache.
public struct EngineFactory: Sendable {
    /// Creates an inference engine for a model, selecting the variant from the model's structure.
    ///
    /// The factory inspects the configuration to detect vision-language bundles, prepares the
    /// model asset, and instantiates the appropriate engine. When `options.variant` is `nil`, the
    /// factory auto-detects the best variant. Otherwise it validates the override against the
    /// model and throws when the combination isn't supported.
    ///
    /// - Parameters:
    ///   - config: The JSON model configuration as raw data.
    ///   - modelURL: The location of the model asset on disk.
    ///   - options: The engine options, including an optional variant override and KV cache
    ///     settings. Defaults to a value with auto-detection enabled and the `.auto` KV cache
    ///     strategy.
    /// - Returns: A configured inference engine.
    public static func createEngine(
        config: Data,
        modelURL: URL,
        options: EngineOptions = EngineOptions()
    ) async throws -> any InferenceEngine {
        // Step 1: Parse config
        let parsedConfig = try parseUnifiedConfig(from: config)

        CLILogger.log("EngineFactory: Creating engine for \(parsedConfig.name)")
        CLILogger.log("  - variant: \(options.variant ?? "auto")")
        CLILogger.log("  - kvCacheStrategy: \(options.kvCacheStrategy.rawValue)")
        if let size = options.kvCacheSize {
            CLILogger.log("  - kvCacheSize: \(size) (override)")
        }

        // Step 2: Resolve model URL for Core AI path
        let coreAIModelURL = PreparedModel.resolveCoreAIModelURL(from: modelURL)

        // Step 3: Prepare model asset via Core AI
        let preparedModel = try await PreparedModel.prepare(at: coreAIModelURL)

        CLILogger.log("  - structure: \(preparedModel.structure.description)")

        // Step 4: Resolve variant with structure detection
        let variant = try resolveVariant(
            override: options.variant,
            detectedStructure: preparedModel.structure
        )

        CLILogger.log("  - resolved variant: \(variant.rawValue)")

        // Step 5: Instantiate the appropriate engine
        return try await selectEngine(
            variant: variant,
            config: parsedConfig,
            preparedModel: preparedModel,
            modelURL: modelURL,
            options: options
        )
    }

    // MARK: - Config Parsing

    /// Parsed config container for engine selection.
    private struct UnifiedConfig {
        let name: String
        let vocabSize: Int
        let maxContextLength: Int
        let tokenizer: String
        let function: String
        let modelDefinition: ModelSource.ModelDefinition
    }

    /// Parses config data using the unified config handler.
    private static func parseUnifiedConfig(from data: Data) throws -> UnifiedConfig {
        let config = try ModelConfig(parsing: data)
        return UnifiedConfig(
            name: config.name,
            vocabSize: config.vocabSize,
            maxContextLength: config.maxContextLength,
            tokenizer: config.tokenizer,
            function: config.function,
            modelDefinition: config.resolvedModelDefinition
        )
    }

    // MARK: - Variant Resolution

    /// Resolves the engine variant based on override or auto-detection.
    private static func resolveVariant(
        override: String?,
        detectedStructure: ModelStructure
    ) throws -> Variant {
        // Handle nil, "auto", or "default" as auto-detection
        guard let overrideStr = override,
            overrideStr != "auto",
            overrideStr != "default"
        else {
            return autoDetectVariant(structure: detectedStructure)
        }

        // Parse override string to variant
        guard let variant = Variant(rawValue: overrideStr) else {
            throw InferenceRuntimeError.unsupportedEngineVariant(
                "Unknown variant '\(overrideStr)'. Valid: auto, coreai-sequential, coreai-pipelined, static-shape"
            )
        }

        // Validate compatibility and warn if mismatch
        let (isCompatible, warning) = checkVariantCompatibility(
            variant: variant,
            structure: detectedStructure
        )

        if let warning = warning {
            CLILogger.log("⚠️ Warning: \(warning)")
        }

        if !isCompatible {
            throw InferenceRuntimeError.unsupportedEngineVariant(
                "Variant '\(overrideStr)' incompatible with model structure"
            )
        }

        return variant
    }

    /// Auto-detects the optimal variant based on model structure.
    private static func autoDetectVariant(
        structure: ModelStructure
    ) -> Variant {
        switch structure {
        case .chunkedStatic:
            return .staticShape
        case .dynamic:
            return .pipelined
        default:
            // EngineFactory drives LLM engines only
            preconditionFailure(
                "EngineFactory only supports chunkedStatic and dynamic model structures."
            )
        }
    }

    /// Checks if a variant override is compatible with the model structure.
    private static func checkVariantCompatibility(
        variant: Variant,
        structure: ModelStructure
    ) -> (compatible: Bool, warning: String?) {
        switch (variant, structure) {
        case (.staticShape, .dynamic):
            return (false, "Static-shape variant requires chunked static model (extend_* functions)")
        case (.pipelined, .chunkedStatic):
            return (false, "Core AI pipelined variant requires dynamic model")
        case (.sequential, .chunkedStatic):
            return (false, "Sequential variant requires dynamic model")
        case (_, .dynamic), (_, .chunkedStatic):
            return (true, nil)
        default:
            // Any other structure isn't an LLM model (e.g. a segmenter asset), so no LLM
            // engine variant can run it.
            return (false, "LLM engine variants are incompatible with this model structure")
        }
    }

    // MARK: - Engine Selection

    /// Instantiates the appropriate engine based on resolved variant.
    ///
    /// Uses `preparedModel` directly for pipelined/sequential engines to avoid
    /// redundant asset loading.
    private static func selectEngine(
        variant: Variant,
        config: UnifiedConfig,
        preparedModel: PreparedModel,
        modelURL: URL,
        options: EngineOptions
    ) async throws -> any InferenceEngine {
        let modelConfig = ModelConfig(
            name: config.name,
            tokenizer: config.tokenizer,
            vocabSize: config.vocabSize,
            maxContextLength: config.maxContextLength,
            source: ModelSource(
                hfModelId: config.name,
                modelDefinition: config.modelDefinition
            ),
            serializedModel: [modelURL.lastPathComponent],
            function: config.function
        )

        switch variant {
        case .staticShape:
            CLILogger.log("Creating static-shape engine")
            return try await StaticShapeEngine(
                configuration: modelConfig,
                preparedModel: preparedModel
            )

        case .sequential:
            CLILogger.log("Creating CoreAI sequential engine (clean, public API)")
            return try await CoreAISequentialEngine(
                config: modelConfig,
                preparedModel: preparedModel,
                options: options
            )

        case .pipelined:
            CLILogger.log("Creating CoreAI pipelined engine (GPU)")
            return try await CoreAIPipelinedEngine(
                config: modelConfig,
                preparedModel: preparedModel,
                options: options
            )
        }
    }
}

/// Options that customize how the factory creates an inference engine and how
/// the engine manages its KV cache.
public struct EngineOptions: Sendable {
    /// A specific engine variant name that overrides auto-detection.
    ///
    /// When `nil`, the factory selects a variant from the model's structure. Set this to
    /// `"coreai-sequential"`, `"coreai-pipelined"`, or `"static-shape"` to bypass auto-detection.
    public let variant: String?

    /// The strategy that determines how the engine sizes the KV cache.
    ///
    /// The default value, `.auto`, balances memory and throughput. It uses a 256-token initial
    /// cache for dynamic-shape models and the full context length for chunked-static models.
    ///
    /// - Important: Avoid `.fixedSize` unless you need a known upper bound. It pre-allocates the
    ///   cache at the full `maxContextLength`, which can consume several gigabytes on long-context
    ///   models and slows each decoding step because every iteration operates on the full-size KV.
    public let kvCacheStrategy: KVCacheStrategy

    /// The KV cache size in tokens that overrides the strategy's default.
    ///
    /// When `nil`, the engine uses `kvCacheStrategy.defaultSize(maxContextLength:)`. The meaning
    /// of this value depends on `kvCacheStrategy`:
    ///
    /// - `.fixedSize`: the total capacity, capped at `maxContextLength`.
    /// - `.growing`: the initial capacity, which grows up to `maxContextLength`.
    /// - `.chunked`: the window size.
    public let kvCacheSize: Int?

    /// Creates an options value with the variant and KV cache settings you specify.
    ///
    /// - Parameters:
    ///   - variant: The name of the engine variant to use, or `nil` to let the factory auto-detect
    ///     from the model. Defaults to `nil`.
    ///   - kvCacheStrategy: The KV cache allocation strategy. Defaults to `.auto`.
    ///   - kvCacheSize: The KV cache size in tokens, or `nil` to use the strategy's default size.
    ///     Defaults to `nil`.
    public init(
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        kvCacheSize: Int? = nil
    ) {
        self.variant = variant
        self.kvCacheStrategy = kvCacheStrategy
        self.kvCacheSize = kvCacheSize
    }

    /// Returns the KV cache size in tokens that the engine uses for a given context length.
    ///
    /// When you set `kvCacheSize`, this method returns that value. Otherwise it returns the size
    /// that `kvCacheStrategy` provides for the context length.
    ///
    /// - Parameter maxContextLength: The maximum context length in tokens, from the model
    ///   configuration.
    /// - Returns: The KV cache size in tokens, or `nil` when the strategy is `.auto` and the
    ///   factory resolves the size at engine creation.
    public func resolvedKVCacheSize(maxContextLength: Int) -> Int? {
        if let explicitSize = kvCacheSize {
            return explicitSize
        }
        return kvCacheStrategy.defaultSize(maxContextLength: maxContextLength)
    }
}

extension EngineFactory {
    /// Determines the appropriate engine variant based on model structure.
    private enum Variant: String, Sendable, CaseIterable {
        /// Core AI sequential engine (clean public API rewrite)
        case sequential = "coreai-sequential"

        /// Core AI pipelined engine (GPU)
        case pipelined = "coreai-pipelined"

        /// Static-shape engine (chunked static, Neural Engine)
        case staticShape = "static-shape"
    }
}
