// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Synchronization

// MARK: - Model Source

/// Model source configuration.
public struct ModelSource: Codable, Sendable {
    public let hfModelId: String?
    public let modelDefinition: ModelDefinition?

    public init(hfModelId: String?, modelDefinition: ModelDefinition? = nil) {
        self.hfModelId = hfModelId
        self.modelDefinition = modelDefinition
    }

    public enum ModelDefinition: String, Codable, Sendable {
        case pyTorch = "torch"
    }

    enum CodingKeys: String, CodingKey {
        case hfModelId = "hf_model_id"
        case modelDefinition = "model_definition"
    }
}

// MARK: - Model Config

/// Unified model configuration.
///
/// ## Required Fields (all engines)
///
/// | JSON key              | Swift property       | Type       | Description                          |
/// |-----------------------|----------------------|------------|--------------------------------------|
/// | `name`                | ``name``             | `String`   | Model display name                   |
/// | `engine`              | ``engine``            | `String`   | One of `coreai`, `static-shape`      |
/// | `tokenizer`           | ``tokenizer``         | `String`   | Tokenizer identifier                 |
/// | `vocab_size`          | ``vocabSize``         | `Int`      | Vocabulary size (must be > 0)        |
/// | `max_context_length`  | ``maxContextLength``  | `Int`      | Maximum context window (must be > 0) |
/// | `source`              | ``source``            | ``ModelSource`` | Model source info (hf_model_id, etc.) |
/// | `serialized_model`    | ``serializedModel``   | `[String]` | Model filenames (.aimodel)           |
/// | `function`            | ``function``          | `String`   | Core AI function entry point            |
///
/// ## Optional Fields (engine-specific)
///
/// | JSON key              | Swift property       | Type         | Default        | Description                         |
/// |-----------------------|----------------------|--------------|----------------|-------------------------------------|
/// | `input_mode`          | ``inputMode``         | ``InputMode``| `nil`          | Input initialization mode           |
/// | `model_definition`    | (on source)           | ``ModelSource/ModelDefinition`` | `.pyTorch` | Model origin framework |
///
/// Use ``resolvedModelDefinition`` for safe access with defaults.
public struct ModelConfig: InferenceConfiguration, Codable, Sendable {
    public let maxContextLength: Int

    var name: String
    var tokenizer: String
    var vocabSize: Int
    var source: ModelSource?
    var serializedModel: [String]
    var function: String

    // Static-shape-specific (optional)
    var inputMode: InputMode?

    // Chunking overrides (not serialized — injected at runtime)
    var prefillChunkSizeOverride: Int?
    var prefillChunkThresholdOverride: Int?

    public enum InputMode: String, Codable, Sendable {
        case random
        case allZeros = "all-zeros"
    }

    public init(
        name: String,
        tokenizer: String,
        vocabSize: Int,
        maxContextLength: Int,
        source: ModelSource? = nil,
        serializedModel: [String],
        function: String,
        inputMode: InputMode? = nil,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil
    ) {
        self.name = name
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.source = source
        self.serializedModel = serializedModel
        self.function = function
        self.inputMode = inputMode
        self.prefillChunkSizeOverride = prefillChunkSize
        self.prefillChunkThresholdOverride = prefillChunkThreshold
    }

    enum CodingKeys: String, CodingKey {
        case name
        case tokenizer
        case vocabSize = "vocab_size"
        case maxContextLength = "max_context_length"
        case source
        case serializedModel = "serialized_model"
        case function
        case inputMode = "input_mode"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
        self.source = try c.decodeIfPresent(ModelSource.self, forKey: .source)
        self.serializedModel = try c.decode([String].self, forKey: .serializedModel)
        self.function = try c.decodeIfPresent(String.self, forKey: .function) ?? "main"
        self.inputMode = try c.decodeIfPresent(InputMode.self, forKey: .inputMode)
        self.prefillChunkSizeOverride = nil
        self.prefillChunkThresholdOverride = nil
    }
}

// MARK: - Convenience accessors

extension ModelConfig {
    /// Resolved model definition — returns the explicit value or `.pyTorch` as default.
    public var resolvedModelDefinition: ModelSource.ModelDefinition {
        source?.modelDefinition ?? .pyTorch
    }
}

// MARK: - Chunking configuration (layered resolution)

extension ModelConfig {
    /// Prefill chunk size with layered resolution:
    /// override → deprecated env var → memory-based default.
    public var prefillChunkSize: Int {
        if let override = prefillChunkSizeOverride, override > 0 {
            return override
        }
        if let value = ProcessInfo.processInfo.environment["COREAI_CHUNK_THRESHOLD"],
            let size = Int(value), size > 0
        {
            Self._emitDeprecationWarning()
            return size
        }
        return defaultPrefillChunkSize()
    }

    /// Chunk threshold with layered resolution:
    /// override → default (2× prefillChunkSize).
    /// When the deprecated env var is active, threshold equals the env var value.
    public var prefillChunkThreshold: Int {
        if let override = prefillChunkThresholdOverride, override > 0 {
            return override
        }
        if ProcessInfo.processInfo.environment["COREAI_CHUNK_THRESHOLD"] != nil {
            Self._emitDeprecationWarning()
            return prefillChunkSize
        }
        return prefillChunkSize * 2
    }

    /// Protocol conformance: engines read this to decide when to chunk.
    public var chunkThreshold: Int { prefillChunkThreshold }

    /// Applies runtime overrides from CLI or FM API.
    public mutating func applyChunkingOverrides(
        prefillChunkSize: Int?,
        prefillChunkThreshold: Int?
    ) {
        self.prefillChunkSizeOverride = prefillChunkSize
        self.prefillChunkThresholdOverride = prefillChunkThreshold
    }

    private static let _deprecationWarned = Mutex(false)

    private static func _emitDeprecationWarning() {
        _deprecationWarned.withLock { warned in
            guard !warned else { return }
            warned = true
            CLILogger.log(
                "COREAI_CHUNK_THRESHOLD is deprecated. "
                    + "Use --chunk-size or metadata.json prefill_chunk_size instead.",
                component: "ModelConfig"
            )
        }
    }
}

/// Accepted serialized-model file extensions.
private let acceptedFileExtensions: [String] = [
    ".aimodel"
]

extension ModelConfig {
    /// Creates a model configuration from raw data.
    public init(parsing data: Data) throws {
        do {
            let decoder = JSONDecoder()
            self = try decoder.decode(ModelConfig.self, from: data)
        } catch DecodingError.keyNotFound(let key, let context) {
            let errorMsg = "Missing required field '\(key.stringValue)' in CoreAI model configuration"
            let location =
                context.codingPath.isEmpty
                ? "" : " at \(context.codingPath.map { $0.stringValue }.joined(separator: " -> "))"
            throw ConfigurationError.decodingError("model config", errorMsg + location)
        } catch DecodingError.typeMismatch(let type, let context) {
            let errorMsg = "Type mismatch - expected \(type) in CoreAI model configuration"
            let location =
                context.codingPath.isEmpty
                ? "" : " at \(context.codingPath.map { $0.stringValue }.joined(separator: " -> "))"
            throw ConfigurationError.decodingError("model config", errorMsg + location)
        } catch {
            throw ConfigurationError.decodingError(
                "model config", "Failed to parse model configuration: \(error.localizedDescription)")
        }
    }

    /// Validates the model configuration.
    public func validate() throws {
        // Validate model name is not empty
        guard !name.isEmpty else {
            throw ConfigurationError.validationError("model config", "Model name cannot be empty")
        }

        // Validate vocab size
        guard vocabSize > 0 else {
            throw ConfigurationError.validationError(
                "model config", "vocab_size must be positive, got \(vocabSize)")
        }

        // Validate max context length
        guard maxContextLength > 0 else {
            throw ConfigurationError.validationError(
                "model config", "max_context_length must be positive, got \(maxContextLength)")
        }

        // Validate source has HF model ID (if source is present)
        if let source = source {
            guard let hfModelId = source.hfModelId, !hfModelId.isEmpty else {
                throw ConfigurationError.validationError(
                    "model config", "source.hf_model_id is required and cannot be empty")
            }
        }

        // Validate serialized model files
        guard !serializedModel.isEmpty else {
            throw ConfigurationError.validationError("model config", "serialized_model array cannot be empty")
        }

        for (index, filename) in serializedModel.enumerated() {
            guard !filename.isEmpty else {
                throw ConfigurationError.validationError("model config", "serialized_model[\(index)] cannot be empty")
            }
        }

        // Validate file extensions
        for (index, filename) in serializedModel.enumerated() {
            guard acceptedFileExtensions.contains(where: { filename.hasSuffix($0) }) else {
                throw ConfigurationError.validationError(
                    "model config",
                    "serialized_model[\(index)] '\(filename)' must have .aimodel extension"
                )
            }
        }

        guard !function.isEmpty else {
            throw ConfigurationError.validationError("model config", "function cannot be empty")
        }
    }
}
