// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Synchronization

// MARK: - Inference Output

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
public typealias LogitsScalarType = Float16
#else
public typealias LogitsScalarType = Float
#endif

/// Single step output from `InferenceEngine.generate()`.
/// Contains the sampled token and optionally raw logits.
public struct InferenceOutput: Sendable {
    public let tokenId: Int32

    /// Populated when `InferenceOptions.includeLogits` is true. Shape: [vocabSize].
    public let logits: [LogitsScalarType]?

    public init(tokenId: Int32, logits: [LogitsScalarType]? = nil) {
        self.tokenId = tokenId
        self.logits = logits
    }
}

// MARK: - Inference Options

/// Controls what the engine produces and how much.
/// Struct-based for additive extensibility (future: embeddings, attention maps).
public struct InferenceOptions: Sendable {
    /// Max tokens to generate. Nil = until EOS or context limit.
    public var maxTokens: Int?
    /// Include raw logits in each `InferenceOutput`. May incur GPU→CPU copy cost.
    public var includeLogits: Bool
    /// When set, engines use these token IDs instead of sampling.
    /// Used by MMLU-style evaluation to compute P(continuation|context).
    public var forcedContinuation: [Int32]?

    public init(
        maxTokens: Int? = nil,
        includeLogits: Bool = false,
        forcedContinuation: [Int32]? = nil
    ) {
        self.maxTokens = maxTokens
        self.includeLogits = includeLogits
        self.forcedContinuation = forcedContinuation
    }
}

// MARK: - Configuration Data Structures

/// Configuration-specific errors with user-friendly messages
public enum ConfigurationError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidJSON(String, String)
    case decodingError(String, String)
    case validationError(String, String)
    case noValidConfigurations

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Configuration file not found: \(path)"
        case .invalidJSON(let file, let reason):
            return "Invalid JSON in \(file): \(reason)"
        case .decodingError(let file, let reason):
            return "Configuration format error in \(file): \(reason)"
        case .validationError(let file, let reason):
            return "Configuration validation failed in \(file): \(reason)"
        case .noValidConfigurations:
            return "No valid configuration files could be loaded"
        }
    }
}

// MARK: - Core Abstractions

/// Interface for inference engines.
///
/// KV cache is preserved between `generate()` calls. Call `reset()` to clear.
public protocol InferenceEngine: Sendable {
    associatedtype OutputSequence: InferenceOutputSequence
    typealias TokenId = Int32

    // MARK: - Primary API

    /// Stream token generation from the given input.
    ///
    /// - Parameters:
    ///   - input: Token IDs (prompt, context, or continuation).
    ///   - sampling: Sampling configuration (temperature, topK, etc.).
    ///   - generation: Inference options (maxTokens, includeLogits).
    /// - Returns: An `InferenceOutputSequence` — iterate for tokens, read
    ///   `stopReason` after the loop to learn why generation ended.
    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> OutputSequence

    // MARK: - Lifecycle

    /// Number of tokens the engine has processed in the current session.
    /// Resets to 0 on full reset. Used by callers to compute shared prefix length.
    var processedTokenCount: Int { get }

    /// Reset KV cache to the state after processing `tokenIndex` tokens.
    /// - tokenIndex == 0: full reset (clear all state, equivalent to reset())
    /// - tokenIndex > 0: partial reset (keep cache for first tokenIndex positions)
    /// Precondition: tokenIndex >= 0 && tokenIndex <= processedTokenCount
    func reset(to tokenIndex: Int) async throws

    /// Run dummy inference to trigger kernel compilation.
    func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws

    // MARK: - Cancellation

    /// Whether the engine has an active generation in progress.
    var isBusy: Bool { get }

    /// Cancel any in-flight generation. Invalidates the current GenerationToken.
    /// For pull-based engines, takes effect on the next `next()` call.
    /// For push-based engines (pipelined), also cancels the background Task.
    func cancel() async throws

    // MARK: - Capabilities

    /// Whether this engine supports per-step logits extraction.
    /// GPU-pipelined engines (which sample on-device) return false.
    var supportsLogits: Bool { get }

    /// How many tokens were reused from cache on the last `generate()` call.
    ///
    /// After each `generate()`, this reflects how many leading tokens of the input
    /// matched the engine's cached history and were skipped (no recomputation needed).
    /// Useful for debugging multi-turn efficiency and verifying prefix caching behavior.
    var lastPrefixHitCount: Int { get }

    // MARK: - Configuration

    associatedtype ConfigType: Codable, InferenceConfiguration
    var config: ConfigType { get }
}

public protocol InferenceConfiguration: Sendable {
    var maxContextLength: Int { get }

    /// Size for prefill chunks. Override in conforming types if needed.
    var prefillChunkSize: Int { get }

    /// Minimum prompt size to trigger chunked processing.
    /// Prompts smaller than this are processed in a single pass.
    /// Default: 1024 tokens.
    var chunkThreshold: Int { get }
}

extension InferenceConfiguration {
    /// Default prefill chunk size: 512 tokens.
    ///
    /// Trade-off: smaller = less memory but more overhead.
    ///
    /// ## Memory Calculation
    /// Logits buffer = batch × seqLen × vocabSize × sizeof(Float16)
    ///
    /// Example with Qwen3 (vocab_size = 151,936):
    /// - 32K prompt without chunking: 1 × 32,768 × 151,936 × 2 = **9.6 GB**
    /// - 512-token chunk:             1 × 512 × 151,936 × 2 = **155 MB** (98% reduction)
    public var prefillChunkSize: Int { 512 }

    /// Default chunk threshold: 1024 tokens.
    /// Prompts <= 1024 tokens are processed in a single pass.
    public var chunkThreshold: Int { 1024 }
}

// MARK: - Default Implementations

extension InferenceEngine {
    /// Default: supportsLogits is false. Engines that can return per-step
    /// logits (sequential, static-shape) override this to true.
    public var supportsLogits: Bool { false }
}

extension InferenceEngine {
    /// Default: no prefix hits (engine doesn't track history).
    public var lastPrefixHitCount: Int { 0 }
}

extension InferenceEngine {
    /// Default: engine is not busy.
    public var isBusy: Bool { false }

    /// Default: no-op cancel. Engines with active generation override this.
    public func cancel() async throws {}
}

extension InferenceEngine {
    /// Default no-op implementation of warmup.
    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        // No-op by default
    }
}

extension InferenceEngine {
    /// Default: processedTokenCount is 0 (engine hasn't processed anything).
    public var processedTokenCount: Int { 0 }
}

extension InferenceEngine {
    /// Default: reset() delegates to reset(to: 0) for full reset.
    public func reset() async throws {
        try await reset(to: 0)
    }
}

extension InferenceEngine {
    /// Default implementation: accepts all sampling configurations.
    ///
    /// Engines that use CPU fallback samplers can accept all configurations
    /// since CompositeSampler handles greedy, temperature, topK, and topP.
    ///
    /// Override this in engines with GPU samplers that have limited support.
    public func validateSamplingStrategy(_ config: SamplingConfiguration) throws {
        // Default: accept all configurations (CPU fallback handles them)
    }
}

// MARK: - Errors

public enum InferenceRuntimeError: Error, LocalizedError {
    case functionNotFound(String)
    case modelNotFound(String)
    case modelLoadingFailed(underlying: Error)
    case invalidState(String)
    case invalidArgument(String)
    case logitsExtractionFailed
    case invalidInputType(String)
    case invalidOutputType(String)
    case unsupportedLogitsType(String)
    case unsupportedTokenType(String)
    case contextLengthExceeded(Int, Int)  // (current_position, max_context_length)
    case unsupportedEngine(String)
    case unsupportedEngineVariant(String)
    case bufferAllocationFailed(String)
    case genericError(String)

    public var errorDescription: String? {
        switch self {
        case .functionNotFound(let name):
            return "Function '\(name)' not found in compiled program"
        case .modelNotFound(let path):
            return "Model not found at path: \(path)"
        case .modelLoadingFailed(let underlying):
            return "Model loading failed: \(underlying.localizedDescription)"
        case .invalidState(let description):
            return "Invalid engine state: \(description)"
        case .invalidArgument(let description):
            return "Invalid argument: \(description)"
        case .logitsExtractionFailed:
            return "Failed to extract logits from model output"
        case .invalidInputType(let name):
            return "Invalid input type for '\(name)'"
        case .invalidOutputType(let name):
            return "Invalid output type for '\(name)'"
        case .unsupportedLogitsType(let type):
            return "Unsupported logits type: \(type)"
        case .unsupportedTokenType(let type):
            return "Unsupported token type: \(type)"
        case .contextLengthExceeded(let currentPosition, let maxContextLength):
            return
                "Context length exceeded: position \(currentPosition) >= max context length \(maxContextLength). Reduce prompt length or increase max context length."
        case .unsupportedEngine(let type):
            return "Unsupported engine type: \(type)"
        case .unsupportedEngineVariant(let variant):
            return "Unknown variant \(variant)"
        case .bufferAllocationFailed(let details):
            return "GPU buffer allocation failed: \(details)"
        case .genericError(let message):
            return message
        }
    }
}

// MARK: - Engine Options

// MARK: - Multimodal

/// Engine that supports vision/audio input in addition to text tokens.
///
/// The typical flow:
/// 1. `encodeImage(at:)` — preprocess + run vision encoder, return embeddings
/// 2. `generate(with: EmbeddedInput, ...)` — scatter-merge embeddings into
///    token sequence and run prefill + decode
///
/// The caller owns the embeddings and decides caching strategy.
public protocol MultimodalInferenceEngine: InferenceEngine {
    /// Encode an image into embeddings suitable for injection into the LLM.
    /// Returns the embedded representation — caller decides whether to cache.
    func encodeImage(at url: URL) async throws -> EmbeddedInput

    /// Generate tokens from a token sequence with embedded image regions.
    /// The engine scatter-merges `input.imageTokenPositions` with the embedded data
    /// during prefill, then continues standard autoregressive decode.
    func generate(
        with input: EmbeddedInput,
        tokens: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> OutputSequence
}

// TODO: Multi-turn — caller can cache EmbeddedInput across turns and pass it
// again with the accumulated token context. Engine keeps image in KV cache
// via reset(to:) preserving the prefill portion.

// MARK: - Engine Options

/// KV cache memory management strategy.
///
/// Determines how the KV cache is allocated and managed at runtime.
/// This is engine-level configuration applicable to any engine that uses KV caching.
public enum KVCacheStrategy: String, Codable, Sendable, CaseIterable {
    /// Auto-select the best strategy based on model capability.
    /// - For models exported with `--dynamic-sized-kvcache-gpu`: uses `growing`
    /// - For legacy models with fixed seqDim: uses `fixedSize`
    /// This is the recommended default for best memory efficiency with backward compatibility.
    case auto = "auto"

    /// Fixed-size allocation at startup.
    /// Allocates the specified size upfront (defaults to `maxContextLength`).
    /// - Pros: No runtime overhead, predictable memory usage
    /// - Cons: May waste memory for short conversations
    case fixedSize = "fixed_size"

    /// Start small, grow exponentially (2×) as needed up to maxContextLength.
    /// Copies existing KV data to new buffer on growth.
    /// - Pros: Memory efficient for short conversations
    /// - Cons: ~20ms stall on growth (amortized O(log₂ N))
    case growing = "growing"

    /// Fixed size with sliding window (not yet implemented).
    /// Drops oldest tokens when capacity is reached.
    /// - Pros: Bounded memory for infinite contexts
    /// - Cons: Loses early context
    case chunked = "chunked"

    /// Returns the default cache size for this strategy.
    ///
    /// - Parameter maxContextLength: Maximum context length from model config
    /// - Returns: Default size in tokens, or `nil` for `.auto` (resolved at factory level)
    ///   - `.auto`: `nil` (size depends on resolved strategy, determined by model shape)
    ///   - `.fixedSize`: `maxContextLength` (make use of full model capacity)
    ///   - `.growing`: 256 (small initial allocation)
    ///   - `.chunked`: `maxContextLength` (disable chunking)
    public func defaultSize(maxContextLength: Int) -> Int? {
        switch self {
        case .auto: return nil  // Resolved at factory level based on model capability
        case .fixedSize: return maxContextLength
        case .growing: return 256
        case .chunked: return maxContextLength
        }
    }
}
