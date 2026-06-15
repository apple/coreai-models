// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

// MARK: - Generation Result

/// Decoded text with optional token ID and logits.
public struct GenerationResult: Sendable {
    public let text: String
    public let tokenId: Int32

    public let rawLogits: [LogitsScalarType]?

    public init(text: String, tokenId: Int32, rawLogits: [LogitsScalarType]?) {
        self.text = text
        self.tokenId = tokenId
        self.rawLogits = rawLogits
    }
}

// MARK: - Stop Sequences

/// Represents a collection of stop token sequences for halting text generation
///
/// This struct supports both single-token and multi-token stop sequences using
/// a unified matching algorithm based on sliding window comparison.
///
/// The recommended way to create stop sequences is using the `init(for:additionalSequences:)`
/// initializer, which automatically includes the tokenizer's EOS token, along with any custom sequences you specify.
///
/// Example with tokenizer (recommended):
/// ```swift
/// // Automatically includes tokenizer EOS + common EOS tokens + custom sequences
/// let stopSequences = StopSequences(
///     for: tokenizer,
///     additionalSequences: [[456, 789]]  // Optional custom sequences
/// )
/// ```
///
/// Example with manual sequences:
/// ```swift
/// let sequences = StopSequences(sequences: [
///     [123],        // Single-token sequence
///     [456, 789]    // Multi-token sequence
/// ])
///
/// var recentTokens: [Int32] = []
/// for token in generatedTokens {
///     recentTokens.append(token)
///     if recentTokens.count > sequences.maxLength {
///         recentTokens.removeFirst()
///     }
///     if sequences.matches(recentTokens: recentTokens) {
///         break  // Stop generation
///     }
/// }
/// ```
public struct StopSequences: Sendable {
    /// All stop sequences (each is an array of token IDs)
    public let sequences: [[Int32]]

    /// Maximum length of any sequence (used for sliding window buffer size)
    public let maxLength: Int

    /// Initialize with token sequences
    /// - Parameter sequences: Array of token ID sequences
    public init(sequences: [[Int32]]) {
        self.sequences = sequences
        self.maxLength = sequences.map { $0.count }.max() ?? 0
    }

    /// Initialize with tokenizer, automatically including EOS tokens
    /// - Parameter tokenizer: Tokenizer to extract EOS token from
    /// - Parameter additionalSequences: Optional additional stop sequences to include
    /// - Parameter additionalEosTokenIds: Optional additional single-token EOS IDs
    ///   (e.g. from tokenizer_config.json's `additional_special_tokens`)
    public init(
        for tokenizer: any Tokenizer,
        additionalSequences: [[Int32]] = [],
        additionalEosTokenIds: [Int32] = []
    ) {
        var allSequences = additionalSequences

        // Collect existing single-token sequences to avoid duplicates
        var existingTokens = Set<Int32>()
        for seq in additionalSequences where seq.count == 1 {
            existingTokens.insert(seq[0])
        }

        // Add tokenizer's EOS token if available and not already present
        if let eosTokenId = tokenizer.eosTokenId {
            let token = Int32(eosTokenId)
            if !existingTokens.contains(token) {
                existingTokens.insert(token)
                allSequences.append([token])
            }
        }

        // Add additional EOS token IDs (e.g. from tokenizer_config.json)
        for token in additionalEosTokenIds {
            if !existingTokens.contains(token) {
                existingTokens.insert(token)
                allSequences.append([token])
            }
        }

        self.sequences = allSequences
        self.maxLength = allSequences.map { $0.count }.max() ?? 0
    }

    /// Check if recent tokens end with any stop sequence
    ///
    /// Uses suffix matching: returns true if the end of recentTokens matches
    /// the complete sequence of any stop sequence.
    ///
    /// - Parameter recentTokens: Buffer of recently generated tokens
    /// - Returns: true if any sequence matches the end of the buffer
    public func matches(recentTokens: [Int32]) -> Bool {
        matchedSequence(recentTokens: recentTokens) != nil
    }

    /// Return the stop sequence whose tokens match the end of `recentTokens`.
    ///
    /// Same suffix-matching logic as `matches(recentTokens:)`, but returns the
    /// matched sequence so callers can report *why* generation stopped (e.g. to
    /// decode the sequence into a `StopReason.stopSequence` string).
    ///
    /// - Parameter recentTokens: Buffer of recently generated tokens
    /// - Returns: The matched sequence, or nil if none matched.
    public func matchedSequence(recentTokens: [Int32]) -> [Int32]? {
        for sequence in sequences {
            if recentTokens.suffix(sequence.count).elementsEqual(sequence) {
                return sequence
            }
        }
        return nil
    }

    /// Check if empty (no stop sequences)
    public var isEmpty: Bool {
        sequences.isEmpty
    }

    /// Number of stop sequences
    public var count: Int {
        sequences.count
    }
}

// MARK: - Decoding Strategy Protocol

/// Decoding strategies produce text + optional enrichments (logits, token IDs)
/// from an inference engine.
public protocol DecodingStrategy: Sendable {
    associatedtype ResultSequence: AsyncSequence<GenerationResult, Error>

    /// Stream decoded text with optional logits.
    ///
    /// Performs any required setup (session creation, tokenization, engine reset) eagerly before returning, so the
    /// returned sequence is ready to iterate.
    ///
    /// - Parameters:
    ///   - input: Input specification (raw text, prompt, or pre-tokenized)
    ///   - tokenizer: Tokenizer for encoding/decoding
    ///   - inferenceEngine: Engine for model inference
    ///   - samplingConfiguration: Sampling parameters (temperature, topK, etc.)
    ///   - options: Inference options (maxTokens, includeLogits)
    ///   - stopSequences: Token sequences that halt generation
    /// - Returns: Stream of `GenerationResult` (text + optional logits)
    func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) async throws -> ResultSequence
}

// MARK: - Decoding Strategy Factory

/// Factory for creating decoding strategies
public struct DecodingStrategyFactory {
    /// Creates a decoding strategy of the specified type
    /// - Parameters:
    ///   - type: The type of decoding strategy to create
    ///   - parameters: Optional parameters for configuring the strategy
    /// - Returns: A configured decoding strategy instance
    public static func create(type: DecodingType, parameters: DecodingParameters = DecodingParameters())
        -> any DecodingStrategy
    {
        switch type {
        case .vanilla:
            return VanillaDecodingStrategy()
        }
    }
}

/// Enumeration of available decoding strategy types
public enum DecodingType {
    /// Standard vanilla decoding strategy (text-only)
    case vanilla
}

/// Parameters for configuring decoding strategies
public struct DecodingParameters: Sendable {
    /// Initializes decoding parameters with default values
    public init() {
        // No parameters needed for vanilla decoding
    }
}
