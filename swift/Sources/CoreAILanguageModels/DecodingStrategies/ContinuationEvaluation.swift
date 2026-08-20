// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

// MARK: - Continuation Encoding

/// Encodes context and continuation for evaluation
/// Uses divergence-point detection to handle tokenization boundary issues
///
/// Key insight: `encode("hello") + encode(" world")` ≠ `encode("hello world")`
/// The continuation may merge with the end of context (e.g., " " + "B" → " B").
/// We find where tokens diverge and consider all tokens from that point as part
/// of the continuation for evaluation purposes.
public struct ContinuationEncoding: Sendable {
    /// Tokens for the context part (tokens that are identical in both encodings)
    public let contextTokens: [Int32]
    /// Tokens for the continuation part (targets for evaluation)
    /// This includes any merged boundary token
    public let continuationTokens: [Int32]
    /// The whole encoding (context + continuation)
    public let tokens: [Int32]
    /// Index where continuation starts in wholeTokens
    public let continuationStartIndex: Int

    /// Initialize continuation encoding
    /// - Parameters:
    ///   - context: The context string (e.g., question + "Answer: ")
    ///   - continuation: The continuation string (e.g., "A")
    ///   - tokenizer: Tokenizer to use for encoding
    public init(context: String, continuation: String, tokenizer: any Tokenizer) {
        // Encode context alone
        let encodedContext = tokenizer.encode(text: context).map { Int32($0) }

        // Encode whole string (context + continuation)
        let encodedTokens = tokenizer.encode(text: context + continuation).map { Int32($0) }

        // Find divergence point - where context-only and whole encoding differ
        // This handles BPE merging at boundaries (e.g., " " + "B" → " B")
        let divergenceIndex =
            (0..<encodedContext.count).first { i in
                encodedTokens[i] != encodedContext[i]
            } ?? encodedContext.count

        // Context tokens are only those that are identical in both encodings
        self.contextTokens = Array(encodedTokens.prefix(divergenceIndex))
        self.continuationTokens = Array(encodedTokens.dropFirst(divergenceIndex))
        self.tokens = encodedTokens
        self.continuationStartIndex = divergenceIndex
    }
}

// MARK: - Continuation Evaluation Result

/// Result of continuation evaluation containing logits for each continuation position
public struct ContinuationEvaluationResult: Sendable {
    /// Tokens for the context part
    public let contextTokens: [Int32]
    /// Tokens for the continuation part (targets)
    public let continuationTokens: [Int32]

    /// Logits for each continuation position [num_continuation_tokens, vocab_size]
    public let logits: [[LogitsScalarType]]

    /// Calculate log probability of the continuation
    public func logProbability() -> Double {
        LogProbabilities.compute(logits: logits, targets: continuationTokens).sum
    }

    /// Calculate average log probability per token
    public func averageLogProbability() -> Double {
        LogProbabilities.compute(logits: logits, targets: continuationTokens).mean
    }

    /// Calculate perplexity of the continuation
    public func perplexity() -> Double {
        LogProbabilities.compute(logits: logits, targets: continuationTokens).perplexity
    }

    /// Get probability of the target token at each position.
    /// Invalid tokens (out-of-bounds) return 0.0.
    public func targetProbabilities() -> [Double] {
        LogProbabilities.compute(logits: logits, targets: continuationTokens)
            .entries.map { $0.value.isFinite ? exp($0.value) : 0.0 }
    }
}

// MARK: - Errors

public enum ContinuationEvaluationError: Error, LocalizedError {
    case requiresDisabledChatTemplate
    case requiresLogitsOutput
    case engineDoesNotSupportLogits
    case emptyContinuation
    case rawTokensNotSupported
    case emptyInput

    public var errorDescription: String? {
        switch self {
        case .requiresDisabledChatTemplate:
            return "--continuation requires --apply-chat-template=false"
        case .requiresLogitsOutput:
            return "--continuation requires --print-logits or --save-logits"
        case .engineDoesNotSupportLogits:
            return "The current inference engine does not support returning logits"
        case .emptyContinuation:
            return "Continuation string cannot be empty"
        case .rawTokensNotSupported:
            return "--continuation requires text prompt (--prompt or --prompt-file), not --raw-tokens"
        case .emptyInput:
            return "Raw token evaluation requires at least 2 tokens"
        }
    }
}
