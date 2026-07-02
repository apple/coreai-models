// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

// MARK: - Text Generator

/// Main orchestrator that coordinates inference engine, sampling strategy, and decoding strategy
public class TextGenerator {
    private let inferenceEngine: any InferenceEngine
    private let samplingConfiguration: SamplingConfiguration
    private let decodingStrategy: any DecodingStrategy
    private let tokenizer: any Tokenizer

    public init(
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        decodingStrategy: any DecodingStrategy,
        tokenizer: any Tokenizer
    ) {
        self.inferenceEngine = inferenceEngine
        self.samplingConfiguration = samplingConfiguration
        self.decodingStrategy = decodingStrategy
        self.tokenizer = tokenizer
    }

    /// Generate text using the configured strategies
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - stopSequences: Stop token sequences that halt generation. If nil, uses tokenizer's EOS tokens and common fallbacks.
    /// - Returns: Generated text
    public func generate(
        input: Input,
        maxTokens: Int = 50,
        stopSequences: StopSequences? = nil
    ) async throws -> String {
        var result: [String] = []

        // Use provided stop sequences, or create default ones from tokenizer
        let effectiveStopSequences = stopSequences ?? StopSequences(for: tokenizer)

        let tokenStream = try await decodingStrategy.decode(
            from: input,
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            options: InferenceOptions(maxTokens: maxTokens),
            stopSequences: effectiveStopSequences
        )

        for try await generationResult in tokenStream {
            result.append(generationResult.text)
        }

        return result.joined()
    }

    /// Generate text with logits using the configured strategies
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - stopSequences: Stop token sequences that halt generation. If nil, uses tokenizer's EOS tokens and common fallbacks.
    /// - Returns: Tuple containing generated text and array of logits for each token
    public func generateWithLogits(
        input: Input,
        maxTokens: Int = 50,
        stopSequences: StopSequences? = nil
    ) async throws -> (text: String, logits: [[LogitsScalarType]]) {
        var textParts: [String] = []
        var allLogits: [[LogitsScalarType]] = []

        // Use provided stop sequences, or create default ones from tokenizer
        let effectiveStopSequences = stopSequences ?? StopSequences(for: tokenizer)

        let resultStream = try await decodingStrategy.decode(
            from: input,
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            options: InferenceOptions(maxTokens: maxTokens, includeLogits: true),
            stopSequences: effectiveStopSequences
        )

        for try await result in resultStream {
            textParts.append(result.text)
            if let logits = result.rawLogits {
                allLogits.append(logits)
            }
        }

        return (text: textParts.joined(), logits: allLogits)
    }

    /// Evaluate continuation probability (no generation)
    /// Returns logits for each continuation token position
    ///
    /// This method runs inference on (context + continuation[:-1]) and extracts
    /// logits for the continuation token positions. Use this for MMLU-style
    /// evaluation where you compute P(continuation|context).
    ///
    /// - Parameters:
    ///   - context: The context string
    ///   - continuation: The continuation string to evaluate
    /// - Returns: ContinuationEvaluationResult with logits and helper methods
    public func evaluateContinuation(
        context: String,
        continuation: String
    ) async throws -> ContinuationEvaluationResult {
        let encoding = ContinuationEncoding(
            context: context,
            continuation: continuation,
            tokenizer: tokenizer
        )

        CLILogger.log(
            "Evaluation mode: context=\(encoding.contextTokens.count) tokens, "
                + "continuation=\(encoding.continuationTokens.count) tokens, "
                + "divergence at index \(encoding.continuationStartIndex)",
            component: "TextGenerator"
        )

        try await inferenceEngine.reset()

        // Feed context + continuation tokens via generate() with forcedContinuation.
        // The engine runs forward passes and returns logits at each position.
        let contextTokens = Array(encoding.tokens[0..<encoding.continuationStartIndex])
        let continuationTokens = Array(
            encoding.tokens[encoding.continuationStartIndex..<encoding.tokens.count])

        let options = InferenceOptions(
            maxTokens: continuationTokens.count,
            includeLogits: true,
            forcedContinuation: continuationTokens
        )

        let stream = try await inferenceEngine.generate(
            with: contextTokens,
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: options
        )

        var allLogits: [[LogitsScalarType]] = []
        for try await output in stream {
            if let logits = output.logits {
                allLogits.append(logits)
            } else {
                CLILogger.log(
                    "Warning: No logits returned for position \(allLogits.count)",
                    component: "TextGenerator"
                )
            }
        }

        try await inferenceEngine.reset()

        return ContinuationEvaluationResult(
            contextTokens: encoding.contextTokens,
            continuationTokens: encoding.continuationTokens,
            logits: allLogits
        )
    }
}

// MARK: - Text Generator Builder

/// Builder pattern for creating TextGenerator instances with different configurations
public class TextGeneratorBuilder {
    private var inferenceEngine: (any InferenceEngine)?
    private var samplingConfiguration: SamplingConfiguration = .greedy
    private var decodingType: DecodingType = .vanilla
    private var decodingParameters: DecodingParameters = DecodingParameters()
    private var tokenizer: (any Tokenizer)?

    public init() {}

    /// Set a preloaded inference engine
    public func withInferenceEngine(_ engine: any InferenceEngine) -> TextGeneratorBuilder {
        self.inferenceEngine = engine
        return self
    }

    /// Set the sampling strategy
    public func withSampling(
        configuration: SamplingConfiguration = SamplingConfiguration.greedy
    ) -> TextGeneratorBuilder {
        self.samplingConfiguration = configuration
        return self
    }

    /// Set the decoding strategy
    public func withDecoding(
        type: DecodingType,
        parameters: DecodingParameters = DecodingParameters()
    ) -> TextGeneratorBuilder {
        self.decodingType = type
        self.decodingParameters = parameters
        return self
    }

    /// Set the tokenizer
    public func withTokenizer(_ tokenizer: any Tokenizer) -> TextGeneratorBuilder {
        self.tokenizer = tokenizer
        return self
    }

    /// Build the TextGenerator instance
    public func build() async throws -> TextGenerator {
        guard let tokenizer = tokenizer else {
            throw TextGeneratorError.missingTokenizer
        }

        guard let inferenceEngine = inferenceEngine else {
            throw TextGeneratorError.invalidConfiguration("Inference engine must be provided")
        }

        // Model loading profiling is handled by UniversalModelLoader
        // Just log memory usage here
        InstrumentsProfiler.logMemoryUsage(phase: "TextGenerator Build")

        // Create decoding strategy
        let decodingStrategy = DecodingStrategyFactory.create(
            type: decodingType,
            parameters: decodingParameters
        )

        return TextGenerator(
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            decodingStrategy: decodingStrategy,
            tokenizer: tokenizer
        )
    }
}

// MARK: - Configuration Presets

/// Predefined configurations for common use cases
public struct TextGeneratorPresets {
    /// Fast generation with greedy sampling
    public static func fastGeneration() -> (SamplingConfiguration, DecodingType, DecodingParameters) {
        return (
            .greedy,
            .vanilla,
            DecodingParameters()
        )
    }

    /// Creative generation with temperature sampling
    public static func creativeGeneration(temperature: Double = 0.8) -> (
        SamplingConfiguration, DecodingType, DecodingParameters
    ) {
        return (
            SamplingConfiguration(temperature: temperature),
            .vanilla,
            DecodingParameters()
        )
    }
}

// MARK: - LLM Input Specification

/// Represents different types of input that can be provided to a language model
///
/// Use this enum to specify whether input text should be processed as raw text
/// or formatted as a prompt with template application.
public enum Input: Sendable {
    /// Raw text input without any template formatting
    /// - Parameter String: The unformatted text to process
    case rawText(String)

    /// Prompt input that may be formatted using a chat template
    /// - Parameter String: The prompt text to be formatted
    case prompt(String)

    /// Pre-tokenized input - bypasses tokenization entirely
    /// - Parameter [Int]: Array of token IDs
    case tokens([Int])
}

// MARK: - Prompt Utilities

/// Utility functions for prompt formatting
public struct PromptUtils {
    /// Apply chat template using tokenizer's built-in functionality
    /// This method tries to use the tokenizer's applyChatTemplate method, falling back to direct encoding
    public static func maybeApplyTokenizerChatTemplate(_ input: Input, tokenizer: any Tokenizer) throws
        -> [Int]
    {
        switch input {
        case .rawText(let text):
            return tokenizer.encode(text: text)
        case .prompt(let prompt):
            // Try to use tokenizer's applyChatTemplate for proper chat formatting
            let messages = [["role": "user", "content": prompt]]
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)

            CLILogger.log(
                "Applied chat template using tokenizer.applyChatTemplate()", component: "Tokenizer")

            return promptTokens
        case .tokens(let tokenIds):
            // Already tokenized - return as-is
            return tokenIds
        }
    }
}

// MARK: - Errors

public enum TextGeneratorError: Error, LocalizedError {
    case missingTokenizer
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .missingTokenizer:
            return "Tokenizer is required"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        }
    }
}

// MARK: - Convenience Extensions

extension TextGenerator {
    /// Quick generation with default parameters
    public func quickGenerate(_ input: Input) async throws -> String {
        return try await generate(input: input, maxTokens: 50)
    }
}
