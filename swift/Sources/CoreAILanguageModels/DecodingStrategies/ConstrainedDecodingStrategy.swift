// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Grammar-constrained decoding strategy using xgrammar.
///
/// Conforms to `DecodingStrategy` so it can be used anywhere a decoding strategy is expected.
/// Streams `GenerationResult` tokens as they are generated, enabling incremental text display.
///
/// Uses xgrammar bitmask enforcement to ensure generated text conforms to a JSON schema.
/// Each step: (1) run one inference step to get logits, (2) apply the grammar bitmask
/// to zero out tokens that would violate the JSON schema, (3) sample from the masked
/// logits, (4) accept the token in the grammar matcher to advance the grammar state.
public struct ConstrainedDecodingStrategy: DecodingStrategy {
    /// The JSON schema that constrains generation output.
    private let jsonSchema: String

    /// Vocabulary size override. If nil, derived from tokenizer at generation time.
    private let vocabSizeOverride: Int?

    /// Initialize with a JSON schema string.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string that constrains output structure
    ///   - vocabSize: Optional vocabulary size override. If nil, derived from tokenizer.
    public init(jsonSchema: String, vocabSize: Int? = nil) {
        self.jsonSchema = jsonSchema
        self.vocabSizeOverride = vocabSize
    }

    // MARK: - DecodingStrategy conformance

    public func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) async throws -> ConstrainedDecodedSequence {
        CLILogger.log(
            "Starting constrained decoding strategy with schema",
            component: "ConstrainedDecodingStrategy"
        )

        // Eager setup
        let session = try Self.createSession(
            jsonSchema: jsonSchema,
            vocabSizeOverride: vocabSizeOverride,
            tokenizer: tokenizer,
            stopSequences: stopSequences
        )
        let inputTokens =
            try PromptUtils
            .maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        let maxTokens = options.maxTokens ?? 512

        try await inferenceEngine.reset()

        return ConstrainedDecodedSequence(
            prepared: ConstrainedDecodedSequence.Prepared(
                session: consume session,
                inputTokens: inputTokens,
                maxTokens: maxTokens
            ),
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            stopSequences: stopSequences
        )
    }

    // MARK: - Private helpers

    /// Create a constrained generation session with stop token extraction.
    fileprivate static func createSession(
        jsonSchema: String,
        vocabSizeOverride: Int?,
        tokenizer: any Tokenizer,
        stopSequences: StopSequences
    ) throws -> ConstrainedGenerationSession {
        guard let vocabSize = vocabSizeOverride ?? Self.deriveVocabSize(from: tokenizer) else {
            throw InferenceRuntimeError.invalidArgument(
                "Cannot determine vocabulary size from tokenizer. "
                    + "Pass vocabSize explicitly via CoreAIRunner or LLMAsset metadata."
            )
        }

        let singleTokenStops = stopSequences.sequences.filter { $0.count == 1 }.map { $0[0] }
        if stopSequences.sequences.contains(where: { $0.count > 1 }) {
            CLILogger.log(
                "Warning: Multi-token stop sequences not supported by xgrammar, using single-token stops only",
                component: "ConstrainedDecodingStrategy")
        }
        let stopTokenIds: [Int32]? = singleTokenStops.isEmpty ? nil : singleTokenStops

        let session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds
        )
        CLILogger.log(
            "Constrained session created (vocabSize=\(vocabSize), stopTokenIds=\(stopTokenIds ?? []))",
            component: "ConstrainedDecodingStrategy")
        return session
    }

    /// Run one inference step: get logits, apply mask, sample, accept.
    /// Returns `(nil, nil)` if generation should stop.
    fileprivate static func generateOneToken(
        inputTokens: [Int32],
        session: inout ConstrainedGenerationSession,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        constrainedOptions: InferenceOptions
    ) async throws -> (Int32?, [LogitsScalarType]?) {
        var rawLogits: [LogitsScalarType]? = nil
        for try await output in try await inferenceEngine.generate(
            with: inputTokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: constrainedOptions
        ) {
            rawLogits = output.logits
            break
        }
        guard let logits = rawLogits else {
            throw ConstrainedGenerationError.generationFailed("No logits returned from engine")
        }

        var maskedLogits = logits
        _ = session.applyMask(to: &maskedLogits)

        let bestToken = CompositeSampler.sample(from: &maskedLogits, config: samplingConfiguration)

        if !session.acceptToken(bestToken) {
            return (nil, nil)
        }
        return (bestToken, logits)
    }

    fileprivate static func computeTextDelta(
        generatedTokens: [Int32],
        previousDecodedText: inout String,
        tokenizer: any Tokenizer,
        tokenStep: Int
    ) -> String {
        let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
        let fullDecodedText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
        decodeSpan.end()

        let common = fullDecodedText.commonPrefix(with: previousDecodedText)
        let delta = String(fullDecodedText.dropFirst(common.count))

        if delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
            return ""
        }

        previousDecodedText = fullDecodedText
        return delta
    }

    // MARK: - Vocabulary size derivation

    /// Derive vocabulary size from a tokenizer using binary search.
    static func deriveVocabSize(from tokenizer: any Tokenizer) -> Int? {
        var low = 0
        var high = 524_288

        // Binary search for the last valid token ID
        while low < high {
            let mid = (low + high) / 2
            if tokenizer.convertIdToToken(mid) != nil {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low == 0 {
            CLILogger.log(
                "Warning: Could not determine vocab size from tokenizer — grammar mask may be wrong",
                component: "ConstrainedDecodingStrategy")
            return nil
        }
        return low
    }
}

// MARK: - ConstrainedDecodedSequence

extension ConstrainedDecodingStrategy {
    /// Async sequence of `GenerationResult` produced by `decode()`.
    public struct ConstrainedDecodedSequence: AsyncSequence {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        fileprivate let prepared: Prepared
        let tokenizer: any Tokenizer
        let inferenceEngine: any InferenceEngine
        let samplingConfiguration: SamplingConfiguration
        let stopSequences: StopSequences

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                prepared: prepared,
                tokenizer: tokenizer,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                stopSequences: stopSequences
            )
        }
    }
}

extension ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
    /// Holds the eagerly-created, move-only generation session together with the tokenized prompt and token budget.
    fileprivate final class Prepared {
        var session: ConstrainedGenerationSession?
        let inputTokens: [Int32]
        let maxTokens: Int

        init(
            session: consuming ConstrainedGenerationSession,
            inputTokens: [Int32],
            maxTokens: Int
        ) {
            self.session = consume session
            self.inputTokens = inputTokens
            self.maxTokens = maxTokens
        }
    }
}

extension ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        private let tokenizer: any Tokenizer
        private let inferenceEngine: any InferenceEngine
        private let samplingConfiguration: SamplingConfiguration
        private let stopSequences: StopSequences
        private let constrainedOptions = InferenceOptions(maxTokens: 1, includeLogits: true)

        // Generation state, seeded eagerly from the prepared setup.
        private var session: ConstrainedGenerationSession?
        private var inputTokens: [Int32]
        private let maxTokens: Int
        private var generatedTokens: [Int32] = []
        private var previousDecodedText: String = ""
        private var tokenStep: Int = 0
        private var finished: Bool = false

        fileprivate init(
            prepared: ConstrainedDecodingStrategy.ConstrainedDecodedSequence.Prepared,
            tokenizer: any Tokenizer,
            inferenceEngine: any InferenceEngine,
            samplingConfiguration: SamplingConfiguration,
            stopSequences: StopSequences
        ) {
            // Take ownership of the move-only session prepared by `decode()`.
            self.session = prepared.session.take()
            self.inputTokens = prepared.inputTokens
            self.maxTokens = prepared.maxTokens
            self.tokenizer = tokenizer
            self.inferenceEngine = inferenceEngine
            self.samplingConfiguration = samplingConfiguration
            self.stopSequences = stopSequences
        }

        public func next() async throws -> GenerationResult? {
            if finished {
                return nil
            }

            while tokenStep < maxTokens {
                try Task.checkCancellation()

                guard var session = self.session.take() else {
                    finished = true
                    return nil
                }
                if session.isTerminated {
                    finished = true
                    return nil
                }

                let result: (Int32?, [LogitsScalarType]?)
                do {
                    result = try await ConstrainedDecodingStrategy.generateOneToken(
                        inputTokens: inputTokens,
                        session: &session,
                        inferenceEngine: inferenceEngine,
                        samplingConfiguration: samplingConfiguration,
                        constrainedOptions: constrainedOptions
                    )
                } catch {
                    finished = true
                    // Drop session — generation failed, no further use.
                    throw error
                }

                let terminatedAfterAccept = session.isTerminated
                self.session = consume session

                guard let bestToken = result.0, let logits = result.1 else {
                    finished = true
                    return nil
                }

                if stopSequences.matches(recentTokens: [bestToken]) {
                    finished = true
                    return nil
                }

                inputTokens.append(bestToken)
                generatedTokens.append(bestToken)
                tokenStep += 1

                let delta = ConstrainedDecodingStrategy.computeTextDelta(
                    generatedTokens: generatedTokens,
                    previousDecodedText: &previousDecodedText,
                    tokenizer: tokenizer,
                    tokenStep: tokenStep
                )

                if terminatedAfterAccept {
                    finished = true
                }

                return GenerationResult(text: delta, tokenId: bestToken, rawLogits: logits)
            }

            finished = true
            return nil
        }
    }
}
