// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers
import os.signpost

/// Standard autoregressive decoding.
///
/// Handles text decoding, stop sequence detection, and Instruments profiling.
/// Uses `InferenceEngine.generate()` for the underlying token stream.
public struct VanillaDecodingStrategy: DecodingStrategy {
    public init() {}

    // MARK: - Primary API

    /// Stream decoded text with optional logits.
    ///
    /// - Parameters:
    ///   - input: Input specification (raw text, prompt, or pre-tokenized)
    ///   - tokenizer: Tokenizer for encoding/decoding
    ///   - inferenceEngine: Engine for model inference
    ///   - samplingConfiguration: Sampling parameters (temperature, topK, etc.)
    ///   - options: Inference options (maxTokens, tokens, logits)
    ///   - stopSequences: Token sequences that halt generation
    /// - Returns: Stream of `GenerationResult` (text + optional logits)
    public func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) async throws -> VanillaDecodedSequence {
        CLILogger.log("🔄 Starting vanilla decoding generation")

        // Eager setup.
        let inputTokens =
            try PromptUtils
            .maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        CLILogger.log("Input tokens: \(inputTokens.prefix(10))... (showing first 10)")

        let stream = try await inferenceEngine.generate(
            with: inputTokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: options
        )
        return decodedSequence(
            of: stream, inputTokenCount: inputTokens.count, tokenizer: tokenizer, stopSequences: stopSequences)
    }

    /// Stream decoded text for a prompt with image placeholders.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings of every image placeholder in `tokens`, in order
    ///   - tokens: Prompt tokens, image placeholders included
    ///   - tokenizer: Tokenizer for decoding
    ///   - inferenceEngine: Engine that accepts multimodal input
    ///   - samplingConfiguration: Sampling parameters (temperature, topK, etc.)
    ///   - options: Inference options (maxTokens, tokens, logits)
    ///   - stopSequences: Token sequences that halt generation
    /// - Returns: Stream of `GenerationResult` (text + optional logits)
    package func decode(
        embeddings: InputEmbeddings,
        tokens: [Int32],
        tokenizer: any Tokenizer,
        inferenceEngine: any MultimodalInferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) async throws -> VanillaDecodedSequence {
        let stream = try await inferenceEngine.generate(
            with: embeddings,
            tokens: tokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: options
        )
        return decodedSequence(
            of: stream, inputTokenCount: tokens.count, tokenizer: tokenizer, stopSequences: stopSequences)
    }

    /// The decoded text of the engine's `stream` for a prompt of `inputTokenCount` tokens.
    private func decodedSequence(
        of stream: any InferenceOutputSequence, inputTokenCount: Int, tokenizer: any Tokenizer,
        stopSequences: StopSequences
    ) -> VanillaDecodedSequence {
        VanillaDecodedSequence(
            prepared: VanillaDecodedSequence.Prepared(
                initialInputTokenCount: inputTokenCount,
                engineSequence: stream,
                upstreamIterator: stream.makeAsyncIterator()
            ),
            tokenizer: tokenizer,
            stopSequences: stopSequences
        )
    }

    // MARK: - UTF-8 Safety

    /// Returns the end index of the longest prefix of `text` that contains
    /// only complete UTF-8 characters (no trailing U+FFFD replacement chars
    /// that indicate the tokenizer produced an incomplete byte sequence).
    fileprivate static func safeUTF8Prefix(of text: String) -> String.Index {
        var end = text.endIndex
        while end > text.startIndex {
            let prev = text.index(before: end)
            if text[prev] == "\u{FFFD}" {
                end = prev
            } else {
                break
            }
        }
        return end
    }
}

// MARK: - VanillaDecodedSequence

extension VanillaDecodingStrategy {
    /// Async sequence of `GenerationResult` produced by `decode()`.
    public struct VanillaDecodedSequence: AsyncSequence, DecodeOutcomeReporting {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        /// Holds the eagerly-opened engine stream (and its single iterator) produced by `decode()`.
        fileprivate final class Prepared {
            let initialInputTokenCount: Int
            var engineSequence: (any InferenceOutputSequence)?
            var upstreamIterator: (any AsyncIteratorProtocol<InferenceOutput, Error>)?
            /// Why the stream ended, kept apart from the engine sequence so that dropping that sequence
            /// to stop the engine does not lose it.
            let outcome = StopReasonStore()
            /// Tokens the engine produced, for ``VanillaDecodedSequence/tokensGenerated``.
            let tokenCount = TokenCountStore()

            init(
                initialInputTokenCount: Int,
                engineSequence: any InferenceOutputSequence,
                upstreamIterator: any AsyncIteratorProtocol<InferenceOutput, Error>
            ) {
                self.initialInputTokenCount = initialInputTokenCount
                self.engineSequence = engineSequence
                self.upstreamIterator = upstreamIterator
            }
        }

        fileprivate let prepared: Prepared
        let tokenizer: any Tokenizer
        let stopSequences: StopSequences

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                prepared: prepared,
                tokenizer: tokenizer,
                stopSequences: stopSequences
            )
        }

        public var stopReason: StopReason? { prepared.outcome.stopReason }

        public var tokensGenerated: Int { prepared.tokenCount.count }
    }
}

extension VanillaDecodingStrategy.VanillaDecodedSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        private let tokenizer: any Tokenizer
        private let stopSequences: StopSequences

        // The engine sequence is retained alongside its iterator so the decoder can record *why* generation ended
        // (e.g. a matched stop sequence) on the shared `stopReason` slot — see `InferenceOutputSequence`.
        private var engineSequence: (any InferenceOutputSequence)?
        private var upstreamIterator: (any AsyncIteratorProtocol<InferenceOutput, Error>)?
        private let initialInputTokenCount: Int
        private let outcome: StopReasonStore
        private let tokenCount: TokenCountStore

        // Decoding state
        private var generatedTokenCount: Int = 0
        private var newlyGeneratedTokens: [Int] = []
        private var recentTokens: [Int32] = []
        private var pendingText: String = ""
        private var inferenceSpan: ProfileSpan?
        private var stopped: Bool = false
        private var flushed: Bool = false
        /// Ids in a row that the tokenizer has no token for; see ``maxConsecutiveUnknownIds``.
        private var consecutiveUnknownIds = 0

        /// Ids in a row with no token in the tokenizer before the stream fails. One is skipped: the
        /// output layer is often padded past the vocabulary, and sampling can land in the padding.
        /// Several in a row is a sampler stuck there, which otherwise runs silently to `maxTokens`.
        static let maxConsecutiveUnknownIds = 4

        fileprivate init(
            prepared: VanillaDecodingStrategy.VanillaDecodedSequence.Prepared,
            tokenizer: any Tokenizer,
            stopSequences: StopSequences
        ) {
            self.tokenizer = tokenizer
            self.stopSequences = stopSequences
            self.initialInputTokenCount = prepared.initialInputTokenCount
            self.outcome = prepared.outcome
            self.tokenCount = prepared.tokenCount
            self.engineSequence = prepared.engineSequence.take()
            self.upstreamIterator = prepared.upstreamIterator.take()
            self.inferenceSpan = InstrumentsProfiler.beginPrompt(tokens: prepared.initialInputTokenCount)
        }

        deinit {
            // If the consumer abandons iteration early, `next()` never reaches the span-end / metrics code. End the
            // still-open inference span here so it isn't orphaned in Instruments, and record the token count.
            if let span = inferenceSpan.take() {
                span.end()
            }
            if !flushed {
                let count = generatedTokenCount
                // Metrics recording is async; fire-and-forget on the abandon path.
                Task { await PerformanceMetrics.shared.recordGeneratedTokens(count) }
            }
        }

        public func next() async throws -> GenerationResult? {
            guard !stopped, var iterator = upstreamIterator else {
                return try await flushTrailingText()
            }

            // Drive the upstream engine iterator until we have a non-empty text delta to emit.
            while let output = try await nextOutput(&iterator) {
                try Task.checkCancellation()
                let step = generatedTokenCount

                endInferenceSpan()

                CLILogger.log("✅ Generated token ID: \(output.tokenId)", level: 2)

                let decodingSpan = InstrumentsProfiler.beginDecode(step: step)
                generatedTokenCount += 1
                tokenCount.set(generatedTokenCount)

                // An id with no token adds no text, but still counts and can still end a stop sequence.
                // BPE and WordPiece tokenizers return nil for an id past their vocabulary. Unigram
                // tokenizers index an array and trap on such an id, as `decode` already does.
                let known = tokenizer.convertIdToToken(Int(output.tokenId)) != nil
                consecutiveUnknownIds = known ? 0 : consecutiveUnknownIds + 1
                if consecutiveUnknownIds >= Self.maxConsecutiveUnknownIds {
                    decodingSpan.end()
                    outcome.setIfUnset(.error)
                    throw InferenceRuntimeError.invalidState(
                        "the sampler returned \(consecutiveUnknownIds) ids in a row that are no token in this "
                            + "tokenizer (last \(output.tokenId)), after \(generatedTokenCount) token(s)")
                }

                var newText = ""
                if known {
                    newlyGeneratedTokens.append(Int(output.tokenId))
                    let fullDecode = tokenizer.decode(tokens: newlyGeneratedTokens)

                    let safeEnd = VanillaDecodingStrategy.safeUTF8Prefix(of: fullDecode)
                    let emittable = String(fullDecode[fullDecode.startIndex..<safeEnd])

                    if emittable.count > pendingText.count, emittable.hasPrefix(pendingText) {
                        newText = String(emittable.dropFirst(pendingText.count))
                        pendingText = emittable
                    } else if emittable != pendingText {
                        newText = emittable
                        pendingText = emittable
                    }

                    if newText.hasSuffix("\n") {
                        newlyGeneratedTokens = []
                        pendingText = ""
                    }
                }

                recentTokens.append(output.tokenId)
                if recentTokens.count > stopSequences.maxLength {
                    recentTokens.removeFirst()
                }

                if let matched = stopSequences.matchedSequence(recentTokens: recentTokens) {
                    CLILogger.log("✅ Stop sequence detected at tokens: \(recentTokens)", level: 2)
                    decodingSpan.end()
                    stopped = true

                    // Record why the engine stream ended before abandoning it.
                    let matchedText = tokenizer.decode(tokens: matched.map(Int.init))
                    engineSequence?.setStopReason(.stopSequence(matchedText))
                    outcome.set(.stopSequence(matchedText))

                    // Release the engine stream instead of draining it. Draining waited for every
                    // remaining token up to maxTokens, full forward passes whose output was thrown
                    // away: with a large maxTokens, minutes of GPU work after the answer was complete.
                    //
                    // Dropping the sequence and its iterator ends the engine's generation between
                    // tokens: the pipelined engine's output stream terminates and its producer sees
                    // the cancellation; the sequential and static-shape engines simply stop being
                    // asked for tokens, and their iterators give back the engine in `deinit`; on the
                    // batched lane, `setStopReason` above cancels the request.
                    engineSequence = nil
                    upstreamIterator = nil

                    // Flush any buffered text and record metrics — flushTrailingText()
                    // sets `flushed`, so subsequent next() calls return nil.
                    return try await flushTrailingText()
                }

                decodingSpan.end()
                InstrumentsProfiler.logTokenGeneration(tokenIndex: step, token: newText)

                inferenceSpan = InstrumentsProfiler.beginExtend(
                    step: step + 1,
                    tokens: initialInputTokenCount + generatedTokenCount
                )

                if !newText.isEmpty {
                    upstreamIterator = iterator
                    return GenerationResult(text: newText, tokenId: output.tokenId, rawLogits: output.logits)
                }
                CLILogger.log("✅ Generated newText: \(newText)", level: 2)
            }

            // Engine stream exhausted naturally — its iterator already recorded `.maxTokens` (or `.cancelled`/`.error`
            // if it threw above).
            outcome.setIfUnset(engineSequence?.stopReason ?? .maxTokens)
            engineSequence = nil
            upstreamIterator = nil
            endInferenceSpan()
            return try await flushTrailingText()
        }

        /// The engine's next output, recording `.error` as the stream's stop reason if it throws.
        private func nextOutput(
            _ iterator: inout any AsyncIteratorProtocol<InferenceOutput, Error>
        ) async throws -> InferenceOutput? {
            do {
                return try await iterator.next()
            } catch {
                outcome.setIfUnset(.error)
                throw error
            }
        }

        private func endInferenceSpan() {
            if let span = inferenceSpan.take() {
                span.end()
            }
        }

        private func flushTrailingText() async throws -> GenerationResult? {
            guard !flushed else {
                return nil
            }
            flushed = true

            let finalDecode = tokenizer.decode(tokens: newlyGeneratedTokens)
            let trailing: GenerationResult?
            if finalDecode.count > pendingText.count, finalDecode.hasPrefix(pendingText) {
                let remaining = String(finalDecode.dropFirst(pendingText.count))
                trailing =
                    remaining.isEmpty
                    ? nil
                    : GenerationResult(text: remaining, tokenId: 0, rawLogits: nil)
            } else {
                trailing = nil
            }

            await PerformanceMetrics.shared.recordGeneratedTokens(generatedTokenCount)
            return trailing
        }
    }
}
