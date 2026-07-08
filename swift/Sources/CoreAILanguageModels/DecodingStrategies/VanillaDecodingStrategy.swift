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
    ///   - options: Inference options (maxTokens, includeLogits)
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

        return VanillaDecodedSequence(
            prepared: VanillaDecodedSequence.Prepared(
                initialInputTokenCount: inputTokens.count,
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
    public struct VanillaDecodedSequence: AsyncSequence {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        /// Holds the eagerly-opened engine stream (and its single iterator) produced by `decode()`.
        fileprivate final class Prepared {
            let initialInputTokenCount: Int
            var engineSequence: (any InferenceOutputSequence)?
            var upstreamIterator: (any AsyncIteratorProtocol<InferenceOutput, Error>)?

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

        // Decoding state
        private var generatedTokenCount: Int = 0
        private var newlyGeneratedTokens: [Int] = []
        private var recentTokens: [Int32] = []
        private var pendingText: String = ""
        private var inferenceSpan: ProfileSpan?
        private var stopped: Bool = false
        private var flushed: Bool = false

        fileprivate init(
            prepared: VanillaDecodingStrategy.VanillaDecodedSequence.Prepared,
            tokenizer: any Tokenizer,
            stopSequences: StopSequences
        ) {
            self.tokenizer = tokenizer
            self.stopSequences = stopSequences
            self.initialInputTokenCount = prepared.initialInputTokenCount
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
                Task { await PerformanceMetrics.shared.setGeneratedTokenCount(count) }
            }
        }

        public func next() async throws -> GenerationResult? {
            guard !stopped, var iterator = upstreamIterator else {
                return try await flushTrailingText()
            }

            // Drive the upstream engine iterator until we have a non-empty text delta to emit.
            while let output = try await iterator.next() {
                try Task.checkCancellation()
                let step = generatedTokenCount

                endInferenceSpan()

                CLILogger.log("✅ Generated token ID: \(output.tokenId)", level: 2)

                let decodingSpan = InstrumentsProfiler.beginDecode(step: step)

                newlyGeneratedTokens.append(Int(output.tokenId))
                generatedTokenCount += 1

                let fullDecode = tokenizer.decode(tokens: newlyGeneratedTokens)

                let safeEnd = VanillaDecodingStrategy.safeUTF8Prefix(of: fullDecode)
                let emittable = String(fullDecode[fullDecode.startIndex..<safeEnd])

                let newText: String
                if emittable.count > pendingText.count, emittable.hasPrefix(pendingText) {
                    newText = String(emittable.dropFirst(pendingText.count))
                    pendingText = emittable
                } else if emittable != pendingText {
                    newText = emittable
                    pendingText = emittable
                } else {
                    newText = ""
                }

                if newText.hasSuffix("\n") {
                    newlyGeneratedTokens = []
                    pendingText = ""
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

                    // Drain remaining tokens so the pipelined engine's producer Task completes.
                    while try await iterator.next() != nil {}
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
                    return GenerationResult(
                        text: newText,
                        tokenId: output.tokenId,
                        rawLogits: output.logits
                    )
                }
                CLILogger.log("✅ Generated newText: \(newText)", level: 2)
            }

            // Engine stream exhausted naturally — its iterator already recorded `.maxTokens` (or `.cancelled`/`.error`
            // if it threw above).
            engineSequence = nil
            upstreamIterator = nil
            endInferenceSpan()
            return try await flushTrailingText()
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

            await PerformanceMetrics.shared.setGeneratedTokenCount(generatedTokenCount)
            return trailing
        }
    }
}
