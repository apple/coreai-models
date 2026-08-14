// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Synchronization
import Tokenizers

private let defaultMaxConstrainedTokens = 512
private let maxConsecutiveDecodeFailures = 10

private final class SingleUseFlag: Sendable {
    private let value = Atomic<Bool>(false)
    func testAndSet() -> Bool {
        value.exchange(true, ordering: .acquiring)
    }
}

/// Grammar-constrained decoding strategy using the GPU pipelined engine.
///
/// Unlike `ConstrainedDecodingStrategy` which forces the sequential engine path
/// (CPU logits -> mask -> CPU sample), this strategy applies the xgrammar bitmask
/// directly inside the MPSGraph sampler on GPU -- eliminating the ~296KB logit
/// transfer and CPU-side softmax/sampling per token.
///
/// Only used when the engine conforms to `ConstrainedGenerationCapable`. The routing logic in
/// `CoreAILanguageModel` selects this strategy based on engine type.
public struct PipelinedConstrainedDecodingStrategy: DecodingStrategy {
    private let jsonSchema: String
    private let vocabSizeOverride: Int?

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
    ) async throws -> PipelinedConstrainedSequence {
        guard let constrainedEngine = inferenceEngine as? any ConstrainedGenerationCapable else {
            throw InferenceRuntimeError.invalidArgument(
                "PipelinedConstrainedDecodingStrategy requires a ConstrainedGenerationCapable engine")
        }

        let vocabSize = vocabSizeOverride ?? ConstrainedDecodingStrategy.deriveVocabSize(from: tokenizer)
        guard let vocabSize else {
            throw InferenceRuntimeError.invalidArgument(
                "Cannot determine vocabulary size from tokenizer. "
                    + "Pass vocabSize explicitly via the model metadata."
            )
        }

        let singleTokenStops = stopSequences.sequences.filter { $0.count == 1 }.map { $0[0] }
        let stopTokenIds: [Int32]? = singleTokenStops.isEmpty ? nil : singleTokenStops

        let session = try constrainedEngine.getOrCreateConstrainedSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds
        )

        let inputTokens = try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        let maxTokens = options.maxTokens ?? defaultMaxConstrainedTokens

        CLILogger.log(
            "Starting GPU pipelined constrained decoding (vocabSize=\(vocabSize), maxTokens=\(maxTokens))",
            component: "PipelinedConstrained")

        return PipelinedConstrainedSequence(
            session: session,
            inputTokens: inputTokens,
            maxTokens: maxTokens,
            tokenizer: tokenizer,
            engine: constrainedEngine,
            samplingConfiguration: samplingConfiguration,
            stopSequences: stopSequences
        )
    }
}

// MARK: - PipelinedConstrainedSequence

/// Single-use async sequence of `GenerationResult` produced by pipelined constrained decoding.
///
/// The engine's background Task owns session lifetime: it returns the session to
/// cache in its `defer` block after generation completes (normal, error, or cancel).
/// The iterator does not manage session lifecycle.
public struct PipelinedConstrainedSequence: AsyncSequence {
    public typealias Element = GenerationResult
    public typealias Failure = Error

    fileprivate let session: ConstrainedSessionHandle
    fileprivate let inputTokens: [Int32]
    fileprivate let maxTokens: Int
    let tokenizer: any Tokenizer
    let engine: any ConstrainedGenerationCapable
    let samplingConfiguration: SamplingConfiguration
    let stopSequences: StopSequences

    /// Guards against multiple `makeAsyncIterator()` calls sharing the same session box.
    private let consumed = SingleUseFlag()

    public func makeAsyncIterator() -> Iterator {
        precondition(!consumed.testAndSet(), "PipelinedConstrainedSequence may only be iterated once")

        return Iterator(
            session: session,
            inputTokens: inputTokens,
            maxTokens: maxTokens,
            tokenizer: tokenizer,
            engine: engine,
            samplingConfiguration: samplingConfiguration,
            stopSequences: stopSequences
        )
    }
}

extension PipelinedConstrainedSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        private let tokenizer: any Tokenizer
        private let engine: any ConstrainedGenerationCapable
        private let samplingConfiguration: SamplingConfiguration
        private let stopSequences: StopSequences
        private let session: ConstrainedSessionHandle

        private let inputTokens: [Int32]
        private let maxTokens: Int

        private var innerIterator: AsyncThrowingStream<Int32, Error>.AsyncIterator?
        private var generatedTokens: [Int32] = []
        private var previousDecodedText: String = ""
        private var recentTokens: [Int32] = []
        private var finished: Bool = false

        fileprivate init(
            session: ConstrainedSessionHandle,
            inputTokens: [Int32],
            maxTokens: Int,
            tokenizer: any Tokenizer,
            engine: any ConstrainedGenerationCapable,
            samplingConfiguration: SamplingConfiguration,
            stopSequences: StopSequences
        ) {
            self.session = session
            self.inputTokens = inputTokens
            self.maxTokens = maxTokens
            self.tokenizer = tokenizer
            self.engine = engine
            self.samplingConfiguration = samplingConfiguration
            self.stopSequences = stopSequences
        }

        public func next() async throws -> GenerationResult? {
            if finished { return nil }

            // Lazily start the engine stream on first iteration
            if innerIterator == nil {
                let stream = try engine.generateConstrained(
                    with: inputTokens,
                    samplingConfiguration: samplingConfiguration,
                    maxTokens: maxTokens,
                    session: session
                )
                self.innerIterator = stream.makeAsyncIterator()
            }

            do {
                var consecutiveDecodeFailures = 0
                while true {
                    try Task.checkCancellation()

                    // Safe to copy: AsyncThrowingStream.AsyncIterator is a class-backed
                    // reference type, so the copy shares the underlying stream state.
                    guard var iterator = self.innerIterator else {
                        finished = true
                        return nil
                    }

                    guard let tokenId = try await iterator.next() else {
                        finished = true
                        self.innerIterator = nil
                        return nil
                    }
                    self.innerIterator = iterator

                    // Check stop sequences
                    recentTokens.append(tokenId)
                    if recentTokens.count > stopSequences.maxLength {
                        recentTokens.removeFirst()
                    }
                    if stopSequences.matches(recentTokens: recentTokens) {
                        finished = true
                        self.innerIterator = nil
                        return nil
                    }

                    // Decode text incrementally
                    generatedTokens.append(tokenId)
                    let fullDecode = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
                    let common = fullDecode.commonPrefix(with: previousDecodedText)
                    let delta = String(fullDecode.dropFirst(common.count))

                    if delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                        consecutiveDecodeFailures += 1
                        if consecutiveDecodeFailures >= maxConsecutiveDecodeFailures {
                            throw InferenceRuntimeError.invalidState(
                                "Decoding produced \(maxConsecutiveDecodeFailures) consecutive "
                                    + "incomplete codepoints — generation stream is likely corrupted")
                        }
                        continue
                    }
                    consecutiveDecodeFailures = 0

                    previousDecodedText = fullDecode

                    if !delta.isEmpty {
                        return GenerationResult(text: delta, tokenId: tokenId, rawLogits: nil)
                    }
                }
            } catch {
                finished = true
                self.innerIterator = nil
                throw error
            }
        }
    }
}
