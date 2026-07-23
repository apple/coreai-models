// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Grammar-constrained decoding strategy using the GPU pipelined engine.
///
/// Unlike `ConstrainedDecodingStrategy` which forces the sequential engine path
/// (CPU logits -> mask -> CPU sample), this strategy applies the xgrammar bitmask
/// directly inside the MPSGraph sampler on GPU -- eliminating the ~296KB logit
/// transfer and CPU-side softmax/sampling per token.
///
/// Only used when the engine is `CoreAIPipelinedEngine`. The routing logic in
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
    ) async throws -> AsyncThrowingStream<GenerationResult, Error> {
        guard let pipelinedEngine = inferenceEngine as? CoreAIPipelinedEngine else {
            throw InferenceRuntimeError.invalidArgument(
                "PipelinedConstrainedDecodingStrategy requires CoreAIPipelinedEngine")
        }

        let vocabSize = vocabSizeOverride ?? ConstrainedDecodingStrategy.deriveVocabSize(from: tokenizer)
        guard let vocabSize else {
            throw InferenceRuntimeError.invalidArgument(
                "Cannot determine vocabulary size from tokenizer. "
                    + "Pass vocabSize explicitly via CoreAIRunner or LLMAsset metadata."
            )
        }

        let singleTokenStops = stopSequences.sequences.filter { $0.count == 1 }.map { $0[0] }
        let stopTokenIds: [Int32]? = singleTokenStops.isEmpty ? nil : singleTokenStops

        var session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds
        )

        let inputTokens = try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        let maxTokens = options.maxTokens ?? 512

        try await pipelinedEngine.reset()

        // Move session into a class box before the closure boundary (session is ~Copyable)
        let sessionBox = ConstrainedSessionBox(session: session)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.runPipelinedConstrained(
                        inputTokens: inputTokens,
                        engine: pipelinedEngine,
                        sessionBox: sessionBox,
                        samplingConfiguration: samplingConfiguration,
                        maxTokens: maxTokens,
                        stopSequences: stopSequences,
                        tokenizer: tokenizer,
                        with: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Core Loop

    private func runPipelinedConstrained(
        inputTokens: [Int32],
        engine: CoreAIPipelinedEngine,
        sessionBox: ConstrainedSessionBox,
        samplingConfiguration: SamplingConfiguration,
        maxTokens: Int,
        stopSequences: StopSequences,
        tokenizer: any Tokenizer,
        with continuation: AsyncThrowingStream<GenerationResult, Error>.Continuation
    ) async throws {
        CLILogger.log("Starting GPU pipelined constrained decoding", component: "PipelinedConstrained")

        var generatedTokens: [Int32] = []
        var previousDecodedText = ""
        var recentTokens: [Int32] = []

        for try await tokenId in try engine.generateConstrained(
            with: inputTokens,
            samplingConfiguration: samplingConfiguration,
            maxTokens: maxTokens,
            session: sessionBox
        ) {
            // Check stop sequences
            recentTokens.append(tokenId)
            if recentTokens.count > stopSequences.maxLength {
                recentTokens.removeFirst()
            }
            if stopSequences.matches(recentTokens: recentTokens) { break }

            // Decode text incrementally
            generatedTokens.append(tokenId)
            let fullDecode = tokenizer.decode(tokens: generatedTokens.map { Int($0) })

            let common = fullDecode.commonPrefix(with: previousDecodedText)
            let delta = String(fullDecode.dropFirst(common.count))

            if delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                continue
            }

            previousDecodedText = fullDecode

            if !delta.isEmpty {
                continuation.yield(GenerationResult(text: delta, tokenId: tokenId, rawLogits: nil))
            }
        }

        continuation.finish()
    }
}
