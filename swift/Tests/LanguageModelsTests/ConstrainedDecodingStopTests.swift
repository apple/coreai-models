// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// How a constrained decode ends: the reason it reports and the tokens it counts.
@Suite("ConstrainedDecodingStrategy — how a stream ends")
struct ConstrainedDecodingStopTests {
    private func decode(
        _ engine: MockEngine, maxTokens: Int, recordTokenIds: Bool = false
    ) async throws -> ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
        let tokenizer = MockTokenizer()
        return try await ConstrainedDecodingStrategy(jsonSchema: #"{"type": "integer"}"#, vocabSize: 256).decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: maxTokens),
            decoderOptions: DecoderOptions(
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: []),
                recordTokenIds: recordTokenIds))
    }

    private func text(_ stream: ConstrainedDecodingStrategy.ConstrainedDecodedSequence) async throws -> String {
        var text = ""
        for try await result in stream { text += result.text }
        return text
    }

    @Test("Running to the limit reports maxTokens and counts every token")
    func limitReportsMaxTokens() async throws {
        // The engine's logits peak at "1" every step.
        let stream = try await decode(MockEngine(tokens: [49], vocabSize: 256), maxTokens: 3)
        #expect(try await text(stream) == "111")
        #expect(stream.stopReason == .maxTokens)
        #expect(stream.tokensGenerated == 3)
    }

    @Test("The stop token is counted but kept out of the reply")
    func stopTokenCountedNotEmitted() async throws {
        // "1", then the end-of-sequence token (2), which the grammar allows once the integer is complete.
        let stream = try await decode(MockEngine(tokens: [49, 2], vocabSize: 256), maxTokens: 10)
        #expect(try await text(stream) == "1")
        guard case .stopSequence = stream.stopReason else {
            Issue.record("expected a stop-sequence end, got \(String(describing: stream.stopReason))")
            return
        }
        #expect(stream.tokensGenerated == 2)
    }

    @Test("Token IDs are recorded only when asked for, the stop token included")
    func tokenIds() async throws {
        let unrecorded = try await decode(MockEngine(tokens: [49, 2], vocabSize: 256), maxTokens: 10)
        _ = try await text(unrecorded)
        #expect(unrecorded.generatedTokenIds == nil)

        let recorded = try await decode(
            MockEngine(tokens: [49, 2], vocabSize: 256), maxTokens: 10, recordTokenIds: true)
        _ = try await text(recorded)
        #expect(recorded.generatedTokenIds == [49, 2])
        #expect(recorded.generatedTokenIds?.count == recorded.tokensGenerated)
    }

    @Test("ConstrainedGenerator refuses to record token IDs rather than ignore the option")
    func generatorRefusesTokenIds() async throws {
        let tokenizer = MockTokenizer()
        let engine = MockEngine(tokens: [49], vocabSize: 256)
        let generator = ConstrainedGenerator(
            engine: engine, tokenizer: tokenizer, vocabSize: 256, jsonSchema: #"{"type": "integer"}"#)
        #expect {
            _ = try generator.decode(
                from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: 3),
                decoderOptions: DecoderOptions(stopSequences: StopSequences(sequences: []), recordTokenIds: true))
        } throws: { error in
            guard case InferenceRuntimeError.unsupportedDecoderOption = error else { return false }
            return true
        }
    }
}
