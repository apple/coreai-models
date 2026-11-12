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
        _ engine: MockEngine, maxTokens: Int
    ) async throws -> ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
        let tokenizer = MockTokenizer()
        return try await ConstrainedDecodingStrategy(jsonSchema: #"{"type": "integer"}"#, vocabSize: 256).decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: maxTokens),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: []))
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
}
