// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// Streamed text deltas must concatenate to the decoded text. A token that extends the
/// previous grapheme cluster (combining mark, ZWJ emoji sequence, Indic vowel sign) leaves
/// the `Character` count unchanged, so a Character-based diff re-emits text.
@Suite("Streaming text deltas")
struct StreamingTextDeltaTests {
    /// MockTokenizer emits one token per UTF-8 byte, so every Unicode scalar arrives on its own.
    private static let texts = [
        "cafe\u{0301} au lait",
        "family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} ok",
        "\u{0B95}\u{0BB3}\u{0BBF}\u{0BB2}\u{0BCD}",
    ]

    private let tokenizer = MockTokenizer()

    private func tokens(for text: String) -> [Int32] {
        tokenizer.encode(text: text).map(Int32.init)
    }

    private func concatenated(_ stream: some AsyncSequence<GenerationResult, Error>) async throws -> String {
        var text = ""
        for try await result in stream {
            text += result.text
        }
        return text
    }

    @Test("Vanilla decoding streams a grapheme extension once", arguments: texts)
    func vanilla(text: String) async throws {
        let scripted = tokens(for: text)
        let stream = try await VanillaDecodingStrategy().decode(
            from: .tokens([1]),
            tokenizer: tokenizer,
            inferenceEngine: MockEngine(tokens: scripted, vocabSize: nil),
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: scripted.count),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )

        let streamed = try await concatenated(stream)
        #expect(streamed == text)
    }

    @Test("Constrained decoding streams a grapheme extension once", arguments: texts)
    func constrained(text: String) async throws {
        let json = "\"\(text)\""
        let scripted = tokens(for: json)
        let stream = try await ConstrainedDecodingStrategy(jsonSchema: #"{"type": "string"}"#, vocabSize: 256)
            .decode(
                from: .tokens([1]),
                tokenizer: tokenizer,
                inferenceEngine: MockEngine(tokens: scripted, vocabSize: 256),
                samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: scripted.count),
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
            )

        let streamed = try await concatenated(stream)
        #expect(streamed == json)
    }

    @Test("Pipelined constrained decoding streams a grapheme extension once", arguments: texts)
    func pipelinedConstrained(text: String) async throws {
        let json = "\"\(text)\""
        let scripted = tokens(for: json)
        let stream = try await PipelinedConstrainedDecodingStrategy(jsonSchema: #"{"type": "string"}"#, vocabSize: 256)
            .decode(
                from: .tokens([1]),
                tokenizer: tokenizer,
                inferenceEngine: MockConstrainedEngine(scriptedTokens: scripted),
                samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: scripted.count),
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
            )

        let streamed = try await concatenated(stream)
        #expect(streamed == json)
    }
}
