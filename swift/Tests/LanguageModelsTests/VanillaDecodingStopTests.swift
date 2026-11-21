// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import TestUtilities
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// How a vanilla decode ends: the reason it reports, the tokens it counts, and that a stop sequence
/// ends the engine's work rather than running it on to `maxTokens`.
@Suite("VanillaDecodingStrategy — how a stream ends")
struct VanillaDecodingStopTests {
    private func decode(
        _ engine: MockEngine, maxTokens: Int, stops: [[Int32]] = [], recordTokenIds: Bool = false
    ) async throws -> VanillaDecodingStrategy.VanillaDecodedSequence {
        try await VanillaDecodingStrategy().decode(
            from: .tokens([1]), tokenizer: MockTokenizer(), inferenceEngine: engine,
            samplingConfiguration: .greedy, options: InferenceOptions(maxTokens: maxTokens),
            decoderOptions: DecoderOptions(
                stopSequences: StopSequences(sequences: stops), recordTokenIds: recordTokenIds))
    }

    /// Every result's text, and how many results there were.
    private func read(_ stream: VanillaDecodingStrategy.VanillaDecodedSequence) async throws -> (String, Int) {
        var text = ""
        var results = 0
        for try await result in stream {
            text += result.text
            results += 1
        }
        return (text, results)
    }

    @Test("Running to the limit reports maxTokens and counts every token")
    func limitReportsMaxTokens() async throws {
        // "h", "i", " ": every token adds text.
        let stream = try await decode(MockEngine(tokens: [104, 105, 32]), maxTokens: 6)
        _ = try await read(stream)
        #expect(stream.stopReason == .maxTokens)
        #expect(stream.tokensGenerated == 6)
    }

    @Test("Tokens that add no text of their own are still counted")
    func splitCharactersCount() async throws {
        // 0xC3 0xA9 is "é": its first byte is generated and paid for, but yields no result.
        let stream = try await decode(MockEngine(tokens: [0xC3, 0xA9]), maxTokens: 6)
        let (text, results) = try await read(stream)
        #expect(text == "ééé")
        #expect(results == 3)
        #expect(stream.tokensGenerated == 6)
    }

    @Test("A stop sequence reports stopSequence, keeps its text out of the reply, and stops the engine there")
    func stopSequenceReleasesTheEngine() async throws {
        // "h", "i", "!" over and over; stop on "!".
        let engine = MockEngine(tokens: [104, 105, 33])
        let stream = try await decode(engine, maxTokens: 100, stops: [[33]])
        let (text, _) = try await read(stream)
        #expect(text == "hi")
        #expect(stream.stopReason == .stopSequence("!"))
        // The stop token was generated too.
        #expect(stream.tokensGenerated == 3)
        // Draining would have run the engine to all 100.
        #expect(engine.inferenceCallCount == 3)
    }

    @Test("Token IDs are recorded only when asked for")
    func tokenIdsOnlyWhenAsked() async throws {
        let stream = try await decode(MockEngine(tokens: [104, 105, 33]), maxTokens: 100, stops: [[33]])
        _ = try await read(stream)
        #expect(stream.generatedTokenIds == nil)
    }

    @Test("Recorded token IDs are every generated token: those with no text and the stop token too")
    func tokenIdsMatchTheCount() async throws {
        // 0xC3 yields no result of its own; "!" ends the reply.
        let stream = try await decode(
            MockEngine(tokens: [104, 0xC3, 0xA9, 33]), maxTokens: 100, stops: [[33]], recordTokenIds: true)
        let (text, _) = try await read(stream)
        #expect(text == "hé")
        #expect(stream.generatedTokenIds == [104, 0xC3, 0xA9, 33])
        #expect(stream.generatedTokenIds?.count == stream.tokensGenerated)
    }

    @Test("Recorded token IDs can be read while the stream runs, and from an offset")
    func tokenIdsWhileStreaming() async throws {
        let stream = try await decode(
            MockEngine(tokens: [104, 105, 33]), maxTokens: 100, stops: [[33]], recordTokenIds: true)
        var seen: [[Int32]] = []
        var read = 0
        for try await _ in stream {
            let ids = try #require(stream.generatedTokenIds(from: read))
            read += ids.count
            seen.append(ids)
        }
        #expect(seen == [[104], [105]])
        #expect(stream.generatedTokenIds(from: read) == [33])
    }

    @Test("A lone id the tokenizer has no token for is skipped; the reply goes on")
    func loneUnknownIdIsSkipped() async throws {
        // MockTokenizer knows ids 0-255; 300 is past it, as a padded output layer's ids are.
        let stream = try await decode(MockEngine(tokens: [104, 300, 105]), maxTokens: 6)
        let (text, _) = try await read(stream)
        #expect(text == "hihi")
        #expect(stream.stopReason == .maxTokens)
        #expect(stream.tokensGenerated == 6)
    }

    @Test("Ids with no token, several in a row, fail the stream instead of running silently to the limit")
    func stuckOnUnknownIdsThrows() async throws {
        let engine = MockEngine(tokens: [300])
        let stream = try await decode(engine, maxTokens: 100)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await read(stream)
        }
        #expect(stream.stopReason == .error)
        #expect(
            engine.inferenceCallCount
                == VanillaDecodingStrategy.VanillaDecodedSequence.Iterator.maxConsecutiveUnknownIds)
    }

    @Test("A stop id the tokenizer has no token for still ends the reply as a stop")
    func unknownStopIdStops() async throws {
        let stream = try await decode(MockEngine(tokens: [104, 300]), maxTokens: 10, stops: [[300]])
        let (text, _) = try await read(stream)
        #expect(text == "h")
        #expect(stream.stopReason == .stopSequence(""))
    }
}

#endif
