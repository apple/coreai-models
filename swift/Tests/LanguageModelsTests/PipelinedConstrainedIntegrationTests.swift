// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Mock Constrained Engine

/// Mock engine conforming to ConstrainedGenerationCapable for testing the
/// pipelined constrained strategy without Metal/GPU.
final class MockConstrainedEngine: InferenceEngine, ConstrainedGenerationCapable, @unchecked Sendable {
    struct MockConfig: Codable, InferenceConfiguration {
        var maxContextLength: Int = 4096
    }

    let config = MockConfig()
    var supportsLogits: Bool { false }
    var processedTokenCount: Int = 0
    private(set) var lastPrefixHitCount: Int = 0
    var isBusy: Bool { false }

    /// Tokens the mock engine will yield from generateConstrained
    var scriptedTokens: [Int32]

    /// When set, the stream ends after this many tokens with this reason, as the real engine's does
    /// when it is cancelled from the engine side.
    var endsEarly: (after: Int, reason: StopReason)?

    /// Track session lifecycle
    private(set) var getSessionCallCount = 0
    private(set) var storeSessionCallCount = 0
    private(set) var generateCallCount = 0
    private(set) var lastSchemaUsed: String?

    /// The cached session handle (simulating the real cache)
    private var cached: ConstrainedSessionHandle?

    init(scriptedTokens: [Int32]) {
        self.scriptedTokens = scriptedTokens
    }

    // MARK: - InferenceEngine conformance (minimal)

    typealias ConfigType = MockConfig
    typealias OutputSequence = MockOutputSequence

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> MockOutputSequence {
        fatalError("Use generateConstrained for this mock")
    }

    func reset(to tokenIndex: Int) async throws {
        processedTokenCount = tokenIndex
    }

    func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {}

    func cancel() async throws {}

    // MARK: - ConstrainedGenerationCapable conformance

    func getOrCreateConstrainedSession(
        jsonSchema: String,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        stopTokenIds: [Int32]?,
        maxWhitespaceCount: Int?
    ) throws -> ConstrainedSessionHandle {
        getSessionCallCount += 1
        lastSchemaUsed = jsonSchema

        if let handle = cached,
            handle.matches(
                schema: jsonSchema, maxWhitespaceCount: maxWhitespaceCount, stopTokenIds: stopTokenIds,
                vocabularySize: vocabSize)
        {
            cached = nil
            handle.reset()
            return handle
        }
        cached = nil

        let session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds,
            maxWhitespaceCount: maxWhitespaceCount
        )
        return ConstrainedSessionHandle(session: session, tokenizer: tokenizer, stopTokenIds: stopTokenIds)
    }

    func generateConstrained(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        maxTokens: Int,
        session: ConstrainedSessionHandle
    ) async throws -> InferenceTokenSequence {
        generateCallCount += 1
        let tokens = scriptedTokens
        let limit = min(tokens.count, maxTokens)
        let endsEarly = self.endsEarly

        let (stream, continuation) = AsyncThrowingStream<TokenId, Error>.makeStream()
        let stopReasonStore = StopReasonStore()

        let task = Task {
            defer { self.storeConstrainedSessionForReuse(session) }
            for i in 0..<limit {
                if let endsEarly, i == endsEarly.after {
                    stopReasonStore.setIfUnset(endsEarly.reason)
                    break
                }
                do { try Task.checkCancellation() } catch { break }
                if !session.acceptToken(tokens[i]) { break }
                if session.isTerminated { break }
                continuation.yield(tokens[i])
            }
            continuation.finish()
        }
        _ = task

        return InferenceTokenSequence(stream: stream, stopReasonStore: stopReasonStore)
    }

    func storeConstrainedSessionForReuse(_ handle: ConstrainedSessionHandle) {
        storeSessionCallCount += 1
        cached = handle
    }

    // MARK: - Mock output sequence (unused but required)

    struct MockOutputSequence: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = Error
        let stopReasonStore = StopReasonStore()
        var stopReason: StopReason? { nil }
        func setStopReason(_ reason: StopReason) {}
        func makeAsyncIterator() -> MockIterator { MockIterator() }
        struct MockIterator: AsyncIteratorProtocol {
            mutating func next() async throws -> InferenceOutput? { nil }
        }
    }
}

// MARK: - Tests

@Suite("PipelinedConstrainedStrategy Integration Tests")
struct PipelinedConstrainedIntegrationTests {
    private var tokenizer: MockTokenizer { MockTokenizer() }

    @Test("Happy path: generates tokens and yields text deltas")
    func happyPath() async throws {
        // UTF-8 for "0123" = [48, 49, 50, 51]
        let engine = MockConstrainedEngine(scriptedTokens: [48, 49, 50, 51, 52])
        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)

        let stream = try await strategy.decode(
            from: .tokens([1, 2, 3]),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 5),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )

        var results: [GenerationResult] = []
        for try await result in stream {
            results.append(result)
        }

        #expect(!results.isEmpty)
        #expect(engine.generateCallCount == 1)

        // Give the Task a moment to complete its defer block
        try await Task.sleep(for: .milliseconds(50))
        #expect(engine.storeSessionCallCount == 1, "Session should be returned to cache")
    }

    @Test("Session cache reuse: same schema reuses session")
    func sessionCacheReuse() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [48, 49])
        let schema = #"{"type": "integer"}"#

        let strategy = PipelinedConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: 256)
        let opts = InferenceOptions(maxTokens: 2)
        let stops = StopSequences(for: tokenizer, additionalEosTokenIds: [])

        // First call
        let stream1 = try await strategy.decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine,
            samplingConfiguration: .greedy, options: opts, stopSequences: stops
        )
        for try await _ in stream1 {}
        try await Task.sleep(for: .milliseconds(50))

        #expect(engine.getSessionCallCount == 1)
        #expect(engine.storeSessionCallCount == 1)

        // Second call with same schema — should reuse cached session
        let stream2 = try await strategy.decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine,
            samplingConfiguration: .greedy, options: opts, stopSequences: stops
        )
        for try await _ in stream2 {}
        try await Task.sleep(for: .milliseconds(50))

        #expect(engine.getSessionCallCount == 2)
        #expect(engine.storeSessionCallCount == 2)
    }

    @Test("Schema change invalidates cache")
    func schemaChangeInvalidatesCache() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [48])
        let opts = InferenceOptions(maxTokens: 1)
        let stops = StopSequences(for: tokenizer, additionalEosTokenIds: [])

        // First call with schema A
        let strategyA = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)
        let stream1 = try await strategyA.decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine,
            samplingConfiguration: .greedy, options: opts, stopSequences: stops
        )
        for try await _ in stream1 {}
        try await Task.sleep(for: .milliseconds(50))

        // Second call with different schema B
        let strategyB = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "string"}"#, vocabSize: 256)
        let stream2 = try await strategyB.decode(
            from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine,
            samplingConfiguration: .greedy, options: opts, stopSequences: stops
        )
        for try await _ in stream2 {}

        #expect(engine.getSessionCallCount == 2)
        #expect(engine.lastSchemaUsed == #"{"type": "string"}"#)
    }

    @Test("maxTokens=1 yields at most one token")
    func maxTokensOne() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [48, 49, 50])
        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)

        let stream = try await strategy.decode(
            from: .tokens([1]),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 1),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )

        var count = 0
        for try await _ in stream { count += 1 }

        #expect(count <= 1)
    }

    @Test("Stop sequence terminates generation")
    func stopSequenceTerminates() async throws {
        // Token 50 = "2" in UTF-8, use it as EOS
        let engine = MockConstrainedEngine(scriptedTokens: [48, 49, 50, 51, 52])
        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)

        let stream = try await strategy.decode(
            from: .tokens([1]),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 10),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [50])
        )

        var tokens: [Int32] = []
        for try await result in stream {
            tokens.append(result.tokenId)
        }

        // Should stop at or before token 50
        #expect(!tokens.contains(where: { $0 > 50 }))
    }

    @Test("Invalid schema throws during decode")
    func invalidSchemaThrows() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [])

        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: "not valid json", vocabSize: 256)

        #expect(throws: (any Error).self) {
            _ = try strategy.getOrCreateSession(engine: engine)
        }
    }

    @Test("Wrong engine type throws descriptive error")
    func wrongEngineTypeThrows() async throws {
        let wrongEngine = MockEngine(tokens: [1, 2, 3])
        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)

        await #expect(throws: (any Error).self) {
            _ = try await strategy.decode(
                from: .tokens([1]),
                tokenizer: tokenizer,
                inferenceEngine: wrongEngine,
                samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: 5),
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
            )
        }
    }

    @Test("Session returned to cache even when generation yields no tokens")
    func emptyGenerationReturnsSession() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [])
        let strategy = PipelinedConstrainedDecodingStrategy(
            jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)

        let stream = try await strategy.decode(
            from: .tokens([1]),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 5),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )

        for try await _ in stream {}

        // Give the Task a moment to complete its defer block
        try await Task.sleep(for: .milliseconds(50))
        #expect(engine.storeSessionCallCount == 1)
    }
}

// MARK: - How a stream ends

@Suite("PipelinedConstrainedSequence — how a stream ends")
struct PipelinedConstrainedStopTests {
    private func decode(
        _ engine: MockConstrainedEngine, maxTokens: Int
    ) async throws -> PipelinedConstrainedSequence {
        let tokenizer = MockTokenizer()
        return try await PipelinedConstrainedDecodingStrategy(jsonSchema: #"{"type": "integer"}"#, vocabSize: 256)
            .decode(
                from: .tokens([1]), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: maxTokens),
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: []))
    }

    private func text(_ stream: PipelinedConstrainedSequence) async throws -> String {
        var text = ""
        for try await result in stream { text += result.text }
        return text
    }

    @Test("Running to the limit reports maxTokens and counts every token")
    func limitReportsMaxTokens() async throws {
        let stream = try await decode(MockConstrainedEngine(scriptedTokens: [49, 50, 51, 52, 53]), maxTokens: 3)
        #expect(try await text(stream) == "123")
        #expect(stream.stopReason == .maxTokens)
        #expect(stream.tokensGenerated == 3)
    }

    @Test("The grammar accepting the end of sequence reports eos")
    func grammarEndReportsEos() async throws {
        // "1", then the end-of-sequence token (2), which completes the integer and ends the stream.
        let stream = try await decode(MockConstrainedEngine(scriptedTokens: [49, 2, 50]), maxTokens: 10)
        #expect(try await text(stream) == "1")
        #expect(stream.stopReason == .eos)
        #expect(stream.tokensGenerated == 1)
    }

    @Test("A stream the engine ends as cancelled reports cancelled, not eos")
    func engineCancellationReportsCancelled() async throws {
        let engine = MockConstrainedEngine(scriptedTokens: [49, 50, 51, 52, 53])
        engine.endsEarly = (after: 2, reason: .cancelled)
        let stream = try await decode(engine, maxTokens: 10)
        #expect(try await text(stream) == "12")
        #expect(stream.stopReason == .cancelled)
        #expect(stream.tokensGenerated == 2)
    }
}

// MARK: - Helpers

extension PipelinedConstrainedDecodingStrategy {
    /// Helper for testing session creation errors without full decode flow.
    fileprivate func getOrCreateSession(engine: MockConstrainedEngine) throws {
        _ = try engine.getOrCreateConstrainedSession(
            jsonSchema: jsonSchema,
            tokenizer: MockTokenizer(),
            vocabSize: 256,
            stopTokenIds: nil,
            maxWhitespaceCount: ConstrainedGenerationSession.defaultMaxWhitespaceCount
        )
    }

    fileprivate var jsonSchema: String {
        Mirror(reflecting: self).children.first(where: { $0.label == "jsonSchema" })?.value as? String ?? ""
    }
}

// MARK: - Session handle identity

@Suite("Constrained session handle identity")
struct ConstrainedSessionHandleIdentityTests {
    private static let schema = #"{"type": "object", "properties": {"a": {"type": "integer"}}}"#

    private func handle(limit: Int?, stops: [Int32]? = [2]) throws -> ConstrainedSessionHandle {
        let tokenizer = MockTokenizer()
        let session = try ConstrainedGenerationSession(
            jsonSchema: Self.schema, tokenizer: tokenizer, vocabSize: 256, vocabType: .raw,
            maxWhitespaceCount: limit)
        return ConstrainedSessionHandle(session: session, tokenizer: tokenizer, stopTokenIds: stops)
    }

    private func matches(
        _ h: ConstrainedSessionHandle, schema: String = Self.schema, limit: Int?, stops: [Int32]? = [2],
        vocab: Int = 256
    ) -> Bool {
        h.matches(schema: schema, maxWhitespaceCount: limit, stopTokenIds: stops, vocabularySize: vocab)
    }

    @Test("A handle matches only the schema, whitespace limit, stop tokens and vocabulary it was built with")
    func matchesSchemaAndLimit() throws {
        let h = try handle(limit: 32)
        #expect(matches(h, limit: 32))
        #expect(!matches(h, limit: 8))
        #expect(!matches(h, limit: nil))
        #expect(!matches(h, schema: "{}", limit: 32))
        let unlimited = try handle(limit: nil)
        #expect(matches(unlimited, limit: nil))
        #expect(!matches(unlimited, limit: 32))
    }

    @Test("Stop tokens decide the match, in any order, and so does the vocabulary")
    func matchesStopTokensAndVocabulary() throws {
        // A request's single-token `stop` strings are grammar stop tokens: a grammar built with "\n"
        // (10) among them ends at a newline, one built without it allows a newline inside the JSON.
        let withNewline = try handle(limit: 32, stops: [2, 10])
        #expect(matches(withNewline, limit: 32, stops: [10, 2]))
        #expect(!matches(withNewline, limit: 32, stops: [2]))
        #expect(!matches(withNewline, limit: 32, stops: nil))
        #expect(!matches(withNewline, limit: 32, stops: [2, 10], vocab: 300))
    }
}
