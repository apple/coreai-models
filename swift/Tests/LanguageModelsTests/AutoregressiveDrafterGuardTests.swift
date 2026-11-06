// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Autoregressive drafter guards")
struct AutoregressiveDrafterGuardTests {
    @Test("A confirmed prefix shorter than the drafter's committed tokens throws instead of trapping")
    func draftAllThrowsWhenPrefixIsBehind() async throws {
        let model = HashModel(vocab: 8, levels: 5, salt: 1)
        let engine = HashEngine(model: model)
        let drafter = AutoregressiveDrafter(engine: engine)
        var rng = SystemRandomNumberGenerator()
        _ = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy, using: &rng)
        #expect(engine.processedTokenCount >= 3)
        await #expect(throws: (any Error).self) {
            var rng = SystemRandomNumberGenerator()
            _ = try await drafter.draftAll(
                confirmedPrefix: [1], count: 2, policy: .greedy, using: &rng)
        }
        await #expect(throws: (any Error).self) {
            var rng = SystemRandomNumberGenerator()
            _ = try await drafter.draftAll(
                confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy, using: &rng)
        }
    }

    @Test("Asking for no drafts proposes nothing and leaves the draft engine untouched", arguments: [0, -3])
    func noDraftsRequested(count: Int) async throws {
        let engine = HashEngine(model: HashModel(vocab: 8, levels: 5, salt: 1))
        let drafter = AutoregressiveDrafter(engine: engine)
        var rng = SystemRandomNumberGenerator()
        let proposal = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3], count: count, policy: .greedy, using: &rng)
        #expect(proposal.tokens.isEmpty)
        #expect(engine.processedTokenCount == 0)
        // The same prefix can still be drafted from afterwards.
        let later = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy, using: &rng)
        #expect(later.tokens.count == 2)
    }

    @Test("A draft engine whose rewind lands elsewhere is reported, not drafted from")
    func draftEngineThatCannotRewind() async throws {
        let engine = HashEngine(model: HashModel(vocab: 8, levels: 5, salt: 1))
        engine.rewindsOnlyToZero = true
        let drafter = AutoregressiveDrafter(engine: engine)
        var rng = SystemRandomNumberGenerator()
        _ = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3, 4], count: 2, policy: .greedy, using: &rng)
        // The draft holds 5 tokens; rewinding to 3 clears the whole cache.
        await #expect(throws: InferenceRuntimeError.self) { try await drafter.reset(to: 3) }
    }

    @Test("Greedy drafting reads each step's argmax and never copies a logits row")
    func greedyDraftsAvoidTheLogitsCopy() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 2)
        let engine = CountingEngine(inner: HashEngine(model: model))
        let drafter = AutoregressiveDrafter(engine: engine)
        var rng = SystemRandomNumberGenerator()
        let greedy = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3], count: 4, policy: .greedy, using: &rng)
        #expect(engine.logitsCalls == 0)
        #expect(engine.argmaxCalls == 4)  // the seed token and the first three drafts
        #expect(greedy.logits == nil)

        // The drafts are the draft model's own greedy continuation.
        let reference = HashEngine(model: model)
        var expected: [Int32] = []
        var context: [Int32] = [1, 2, 3]
        try await reference.advance([1, 2])
        for _ in 0..<4 {
            let next = try await reference.forwardPerPositionArgmax([context.last!])[0]
            expected.append(next)
            context.append(next)
        }
        #expect(greedy.tokens == expected)

        // Sampling needs the rows: they are the proposal's `q` distributions.
        engine.resetCounts()
        try await drafter.reset(to: 0)
        let sampled = try await drafter.draftAll(
            confirmedPrefix: [1, 2, 3], count: 4, policy: .sampling(temperature: 1), using: &rng)
        #expect(engine.logitsCalls == 4)
        #expect(sampled.logits?.count == 4)
    }
}

/// Counts which verify primitive a drafter reads its draft model through.
private final class CountingEngine: SpeculativeEngine, @unchecked Sendable {
    let inner: HashEngine
    private(set) var logitsCalls = 0
    private(set) var argmaxCalls = 0
    init(inner: HashEngine) { self.inner = inner }

    func resetCounts() {
        logitsCalls = 0
        argmaxCalls = 0
    }

    var config: RowEngine.Config { inner.config }
    var processedTokenCount: Int { inner.processedTokenCount }
    func reset(to tokenIndex: Int) async throws { try await inner.reset(to: tokenIndex) }
    func advance(_ tokens: [Int32]) async throws { try await inner.advance(tokens) }
    func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32] {
        argmaxCalls += 1
        return try await inner.forwardPerPositionArgmax(tokens)
    }
    func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]] {
        logitsCalls += 1
        return try await inner.forwardWithPerPositionLogits(tokens)
    }
    func generate(
        with input: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> EmptyOutputSequence {
        throw InferenceRuntimeError.invalidState("CountingEngine implements only the speculative primitives")
    }
}
