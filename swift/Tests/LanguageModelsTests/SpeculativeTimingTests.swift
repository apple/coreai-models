// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Testing

@testable import CoreAILanguageModels

/// A drafter that takes a fixed time per `draftAll` and counts the calls.
private final class SlowDrafter: DrafterEngine, @unchecked Sendable {
    let inner: FuzzDrafter
    let delay: Duration
    private(set) var calls = 0
    var maxContextLength: Int { inner.maxContextLength }

    init(inner: FuzzDrafter, delay: Duration) {
        self.inner = inner
        self.delay = delay
    }

    func reset(to tokenIndex: Int) async throws { try await inner.reset(to: tokenIndex) }
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {
        try await inner.commit(confirmedLength: confirmedLength, features: features)
    }

    func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        calls += 1
        try await Task.sleep(for: delay)
        return try await inner.draftAll(
            confirmedPrefix: confirmedPrefix, count: count, policy: policy, using: &rng)
    }
}

@Suite("Speculative decoding timing stats")
struct SpeculativeTimingTests {
    private func run(delay: Duration) async throws -> (SpeculativeDecoder.Stats, drafterCalls: Int) {
        let model = HashModel(vocab: 64, levels: 1000, salt: 7)
        let drafter = SlowDrafter(inner: FuzzDrafter(model: model, seed: 3), delay: delay)
        let decoder = SpeculativeDecoder(
            drafter: drafter, target: HashEngine(model: model), numDraftTokens: 4)
        let stats = try await decoder.generateGreedy(
            promptTokens: [1, 2, 3], maxNewTokens: 40, stopTokenIds: [], onToken: { _ in })
        return (stats, drafter.calls)
    }

    @Test("Drafter time covers the drafter's calls, and verify time the target's passes")
    func splitsTheRound() async throws {
        let (stats, calls) = try await run(delay: .milliseconds(3))
        #expect(calls > 0)
        #expect(stats.draftSeconds >= 0.003 * Double(calls) * 0.9)
        #expect(stats.verifySeconds > 0)
        // The two are parts of the generation loop, which also rewinds, commits and emits tokens.
        #expect(stats.draftSeconds + stats.verifySeconds <= stats.generationSeconds + 0.001)
    }

    @Test("A round proposed by prompt lookup has no drafter time")
    func lookupRoundsHaveNoDrafterTime() async throws {
        let model = HashModel(vocab: 64, levels: 1000, salt: 7)
        let decoder = SpeculativeDecoder(
            drafter: PromptLookupDrafter(maxContextLength: 100_000), target: HashEngine(model: model),
            numDraftTokens: 4)
        let stats = try await decoder.generateGreedy(
            promptTokens: [1, 2, 3], maxNewTokens: 20, stopTokenIds: [], onToken: { _ in })
        #expect(stats.verifySeconds > 0)
        #expect(stats.draftSeconds < stats.verifySeconds)
    }
}
