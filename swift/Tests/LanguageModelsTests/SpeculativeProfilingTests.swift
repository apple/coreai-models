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

private struct Failure: Error {}

/// A greedy run whose drafter takes 3 ms per call, off the main actor like the decoder's callers.
private func runWithSlowDrafter() async throws -> (SpeculativeDecoder.Stats, drafterCalls: Int) {
    let model = HashModel(vocab: 64, levels: 1000, salt: 7)
    let drafter = SlowDrafter(inner: FuzzDrafter(model: model, seed: 3), delay: .milliseconds(3))
    let decoder = SpeculativeDecoder(drafter: drafter, target: HashEngine(model: model), numDraftTokens: 4)
    let stats = try await decoder.generateGreedy(
        promptTokens: [1, 2, 3], maxNewTokens: 40, stopTokenIds: [], onToken: { _ in })
    return (stats, drafter.calls)
}

/// Spans reach the shared statistics through a task, and other suites record into the same storage
/// concurrently, so these tests wait for each count to grow by at least what they caused.
@MainActor
@Suite("Speculative decoding spans")
struct SpeculativeProfilingTests {
    private let storage = StatsStorage.shared

    private func counts() -> [SignpostCategory: Int] {
        let categories: [SignpostCategory] = [.prompt, .extend, .speculativeDraft, .speculativeVerify]
        return Dictionary(uniqueKeysWithValues: categories.map { ($0, storage.count(for: $0)) })
    }

    private func waitUntil(_ condition: () -> Bool) async {
        for _ in 0..<300 where !condition() { try? await Task.sleep(for: .milliseconds(10)) }
    }

    @Test("A run records one prompt span and one loop span, and a draft and a verify span per round")
    func spansOfARun() async throws {
        let before = counts()
        let draftTimeBefore = storage.totalDuration(for: .speculativeDraft)
        let (stats, drafted) = try await runWithSlowDrafter()

        #expect(drafted > 0 && stats.iterations >= drafted)
        await waitUntil {
            let now = counts()
            return now[.speculativeDraft]! >= before[.speculativeDraft]! + drafted
                && now[.speculativeVerify]! >= before[.speculativeVerify]! + stats.iterations
                && now[.prompt]! >= before[.prompt]! + 1 && now[.extend]! >= before[.extend]! + 1
        }
        let now = counts()
        #expect(now[.speculativeDraft]! >= before[.speculativeDraft]! + drafted)
        #expect(now[.speculativeVerify]! >= before[.speculativeVerify]! + stats.iterations)
        #expect(now[.prompt]! >= before[.prompt]! + 1)
        #expect(now[.extend]! >= before[.extend]! + 1)
        // The drafter slept 3 ms in each of its calls, all of which the draft spans cover.
        #expect(storage.totalDuration(for: .speculativeDraft) - draftTimeBefore >= 0.003 * Double(drafted) * 0.9)
    }

    @Test("profile returns the body's value and ends the span")
    func profileReturns() async {
        let before = storage.count(for: .speculativeDraft)
        let value = await InstrumentsProfiler.profile(InstrumentsProfiler.beginSpeculativeDraft(tokens: 1)) { 42 }
        #expect(value == 42)
        await waitUntil { storage.count(for: .speculativeDraft) >= before + 1 }
        #expect(storage.count(for: .speculativeDraft) >= before + 1)
    }

    @Test("profile ends the span when the body throws, and rethrows")
    func profileEndsOnThrow() async {
        let before = storage.count(for: .speculativeVerify)
        await #expect(throws: Failure.self) {
            try await InstrumentsProfiler.profile(InstrumentsProfiler.beginSpeculativeVerify(tokens: 1)) {
                throw Failure()
            }
        }
        await waitUntil { storage.count(for: .speculativeVerify) >= before + 1 }
        #expect(storage.count(for: .speculativeVerify) >= before + 1)
    }
}
