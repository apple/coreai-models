// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The accept/reject rule (Chen et al. 2023) is what makes `--draft-model` sampling follow the
/// target's distribution. These tests fail if a draft is accepted regardless of the ratio
/// min(1, p / q).
@Suite("Speculative sampling rule")
struct SpeculativeSamplingRuleTests {
    private func decoder(target: [[Float]], draft: [[Float]], numDraft: Int) -> SpeculativeDecoder {
        SpeculativeDecoder(
            draft: TableEngine(table: draft),
            target: TableEngine(table: target), numDraftTokens: numDraft)
    }

    private func softmax(_ l: [Float]) -> [Double] {
        let m = l.max()!
        let e = l.map { exp(Double($0 - m)) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }

    private func chiSquare(observed: [Int], expected: [Double]) -> Double {
        let n = Double(observed.reduce(0, +))
        var x = 0.0
        for (o, p) in zip(observed, expected) where p > 0 {
            let e = n * p
            x += (Double(o) - e) * (Double(o) - e) / e
        }
        return x
    }

    /// Marginals of the first two emitted tokens against the target's chain.
    private func scenario(
        _ name: String, target: [[Float]], draft: [[Float]], numDraft: Int
    ) async throws {
        let v = target.count
        let trials = 20_000
        var first = [Int](repeating: 0, count: v)
        var second = [Int](repeating: 0, count: v)
        let decoder = decoder(target: target, draft: draft, numDraft: numDraft)
        for _ in 0..<trials {
            var got: [Int32] = []
            _ = try await decoder.generateSampling(
                promptTokens: [0], maxNewTokens: 2, stopTokenIds: [], temperature: 1,
                onToken: { got.append($0) })
            first[Int(got[0])] += 1
            second[Int(got[1])] += 1
        }
        let p0 = softmax(target[0])
        var p1 = [Double](repeating: 0, count: v)
        for t in 0..<v {
            let r = softmax(target[t])
            for u in 0..<v { p1[u] += p0[t] * r[u] }
        }
        // 5 degrees of freedom; 40 is far beyond the 1e-6 tail, so a correct rule essentially
        // never fails while a wrong one moves the marginals by far more.
        let c1 = chiSquare(observed: first, expected: p0)
        let c2 = chiSquare(observed: second, expected: p1)
        #expect(c1 < 40 && c2 < 40, "\(name): chi2 first \(c1) second \(c2); first \(first) vs \(p0)")
    }

    @Test("Draft equal to target: marginals follow the target")
    func sameDistribution() async throws {
        let t: [[Float]] = (0..<6).map { r in (0..<6).map { c in Float((r * 3 + c * 5) % 7) * 0.4 } }
        try await scenario("same", target: t, draft: t, numDraft: 3)
    }

    @Test("Draft peaked where the target is flat: rejections restore the target's marginals")
    func peakedDraft() async throws {
        let t: [[Float]] = (0..<6).map { r in (0..<6).map { c in Float((r + c) % 3) * 0.3 } }
        let d: [[Float]] = (0..<6).map { r in (0..<6).map { c in c == (r + 1) % 6 ? 6 : 0 } }
        try await scenario("peaked", target: t, draft: d, numDraft: 4)
    }

    @Test("Zero-probability entries on either side are handled")
    func zeroProbabilityEntries() async throws {
        let ninf = -Float.infinity
        let t: [[Float]] = (0..<6).map { r in (0..<6).map { c in c == (r + 2) % 6 ? ninf : Float(c) * 0.2 } }
        let d: [[Float]] = (0..<6).map { r in (0..<6).map { c in c % 2 == 0 ? ninf : Float(r) * 0.1 } }
        try await scenario("zeros", target: t, draft: d, numDraft: 3)
    }

    @Test("A draft token the target gives no probability is never emitted")
    func impossibleDraftIsRejected() async throws {
        // The draft always proposes token 3. The target forbids it everywhere.
        let ninf = -Float.infinity
        let target: [[Float]] = (0..<4).map { _ in [0, 0, 0, ninf] }
        let draft: [[Float]] = (0..<4).map { _ in [ninf, ninf, ninf, 0] }
        let decoder = decoder(target: target, draft: draft, numDraft: 3)
        for _ in 0..<300 {
            var got: [Int32] = []
            _ = try await decoder.generateSampling(
                promptTokens: [0], maxNewTokens: 6, stopTokenIds: [], temperature: 1,
                onToken: { got.append($0) })
            #expect(!got.contains(3), "emitted a token with zero target probability: \(got)")
        }
    }

    @Test("A draft token the target prefers more than the draft does is always accepted")
    func favouredDraftIsAccepted() async throws {
        // p(x) >= q(x) for the proposed token: acceptance probability 1, so no draft is rejected.
        let target: [[Float]] = (0..<3).map { _ in [0, 0, 20] }
        let draft: [[Float]] = (0..<3).map { _ in [0, 0, 0] }
        let decoder = decoder(target: target, draft: draft, numDraft: 3)
        var proposed = 0
        var accepted = 0
        for _ in 0..<200 {
            let stats = try await decoder.generateSampling(
                promptTokens: [0], maxNewTokens: 4, stopTokenIds: [], temperature: 1, onToken: { _ in })
            proposed += stats.draftTokensProposed
            accepted += stats.draftTokensAccepted
        }
        // The draft proposes token 2 with probability 1/3; every time it does, it is accepted,
        // and any other proposal is rejected almost surely (p is tiny there).
        #expect(accepted > 0)
        #expect(accepted < proposed)
    }

    @Test("Residual sampling excludes tokens the draft over-weights")
    func residualSampling() {
        var rng = SplitMix(1)
        // p = [0.5, 0.5, 0], q = [0, 0.5, 0.5]: residual relu(p - q) is all on token 0.
        for _ in 0..<200 {
            #expect(SpeculativeMath.sampleResidual(p: [0.5, 0.5, 0], q: [0, 0.5, 0.5], using: &rng) == 0)
        }
    }
}
