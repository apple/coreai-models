// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The accept/reject sampling path draws from the sampler's softmax and multinomial, so these
/// pin that the shared pieces behave as the speculative decoder needs them to.
@Suite("Speculative decoding math")
struct SpeculativeMathTests {
    private func logits(_ values: [Double]) -> [LogitsScalarType] { values.map { LogitsScalarType($0) } }

    @Test("argmax picks the largest logit and tolerates an empty array")
    func argmaxPicksLargest() {
        #expect(SpeculativeMath.argmax(logits([0.5, 3, -1, 2.5])) == 1)
        #expect(SpeculativeMath.argmax(logits([-4, -3, -2])) == 2)
        #expect(SpeculativeMath.argmax([]) == 0)
    }

    @Test("A temperature too small for Float is the greedy limit, not a division by zero")
    func temperatureBelowFloatRange() {
        let probs = SpeculativeMath.temperatureSoftmax(logits([0, 3, 1, 3]), temperature: 1e-50)
        #expect(probs == [0, 1, 0, 0])  // one-hot at the first maximum
        #expect(probs.allSatisfy { $0.isFinite })
    }

    @Test("temperatureSoftmax matches a Double reference and sums to one", arguments: [0.5, 1.0, 2.0])
    func softmaxMatchesReference(temperature: Double) {
        let values: [Double] = [2, -1, 0.5, 3, 3, -6, 0]
        let probs = SpeculativeMath.temperatureSoftmax(logits(values), temperature: temperature)
        let scaled = values.map { $0 / temperature }
        let peak = scaled.max()!
        let exps = scaled.map { exp($0 - peak) }
        let total = exps.reduce(0, +)
        #expect(probs.count == values.count)
        for (p, e) in zip(probs, exps) { #expect(abs(Double(p) - e / total) < 1e-5) }
        #expect(abs(probs.reduce(0, +) - 1) < 1e-5)
    }

    @Test("A lower temperature sharpens the distribution")
    func lowerTemperatureSharpens() {
        let values = logits([1, 2, 3])
        let warm = SpeculativeMath.temperatureSoftmax(values, temperature: 2)
        let cold = SpeculativeMath.temperatureSoftmax(values, temperature: 0.5)
        #expect(cold[2] > warm[2])
    }

    @Test("softmax of an empty array is empty, and masked (-inf) entries get no mass")
    func softmaxEdgeCases() {
        #expect(SpeculativeMath.temperatureSoftmax([], temperature: 1).isEmpty)
        let masked = SpeculativeMath.temperatureSoftmax(
            [LogitsScalarType(1), -LogitsScalarType.infinity, LogitsScalarType(1)], temperature: 1)
        #expect(masked[1] == 0)
        #expect(abs(masked[0] - 0.5) < 1e-5)
    }

    @Test("sampleFromProbs is reproducible under a seeded generator and respects a point mass")
    func sampleFromProbsIsSeeded() {
        let probs: [Float] = [0.1, 0.2, 0.3, 0.4]
        var a = SeededRandomNumberGenerator(seed: 7)
        var b = SeededRandomNumberGenerator(seed: 7)
        let draws = (0..<64).map { _ in SpeculativeMath.sampleFromProbs(probs, using: &a) }
        #expect(draws == (0..<64).map { _ in SpeculativeMath.sampleFromProbs(probs, using: &b) })
        var rng = SeededRandomNumberGenerator(seed: 1)
        #expect((0..<32).allSatisfy { _ in SpeculativeMath.sampleFromProbs([0, 0, 1, 0], using: &rng) == 2 })
        #expect(SpeculativeMath.sampleFromProbs([], using: &rng) == 0)
    }

    @Test("sampleFromProbs follows the distribution")
    func sampleFromProbsFollowsDistribution() {
        let probs: [Float] = [0.1, 0.2, 0.3, 0.4]
        var rng = SeededRandomNumberGenerator(seed: 42)
        var counts = [Int](repeating: 0, count: probs.count)
        let n = 20_000
        for _ in 0..<n { counts[Int(SpeculativeMath.sampleFromProbs(probs, using: &rng))] += 1 }
        for (count, p) in zip(counts, probs) { #expect(abs(Double(count) / Double(n) - Double(p)) < 0.02) }
    }

    @Test("sampleResidual draws from norm(relu(p − q)) and never from a slot where p ≤ q")
    func residualFollowsDistribution() {
        let p: [Float] = [0.5, 0.1, 0.3, 0.1]
        let q: [Float] = [0.2, 0.4, 0.1, 0.3]
        // relu(p − q) = [0.3, 0, 0.2, 0] → [0.6, 0, 0.4, 0]
        var rng = SeededRandomNumberGenerator(seed: 9)
        var counts = [Int](repeating: 0, count: 4)
        let n = 20_000
        for _ in 0..<n { counts[Int(SpeculativeMath.sampleResidual(p: p, q: q, using: &rng))] += 1 }
        #expect(counts[1] == 0 && counts[3] == 0)
        #expect(abs(Double(counts[0]) / Double(n) - 0.6) < 0.02)
        #expect(abs(Double(counts[2]) / Double(n) - 0.4) < 0.02)
    }

    @Test("A degenerate residual (p ≤ q everywhere) falls back to p, and empty inputs do not trap")
    func residualDegenerateCases() {
        var rng = SeededRandomNumberGenerator(seed: 3)
        let p: [Float] = [0, 1, 0]
        #expect(SpeculativeMath.sampleResidual(p: p, q: [0.5, 1, 0.5], using: &rng) == 1)
        #expect(SpeculativeMath.sampleResidual(p: [], q: [], using: &rng) == 0)
    }
}
