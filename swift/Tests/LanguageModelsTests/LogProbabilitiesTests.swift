// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("LogProbabilities")
struct LogProbabilitiesTests {
    @Test("Uniform logits produce equal log probabilities")
    func uniformLogits() {
        let logits: [[LogitsScalarType]] = [[1.0, 1.0, 1.0, 1.0]]
        let targets: [Int32] = [0]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries.count == 1)
        let expected = -log(4.0)
        #expect(abs(result.entries[0].value - expected) < 1e-5)
    }

    @Test("Dominant logit gets near-zero log probability")
    func dominantLogit() {
        let logits: [[LogitsScalarType]] = [[100.0, 0.0, 0.0]]
        let targets: [Int32] = [0]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries[0].value > -0.01)
    }

    @Test("Suppressed logit gets very negative log probability")
    func suppressedLogit() {
        let logits: [[LogitsScalarType]] = [[100.0, 0.0, 0.0]]
        let targets: [Int32] = [1]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries[0].value < -90)
    }

    @Test("Sum of log probabilities across positions")
    func sumAcrossPositions() {
        let logits: [[LogitsScalarType]] = [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
        let targets: [Int32] = [0, 1]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries.count == 2)
        #expect(result.sum > -0.1)
    }

    @Test("Perplexity of uniform distribution")
    func perplexityUniform() {
        let vocabSize = 100
        let logits: [[LogitsScalarType]] = [Array(repeating: LogitsScalarType(1.0), count: vocabSize)]
        let targets: [Int32] = [42]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(abs(result.perplexity - Double(vocabSize)) < 0.5)
    }

    @Test("Top-K alternatives are sorted by probability")
    func topKSorted() {
        let logits: [[LogitsScalarType]] = [[5.0, 3.0, 1.0, 10.0, 0.0]]
        let targets: [Int32] = [2]
        let result = LogProbabilities.compute(logits: logits, targets: targets, topK: 3)

        let alts = result.entries[0].alternatives
        #expect(alts.count == 3)
        #expect(alts[0].tokenId == 3)
        #expect(alts[0].value > alts[1].value)
        #expect(alts[1].value > alts[2].value)
    }

    @Test("Out of bounds target token returns -infinity")
    func outOfBoundsTarget() {
        let logits: [[LogitsScalarType]] = [[1.0, 2.0, 3.0]]
        let targets: [Int32] = [999]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries[0].value == -.infinity)
    }

    @Test("Empty inputs produce empty result")
    func emptyInputs() {
        let result = LogProbabilities.compute(logits: [], targets: [])
        #expect(result.entries.isEmpty)
        #expect(result.sum == 0)
    }

    @Test("Mean and perplexity with no entries")
    func emptyMeanPerplexity() {
        let result = LogProbabilities.compute(logits: [], targets: [])
        #expect(result.mean == 0)
        #expect(result.perplexity == 1.0)
    }

    // MARK: - Numerical Robustness (Critical Fix #4)

    @Test("+Infinity logit does not produce NaN")
    func infinityLogitDoesNotNaN() {
        let logits: [[LogitsScalarType]] = [[LogitsScalarType.infinity, 0.0, 0.0]]
        let targets: [Int32] = [0]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(!result.entries[0].value.isNaN, "+Inf logit should not produce NaN")
        #expect(result.entries[0].value.isFinite, "+Inf logit target should get a finite log-prob")
        #expect(result.entries[0].value > -0.01, "+Inf logit should dominate (near 0 log-prob)")
    }

    @Test("-Infinity logit does not produce NaN")
    func negInfinityLogit() {
        let logits: [[LogitsScalarType]] = [[0.0, -LogitsScalarType.infinity, 1.0]]
        let targets: [Int32] = [1]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(!result.entries[0].value.isNaN, "-Inf logit should not produce NaN")
        #expect(result.entries[0].value < -100, "-Inf logit should be heavily suppressed")
    }

    @Test("Mixed infinity logits produce valid probabilities")
    func mixedInfinityLogits() {
        // When +Inf exists, the +Inf token dominates (log-prob = 0.0)
        // and all other tokens have probability 0 (log-prob = -Inf).
        let logits: [[LogitsScalarType]] = [[LogitsScalarType.infinity, -LogitsScalarType.infinity, 5.0]]

        // Target the +Inf token: should get log-prob 0.0
        let resultInf = LogProbabilities.compute(logits: logits, targets: [0])
        #expect(!resultInf.entries[0].value.isNaN)
        #expect(resultInf.entries[0].value == 0.0)

        // Target a non-Inf token: -Inf is valid (not NaN)
        let resultOther = LogProbabilities.compute(logits: logits, targets: [2])
        #expect(!resultOther.entries[0].value.isNaN)
        #expect(resultOther.entries[0].value == -.infinity)
    }

    @Test("Very large logits (near overflow) remain stable")
    func veryLargeLogits() {
        let big = LogitsScalarType(1e15)
        let logits: [[LogitsScalarType]] = [[big, big - 10, big - 20]]
        let targets: [Int32] = [0]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(!result.entries[0].value.isNaN)
        #expect(result.entries[0].value > -0.01, "Dominant large logit should be near 0")
    }

    @Test("Single-element vocabulary produces log-prob 0")
    func singleElement() {
        let logits: [[LogitsScalarType]] = [[42.0]]
        let targets: [Int32] = [0]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(abs(result.entries[0].value) < 1e-10, "Only token must have log-prob 0")
    }

    @Test("Negative target token returns -infinity")
    func negativeTargetToken() {
        let logits: [[LogitsScalarType]] = [[1.0, 2.0, 3.0]]
        let targets: [Int32] = [-1]
        let result = LogProbabilities.compute(logits: logits, targets: targets)

        #expect(result.entries[0].value == -.infinity)
    }
}
