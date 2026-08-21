// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Testing

@testable import CoreAILanguageModels

@Suite("RepetitionPenaltyProcessor")
struct RepetitionPenaltyProcessorTests {
    @Test("Positive logits are divided by penalty")
    func positiveLogitsDivided() {
        var logits: [LogitsScalarType] = [0, 0, LogitsScalarType(2.0), 0, 0]
        RepetitionPenaltyProcessor.apply(to: &logits, recentTokenIds: [2], penalty: 2.0)
        #expect(abs(Float(logits[2]) - 1.0) < 1e-3)
    }

    @Test("Negative logits are multiplied by penalty")
    func negativeLogitsMultiplied() {
        var logits: [LogitsScalarType] = [0, 0, LogitsScalarType(-2.0), 0, 0]
        RepetitionPenaltyProcessor.apply(to: &logits, recentTokenIds: [2], penalty: 2.0)
        #expect(abs(Float(logits[2]) - (-4.0)) < 1e-2)
    }

    @Test("Zero logits unchanged")
    func zeroLogitsUnchanged() {
        var logits: [LogitsScalarType] = [0, 0, 0, 0, 0]
        RepetitionPenaltyProcessor.apply(to: &logits, recentTokenIds: [0, 1, 2, 3, 4], penalty: 1.5)
        for l in logits {
            #expect(l == 0)
        }
    }

    @Test("Penalty of 1.0 is a no-op")
    func penaltyOneIsNoop() {
        var logits: [LogitsScalarType] = [LogitsScalarType(3.0), LogitsScalarType(-1.0)]
        let original = logits
        RepetitionPenaltyProcessor.apply(to: &logits, recentTokenIds: [0, 1], penalty: 1.0)
        #expect(logits == original)
    }

    @Test("Duplicate token IDs penalized only once")
    func deduplication() {
        var logits1: [LogitsScalarType] = [LogitsScalarType(4.0), 0, 0]
        var logits2: [LogitsScalarType] = [LogitsScalarType(4.0), 0, 0]
        RepetitionPenaltyProcessor.apply(to: &logits1, recentTokenIds: [0], penalty: 2.0)
        RepetitionPenaltyProcessor.apply(to: &logits2, recentTokenIds: [0, 0, 0], penalty: 2.0)
        #expect(logits1[0] == logits2[0])
    }

    @Test("Out-of-range token IDs are ignored")
    func outOfRangeIgnored() {
        var logits: [LogitsScalarType] = [LogitsScalarType(1.0), LogitsScalarType(2.0)]
        RepetitionPenaltyProcessor.apply(to: &logits, recentTokenIds: [-1, 5, 100], penalty: 2.0)
        #expect(Float(logits[0]) == 1.0)
        #expect(Float(logits[1]) == 2.0)
    }

    @Test("fallbackSampler with tokenHistory applies penalty")
    func fallbackSamplerWithHistory() {
        let config = SamplingConfiguration(temperature: 0, repetitionPenalty: 2.0)
        // Token 0 has highest logit but is penalized; token 1 should win
        var logits: [LogitsScalarType] = [LogitsScalarType(3.0), LogitsScalarType(2.0), LogitsScalarType(1.0)]
        let token = config.fallbackSampler(from: &logits, tokenHistory: [0] as [Int32])
        #expect(token == 1)
    }

    @Test("Window limits which tokens are penalized")
    func windowLimitsScope() {
        let config = SamplingConfiguration(temperature: 0, repetitionPenalty: 2.0, repetitionPenaltyWindow: 1)
        // History: [0, 1] but window=1, so only token 1 is penalized
        var logits: [LogitsScalarType] = [LogitsScalarType(2.0), LogitsScalarType(3.0), LogitsScalarType(1.0)]
        let token = config.fallbackSampler(from: &logits, tokenHistory: [0, 1] as [Int32])
        // Token 1 (3.0/2.0=1.5) penalized, token 0 (2.0) not penalized → token 0 wins
        #expect(token == 0)
    }
}
