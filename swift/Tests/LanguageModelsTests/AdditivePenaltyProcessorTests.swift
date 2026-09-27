// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Testing

@testable import CoreAILanguageModels

@Suite("AdditivePenaltyProcessor")
struct AdditivePenaltyProcessorTests {
    @Test("Frequency penalty subtracts count times penalty")
    func frequencyScalesWithCount() {
        var logits: [LogitsScalarType] = [LogitsScalarType(5.0), LogitsScalarType(5.0), 0]
        // Token 0 appears 3 times, token 1 appears once, penalty 0.5.
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [0, 0, 0, 1] as [Int32],
            frequencyPenalty: 0.5, presencePenalty: 0, logitBias: nil)
        #expect(abs(Float(logits[0]) - (5.0 - 1.5)) < 1e-2)  // 5 - 0.5*3
        #expect(abs(Float(logits[1]) - (5.0 - 0.5)) < 1e-2)  // 5 - 0.5*1
        #expect(Float(logits[2]) == 0)
    }

    @Test("Presence penalty subtracts once regardless of count")
    func presenceAppliedOnce() {
        var logits: [LogitsScalarType] = [LogitsScalarType(5.0), LogitsScalarType(5.0), 0]
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [0, 0, 0, 1] as [Int32],
            frequencyPenalty: 0, presencePenalty: 1.0, logitBias: nil)
        #expect(abs(Float(logits[0]) - 4.0) < 1e-2)  // 5 - 1.0 (once, despite 3 occurrences)
        #expect(abs(Float(logits[1]) - 4.0) < 1e-2)  // 5 - 1.0
        #expect(Float(logits[2]) == 0)  // not seen
    }

    @Test("Frequency and presence combine additively")
    func frequencyAndPresenceCombine() {
        var logits: [LogitsScalarType] = [LogitsScalarType(10.0), 0]
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [0, 0] as [Int32],
            frequencyPenalty: 0.5, presencePenalty: 1.0, logitBias: nil)
        // 10 - (0.5*2 + 1.0) = 8.0
        #expect(abs(Float(logits[0]) - 8.0) < 1e-2)
    }

    @Test("Logit bias adds to specified tokens")
    func logitBiasAdds() {
        var logits: [LogitsScalarType] = [LogitsScalarType(1.0), LogitsScalarType(2.0), LogitsScalarType(3.0)]
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [] as [Int32],
            frequencyPenalty: 0, presencePenalty: 0, logitBias: [0: 10.0, 2: -5.0])
        #expect(abs(Float(logits[0]) - 11.0) < 1e-2)
        #expect(Float(logits[1]) == 2.0)
        #expect(abs(Float(logits[2]) - (-2.0)) < 1e-2)
    }

    @Test("Out-of-range token ids are ignored")
    func outOfRangeIgnored() {
        var logits: [LogitsScalarType] = [LogitsScalarType(1.0), LogitsScalarType(2.0)]
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [-1, 5, 100] as [Int32],
            frequencyPenalty: 1.0, presencePenalty: 1.0, logitBias: [-1: 50.0, 9: 50.0])
        #expect(Float(logits[0]) == 1.0)
        #expect(Float(logits[1]) == 2.0)
    }

    @Test("Zero penalties with nil bias is a no-op")
    func zeroIsNoop() {
        var logits: [LogitsScalarType] = [LogitsScalarType(3.0), LogitsScalarType(-1.0)]
        let original = logits
        AdditivePenaltyProcessor.apply(
            to: &logits, recentTokenIds: [0, 1] as [Int32],
            frequencyPenalty: 0, presencePenalty: 0, logitBias: nil)
        #expect(logits == original)
    }

    @Test("Greedy fallbackSampler honors logit bias to ban a token")
    func greedyBiasBansToken() {
        // Token 0 has the highest logit; a large negative bias should push token 1 to win.
        let config = SamplingConfiguration(
            temperature: 0, logitBias: [0: -100.0])
        var logits: [LogitsScalarType] = [LogitsScalarType(5.0), LogitsScalarType(4.0), LogitsScalarType(1.0)]
        let token = config.fallbackSampler(from: &logits, tokenHistory: [] as [Int32])
        #expect(token == 1)
    }

    @Test("Greedy fallbackSampler honors frequency penalty via history")
    func greedyFrequencyShiftsArgmax() {
        // Token 0 leads by 1.0 but appeared twice; frequency 1.0 * 2 = 2.0 penalty flips it.
        let config = SamplingConfiguration(temperature: 0, frequencyPenalty: 1.0)
        var logits: [LogitsScalarType] = [LogitsScalarType(5.0), LogitsScalarType(4.0), LogitsScalarType(1.0)]
        let token = config.fallbackSampler(from: &logits, tokenHistory: [0, 0] as [Int32])
        #expect(token == 1)
    }
}
