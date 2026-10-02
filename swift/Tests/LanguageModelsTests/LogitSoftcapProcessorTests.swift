// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The Gemma 4 iOS graph emits *uncapped* logits, so the runner applies
/// `c · tanh(logits / c)` on the CPU before sampling.
@Suite("LogitSoftcapProcessor")
struct LogitSoftcapProcessorTests {
    @Test("Matches c·tanh(logits/c) elementwise")
    func matchesReference() {
        let cap: Float = 30
        var logits: [LogitsScalarType] = [-100, -30, -7.5, -1, 0, 1, 7.5, 30, 100]
        let inputs = logits.map { Double($0) }

        LogitSoftcapProcessor.apply(to: &logits, cap: cap)

        for (actual, input) in zip(logits, inputs) {
            // Tolerance covers the round back to LogitsScalarType (Float16 on ARM),
            // whose spacing near 30 is ~0.016.
            #expect(abs(Double(actual) - tanh(input / Double(cap)) * Double(cap)) < 0.02)
        }
    }

    @Test("fallbackSampler caps before the repetition penalty")
    func fallbackSamplerCapsFirst() {
        // Uncapped, the penalty (÷2) leaves token 0 on top: 100 / 2 = 50 > 40. Capped at
        // 30 first, 30·tanh(100/30) / 2 ≈ 15 < 30·tanh(40/30) ≈ 26, so token 1 wins.
        let config = SamplingConfiguration(temperature: 0, repetitionPenalty: 2.0)
        var logits: [LogitsScalarType] = [100, 40, 0]
        let token = config.fallbackSampler(from: &logits, tokenHistory: [0] as [Int32], logitSoftcap: 30)
        #expect(token == 1)

        var uncapped: [LogitsScalarType] = [100, 40, 0]
        #expect(config.fallbackSampler(from: &uncapped, tokenHistory: [0] as [Int32]) == 0)
    }
}
