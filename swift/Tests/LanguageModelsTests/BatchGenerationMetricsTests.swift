// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("BatchGenerationMetrics")
struct BatchGenerationMetricsTests {
    private func result(tokens: Int, prompt: Int = 100, ttft: Double, decode: Double) -> BatchGenerationResult {
        BatchGenerationResult(
            tokens: Array(repeating: 1, count: tokens), stopReason: .maxTokens, promptTokenCount: prompt,
            timeToFirstToken: ttft, decodeSeconds: decode)
    }

    @Test("Rates divide by the span the requests ran, never sum each request's own rate")
    func ratesOverTheSpan() {
        // Two rows, three requests: the third starts when the first finishes.
        let results = [
            result(tokens: 11, ttft: 1, decode: 4),  // first tokens 1...5
            result(tokens: 11, ttft: 2, decode: 4),  // 2...6
            result(tokens: 11, ttft: 5, decode: 4),  // 5...9
        ]
        let metrics = BatchGenerationMetrics(results: results, wallSeconds: 10, rows: 2)
        #expect(metrics.generatedTokens == 33)
        // 30 tokens after each first, over 1...9.
        #expect(metrics.decodeTokensPerSecond == 30.0 / 8)
        // The first two requests' 200 prompt tokens, by the second's first token at 2 s.
        #expect(metrics.promptTokensPerSecond == 100)
        #expect(metrics.meanRequestTokensPerSecond == 2.5)
        #expect(metrics.meanTimeToFirstToken == 8.0 / 3)
        #expect(metrics.endToEndTokensPerSecond == 3.3)
    }

    @Test("Replies too short for a rate are left out of the per-request mean")
    func shortRepliesHaveNoRate() {
        let results = [result(tokens: 5, ttft: 1, decode: 2), result(tokens: 1, ttft: 1, decode: 0)]
        #expect(results[1].tokensPerSecond == nil)
        #expect(BatchGenerationMetrics(results: results, wallSeconds: 3, rows: 2).meanRequestTokensPerSecond == 2)
    }

    @Test("A reply's EOS leaves its tokens and its span; an EOS-only reply keeps its TTFT")
    func eosLeavesTheReply() {
        let started = SuspendingClock.now
        let at = { (seconds: Double) in started + .milliseconds(Int(seconds * 1000)) }
        let ended = RowResult(tokens: [7, 8, 9, 2], stopReason: .eos, tokenTimes: [at(1), at(2), at(3), at(4)])
        let reply = BatchGenerationResult(ended, promptTokenCount: 10, started: started)
        #expect(reply.tokens == [7, 8, 9])
        #expect(reply.timeToFirstToken == 1)
        #expect(reply.decodeSeconds == 2)

        let onlyEOS = BatchGenerationResult(
            RowResult(tokens: [2], stopReason: .eos, tokenTimes: [at(1.5)]), promptTokenCount: 10, started: started)
        #expect(onlyEOS.tokens.isEmpty)
        #expect(onlyEOS.timeToFirstToken == 1.5)
        #expect(onlyEOS.decodeSeconds == 0)
        #expect(onlyEOS.tokensPerSecond == nil)
    }
}
