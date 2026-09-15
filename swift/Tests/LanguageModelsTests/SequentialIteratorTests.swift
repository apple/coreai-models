// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Covers the pure-value iterator helpers both sequential engines share: the generation-length
/// clamp and next-token selection.
@Suite("SequentialIterator")
struct SequentialIteratorTests {
    @Test("clampMaxTokens: forced replays its own count, else clamps the request to remaining context")
    func clampMaxTokens() {
        // Forced continuation ignores the request and the context budget.
        #expect(
            SequentialIterator.clampMaxTokens(
                requested: 999, forcedCount: 5, inputCount: 100, maxContextLength: 128) == 5)
        // No request ("unbounded") clamps to the context left after the prompt.
        #expect(
            SequentialIterator.clampMaxTokens(
                requested: nil, forcedCount: nil, inputCount: 100, maxContextLength: 128) == 28)
        // A request smaller than the remaining context is honored as-is.
        #expect(
            SequentialIterator.clampMaxTokens(
                requested: 10, forcedCount: nil, inputCount: 100, maxContextLength: 128) == 10)
    }

    @Test("nextToken: replays the forced token, else returns the sampler's greedy argmax")
    func nextToken() {
        let logits: [LogitsScalarType] = [0.1, 0.2, 0.9, 0.3]
        // Forced continuation replays the token at `step`, ignoring the logits.
        #expect(
            SequentialIterator.nextToken(
                fromLogits: logits, forced: [7, 42], step: 1,
                sampling: SamplingConfiguration(temperature: 0), tokenHistory: []) == 42)
        // No forced tokens: greedy sampling (temperature 0) picks the argmax.
        #expect(
            SequentialIterator.nextToken(
                fromLogits: logits, forced: nil, step: 0,
                sampling: SamplingConfiguration(temperature: 0), tokenHistory: []) == 2)
    }
}
