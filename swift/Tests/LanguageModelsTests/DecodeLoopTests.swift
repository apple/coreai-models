// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Covers the decode loop the sequential and static-shape engines share: the generation-length
/// clamp, next-token selection, and the per-engine sampling policy.
@Suite("DecodeLoop")
struct DecodeLoopTests {
    @Test("clampMaxTokens: forced replays its own count, else clamps the request to remaining context")
    func clampMaxTokens() {
        // Forced continuation ignores the request and the context budget.
        #expect(
            DecodeLoop.clampMaxTokens(
                requested: 999, forcedCount: 5, inputCount: 100, maxContextLength: 128) == 5)
        // No request ("unbounded") clamps to the context left after the prompt.
        #expect(
            DecodeLoop.clampMaxTokens(
                requested: nil, forcedCount: nil, inputCount: 100, maxContextLength: 128) == 28)
        // A request smaller than the remaining context is honored as-is.
        #expect(
            DecodeLoop.clampMaxTokens(
                requested: 10, forcedCount: nil, inputCount: 100, maxContextLength: 128) == 10)
    }

    @Test("nextToken: replays the forced token, else returns the sampler's greedy argmax")
    func nextToken() {
        let logits: [LogitsScalarType] = [0.1, 0.2, 0.9, 0.3]
        // Forced continuation replays the token at `step`, ignoring the logits.
        #expect(
            DecodeLoop.nextToken(
                fromLogits: logits, forced: [7, 42], step: 1,
                sampling: SamplingConfiguration(temperature: 0), tokenHistory: []) == 42)
        // No forced tokens: greedy sampling (temperature 0) picks the argmax.
        #expect(
            DecodeLoop.nextToken(
                fromLogits: logits, forced: nil, step: 0,
                sampling: SamplingConfiguration(temperature: 0), tokenHistory: []) == 2)
    }

    @Test("a soft-cap policy caps the returned logits and samples from the capped values")
    func softcapPolicy() async throws {
        let raw: [LogitsScalarType] = [1, 30, 20]
        let cap = 10.0
        var loop = DecodeLoop(
            input: [5],
            samplingConfiguration: SamplingConfiguration(temperature: 0),
            samplingPolicy: SamplingPolicy(logitSoftcap: cap, normalizesSampling: false),
            inferenceOptions: InferenceOptions(maxTokens: 1, logits: .lastPosition),
            maxContextLength: 16,
            stopReasonStore: StopReasonStore(),
            generationToken: GenerationToken(),
            release: {}
        )
        let output = try #require(try await loop.decodeStep { _ in raw })
        var capped = raw
        LogitSoftcapProcessor.apply(to: &capped, cap: Float(cap))
        #expect(output.logits == capped)
        #expect(output.tokenId == 1)
    }
}
