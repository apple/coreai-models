// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("Runner option checks")
struct RunnerOptionChecksTests {
    @Test("Greedy speculative decoding is accepted with prompt lookup")
    func greedyAccepted() {
        #expect(SpeculativeDecodingOptions.samplingProblem(greedy: true, promptLookup: true) == nil)
    }

    @Test("Sampling is accepted when prompt lookup is off")
    func samplingWithoutPromptLookup() {
        #expect(SpeculativeDecodingOptions.samplingProblem(greedy: false, promptLookup: false) == nil)
    }

    @Test("Prompt lookup with sampling is rejected, naming --prompt-lookup")
    func promptLookupWithSampling() throws {
        let message = try #require(SpeculativeDecodingOptions.samplingProblem(greedy: false, promptLookup: true))
        #expect(message.contains("--prompt-lookup"))
    }

    @Test("An image with raw tokens is rejected")
    func imageWithRawTokens() throws {
        let message = try #require(SpeculativeDecodingOptions.imageProblem(hasImage: true, hasRawTokens: true))
        #expect(message.contains("--image"))
        #expect(message.contains("--raw-tokens"))
        #expect(SpeculativeDecodingOptions.imageProblem(hasImage: true, hasRawTokens: false) == nil)
        #expect(SpeculativeDecodingOptions.imageProblem(hasImage: false, hasRawTokens: true) == nil)
    }

    @Test("A prompt shorter than the context is accepted")
    func shortPrompt() {
        #expect(SpeculativeDecodingOptions.promptLengthProblem(promptTokens: 16383, contextLength: 16384) == nil)
        #expect(SpeculativeDecodingOptions.promptLengthProblem(promptTokens: 0, contextLength: 16384) == nil)
    }

    @Test("A prompt at or over the context length is rejected with both sizes")
    func longPrompt() throws {
        for n in [16384, 16390] {
            let message = try #require(
                SpeculativeDecodingOptions.promptLengthProblem(promptTokens: n, contextLength: 16384))
            #expect(message.contains("\(n)"))
            #expect(message.contains("16384"))
        }
    }
}
