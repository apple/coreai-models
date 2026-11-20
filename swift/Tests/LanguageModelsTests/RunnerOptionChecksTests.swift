// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("Runner option checks")
struct RunnerOptionChecksTests {
    @Test("--speculative excludes --draft-model, and --drafter-top-k needs it and a positive count")
    func draftHeadOptions() throws {
        #expect(
            SpeculativeDecodingOptions.draftHeadProblem(draftHead: true, draftModel: false, drafterTopK: nil) == nil)
        #expect(
            SpeculativeDecodingOptions.draftHeadProblem(draftHead: false, draftModel: true, drafterTopK: nil) == nil)
        let both = try #require(
            SpeculativeDecodingOptions.draftHeadProblem(draftHead: true, draftModel: true, drafterTopK: nil))
        #expect(both.contains("--speculative") && both.contains("--draft-model"))

        #expect(SpeculativeDecodingOptions.draftHeadProblem(draftHead: true, draftModel: false, drafterTopK: 16) == nil)
        let alone = try #require(
            SpeculativeDecodingOptions.draftHeadProblem(draftHead: false, draftModel: false, drafterTopK: 16))
        #expect(alone.contains("--drafter-top-k") && alone.contains("--speculative"))
        for bad in [0, -3] {
            let message = try #require(
                SpeculativeDecodingOptions.draftHeadProblem(draftHead: true, draftModel: false, drafterTopK: bad))
            #expect(message.contains("--drafter-top-k must be >= 1"))
        }
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
