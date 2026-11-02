// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("Speculative decoding option checks")
struct SpeculativeDecodingOptionsTests {
    @Test("An unset draft count is accepted")
    func unsetDraftCount() {
        #expect(SpeculativeDecodingOptions.draftCountProblem(nil) == nil)
    }

    @Test("A draft count from 1 up to the limit is accepted")
    func validDraftCounts() {
        #expect(SpeculativeDecodingOptions.draftCountProblem(1) == nil)
        #expect(SpeculativeDecodingOptions.draftCountProblem(7) == nil)
        #expect(SpeculativeDecodingOptions.draftCountProblem(SpeculativeDefaults.maxDraftTokens) == nil)
    }

    @Test("A zero or negative draft count is rejected with the accepted range")
    func nonPositiveDraftCount() throws {
        for n in [0, -1, -100] {
            let message = try #require(SpeculativeDecodingOptions.draftCountProblem(n))
            #expect(message.contains("--num-draft-tokens"))
            #expect(message.contains("1"))
            #expect(message.contains("\(SpeculativeDefaults.maxDraftTokens)"))
        }
    }

    @Test("A draft count above the limit is rejected")
    func absurdDraftCount() throws {
        let message = try #require(
            SpeculativeDecodingOptions.draftCountProblem(SpeculativeDefaults.maxDraftTokens + 1))
        #expect(message.contains("--num-draft-tokens"))
        #expect(SpeculativeDecodingOptions.draftCountProblem(Int.max) != nil)
    }

    @Test("Options that need another generation mode are named, the first one first")
    func unsupportedOptions() {
        #expect(SpeculativeDecodingOptions.unsupportedProblem(used: []) == nil)
        #expect(
            SpeculativeDecodingOptions.unsupportedProblem(used: ["--video", "--continuation"])
                == "--video cannot be used with speculative decoding")
    }

    @Test("Ignored options are listed in one warning, and nothing is said when there are none")
    func ignoredOptions() throws {
        #expect(SpeculativeDecodingOptions.ignoredWarning(ignored: []) == nil)
        let warning = try #require(
            SpeculativeDecodingOptions.ignoredWarning(ignored: ["--top-k", "--chunk-size"]))
        #expect(warning.contains("--top-k, --chunk-size"))
        #expect(warning.hasPrefix("Warning: speculative decoding ignores"))
    }

    @Test("A schema with speculative decoding needs greedy decoding")
    func grammarNeedsGreedy() throws {
        #expect(SpeculativeDecodingOptions.grammarSamplingProblem(greedy: true, hasSchema: true) == nil)
        #expect(SpeculativeDecodingOptions.grammarSamplingProblem(greedy: false, hasSchema: false) == nil)
        let message = try #require(
            SpeculativeDecodingOptions.grammarSamplingProblem(greedy: false, hasSchema: true))
        #expect(message.contains("--json-schema"))
        #expect(message.contains("greedy"))
    }

    @Test("The grammar proposal limit is checked against its range and where it applies")
    func grammarProposalLimit() throws {
        let top = GrammarConstraint.maxForcedRunLimit
        #expect(SpeculativeDecodingOptions.grammarProposalProblem(nil, speculating: false, hasSchema: false) == nil)
        for limit in [0, 7, top] {
            #expect(
                SpeculativeDecodingOptions.grammarProposalProblem(limit, speculating: true, hasSchema: true) == nil)
        }
        for limit in [-1, top + 1] {
            let message = try #require(
                SpeculativeDecodingOptions.grammarProposalProblem(limit, speculating: true, hasSchema: true))
            #expect(message.contains("--grammar-proposal-max must be between 0 and \(top)"))
        }
        #expect(SpeculativeDecodingOptions.grammarProposalProblem(7, speculating: false, hasSchema: true) != nil)
        #expect(SpeculativeDecodingOptions.grammarProposalProblem(7, speculating: true, hasSchema: false) != nil)
    }
}
