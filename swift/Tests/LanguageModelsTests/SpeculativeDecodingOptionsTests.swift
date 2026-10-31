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
}
