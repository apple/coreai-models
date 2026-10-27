// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("Speculative decoding option checks")
struct SpeculativeOptionsTests {
    @Test("An unset draft count is accepted")
    func unsetDraftCount() {
        #expect(SpeculativeOptions.draftCountProblem(nil) == nil)
    }

    @Test("A draft count from 1 up to the limit is accepted")
    func validDraftCounts() {
        #expect(SpeculativeOptions.draftCountProblem(1) == nil)
        #expect(SpeculativeOptions.draftCountProblem(7) == nil)
        #expect(SpeculativeOptions.draftCountProblem(SpeculativeDefaults.maxDraftTokens) == nil)
    }

    @Test("A zero or negative draft count is rejected with the accepted range")
    func nonPositiveDraftCount() throws {
        for n in [0, -1, -100] {
            let message = try #require(SpeculativeOptions.draftCountProblem(n))
            #expect(message.contains("--num-draft-tokens"))
            #expect(message.contains("1"))
            #expect(message.contains("\(SpeculativeDefaults.maxDraftTokens)"))
        }
    }

    @Test("A draft count above the limit is rejected")
    func absurdDraftCount() throws {
        let message = try #require(
            SpeculativeOptions.draftCountProblem(SpeculativeDefaults.maxDraftTokens + 1))
        #expect(message.contains("--num-draft-tokens"))
        #expect(SpeculativeOptions.draftCountProblem(Int.max) != nil)
    }
}
