// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Draft proposal and feature size validation")
struct DraftProposalValidationTests {
    @Test("A proposal with fewer logits rows than tokens is rejected")
    func proposalRejectsMismatchedLogits() async {
        await #expect(processExitsWith: .failure) {
            _ = DraftProposal(tokens: [1, 2], logits: [[0, 1]])
        }
    }

    @Test("A consistent proposal is accepted")
    func proposalAcceptsMatchingSizes() {
        let p = DraftProposal(tokens: [1, 2], logits: [[0], [1]])
        #expect(p.tokens.count == 2 && p.logits?.count == 2)
        #expect(DraftProposal(tokens: []).tokens.isEmpty)
    }

    @Test("Features with fewer values than positionCount * hiddenDim are rejected")
    func featuresRejectShortValues() async {
        await #expect(processExitsWith: .failure) {
            _ = DrafterFeatures(values: [1, 2, 3], positionCount: 2, hiddenDim: 2, startPosition: 0)
        }
    }

    @Test("A negative position count or width is rejected")
    func featuresRejectNegativeSizes() async {
        await #expect(processExitsWith: .failure) {
            _ = DrafterFeatures(values: [], positionCount: -1, hiddenDim: 2, startPosition: 0)
        }
        await #expect(processExitsWith: .failure) {
            _ = DrafterFeatures(values: [], positionCount: 1, hiddenDim: -2, startPosition: 0)
        }
    }
}
