// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import FoundationModels
import Testing

@testable import CoreAILanguageModels

// `reasoningEffortString` is gated `@available(FoundationModels 2.0, *)`, so the test guards on it
// at runtime; the swift-testing `@Suite`/`@Test` macros reject a declaration-level `@available`.
@Suite("CoreAIExecutor.reasoningEffortString")
struct ReasoningEffortStringTests {
    private typealias Executor = CoreAILanguageModel.CoreAIExecutor

    @Test("maps reasoning levels to canonical effort strings")
    func mapping() {
        guard #available(FoundationModels 2.0, *) else { return }
        #expect(Executor.reasoningEffortString(from: .light) == "low")
        #expect(Executor.reasoningEffortString(from: .moderate) == "medium")
        #expect(Executor.reasoningEffortString(from: .deep) == "high")
        #expect(Executor.reasoningEffortString(from: .custom("aggressive")) == "aggressive")
        #expect(Executor.reasoningEffortString(from: nil) == nil)
    }
}
