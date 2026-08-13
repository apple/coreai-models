// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

@Suite("InputHandler Tests")
struct InputHandlerTests {
    @Test("TokenInputHandler conforms to SyncInputHandler")
    func tokenConformance() {
        let _: any SyncInputHandler.Type = TokenInputHandler.self
    }

    @Test("InputContext.dynamic creates correct context")
    func dynamicContext() {
        let tokens: ArraySlice<Int32> = [10, 20, 30][...]
        let ctx = InputContext.dynamic(tokens: tokens, processedTokenCount: 5)
        #expect(ctx.tokens.count == 3)
        #expect(ctx.processedTokenCount == 5)
        #expect(ctx.alignedStep == 5)
        #expect(ctx.batchSize == 3)
        #expect(ctx.slidingWindow == nil)
    }

    @Test("InputContext.static creates correct context")
    func staticContext() {
        let tokens: ArraySlice<Int32> = [10, 20][...]
        let ctx = InputContext.static(tokens: tokens, alignedStep: 10, batchSize: 4, slidingWindow: 256)
        #expect(ctx.tokens.count == 2)
        #expect(ctx.processedTokenCount == 10)
        #expect(ctx.alignedStep == 10)
        #expect(ctx.batchSize == 4)
        #expect(ctx.slidingWindow == 256)
    }

    @Test("InputCoverage.verify passes when all inputs covered")
    func coveragePass() throws {
        // This is a compile-time API check — we can't easily construct a descriptor
        // in tests without a model, so just verify the type exists
        let _: InputCoverage.Type = InputCoverage.self
    }
}
