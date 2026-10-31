// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Prefix-cache routing on a vision-language model: the token match against history is cut at the
/// first image placeholder, then the text plan (`prefixResetPlan`) decides the rewind.
@Suite("VLM prefix-cache policy")
struct VLMResetPlanTests {
    private let image: Int32 = 9

    /// `count` tokens with image placeholders at `imageAt`.
    private func tokens(_ count: Int, imageAt positions: Range<Int>? = nil) -> [Int32] {
        (0..<count).map { positions?.contains($0) == true ? image : Int32(100 + $0) }
    }

    private func plan(
        commonPrefix: Int, tokens: [Int32], historyCount: Int, processedTokenCount: Int
    ) -> (resetTo: Int?, prefixHitCount: Int) {
        CoreAISequentialEngine.prefixResetPlan(
            commonPrefix: CoreAISequentialEngine.reusablePrefix(
                commonPrefix: commonPrefix, tokens: tokens, imageTokenId: image),
            historyCount: historyCount,
            processedTokenCount: processedTokenCount,
            inputCount: tokens.count,
            hasNonTruncatableStates: false)
    }

    @Test("the token match is cut at the first image placeholder it covers")
    func cutsAtFirstImage() {
        let input = tokens(30, imageAt: 5..<9)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 20, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 7, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 5, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 4, tokens: input, imageTokenId: image) == 4)
    }

    @Test("a text model, or a match with no placeholders, is not cut")
    func noCutWithoutImages() {
        let input = tokens(30, imageAt: 5..<9)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 20, tokens: input, imageTokenId: nil) == 20)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 20, tokens: tokens(30), imageTokenId: image) == 20)
    }

    @Test("the same image extended with new text full-resets")
    func sameImageExtensionResets() {
        let p = plan(commonPrefix: 20, tokens: tokens(30, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 0)
        #expect(p.prefixHitCount == 5)
    }

    @Test("an identical image re-request full-resets instead of rewinding one token")
    func identicalImageRequestResets() {
        let p = plan(commonPrefix: 20, tokens: tokens(20, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 0)
        #expect(p.prefixHitCount == 5)
    }

    @Test("an image after the cached text reuses the text")
    func imageAfterCachedTextReuses() {
        let p = plan(commonPrefix: 10, tokens: tokens(30, imageAt: 12..<16), historyCount: 10, processedTokenCount: 10)
        #expect(p.resetTo == nil)
        #expect(p.prefixHitCount == 10)
    }

    @Test("an image at position 0 never reuses")
    func leadingImageNeverReuses() {
        let p = plan(commonPrefix: 20, tokens: tokens(30, imageAt: 0..<4), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 0)
        #expect(p.prefixHitCount == 0)
    }

    @Test("text-only requests follow the text engine's plan")
    func textOnlyMatchesTextEngine() {
        let cases: [(commonPrefix: Int, historyCount: Int, processedTokenCount: Int, inputCount: Int)] = [
            (10, 10, 10, 15),  // pure extension with new tokens
            (10, 10, 10, 10),  // identical re-request: rewind one to reseed
            (4, 10, 10, 12),  // divergence
            (10, 12, 12, 10),  // input is a strict prefix of history
        ]
        for c in cases {
            let vlm = plan(
                commonPrefix: c.commonPrefix, tokens: tokens(c.inputCount), historyCount: c.historyCount,
                processedTokenCount: c.processedTokenCount)
            let text = CoreAISequentialEngine.prefixResetPlan(
                commonPrefix: c.commonPrefix, historyCount: c.historyCount,
                processedTokenCount: c.processedTokenCount, inputCount: c.inputCount,
                hasNonTruncatableStates: false)
            #expect(vlm.resetTo == text.resetTo)
            #expect(vlm.prefixHitCount == text.prefixHitCount)
        }
    }
}
