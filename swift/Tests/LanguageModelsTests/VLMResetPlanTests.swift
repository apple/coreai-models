// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Prefix-cache routing on a vision-language session: a multimodal request cuts the history match
/// at its first image placeholder and rewinds there; a text-only request keeps cached images and
/// throws for placeholders past them.
@Suite("VLM prefix-cache policy")
struct VLMResetPlanTests {
    private let image: Int32 = 9

    /// `count` tokens with image placeholders at `imageAt`.
    private func tokens(_ count: Int, imageAt positions: Range<Int>? = nil) -> [Int32] {
        (0..<count).map { positions?.contains($0) == true ? image : Int32(100 + $0) }
    }

    private func multimodal(
        commonPrefix: Int, tokens: [Int32], historyCount: Int, processedTokenCount: Int,
        hybrid: Bool = false, ring: Int? = nil
    ) throws -> (resetTo: Int?, prefixHitCount: Int) {
        try CoreAISequentialEngine.generationResetPlan(
            tokens: tokens, commonPrefix: commonPrefix, isMultimodal: true, imageTokenId: image,
            historyCount: historyCount, processedTokenCount: processedTokenCount, hasNonTruncatableStates: hybrid,
            ringCapacity: ring)
    }

    private func textOnly(
        commonPrefix: Int, tokens: [Int32], historyCount: Int, processedTokenCount: Int, hybrid: Bool = false
    ) throws -> (resetTo: Int?, prefixHitCount: Int) {
        try CoreAISequentialEngine.generationResetPlan(
            tokens: tokens, commonPrefix: commonPrefix, isMultimodal: false, imageTokenId: image,
            historyCount: historyCount, processedTokenCount: processedTokenCount, hasNonTruncatableStates: hybrid)
    }

    @Test("the token match is cut at the first image placeholder it covers")
    func cutsAtFirstImage() {
        let input = tokens(30, imageAt: 5..<9)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 20, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 7, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 5, tokens: input, imageTokenId: image) == 5)
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 4, tokens: input, imageTokenId: image) == 4)
    }

    @Test("a match with no placeholders is not cut")
    func noCutWithoutImages() {
        #expect(CoreAISequentialEngine.reusablePrefix(commonPrefix: 20, tokens: tokens(30), imageTokenId: image) == 20)
    }

    @Test("the same image extended with new text rewinds to the image")
    func sameImageExtensionRewinds() throws {
        let p = try multimodal(
            commonPrefix: 20, tokens: tokens(30, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 5)
        #expect(p.prefixHitCount == 5)
    }

    @Test("an identical image re-request rewinds to the image")
    func identicalImageRequestRewinds() throws {
        let p = try multimodal(
            commonPrefix: 20, tokens: tokens(20, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 5)
        #expect(p.prefixHitCount == 5)
    }

    @Test("an image after the cached text reuses the text")
    func imageAfterCachedTextReuses() throws {
        let p = try multimodal(
            commonPrefix: 10, tokens: tokens(30, imageAt: 12..<16), historyCount: 10, processedTokenCount: 10)
        #expect(p.resetTo == nil)
        #expect(p.prefixHitCount == 10)
    }

    @Test("an image at position 0 never reuses")
    func leadingImageNeverReuses() throws {
        let p = try multimodal(
            commonPrefix: 20, tokens: tokens(30, imageAt: 0..<4), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 0)
        #expect(p.prefixHitCount == 0)
    }

    @Test("a text-only match covering an image is not cut")
    func textMatchOverImageKeepsImage() throws {
        let p = try textOnly(
            commonPrefix: 20, tokens: tokens(30, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == nil)
        #expect(p.prefixHitCount == 20)
    }

    @Test("a text-only request diverging after the image rewinds to the divergence")
    func textDivergenceAfterImageRewinds() throws {
        let p = try textOnly(
            commonPrefix: 12, tokens: tokens(30, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        #expect(p.resetTo == 12)
        #expect(p.prefixHitCount == 12)
    }

    @Test("a text-only request diverging before the image throws")
    func textDivergenceBeforeImageThrows() {
        #expect(throws: InferenceRuntimeError.self) {
            try textOnly(
                commonPrefix: 3, tokens: tokens(30, imageAt: 5..<9), historyCount: 20, processedTokenCount: 20)
        }
    }

    @Test("text-only placeholders on a fresh session throw")
    func textPlaceholdersOnFreshSessionThrow() {
        #expect(throws: InferenceRuntimeError.self) {
            try textOnly(commonPrefix: 0, tokens: tokens(30, imageAt: 5..<9), historyCount: 0, processedTokenCount: 0)
        }
    }

    @Test("text-only extension, re-request and strict prefix over a cached image follow the text plan")
    func textOnlyMatchesTextEngine() throws {
        let cases: [(commonPrefix: Int, historyCount: Int, processedTokenCount: Int, inputCount: Int)] = [
            (10, 10, 10, 15),  // pure extension with new tokens
            (10, 10, 10, 10),  // identical re-request: rewind one to reseed
            (10, 12, 12, 10),  // input is a strict prefix of history
        ]
        for c in cases {
            let input = tokens(c.inputCount, imageAt: 2..<6)
            let vlm = try textOnly(
                commonPrefix: c.commonPrefix, tokens: input, historyCount: c.historyCount,
                processedTokenCount: c.processedTokenCount)
            let text = CoreAISequentialEngine.prefixResetPlan(
                commonPrefix: c.commonPrefix, historyCount: c.historyCount,
                processedTokenCount: c.processedTokenCount, inputCount: c.inputCount,
                hasNonTruncatableStates: false)
            #expect(vlm.resetTo == text.resetTo)
            #expect(vlm.prefixHitCount == text.prefixHitCount)
        }
    }

    @Test("a hybrid session resets on a rewind and keeps its state on an extension")
    func hybridResetsOnRewind() throws {
        let input = tokens(30, imageAt: 5..<9)
        let cut = try multimodal(
            commonPrefix: 20, tokens: input, historyCount: 20, processedTokenCount: 20, hybrid: true)
        #expect(cut.resetTo == 0)
        #expect(cut.prefixHitCount == 0)

        let extended = try textOnly(
            commonPrefix: 20, tokens: input, historyCount: 20, processedTokenCount: 20, hybrid: true)
        #expect(extended.resetTo == nil)
        #expect(extended.prefixHitCount == 0)

        // The reset drops the cached image, so its placeholders would be prefilled as text.
        #expect(throws: InferenceRuntimeError.self) {
            try textOnly(commonPrefix: 12, tokens: input, historyCount: 20, processedTokenCount: 20, hybrid: true)
        }
    }

    @Test("a rewind past a wrapped ring replays")
    func wrappedRingReplays() throws {
        let wrapped = try multimodal(
            commonPrefix: 2600, tokens: tokens(2700, imageAt: 5..<9), historyCount: 2600,
            processedTokenCount: 2600, ring: 2048)
        #expect(wrapped.resetTo == 0)
        #expect(wrapped.prefixHitCount == 0)

        let unwrapped = try multimodal(
            commonPrefix: 1000, tokens: tokens(1100, imageAt: 5..<9), historyCount: 1000,
            processedTokenCount: 1000, ring: 2048)
        #expect(unwrapped.resetTo == 5)
        #expect(unwrapped.prefixHitCount == 5)
    }
}
