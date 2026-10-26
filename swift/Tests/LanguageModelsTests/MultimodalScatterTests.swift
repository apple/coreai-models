// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Testing

@testable import CoreAILanguageModels

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
private let testScalarType: NDArray.ScalarType = .float16
#else
private let testScalarType: NDArray.ScalarType = .float32
#endif

/// Tests for `StaticShapeEngine.scatterImageSoftTokens`, the Gemma4 multimodal
/// scatter used to splice encoded image soft tokens into a batch's gathered token
/// embeddings before they enter the decoder. This math is correctness-critical
/// (a wrong offset or missed embed-scale compensation silently corrupts every image
/// prompt) and was previously untested — these exercise it against synthetic
/// `NDArray`s, with no model asset required.
///
/// Built generically over `LogitsScalarType` (not hardcoded `Float16`) because the
/// scatter itself views its `NDArray`s as `LogitsScalarType`, which is `Float` on
/// x86_64 macOS and `Float16` everywhere else — a fixture that hardcoded one would
/// silently reinterpret the wrong element size on the other platform.
@Suite("Multimodal scatter Tests")
struct MultimodalScatterTests {
    private static let hiddenDim = 4

    /// `gathered`: `[1, batchSize, 1, hiddenDim]`, every row filled with `sentinel`.
    private static func makeGathered(batchSize: Int, sentinel: LogitsScalarType = 100) -> NDArray {
        var array = NDArray(shape: [1, batchSize, 1, hiddenDim], scalarType: testScalarType)
        fillNDArray(&array, as: LogitsScalarType.self, count: batchSize * hiddenDim) { _ in sentinel }
        return array
    }

    /// `image`: `[1, tokenCount, hiddenDim]`, token `i`'s row filled with `value(i)`.
    private static func makeImage(
        tokenCount: Int, hiddenDim: Int? = nil, value: (Int) -> LogitsScalarType
    ) -> NDArray {
        let dim = hiddenDim ?? Self.hiddenDim
        var array = NDArray(shape: [1, tokenCount, dim], scalarType: testScalarType)
        fillNDArray(&array, as: LogitsScalarType.self, count: tokenCount * dim) { value($0 / dim) }
        return array
    }

    @Test("Scatters a single placeholder at the embed-scale-compensated value")
    func scattersSinglePlaceholder() throws {
        let batchSize = 3
        var gathered = Self.makeGathered(batchSize: batchSize)
        let image = Self.makeImage(tokenCount: 1) { _ in 10 }

        try StaticShapeEngine.scatterImageSoftTokens(
            into: &gathered, image: image, positions: [1], alignedStep: 0, batchSize: batchSize)

        let rows = readNDArray(gathered, as: LogitsScalarType.self, count: batchSize * Self.hiddenDim)
        // Row 1 (the placeholder) becomes image_value / sqrt(hiddenDim) = 10 / 2 = 5.
        #expect(Array(rows[0..<4]) == [100, 100, 100, 100])
        #expect(Array(rows[4..<8]) == [5, 5, 5, 5])
        #expect(Array(rows[8..<12]) == [100, 100, 100, 100])
    }

    @Test("Only writes positions inside [alignedStep, alignedStep + batchSize)")
    func windowsToCurrentBatch() throws {
        let batchSize = 3
        let alignedStep = 3
        var gathered = Self.makeGathered(batchSize: batchSize)
        // Two placeholders: absolute position 0 (before this batch) and 5 (inside it,
        // local row 5-3=2). Only the second should land.
        let image = Self.makeImage(tokenCount: 2) { i in i == 0 ? 10 : 20 }

        try StaticShapeEngine.scatterImageSoftTokens(
            into: &gathered, image: image, positions: [0, 5], alignedStep: alignedStep, batchSize: batchSize)

        let rows = readNDArray(gathered, as: LogitsScalarType.self, count: batchSize * Self.hiddenDim)
        #expect(Array(rows[0..<4]) == [100, 100, 100, 100])
        #expect(Array(rows[4..<8]) == [100, 100, 100, 100])
        // image row 1 (value 20) / sqrt(4) = 10.
        #expect(Array(rows[8..<12]) == [10, 10, 10, 10])
    }

    @Test("No placeholders in range leaves the batch untouched")
    func noPlaceholdersInRangeIsNoOp() throws {
        let batchSize = 2
        var gathered = Self.makeGathered(batchSize: batchSize)
        let image = Self.makeImage(tokenCount: 1) { _ in 10 }

        try StaticShapeEngine.scatterImageSoftTokens(
            into: &gathered, image: image, positions: [99], alignedStep: 0, batchSize: batchSize)

        let rows = readNDArray(gathered, as: LogitsScalarType.self, count: batchSize * Self.hiddenDim)
        #expect(rows.allSatisfy { $0 == 100 })
    }

    @Test("Throws when the image's hidden dim doesn't match the gathered buffer's")
    func throwsOnHiddenDimMismatch() throws {
        let batchSize = 2
        var gathered = Self.makeGathered(batchSize: batchSize)
        let mismatched = Self.makeImage(tokenCount: 1, hiddenDim: Self.hiddenDim + 1) { _ in 10 }

        #expect(throws: (any Error).self) {
            try StaticShapeEngine.scatterImageSoftTokens(
                into: &gathered, image: mismatched, positions: [0], alignedStep: 0, batchSize: batchSize)
        }
    }

    // MARK: - Prefix reuse

    @Test("Without a pending image the whole token match is reusable")
    func prefixReuseWithoutImage() {
        #expect(StaticShapeEngine.reusablePrefixLength(matchedPrefix: 12, firstImagePosition: nil) == 12)
    }

    @Test("A pending image cuts the reusable prefix at its first position")
    func prefixReuseStopsAtFirstImagePosition() {
        // Same-size image in the same slots: every token id matches, none of the image rows may be reused.
        #expect(StaticShapeEngine.reusablePrefixLength(matchedPrefix: 12, firstImagePosition: 5) == 5)
        #expect(StaticShapeEngine.reusablePrefixLength(matchedPrefix: 12, firstImagePosition: 0) == 0)
    }

    @Test("Text before the first image position stays reusable")
    func prefixReuseKeepsTextBeforeImage() {
        #expect(StaticShapeEngine.reusablePrefixLength(matchedPrefix: 3, firstImagePosition: 5) == 3)
    }
}
