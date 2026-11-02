// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

/// Covers the image scatter-merge `EmbeddingsInputHandler` applies on the multimodal prefill step.
@Suite("EmbeddingsInputHandler")
struct EmbeddingsInputHandlerTests {
    private let imageTokenId: Int32 = 99
    private let hiddenDim = 2

    /// `[1, rows.count, hiddenDim]` Float16 array with the given row values.
    private func embeddings(_ rows: [[Float16]]) -> NDArray {
        var array = NDArray(shape: [1, rows.count, hiddenDim], scalarType: .float16)
        fillNDArray(&array, as: Float16.self, with: rows.flatMap { $0 })
        return array
    }

    private func rows(of array: NDArray) -> [[Float16]] {
        let flat = readNDArray(array, as: Float16.self, count: array.shape[1] * hiddenDim)
        return stride(from: 0, to: flat.count, by: hiddenDim).map { Array(flat[$0..<($0 + hiddenDim)]) }
    }

    @Test("scatterMerge places image rows at the placeholder positions, in order")
    func placesImageRows() throws {
        let text = embeddings([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        let image = embeddings([[70, 71], [80, 81]])
        let tokens: [Int32] = [5, imageTokenId, imageTokenId, 6, 7]

        let merged = try EmbeddingsInputHandler.scatterMerge(
            textEmbeddings: text, imageEmbeddings: image, tokens: tokens[...], imageTokenId: imageTokenId)

        #expect(rows(of: merged) == [[1, 1], [70, 71], [80, 81], [4, 4], [5, 5]])
    }

    @Test("scatterMerge rejects a prompt with no image placeholders")
    func noPlaceholdersThrows() {
        let text = embeddings([[1, 1], [2, 2], [3, 3]])
        let image = embeddings([[70, 71]])
        let tokens: [Int32] = [5, 6, 7]

        #expect(throws: InferenceRuntimeError.self) {
            _ = try EmbeddingsInputHandler.scatterMerge(
                textEmbeddings: text, imageEmbeddings: image, tokens: tokens[...], imageTokenId: imageTokenId)
        }
    }

    @Test("scatterMerge indexes positions from the start of a non-zero-based slice")
    func sliceOffsets() throws {
        let text = embeddings([[1, 1], [2, 2]])
        let image = embeddings([[70, 71]])
        let tokens: [Int32] = [0, 0, 5, imageTokenId]

        let merged = try EmbeddingsInputHandler.scatterMerge(
            textEmbeddings: text, imageEmbeddings: image, tokens: tokens[2...], imageTokenId: imageTokenId)

        #expect(rows(of: merged) == [[1, 1], [70, 71]])
    }

    @Test("scatterMerge rejects a placeholder count that does not match the image embeddings")
    func countMismatchThrows() {
        let text = embeddings([[1, 1], [2, 2], [3, 3]])
        let image = embeddings([[70, 71], [80, 81]])
        let tokens: [Int32] = [5, imageTokenId, 7]

        #expect(throws: InferenceRuntimeError.self) {
            _ = try EmbeddingsInputHandler.scatterMerge(
                textEmbeddings: text, imageEmbeddings: image, tokens: tokens[...], imageTokenId: imageTokenId)
        }
    }

    @Test("scatterMerge rejects image embeddings of a different hidden size")
    func hiddenSizeMismatchThrows() {
        let text = embeddings([[1, 1], [2, 2]])
        let image = NDArray(shape: [1, 1, 3], scalarType: .float16)
        let tokens: [Int32] = [5, imageTokenId]

        #expect(throws: InferenceRuntimeError.self) {
            _ = try EmbeddingsInputHandler.scatterMerge(
                textEmbeddings: text, imageEmbeddings: image, tokens: tokens[...], imageTokenId: imageTokenId)
        }
    }

    @Test("scatterMerge rejects non-float16 image embeddings")
    func nonFloat16Throws() {
        let text = embeddings([[1, 1], [2, 2]])
        let image = NDArray(shape: [1, 1, hiddenDim], scalarType: .float32)
        let tokens: [Int32] = [5, imageTokenId]

        #expect(throws: InferenceRuntimeError.self) {
            _ = try EmbeddingsInputHandler.scatterMerge(
                textEmbeddings: text, imageEmbeddings: image, tokens: tokens[...], imageTokenId: imageTokenId)
        }
    }
}
