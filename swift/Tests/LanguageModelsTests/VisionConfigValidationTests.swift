// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics
import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Vision and language config validation")
struct VisionConfigValidationTests {
    private func decode(vision: String? = nil, vocab: Int = 100, context: Int = 1024) throws -> LanguageConfig {
        let extra = vision.map { ", \"vision\": \($0)" } ?? ""
        let json = "{\"tokenizer\": \"t\", \"vocab_size\": \(vocab), \"max_context_length\": \(context)\(extra)}"
        return try JSONDecoder().decode(LanguageConfig.self, from: Data(json.utf8))
    }

    private func vision(
        patch: Int = 16, pool: Int = 3, sizes: String = "[384, 768]", imageSize: Int = 768
    ) -> String {
        """
        {"image_size": \(imageSize), "patch_size": \(patch), "pooling_kernel_size": \(pool),
         "image_token_count": 256, "image_token_id": 7, "image_sizes": \(sizes)}
        """
    }

    @Test("A valid size ladder decodes")
    func validLadder() throws {
        let config = try decode(vision: vision())
        #expect(config.vision?.imageSizes == [384, 768])
    }

    @Test("A vision block without a ladder decodes (fixed-size encoders)")
    func noLadder() throws {
        let json = "{\"image_size\": 448, \"patch_size\": 14, \"image_token_count\": 256, \"image_token_id\": 7}"
        _ = try decode(vision: json)
    }

    @Test("A zero or negative patch size or pooling kernel is rejected")
    func nonPositiveBlock() {
        #expect(throws: DecodingError.self) { try decode(vision: vision(patch: 0)) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(patch: -16)) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(pool: 0)) }
    }

    @Test("Sizes must be positive multiples of the block size")
    func sizesMultipleOfBlock() {
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[100]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[0]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[-48]")) }
    }

    @Test("Sizes must be sorted ascending, unique and non-empty")
    func sizesSorted() {
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[768, 384]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[384, 384]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(sizes: "[]")) }
    }

    @Test("The pooled grid and soft-token count follow from the size and the block")
    func derivedGeometry() throws {
        let config = try #require(try decode(vision: vision(patch: 14, pool: 2, sizes: "[28, 56]")).vision)
        #expect(config.blockPixels == 28)
        #expect(config.pooledGrid(imageSize: 56) == 2)
        #expect(config.softTokenCount(imageSize: 56) == 4)
    }

    @Test("A non-positive vocabulary or context length is rejected")
    func nonPositiveLanguageSizes() {
        #expect(throws: DecodingError.self) { try decode(vocab: -1) }
        #expect(throws: DecodingError.self) { try decode(context: 0) }
    }

    @Test("token_merge defaults to a separate embedding asset when absent")
    func tokenMergeDefault() throws {
        let config = try #require(try decode(vision: vision()).vision)
        #expect(config.tokenMerge == .separateEmbedding)
    }

    @Test("token_merge decodes both values and rejects an unknown one")
    func tokenMergeValues() throws {
        func withMerge(_ value: String) -> String {
            """
            {"image_size": 768, "patch_size": 16, "pooling_kernel_size": 3, "image_token_count": 256,
             "image_token_id": 7, "token_merge": "\(value)"}
            """
        }
        #expect(try decode(vision: withMerge("decoder_gather")).vision?.tokenMerge == .decoderGather)
        #expect(try decode(vision: withMerge("separate_embedding")).vision?.tokenMerge == .separateEmbedding)
        #expect(throws: DecodingError.self) { try decode(vision: withMerge("scatter")) }
    }

    private func image(_ side: Int) -> CGImage {
        CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            .makeImage()!
    }

    @Test("Letterboxing with a zero block, or a bucket that is not a multiple of it, throws")
    func letterboxGeometry() {
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.letterbox(cgImage: image(4), bucketSize: 384, blockPx: 0)
        }
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.letterbox(cgImage: image(4), bucketSize: 100, blockPx: 48)
        }
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.letterbox(cgImage: image(4), bucketSize: 0, blockPx: 48)
        }
    }

    @Test("Compaction throws when the pooled grid does not match the encoder output")
    func compactionGridMismatch() {
        let out = NDArray(shape: [1, 4, 8], scalarType: .float16)
        // 4 soft tokens form a 2x2 grid; a claimed 2^26 grid would read far past the array.
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.compactRealSoftTokens(
                from: out, pooledGrid: 1 << 26, realPooledWidth: 2, realPooledHeight: 2)
        }
        // A real rectangle larger than the grid.
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.compactRealSoftTokens(
                from: out, pooledGrid: 2, realPooledWidth: 3, realPooledHeight: 1)
        }
        // A modest mismatch (24 vs 2) used to read out of bounds silently.
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.compactRealSoftTokens(
                from: out, pooledGrid: 24, realPooledWidth: 2, realPooledHeight: 2)
        }
    }

    @Test("Compaction throws when the encoder output scalar type is not the logits type")
    func compactionScalarTypeMismatch() {
        let out = NDArray(
            shape: [1, 4, 2], scalarType: LogitsScalarType.self == Float16.self ? .float32 : .float16)
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.compactRealSoftTokens(
                from: out, pooledGrid: 2, realPooledWidth: 1, realPooledHeight: 1)
        }
    }

    @Test("Compaction keeps the real rectangle of a matching grid")
    func compactionKeepsRealCells() throws {
        var out = NDArray(shape: [1, 4, 2], scalarType: .float16)
        out.mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in
            for i in 0..<8 { p[i] = Float16(i) }
        }
        let compacted = try StaticShapeEngine.compactRealSoftTokens(
            from: out, pooledGrid: 2, realPooledWidth: 1, realPooledHeight: 2)
        #expect(compacted.shape == [1, 2, 2])
        var values: [Float16] = []
        compacted.view(as: Float16.self).withUnsafePointer { p, _, _ in values = (0..<4).map { p[$0] } }
        #expect(values == [0, 1, 4, 5])
    }
}
