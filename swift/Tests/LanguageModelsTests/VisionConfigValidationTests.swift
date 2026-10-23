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
        patch: Int = 16, pool: Int = 3, buckets: String? = nil, imageSize: Int = 768
    ) -> String {
        let ladder =
            buckets
                ?? """
                [{"image_size": 384, "num_soft_tokens": 64, "pooled_grid": 8, "function": "encode_image_384"},
                 {"image_size": 768, "num_soft_tokens": 256, "pooled_grid": 16, "function": "encode_image_768"}]
                """
        return """
            {"image_size": \(imageSize), "patch_size": \(patch), "pooling_kernel_size": \(pool),
             "image_token_count": 256, "image_token_id": 7, "image_buckets": \(ladder)}
            """
    }

    @Test("A valid size ladder decodes")
    func validLadder() throws {
        let config = try decode(vision: vision())
        #expect(config.vision?.imageBuckets?.map(\.imageSize) == [384, 768])
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

    @Test("Buckets must be positive multiples of the block size")
    func bucketsMultipleOfBlock() {
        let unaligned = """
            [{"image_size": 100, "num_soft_tokens": 4, "pooled_grid": 2, "function": "f"}]
            """
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: unaligned)) }
        let zero = """
            [{"image_size": 0, "num_soft_tokens": 0, "pooled_grid": 0, "function": "f"}]
            """
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: zero)) }
    }

    @Test("Buckets must be sorted ascending and unique")
    func bucketsSorted() {
        let b384 = "{\"image_size\": 384, \"num_soft_tokens\": 64, \"pooled_grid\": 8, \"function\": \"a\"}"
        let b768 = "{\"image_size\": 768, \"num_soft_tokens\": 256, \"pooled_grid\": 16, \"function\": \"b\"}"
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: "[\(b768), \(b384)]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: "[\(b384), \(b384)]")) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: "[]")) }
    }

    @Test("A bucket whose pooled grid or soft-token count disagrees with its size is rejected")
    func inconsistentBucket() {
        let wrongGrid = """
            [{"image_size": 384, "num_soft_tokens": 64, "pooled_grid": 7, "function": "f"}]
            """
        let wrongCount = """
            [{"image_size": 384, "num_soft_tokens": 60, "pooled_grid": 8, "function": "f"}]
            """
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: wrongGrid)) }
        #expect(throws: DecodingError.self) { try decode(vision: vision(buckets: wrongCount)) }
    }

    @Test("A non-positive vocabulary or context length is rejected")
    func nonPositiveLanguageSizes() {
        #expect(throws: DecodingError.self) { try decode(vocab: -1) }
        #expect(throws: DecodingError.self) { try decode(context: 0) }
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
