// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIEmbeddings

// MARK: - Function planning

@Suite struct FunctionPlanTests {
    @Test func embeddingGemma2Buckets() throws {
        let text = try #require(try Fixtures.block("embeddinggemma-2").components["text_encoder"])
        let nine = try text.plan(lengths: Array(repeating: 100, count: 9), name: "text_encoder")
        #expect(nine.map(\.function) == ["len128_b8", "len128_b1"])
        let mixed = try text.plan(lengths: [600, 100, 12], name: "text_encoder")
        #expect(mixed.map(\.function) == ["len128_b1", "len128_b1", "len2048_b1"])
        #expect(mixed.map(\.rows) == [[1], [2], [0]])
        #expect(throws: EmbeddingError.self) { try text.plan(lengths: [3000], name: "text_encoder") }
        let image = try #require(try Fixtures.block("embeddinggemma-2").components["image_encoder"])
        let calls = try image.plan(lengths: [260, 12], name: "image_encoder")
        #expect(calls.map(\.rows) == [[0], [1]])
        #expect(calls.allSatisfy { $0.length == nil && $0.function == "main" })
    }

    @Test func padsBelowMinimumBatch() throws {
        let component = Component(
            asset: "a", inputs: [:], outputs: [:], computeUnits: [.gpu],
            functions: [FunctionEntry(function: "b4", batch: 4...4, length: nil)])
        let calls = try component.plan(lengths: [0, 0, 0, 0, 0], name: "c")
        #expect(calls.map(\.rows.count) == [4, 1])
        #expect(calls.map(\.batch) == [4, 4])
    }

    /// A fused CLIP export's `similarity` component, with its `N` image axis, decodes and plans.
    @Test func fusedSimilarityComponent() throws {
        let block = try Fixtures.block("clip-vit-b32") {
            $0.edit("components.similarity") {
                $0["asset"] = "text_encoder"
                $0["inputs"] = ["pixel_values": "float32[N,3,224,224]", "input_ids": "int32[B,L]"]
                $0["outputs"] = ["logits_per_image": "float32[N,B]", "text_embeds": "float32[B,512]"]
                $0["compute_units"] = ["gpu"]
                $0["functions"] = [["function": "main", "batch": ["min": 1, "max": 64], "length": 77]]
            }
        }
        let similarity = try #require(block.components["similarity"])
        #expect(similarity.inputs["pixel_values"]?.shape.first == .secondBatch)
        #expect(
            try similarity.plan(lengths: [5, 9], name: "similarity") == [
                Call(function: "main", rows: [0, 1], batch: 2, length: 77)
            ])
    }

    /// A batch or length range must be positive and ordered; `batch: 0` would never finish a split.
    @Test func refusesEmptyRanges() {
        for batch: Any in [0, ["min": 0, "max": 4], ["min": 4, "max": 2]] {
            #expect(throws: EmbeddingError.self) {
                try Fixtures.block("clip-vit-b32") {
                    $0.edit("components.image_encoder") { $0["functions"] = [["function": "main", "batch": batch]] }
                }
            }
        }
    }
}

// MARK: - Driver planning

@Suite struct PipelineDriverPlanTests {
    static func driver(_ name: String, _ modality: EmbeddingModality) throws -> PipelineDriver {
        let bundle = try FakeModels.bundle(name)
        let tower = try #require(bundle.block.tower(for: modality))
        return try PipelineDriver(tower: tower, block: bundle.block)
    }

    @Test func derivations() throws {
        let driver = try Self.driver("embeddinggemma-2-ios", .image)
        let encoder = driver.stages[2].component
        var row = driver.rows(tokens: [[2, 255999, 258880, 258880, 258882, 1]])[0]
        row.tensors["inputs_embeds"] = HostTensor(float32: [Float](repeating: -1, count: 6 * 2), shape: [6, 2])
        row.tensors["soft_tokens"] = HostTensor(float32: [10, 11, 20, 21, 30, 31], shape: [3, 2])
        let merged = try driver.value("inputs_embeds", of: encoder, row: row, length: 8)
        #expect(merged.floats == [-1, -1, -1, -1, 10, 11, 20, 21, -1, -1, -1, -1])

        var masked = encoder
        masked.bias = AdditiveBias(masked: -9, unmasked: 0)
        let bias = try driver.value("bias", of: masked, row: row, length: 8)
        #expect(bias.shape == [8, 1, 8])
        #expect(Array(bias.floats[0..<8]) == [Float](repeating: 0, count: 8))
        #expect(Array(bias.floats[48..<56]) == [Float](repeating: -9, count: 8))

        let pool = try driver.value("pool_weights", of: encoder, row: row, length: 8)
        #expect(pool.floats == [1, 1, 1, 1, 1, 1, 0, 0].map { $0 / 6 })

        #expect(try driver.value("input_ids", of: encoder, row: row, length: 8) == row.tensors["input_ids"])
        let missing = #expect(throws: EmbeddingError.self) {
            try driver.value("token_type_ids", of: encoder, row: row, length: 8)
        }
        #expect(missing?.description.contains("nothing provides input 'token_type_ids'") == true)
    }

    @Test func refusesPipelineWithoutEmbedding() throws {
        let block = try Fixtures.block("clip-vit-b32") {
            $0.edit("components.image_encoder") { $0["outputs"] = ["features": "float32[B,512]"] }
        }
        #expect(throws: EmbeddingError.self) {
            try PipelineDriver(tower: block.tower(for: .image)!, block: block)
        }
    }
}
