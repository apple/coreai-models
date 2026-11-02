// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// A pack that is malformed, stale or foreign fails at load with a message, never by a trap or an
/// out-of-bounds read at the first draft.
@Suite("Draft pack validation")
struct DraftPackValidationTests {
    private struct Tensor {
        var name: String
        var dtype: String
        var shape: [Int]
        var bytes: Int?
        var offset: Int?
    }

    private static let sizes = ["float16": 2, "float32": 4, "int8": 1, "int32": 4]

    /// Writes a blob and index holding `tensors` laid out back to back, and returns the blob.
    private func write(
        tensors: [Tensor], vocab: Int = 8, centroids: Int = 2, perCentroid: Int = 4, layers: Int = 1,
        extra: String = "", blobBytes: Int? = nil
    ) throws -> URL {
        var offset = 0
        var entries: [String] = []
        for t in tensors {
            let natural = t.shape.reduce(1, *) * (Self.sizes[t.dtype] ?? 2)
            let at = t.offset ?? offset
            entries.append(
                "\"\(t.name)\": {\"offset\": \(at), \"bytes\": \(t.bytes ?? natural), "
                    + "\"shape\": \(t.shape), \"dtype\": \"\(t.dtype)\"}")
            offset = at + natural
        }
        let json = """
            {\(extra) "vocab": \(vocab), "centroids": \(centroids), "per_centroid": \(perCentroid),
             "layers": \(layers), "tensors": {\(entries.joined(separator: ","))}}
            """
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("draft-pack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let blob = dir.appendingPathComponent("pack.bin")
        // Token ordering of 0 everywhere is in range for any vocabulary.
        try Data(count: blobBytes ?? max(offset, 16)).write(to: blob)
        try Data(json.utf8).write(to: dir.appendingPathComponent("pack.json"))
        return blob
    }

    private var scanTensors: [Tensor] {
        [
            Tensor(name: "embed_perm_f16", dtype: "float16", shape: [8, 4]),
            Tensor(name: "token_ordering_i32", dtype: "int32", shape: [8]),
        ]
    }

    private func message(_ body: () throws -> Void) -> String {
        do {
            try body()
            return ""
        } catch let error as DraftPackError {
            return error.description
        } catch {
            return "unexpected \(error)"
        }
    }

    @Test("A well-formed scan pack loads for the scan role")
    func scanPackLoads() throws {
        let pack = try DraftPack(blob: try write(tensors: scanTensors), role: .scan)
        #expect(pack.has("embed_perm_f16"))
    }

    @Test("A shape that needs more bytes than the entry declares is rejected")
    func shapeLargerThanBytes() throws {
        let blob = try write(
            tensors: [Tensor(name: "t", dtype: "float16", shape: [1_000_000, 1], bytes: 16)])
        let text = message { _ = try DraftPack(blob: blob) }
        #expect(text.contains("declares 16 bytes"))
    }

    @Test("A declared byte count that exceeds the shape's is rejected")
    func bytesLargerThanShape() throws {
        let blob = try write(tensors: [Tensor(name: "t", dtype: "float16", shape: [2], bytes: 8)])
        #expect(message { _ = try DraftPack(blob: blob) }.contains("declares 8 bytes"))
    }

    @Test("An unknown dtype is rejected at load")
    func unknownDtype() throws {
        let blob = try write(tensors: [Tensor(name: "t", dtype: "bfloat16", shape: [4])])
        #expect(message { _ = try DraftPack(blob: blob) }.contains("unsupported dtype 'bfloat16'"))
    }

    @Test("A misaligned tensor offset is rejected")
    func misalignedOffset() throws {
        let blob = try write(
            tensors: [Tensor(name: "t", dtype: "float16", shape: [4], offset: 1)], blobBytes: 32)
        #expect(message { _ = try DraftPack(blob: blob) }.contains("multiple"))
    }

    @Test("A pack with a zero-size geometry is rejected")
    func degenerateGeometry() throws {
        let blob = try write(tensors: [], vocab: 0, centroids: 0, perCentroid: 0, layers: 0)
        #expect(message { _ = try DraftPack(blob: blob) }.contains("empty geometry"))
    }

    @Test("A missing scan tensor is rejected for the scan role, naming the tensor")
    func missingScanTensor() throws {
        let blob = try write(tensors: [scanTensors[0]])
        #expect(
            message { _ = try DraftPack(blob: blob, role: .scan) }.contains("no tensor 'token_ordering_i32'"))
    }

    @Test("A scan table stored as float32 is rejected for the scan role")
    func wrongScanDtype() throws {
        let blob = try write(tensors: [
            Tensor(name: "embed_perm_f16", dtype: "float32", shape: [8, 4]), scanTensors[1],
        ])
        #expect(message { _ = try DraftPack(blob: blob, role: .scan) }.contains("expected float16"))
    }

    @Test("A scan table whose row count disagrees with the centroids is rejected")
    func wrongScanRows() throws {
        let blob = try write(
            tensors: [Tensor(name: "embed_perm_f16", dtype: "float16", shape: [6, 4]), scanTensors[1]])
        #expect(message { _ = try DraftPack(blob: blob, role: .scan) }.contains("shape [6, 4]"))
    }

    @Test("A format version newer than the reader's is rejected with a re-export hint")
    func newerVersion() throws {
        let blob = try write(tensors: scanTensors, extra: "\"version\": 2,")
        let text = message { _ = try DraftPack(blob: blob, role: .scan) }
        #expect(text.contains("version 2"))
        #expect(text.contains("update the runner"))
    }

    @Test("A missing version reads as version 1, and version 1 loads")
    func versionOne() throws {
        _ = try DraftPack(blob: try write(tensors: scanTensors), role: .scan)
        _ = try DraftPack(blob: try write(tensors: scanTensors, extra: "\"version\": 1,"), role: .scan)
        let bad = try write(tensors: scanTensors, extra: "\"version\": 0,")
        #expect(message { _ = try DraftPack(blob: bad, role: .scan) }.contains("invalid format version"))
    }

    @Test("The draft head's top-k must be positive, defaults to 32 and is capped at the centroid count")
    func draftTopK() throws {
        #expect(try DraftScan.topK(requested: nil, centroids: 2048) == 32)
        #expect(try DraftScan.topK(requested: nil, centroids: 8) == 8)
        #expect(try DraftScan.topK(requested: 5, centroids: 2048) == 5)
        #expect(try DraftScan.topK(requested: 3000, centroids: 2048) == 2048)
        for bad in [0, -1] {
            #expect(throws: DraftPackError.self) { try DraftScan.topK(requested: bad, centroids: 2048) }
        }
    }
}
