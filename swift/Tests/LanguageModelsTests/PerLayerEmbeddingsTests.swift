// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

/// Writes small safetensors files in the layouts the export produces, and reads them back through
/// ``PerLayerEmbeddings``.
@Suite("Per-layer embeddings table")
struct PerLayerEmbeddingsTests {
    private struct Tensor {
        var name: String
        var dtype: String
        var shape: [Int]
        var bytes: [UInt8]
    }

    private func write(tensors: [Tensor], metadata: [String: String] = [:]) throws -> URL {
        var header: [String: Any] = [:]
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var offset = 0
        var body: [UInt8] = []
        for t in tensors {
            header[t.name] = [
                "dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.count],
            ]
            offset += t.bytes.count
            body += t.bytes
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(contentsOf: body)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ple-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    private func bytes(_ values: [Float16]) -> [UInt8] {
        values.flatMap { value -> [UInt8] in
            withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
        }
    }

    /// Packs signed values the way the checkpoint does: `8 / nbits` per byte, low-order bits first,
    /// each stored as an unsigned field holding the value plus `2^(nbits - 1)`.
    private func pack(_ rows: [[Int]], nbits: Int) -> [UInt8] {
        let perByte = 8 / nbits
        let bias = 1 << (nbits - 1)
        return rows.flatMap { row -> [UInt8] in
            stride(from: 0, to: row.count, by: perByte).map { start in
                var byte = 0
                for k in 0..<perByte where start + k < row.count {
                    byte |= (row[start + k] + bias) << (k * nbits)
                }
                return UInt8(byte)
            }
        }
    }

    private func nativeTable(
        rows: [[Int]], scales: [[Float16]], nbits: Int, totalDim: Int? = nil, blocks: Int? = nil
    ) throws -> URL {
        let width = rows[0].count
        let packedWidth = (width + (8 / nbits) - 1) / (8 / nbits)
        return try write(
            tensors: [
                Tensor(
                    name: "embed_tokens_per_layer", dtype: "U8", shape: [rows.count, packedWidth],
                    bytes: pack(rows, nbits: nbits)),
                Tensor(
                    name: "embed_tokens_per_layer_scale", dtype: "F16",
                    shape: [rows.count, blocks ?? scales[0].count], bytes: bytes(scales.flatMap { $0 })),
            ],
            metadata: [
                "dtype": "native_quantized", "nbits": "\(nbits)", "ple_total_dim": "\(totalDim ?? width)",
            ])
    }

    private func message(_ body: () throws -> Void) -> String {
        do {
            try body()
            return ""
        } catch {
            return "\(error)"
        }
    }

    @Test("An INT8 table loads and a gather copies the rows of the requested tokens")
    func int8Gather() throws {
        let url = try write(
            tensors: [
                Tensor(
                    name: "embed_tokens_per_layer", dtype: "I8", shape: [3, 4],
                    bytes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map { UInt8(bitPattern: Int8($0)) })
            ], metadata: ["dtype": "SI8"])
        let table = try PerLayerEmbeddings(contentsOf: url)
        #expect(!table.isNativeQuantized && table.vocabSize == 3 && table.rowWidth == 4)

        var out = [Int8](repeating: 0, count: 3 * 4)
        out.withUnsafeMutableBufferPointer { table.gather(tokenIDs: [2, 0], batchSize: 3, into: $0) }
        // Token 2 then 0; the third slot is padding and stays zero.
        #expect(out == [9, 10, 11, 12, 1, 2, 3, 4, 0, 0, 0, 0])
    }

    @Test("A token id outside the table leaves its slot as it was, instead of reading past the file")
    func outOfRangeToken() throws {
        let table = try PerLayerEmbeddings(
            contentsOf: try write(
                tensors: [Tensor(name: "embed_tokens_per_layer", dtype: "I8", shape: [2, 2], bytes: [1, 2, 3, 4])]))
        var out = [Int8](repeating: 7, count: 3 * 2)
        out.withUnsafeMutableBufferPointer { table.gather(tokenIDs: [-1, 5, 1], batchSize: 3, into: $0) }
        #expect(out == [7, 7, 7, 7, 3, 4])
    }

    @Test("A native-quantized table unpacks and dequantizes each element with its block's scale", arguments: [2, 4, 8])
    func nativeGather(nbits: Int) throws {
        let half = 1 << (nbits - 1)
        // Two vocabulary rows of 8 values spanning the signed range, two blocks of 4 per row.
        let rows = [
            (0..<8).map { ($0 % (2 * half)) - half },
            (0..<8).map { half - 1 - ($0 % (2 * half)) },
        ]
        let scales: [[Float16]] = [[0.5, 2], [1, 0.25]]
        let table = try PerLayerEmbeddings(contentsOf: try nativeTable(rows: rows, scales: scales, nbits: nbits))
        #expect(table.isNativeQuantized && table.vocabSize == 2 && table.rowWidth == 8)

        var out = [Float16](repeating: 0, count: 3 * 8)
        out.withUnsafeMutableBufferPointer { table.gather(tokenIDs: [1, 0], batchSize: 3, into: $0) }
        for (slot, token) in [1, 0].enumerated() {
            for e in 0..<8 {
                let expected = Float16(rows[token][e]) * scales[token][e / 4]
                #expect(out[slot * 8 + e] == expected, "nbits \(nbits), token \(token), element \(e)")
            }
        }
        #expect(out[16..<24].allSatisfy { $0 == 0 }, "the padding slot is untouched")
    }

    @Test("A native-quantized row whose width is not a whole number of bytes keeps its tail")
    func nativeRowNotByteAligned() throws {
        // 6 values of 4 bits = 3 bytes, but 2-bit packing of 6 values needs 2 bytes with 2 unused slots.
        let rows = [[-2, -1, 0, 1, -2, 1]]
        let table = try PerLayerEmbeddings(
            contentsOf: try nativeTable(rows: rows, scales: [[1, 1, 1]], nbits: 2, totalDim: 6))
        var out = [Float16](repeating: 0, count: 6)
        out.withUnsafeMutableBufferPointer { table.gather(tokenIDs: [0], batchSize: 1, into: $0) }
        #expect(out == rows[0].map { Float16($0) })
    }

    @Test("The input handler rejects a graph whose ple_embeddings type or width does not match the table")
    func handlerValidatesInput() throws {
        let native = try PerLayerEmbeddings(
            contentsOf: try nativeTable(rows: [[1, -2, 0, 1]], scales: [[1]], nbits: 2))
        func check(_ width: Int, _ type: NDArray.ScalarType) -> String {
            message { try PerLayerEmbeddingsInputHandler.validate(table: native, rowWidth: width, scalarType: type) }
        }
        #expect(check(native.rowWidth, .float16).isEmpty)
        #expect(check(native.rowWidth, .int8).contains("PLE type mismatch"))
        #expect(check(native.rowWidth + 1, .float16).contains("PLE row width mismatch"))
    }

    @Test("Malformed tables are reported by name, never trapped on or read out of bounds")
    func malformedTables() throws {
        let good = [[0, 1, -1, 0]]
        let scales: [[Float16]] = [[1, 1]]

        func load(_ url: URL) -> String { message { _ = try PerLayerEmbeddings(contentsOf: url) } }

        // A bit width the packing cannot represent.
        #expect(
            load(try nativeTable(rows: good, scales: scales, nbits: 4).patchingMetadata(nbits: "3")).contains("nbits"))
        // Blocks that do not divide a row.
        #expect(load(try nativeTable(rows: good, scales: [[1, 1, 1]], nbits: 4)).contains("do not divide"))
        // A scale tensor missing from the file.
        let noScale = try write(
            tensors: [Tensor(name: "embed_tokens_per_layer", dtype: "U8", shape: [1, 2], bytes: pack(good, nbits: 4))],
            metadata: ["dtype": "native_quantized", "nbits": "4", "ple_total_dim": "4"])
        #expect(load(noScale).contains("embed_tokens_per_layer_scale"))
        // Fewer packed bytes than the shape declares.
        let short = try write(
            tensors: [
                Tensor(name: "embed_tokens_per_layer", dtype: "U8", shape: [2, 2], bytes: pack(good, nbits: 4)),
                Tensor(name: "embed_tokens_per_layer_scale", dtype: "F16", shape: [2, 2], bytes: bytes([1, 1, 1, 1])),
            ], metadata: ["dtype": "native_quantized", "nbits": "4", "ple_total_dim": "4"])
        #expect(!load(short).isEmpty)
        // A wrong dtype for either format.
        #expect(
            load(
                try write(tensors: [
                    Tensor(name: "embed_tokens_per_layer", dtype: "F16", shape: [1, 2], bytes: [0, 0, 0, 0])
                ])
            )
            .contains("INT8"))
        #expect(load(try nativeTable(rows: good, scales: scales, nbits: 4).patchingDType("I8")).contains("U8"))
    }
}

extension URL {
    /// A copy of a safetensors file whose `__metadata__` has `nbits` replaced.
    fileprivate func patchingMetadata(nbits: String) throws -> URL {
        try patchingHeader { header in
            var metadata = header["__metadata__"] as? [String: String] ?? [:]
            metadata["nbits"] = nbits
            header["__metadata__"] = metadata
        }
    }

    /// A copy whose table tensor declares another dtype.
    fileprivate func patchingDType(_ dtype: String) throws -> URL {
        try patchingHeader { header in
            var entry = header["embed_tokens_per_layer"] as? [String: Any] ?? [:]
            entry["dtype"] = dtype
            header["embed_tokens_per_layer"] = entry
        }
    }

    private func patchingHeader(_ edit: (inout [String: Any]) -> Void) throws -> URL {
        let data = try Data(contentsOf: self)
        let length = Int(UInt64(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }))
        let headerData = data.subdata(in: 8..<(8 + length))
        var header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        edit(&header)
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var newLength = UInt64(json.count).littleEndian
        var out = Data(bytes: &newLength, count: 8)
        out.append(json)
        out.append(data.subdata(in: (8 + length)..<data.count))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ple-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }
}
