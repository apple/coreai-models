// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared

/// `NpyArray` is read by three parity tools (image, video, speech), each of which used to
/// carry its own copy with subtly different dtype handling. These pin the union of what
/// those copies accepted, so consolidating them can't silently narrow one.
@Suite("NpyArray")
struct NpyArrayTests {
    /// Build a `.npy` file in memory. Version 1 header, C order.
    private func npy(descr: String, shape: [Int], payload: Data) -> Data {
        let shapeText = shape.count == 1 ? "(\(shape[0]),)" : "(\(shape.map(String.init).joined(separator: ", ")))"
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': \(shapeText), }"
        // numpy pads the header so the data starts on a 64-byte boundary.
        while (10 + header.count + 1) % 64 != 0 { header += " " }
        header += "\n"

        var out = Data([0x93])
        out.append("NUMPY".data(using: .ascii)!)
        out.append(contentsOf: [0x01, 0x00])
        out.append(contentsOf: [UInt8(header.count & 0xFF), UInt8((header.count >> 8) & 0xFF)])
        out.append(header.data(using: .ascii)!)
        out.append(payload)
        return out
    }

    private func write(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coreai-npy-\(UUID().uuidString).npy")
        try data.write(to: url)
        return url
    }

    private func bytes<T>(_ values: [T]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    @Test("float32 round-trips with its shape")
    func float32() throws {
        let url = try write(
            npy(descr: "<f4", shape: [2, 3], payload: bytes([Float](arrayLiteral: 1, 2, 3, 4, 5, 6))))
        defer { try? FileManager.default.removeItem(at: url) }
        let array = try NpyArray.load(url)
        #expect(array.dtype == .float32)
        #expect(array.shape == [2, 3])
        #expect(array.count == 6)
        #expect(array.asFloat() == [1, 2, 3, 4, 5, 6])
    }

    @Test("int64 narrows to Int32, which is how token dumps arrive")
    func int64Tokens() throws {
        // numpy's default integer dtype is int64, so a list of token ids saves as `<i8`.
        let url = try write(
            npy(descr: "<i8", shape: [4], payload: bytes([Int64](arrayLiteral: 49406, 2533, 49407, 49407))))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try NpyArray.load(url).asInt32() == [49406, 2533, 49407, 49407])
    }

    @Test("A float dtype is accepted by asInt32, truncating toward zero")
    func floatToInt32() throws {
        // The speech parity reader this replaced accepted floats here. Refusing them would
        // turn a readable dump into a failed run for a generator that happened to compute
        // its ids through a float tensor.
        let url = try write(
            npy(descr: "<f4", shape: [4], payload: bytes([Float](arrayLiteral: 3.0, 3.9, -2.7, 0.0))))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try NpyArray.load(url).asInt32() == [3, 3, -2, 0])
    }

    @Test("bool reads as bytes, which is how packbits masks come back")
    func boolAsBytes() throws {
        let url = try write(npy(descr: "|b1", shape: [4], payload: Data([1, 0, 1, 1])))
        defer { try? FileManager.default.removeItem(at: url) }
        let array = try NpyArray.load(url)
        #expect(array.dtype == .bool)
        #expect(try array.asUInt8() == [1, 0, 1, 1])
        #expect(array.asFloat() == [1, 0, 1, 1])
    }

    @Test("Reading a float array as raw bytes is refused")
    func rejectsFloatAsUInt8() throws {
        let url = try write(npy(descr: "<f4", shape: [2], payload: bytes([Float](arrayLiteral: 1, 2))))
        defer { try? FileManager.default.removeItem(at: url) }
        let array = try NpyArray.load(url)
        #expect(throws: NpyArray.NpyError.self) { try array.asUInt8() }
    }

    @Test("A Fortran-ordered file is refused, not read as C-order")
    func rejectsFortranOrder() throws {
        // Reading a transposed view as row-major yields plausible garbage — near-zero
        // cosine on data that is actually correct — so this has to fail loudly.
        var data = npy(descr: "<f4", shape: [2, 2], payload: bytes([Float](arrayLiteral: 1, 2, 3, 4)))
        let text = String(data: data, encoding: .isoLatin1)!
            .replacingOccurrences(of: "'fortran_order': False", with: "'fortran_order': True ")
        data = text.data(using: .isoLatin1)!
        let url = try write(data)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NpyArray.NpyError.self) { try NpyArray.load(url) }
    }

    @Test("A file that isn't .npy is refused")
    func rejectsNonNpy() throws {
        let url = try write(Data(repeating: 0x41, count: 128))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NpyArray.NpyError.self) { try NpyArray.load(url) }
    }

    @Test("An unsupported dtype is named rather than guessed at")
    func rejectsUnsupportedDType() throws {
        // uint32 in particular: it used to be folded into int32, which misreads anything
        // at or above 2^31 as negative.
        let url = try write(npy(descr: "<u4", shape: [2], payload: bytes([UInt32](arrayLiteral: 1, 2))))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NpyArray.NpyError.self) { try NpyArray.load(url) }
    }
}
