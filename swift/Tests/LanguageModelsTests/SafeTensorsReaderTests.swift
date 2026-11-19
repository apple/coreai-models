// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// A crafted `.safetensors` header is rejected with a message, never read past the file: the draft
/// head pack turns each entry's offsets into pointer offsets into a mapping of the file.
@Suite("SafeTensors reader")
struct SafeTensorsReaderTests {
    /// Writes `header` (a JSON value) and `body` as a safetensors file. `headerLength` overrides the
    /// 8-byte length prefix.
    private func write(header: Any, body: Data = Data(count: 16), headerLength: UInt64? = nil) throws -> URL {
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys, .fragmentsAllowed])
        var file = Data()
        withUnsafeBytes(of: (headerLength ?? UInt64(json.count)).littleEndian) { file.append(contentsOf: $0) }
        file.append(json)
        file.append(body)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("st-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    private func entry(shape: [Int] = [2, 2], offsets: [Int] = [0, 8]) -> [String: Any] {
        ["dtype": "F16", "shape": shape, "data_offsets": offsets]
    }

    private func headerError(_ url: URL) -> String {
        do {
            _ = try SafeTensorsReader.header(of: url)
            return ""
        } catch {
            return "\(error)"
        }
    }

    @Test("A well-formed file reads every tensor, its offset from the start of the file, and its metadata")
    func wellFormed() throws {
        let url = try write(header: ["a": entry(), "b": entry(offsets: [8, 16]), "__metadata__": ["k": "v"]])
        let header = try SafeTensorsReader.header(of: url)
        #expect(header.metadata == ["k": "v"])
        #expect(header.tensors["b"]?.byteCount == 8)
        let dataStart = header.fileSize - 16
        #expect(header.tensors["a"]?.fileOffset == dataStart && header.tensors["b"]?.fileOffset == dataStart + 8)
        let single = try SafeTensorsReader(url: url, singleTensor: "b")
        #expect(single.dataStart == dataStart + 8 && single.byteCount == 8 && single.shape == [2, 2])
    }

    @Test("A header length past the end of the file is rejected")
    func headerPastFile() throws {
        #expect(!headerError(try write(header: ["a": entry()], headerLength: 1 << 20)).isEmpty)
    }

    @Test("Offsets past the data region, or reversed, are rejected", arguments: [[0, 17], [8, 4], [-1, 4]])
    func badOffsets(offsets: [Int]) throws {
        #expect(headerError(try write(header: ["a": entry(offsets: offsets)])).contains("data_offsets"))
    }

    @Test("A negative or zero dimension is rejected", arguments: [[-1, 2], [0, 2]])
    func badShape(shape: [Int]) throws {
        #expect(headerError(try write(header: ["a": entry(shape: shape)])).contains("'a'"))
    }

    @Test("Metadata that is not a map of strings is rejected")
    func badMetadata() throws {
        #expect(headerError(try write(header: ["a": entry(), "__metadata__": ["k": 1]])).contains("__metadata__"))
    }

    @Test("A header that is not a JSON object is rejected")
    func notAnObject() throws {
        #expect(headerError(try write(header: [1, 2, 3])).contains("not a JSON object"))
    }

    @Test("Both readers reject the same file, and the single-tensor reader names a missing tensor")
    func readersAgree() throws {
        let url = try write(header: ["a": entry(shape: [0, 2])])
        #expect(throws: (any Error).self) { _ = try SafeTensorsReader(url: url, singleTensor: "a") }
        let good = try write(header: ["a": entry()])
        #expect(throws: (any Error).self) { _ = try SafeTensorsReader(url: good, singleTensor: "missing") }
    }
}
