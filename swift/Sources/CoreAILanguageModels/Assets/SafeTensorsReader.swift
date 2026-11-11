// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Reads one named tensor from a `.safetensors` file, memory-mapped.
///
/// Layout: `[8-byte little-endian header length][JSON header][raw tensor bytes]`.
/// Only the named tensor's header entry is read; other tensors in the file are
/// ignored.
struct SafeTensorsReader: Sendable {
    /// The mapped file contents.
    let data: Data
    /// Byte offset of the tensor's first element in `data`.
    let dataStart: Int
    /// Size of the tensor's data in bytes.
    let byteCount: Int
    /// Safetensors dtype string, e.g. `"I8"` or `"F16"`.
    let dtype: String
    let shape: [Int]
    /// The file's `__metadata__` strings; empty when the header has none.
    let metadata: [String: String]

    enum ReadError: Error, CustomStringConvertible {
        case tooSmall
        case missingTensor(String)
        case badHeader(String)

        var description: String {
            switch self {
            case .tooSmall: return "file is too small to contain a safetensors header"
            case .missingTensor(let name): return "safetensors file has no '\(name)' tensor"
            case .badHeader(let message): return "safetensors header invalid: \(message)"
            }
        }
    }

    init(url: URL, singleTensor name: String) throws {
        // Always map: `.mappedIfSafe` silently reads a multi-GB file into memory when
        // it judges mapping unsafe.
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 8 else { throw ReadError.tooSmall }

        let length = UInt64(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard let headerLength = Int(exactly: length), headerLength <= data.count - 8 else {
            throw ReadError.tooSmall
        }
        let header = data.subdata(in: (data.startIndex + 8)..<(data.startIndex + 8 + headerLength))
        guard
            let json = try JSONSerialization.jsonObject(with: header) as? [String: Any],
            let entry = json[name] as? [String: Any]
        else {
            throw ReadError.missingTensor(name)
        }

        // Both offsets are bounded by the data region, so a crafted header throws
        // rather than trapping on overflow or reading past the mapping.
        let dataRegion = data.count - 8 - headerLength
        guard
            let dtype = entry["dtype"] as? String,
            let shape = entry["shape"] as? [Int], shape.allSatisfy({ $0 > 0 }),
            let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
            0 <= offsets[0], offsets[0] <= offsets[1], offsets[1] <= dataRegion
        else {
            throw ReadError.badHeader("missing or invalid dtype, shape or data_offsets for '\(name)'")
        }

        self.data = data
        self.dataStart = 8 + headerLength + offsets[0]
        self.byteCount = offsets[1] - offsets[0]
        self.dtype = dtype
        self.shape = shape
        self.metadata = json["__metadata__"] as? [String: String] ?? [:]
    }

    /// Every tensor's entry and the `__metadata__` of a `.safetensors` file, read from its header
    /// alone.
    struct Header {
        struct Entry {
            /// Safetensors dtype string, e.g. `"F16"`.
            let dtype: String
            let shape: [Int]
            /// Byte offset of the tensor's first element from the start of the file.
            let fileOffset: Int
            let byteCount: Int
        }
        let tensors: [String: Entry]
        /// The header's `__metadata__` strings; empty when it has none.
        let metadata: [String: String]
        /// Size of the whole file in bytes.
        let fileSize: Int
    }

    /// Reads the header of the `.safetensors` file at `url`. Every entry's offsets are checked
    /// against the data region, so a crafted header throws rather than pointing past the file.
    static func header(of url: URL) throws -> Header {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = Int(try handle.seekToEnd())
        try handle.seek(toOffset: 0)
        guard fileSize >= 8, let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw ReadError.tooSmall
        }
        let length = UInt64(littleEndian: prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard let headerLength = Int(exactly: length), headerLength <= fileSize - 8,
            let headerData = try handle.read(upToCount: headerLength), headerData.count == headerLength
        else {
            throw ReadError.tooSmall
        }
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw ReadError.badHeader("the header is not a JSON object")
        }
        let dataStart = 8 + headerLength
        var tensors: [String: Header.Entry] = [:]
        var metadata: [String: String] = [:]
        for (name, value) in json {
            if name == "__metadata__" {
                guard let strings = value as? [String: String] else {
                    throw ReadError.badHeader("__metadata__ must map strings to strings")
                }
                metadata = strings
                continue
            }
            guard
                let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int], shape.allSatisfy({ $0 >= 0 }),
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                0 <= offsets[0], offsets[0] <= offsets[1], offsets[1] <= fileSize - dataStart
            else {
                throw ReadError.badHeader("missing or invalid dtype, shape or data_offsets for '\(name)'")
            }
            tensors[name] = Header.Entry(
                dtype: dtype, shape: shape, fileOffset: dataStart + offsets[0], byteCount: offsets[1] - offsets[0])
        }
        return Header(tensors: tensors, metadata: metadata, fileSize: fileSize)
    }
}
