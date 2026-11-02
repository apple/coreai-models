// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

// The draft head's candidate scan: the masked-embedder table (`draft_scan.bin` and its
// JSON index) and the CPU arithmetic that scores candidate rows against the head's state.

// MARK: - Weight pack (flat blob + json index)

enum DraftPackError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self {
        case .message(let m): return m
        }
    }
}

/// mmap over the scan table blob (`draft_scan.bin`), plus its json index (`draft_scan.json`). fp16 tensors, raw pointers into page cache — the kernels walk the bytes
/// directly with no per-step parsing.
///
/// Read-only after init, so the mapping may be read from any task.
///
/// The index is validated before anything is read through it (see ``validate(index:length:name:)``
/// and ``require(_:index:name:)``), so a stale, foreign or hand-edited pack fails to load with a
/// message that names the problem instead of trapping or reading out of bounds at the first draft.
final class DraftPack: @unchecked Sendable {
    struct Entry: Decodable {
        let offset: Int
        let bytes: Int
        let shape: [Int]
        let dtype: String
    }
    struct Index: Decodable {
        /// Sidecar format version; absent in packs written before it existed, which read as 1.
        let version: Int?
        let vocab: Int
        let centroids: Int
        let perCentroid: Int
        let layers: Int
        let hidden: Int?
        let heads: Int?
        let headDim: Int?
        let fullHeadDim: Int?
        let ffn: Int?
        let tensors: [String: Entry]

        enum CodingKeys: String, CodingKey {
            case version, vocab, centroids, layers, hidden, heads, ffn, tensors
            case perCentroid = "per_centroid"
            case headDim = "head_dim"
            case fullHeadDim = "full_head_dim"
        }
    }

    /// Newest sidecar format this reader understands. Exporters write the same number.
    static let supportedVersion = 1

    /// What a consumer reads from the pack.
    enum Role {
        /// The masked embedder's candidate scan (`draft_scan.bin`): the permuted output table and
        /// the permutation.
        case scan
    }

    private let base: UnsafeMutableRawPointer
    private let length: Int
    let index: Index

    /// - Parameter role: when given, the pack must also hold the tensors that role reads, with
    ///   shapes and dtypes that agree with its geometry.
    /// - Parameters:
    ///   - blob: the flat tensor file.
    ///   - indexURL: its JSON index; by default the blob's path with a `.json` extension.
    init(blob: URL, index indexURL: URL? = nil, role: Role? = nil) throws {
        let name = blob.lastPathComponent
        let meta = indexURL ?? blob.deletingPathExtension().appendingPathExtension("json")
        let metaData: Data
        do {
            metaData = try Data(contentsOf: meta)
        } catch {
            throw DraftPackError.message("cannot read \(meta.lastPathComponent): \(error.localizedDescription)")
        }
        do {
            index = try JSONDecoder().decode(Index.self, from: metaData)
        } catch {
            throw DraftPackError.message(
                "\(meta.lastPathComponent) is not a draft pack index (\(error.localizedDescription)); "
                    + "re-export it with models/gemma4/export_drafter.py")
        }
        let fd = open(blob.path, O_RDONLY)
        guard fd >= 0 else { throw DraftPackError.message("cannot open \(blob.path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw DraftPackError.message("cannot stat \(blob.path)") }
        length = Int(st.st_size)
        try Self.validate(index: index, length: length, name: name)
        if let role { try Self.require(role, index: index, name: name) }
        guard length > 0,
            let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED
        else {
            throw DraftPackError.message("cannot mmap \(blob.path)")
        }
        base = p
    }

    deinit { munmap(base, length) }

    func has(_ name: String) -> Bool { index.tensors[name] != nil }
    func shape(_ name: String) -> [Int] { index.tensors[name]?.shape ?? [] }

    /// fp16 view of a tensor. A pack loaded with a ``Role`` holds every tensor that role reads,
    /// with the dtype it reads it as, so a violation here is a programming error.
    func f16(_ name: String) -> UnsafePointer<Float16> {
        guard let e = index.tensors[name], e.dtype == "float16" else {
            preconditionFailure("drafter weight '\(name)' is not a float16 tensor of the loaded pack")
        }
        return UnsafePointer(base.advanced(by: e.offset).assumingMemoryBound(to: Float16.self))
    }

    func i8(_ name: String) -> UnsafePointer<Int8>? {
        guard let e = index.tensors[name], e.dtype == "int8" else { return nil }
        return UnsafePointer(base.advanced(by: e.offset).assumingMemoryBound(to: Int8.self))
    }

    func i32(_ name: String) -> UnsafePointer<Int32> {
        guard let e = index.tensors[name], e.dtype == "int32" else {
            preconditionFailure("drafter weight '\(name)' is not an int32 tensor of the loaded pack")
        }
        return UnsafePointer(base.advanced(by: e.offset).assumingMemoryBound(to: Int32.self))
    }

    // MARK: Validation

    private static let elementSize = ["float16": 2, "float32": 4, "int8": 1, "int32": 4]

    /// Checks the index against the blob: format version, geometry, and that every tensor has a
    /// known dtype, a byte count equal to its shape's, an aligned offset, and lies inside the blob.
    static func validate(index: Index, length: Int, name: String) throws {
        let repack = "re-export it with models/gemma4/export_drafter.py"
        let version = index.version ?? 1
        guard version >= 1, version <= supportedVersion else {
            throw DraftPackError.message(
                version > supportedVersion
                    ? "\(name) has format version \(version), newer than the \(supportedVersion) this runner "
                        + "reads; update the runner or \(repack) with a matching exporter"
                    : "\(name) has invalid format version \(version); \(repack)")
        }
        guard index.centroids > 0, index.perCentroid > 0, index.vocab > 0, index.layers >= 0 else {
            throw DraftPackError.message(
                "\(name) declares an empty geometry (vocab \(index.vocab), centroids \(index.centroids), "
                    + "per_centroid \(index.perCentroid), layers \(index.layers)); \(repack)")
        }
        _ = try candidateRows(index: index, name: name)
        for (tensor, entry) in index.tensors.sorted(by: { $0.key < $1.key }) {
            guard let size = elementSize[entry.dtype] else {
                throw DraftPackError.message(
                    "tensor '\(tensor)' in \(name) has unsupported dtype '\(entry.dtype)' "
                        + "(supported: \(elementSize.keys.sorted().joined(separator: ", ")))")
            }
            var count = 1
            for dim in entry.shape {
                let (product, overflow) = count.multipliedReportingOverflow(by: dim)
                guard dim >= 0, !overflow else {
                    throw DraftPackError.message("tensor '\(tensor)' in \(name) has invalid shape \(entry.shape)")
                }
                count = product
            }
            let (expected, overflow) = count.multipliedReportingOverflow(by: size)
            guard !overflow, entry.bytes == expected else {
                throw DraftPackError.message(
                    "tensor '\(tensor)' in \(name) declares \(entry.bytes) bytes, but shape \(entry.shape) "
                        + "of \(entry.dtype) is \(overflow ? -1 : expected) bytes")
            }
            let (end, endOverflow) = entry.offset.addingReportingOverflow(entry.bytes)
            guard entry.offset >= 0, !endOverflow, end <= length else {
                throw DraftPackError.message(
                    "tensor '\(tensor)' (offset \(entry.offset), \(entry.bytes) bytes) lies outside "
                        + "the \(length)-byte pack \(name)")
            }
            guard entry.offset % size == 0 else {
                throw DraftPackError.message(
                    "tensor '\(tensor)' in \(name) is at offset \(entry.offset), not a multiple of its "
                        + "\(size)-byte element size")
            }
        }
    }

    /// The number of candidate rows, `centroids * per_centroid`. The index is read from a file, so
    /// a product that does not fit an `Int` is reported, not trapped on.
    static func candidateRows(index: Index, name: String) throws -> Int {
        let (rows, overflow) = index.centroids.multipliedReportingOverflow(by: index.perCentroid)
        guard !overflow else {
            throw DraftPackError.message(
                "\(name) declares \(index.centroids) centroids of \(index.perCentroid) candidates, "
                    + "which is more rows than fit in an Int; re-export it with models/gemma4/export_drafter.py")
        }
        return rows
    }

    /// Checks the tensors `role` reads are present with the right dtype and shape.
    static func require(_ role: Role, index: Index, name: String) throws {
        let rows = try candidateRows(index: index, name: name)
        func tensor(_ tensorName: String, _ dtype: String, shape: [Int?]) throws {
            guard let entry = index.tensors[tensorName] else {
                throw DraftPackError.message(
                    "\(name) has no tensor '\(tensorName)'; re-export it with models/gemma4/export_drafter.py")
            }
            guard entry.dtype == dtype else {
                throw DraftPackError.message(
                    "tensor '\(tensorName)' in \(name) is \(entry.dtype), expected \(dtype)")
            }
            guard entry.shape.count == shape.count,
                zip(entry.shape, shape).allSatisfy({ $1 == nil || $0 == $1 })
            else {
                throw DraftPackError.message(
                    "tensor '\(tensorName)' in \(name) has shape \(entry.shape), expected "
                        + "\(shape.map { $0.map(String.init) ?? "_" })")
            }
        }
        // The scan: one row per candidate, as wide as the head's state.
        try tensor("embed_perm_f16", "float16", shape: [rows, nil])
        try tensor("token_ordering_i32", "int32", shape: [rows])
    }
}

package enum DraftScan {
    /// Centroids whose candidates the masked embedder scores when no count is requested.
    package static let defaultTopK = 32

    /// The centroid count to score: `requested` (or ``defaultTopK``), at most `centroids`.
    static func topK(requested: Int?, centroids: Int) throws -> Int {
        guard let requested else { return min(defaultTopK, centroids) }
        guard requested >= 1 else {
            throw DraftPackError.message("the draft head's top-k must be a positive integer (got \(requested))")
        }
        return min(requested, centroids)
    }

    /// dot(x, weights) with weights fp16 contiguous: the masked embedder's per-candidate score. Reads
    /// the first `n` entries of `x` and of `weights`, so `x` must hold at least `n`.
    static func dot(_ x: [Float], _ weights: UnsafePointer<Float16>, n: Int) -> Float {
        guard n > 0 else { return 0 }
        precondition(x.count >= n, "dot reads \(n) values from a vector of \(x.count)")
        return x.withUnsafeBufferPointer { xp -> Float in
            let xb = xp.baseAddress!
            let row = UnsafeRawPointer(weights)
            var a0 = SIMD8<Float>()
            var i = 0
            while i + 8 <= n {
                a0 +=
                    SIMD8<Float>(row.loadUnaligned(fromByteOffset: i * 2, as: SIMD8<Float16>.self))
                    * UnsafeRawPointer(xb + i).loadUnaligned(as: SIMD8<Float>.self)
                i += 8
            }
            var acc = a0.sum()
            while i < n {
                acc += Float(weights[i]) * xb[i]
                i += 1
            }
            return acc
        }
    }
}
