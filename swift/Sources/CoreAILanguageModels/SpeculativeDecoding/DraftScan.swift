// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
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
        /// Positions the head's sliding layers attend to (a ``Role/head`` pack).
        let slidingWindow: Int?
        /// RoPE base of the sliding and of the full-attention layers (a ``Role/head`` pack).
        let slidingTheta: Float?
        let fullTheta: Float?
        /// Share of the full-attention layer's frequency pairs that rotate (a ``Role/head`` pack).
        let fullPartialRotaryFactor: Float?
        let tensors: [String: Entry]

        enum CodingKeys: String, CodingKey {
            case version, vocab, centroids, layers, hidden, heads, ffn, tensors
            case perCentroid = "per_centroid"
            case headDim = "head_dim"
            case fullHeadDim = "full_head_dim"
            case slidingWindow = "sliding_window"
            case slidingTheta = "sliding_theta"
            case fullTheta = "full_theta"
            case fullPartialRotaryFactor = "full_partial_rotary_factor"
        }
    }

    /// Newest sidecar format this reader understands. Exporters write the same number.
    static let supportedVersion = 1

    /// What a consumer reads from the pack.
    enum Role {
        /// The masked embedder's candidate scan (`draft_scan.bin`): the permuted output table and
        /// the permutation.
        case scan
        /// The whole draft head for ``CPUDrafter`` (`draft_head.safetensors`, from
        /// `models/gemma4/export_draft_pack.py`): the scan, the centroids and every layer.
        case head
    }

    private let base: UnsafeMutableRawPointer
    private let length: Int
    let index: Index

    /// - Parameter role: when given, the pack must also hold the tensors that role reads, with
    ///   shapes and dtypes that agree with its geometry.
    /// - Parameters:
    ///   - blob: the flat tensor file.
    ///   - indexURL: its JSON index; by default the blob's path with a `.json` extension.
    convenience init(blob: URL, index indexURL: URL? = nil, role: Role? = nil) throws {
        let meta = indexURL ?? blob.deletingPathExtension().appendingPathExtension("json")
        let metaData: Data
        do {
            metaData = try Data(contentsOf: meta)
        } catch {
            throw DraftPackError.message("cannot read \(meta.lastPathComponent): \(error.localizedDescription)")
        }
        let index: Index
        do {
            index = try JSONDecoder().decode(Index.self, from: metaData)
        } catch {
            throw DraftPackError.message(
                "\(meta.lastPathComponent) is not a draft pack index (\(error.localizedDescription)); "
                    + "re-export it with models/gemma4/export_drafter.py")
        }
        try self.init(index: index, mapping: blob, role: role)
    }

    /// The `__metadata__["format"]` of a draft head written by `models/gemma4/export_draft_pack.py`.
    static let safetensorsFormat = "gemma4_draft_head"

    /// A pack stored as one `.safetensors` file, whose `__metadata__` holds the index's geometry.
    convenience init(safetensors url: URL, role: Role? = nil) throws {
        let header: SafeTensorsReader.Header
        do {
            header = try SafeTensorsReader.header(of: url)
        } catch {
            throw DraftPackError.message("\(url.lastPathComponent) is not a readable safetensors file: \(error)")
        }
        try self.init(index: Self.index(of: header, name: url.lastPathComponent), mapping: url, role: role)
    }

    /// The index a safetensors header describes: its tensors, and the geometry in `__metadata__`.
    static func index(of header: SafeTensorsReader.Header, name: String) throws -> Index {
        let repack = "re-export it with models/gemma4/export_draft_pack.py"
        let meta = header.metadata
        guard meta["format"] == safetensorsFormat else {
            throw DraftPackError.message(
                "\(name) is not a draft head pack (its metadata format is \(meta["format"].map { "'\($0)'" } ?? "missing")); "
                    + repack)
        }
        func int(_ key: String) throws -> Int? {
            guard let raw = meta[key] else { return nil }
            guard let value = Int(raw) else {
                throw DraftPackError.message("\(name) metadata '\(key)' is '\(raw)', not an integer; \(repack)")
            }
            return value
        }
        func float(_ key: String) throws -> Float? {
            guard let raw = meta[key] else { return nil }
            guard let value = Float(raw) else {
                throw DraftPackError.message("\(name) metadata '\(key)' is '\(raw)', not a number; \(repack)")
            }
            return value
        }
        func required(_ key: String) throws -> Int {
            guard let value = try int(key) else {
                throw DraftPackError.message("\(name) metadata has no '\(key)'; \(repack)")
            }
            return value
        }
        let dtypes = ["F16": "float16", "F32": "float32", "I8": "int8", "I32": "int32"]
        return Index(
            version: try int("version"), vocab: try required("vocab"), centroids: try required("centroids"),
            perCentroid: try required("per_centroid"), layers: try required("layers"), hidden: try int("hidden"),
            heads: try int("heads"), headDim: try int("head_dim"), fullHeadDim: try int("full_head_dim"),
            ffn: try int("ffn"), slidingWindow: try int("sliding_window"), slidingTheta: try float("sliding_theta"),
            fullTheta: try float("full_theta"), fullPartialRotaryFactor: try float("full_partial_rotary_factor"),
            tensors: header.tensors.mapValues {
                Entry(offset: $0.fileOffset, bytes: $0.byteCount, shape: $0.shape, dtype: dtypes[$0.dtype] ?? $0.dtype)
            })
    }

    /// Validates `index` against the file at `url` and maps it.
    private init(index: Index, mapping url: URL, role: Role?) throws {
        let name = url.lastPathComponent
        self.index = index
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw DraftPackError.message("cannot open \(url.path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw DraftPackError.message("cannot stat \(url.path)") }
        length = Int(st.st_size)
        try Self.validate(index: index, length: length, name: name)
        if let role { try Self.require(role, index: index, name: name) }
        guard length > 0,
            let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED
        else {
            throw DraftPackError.message("cannot mmap \(url.path)")
        }
        base = p
        // Start reading the table in now, while the target loads and prefills. Each drafted token
        // scores scattered blocks of it; mapped lazily, the first decode after a load stalls on
        // each block's first read from storage. The hint is advisory: a failure only loses the
        // head start.
        _ = madvise(p, length, MADV_WILLNEED)
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
        let exporter = role == .head ? "export_draft_pack.py" : "export_drafter.py"
        func tensor(_ tensorName: String, _ dtype: String, shape: [Int?]) throws {
            guard let entry = index.tensors[tensorName] else {
                throw DraftPackError.message(
                    "\(name) has no tensor '\(tensorName)'; re-export it with models/gemma4/\(exporter)")
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
        guard role == .head else { return }

        let repack = "re-export it with models/gemma4/export_draft_pack.py"
        func positive<T: Comparable & Numeric>(_ value: T?, _ key: String) throws -> T {
            guard let value, value > 0 else {
                throw DraftPackError.message("\(name) does not declare a positive '\(key)'; \(repack)")
            }
            return value
        }
        let hidden = try positive(index.hidden, "hidden")
        let heads = try positive(index.heads, "heads")
        let headDim = try positive(index.headDim, "head_dim")
        let fullHeadDim = try positive(index.fullHeadDim, "full_head_dim")
        let ffn = try positive(index.ffn, "ffn")
        _ = try positive(index.slidingWindow, "sliding_window")
        _ = try positive(index.slidingTheta, "sliding_theta")
        _ = try positive(index.fullTheta, "full_theta")
        _ = try positive(index.fullPartialRotaryFactor, "full_partial_rotary_factor")
        guard index.layers > 0 else { throw DraftPackError.message("\(name) declares no layers; \(repack)") }
        guard headDim % 2 == 0, fullHeadDim % 2 == 0 else {
            throw DraftPackError.message("\(name) declares an odd head width, which RoPE cannot rotate; \(repack)")
        }
        try tensor("embed_perm_f16", "float16", shape: [rows, hidden])
        try tensor("centroids_f16", "float16", shape: [index.centroids, hidden])
        try tensor("pre_projection.weight", "float16", shape: [hidden, nil])
        try tensor("post_projection.weight", "float16", shape: [nil, hidden])
        try tensor("model.norm.weight", "float16", shape: [hidden])
        func projection(_ tensorName: String, rows: Int, columns: Int) throws {
            try tensor(tensorName, "float16", shape: [rows, columns])
        }
        // The last layer attends to the full cache, the others to the sliding one.
        for layer in 0..<index.layers {
            let dim = layer == index.layers - 1 ? fullHeadDim : headDim
            let p = "model.layers.\(layer)."
            try projection(p + "self_attn.q_proj.weight", rows: heads * dim, columns: hidden)
            try projection(p + "self_attn.o_proj.weight", rows: hidden, columns: heads * dim)
            try projection(p + "mlp.gate_proj.weight", rows: ffn, columns: hidden)
            try projection(p + "mlp.up_proj.weight", rows: ffn, columns: hidden)
            try projection(p + "mlp.down_proj.weight", rows: hidden, columns: ffn)
            try tensor(p + "self_attn.q_norm.weight", "float16", shape: [dim])
            for norm in [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                try tensor(p + norm + ".weight", "float16", shape: [hidden])
            }
            try tensor(p + "layer_scalar", "float16", shape: [1])
        }
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

    /// The masked embedder's candidate scan over a pack's permuted output table: score only the
    /// tokens of the `topK` best centroids and take the best. Holds its scratch rows, so one
    /// drafter owns one.
    struct Scanner {
        let scan: DraftPack
        let topK: Int
        /// One centroid's candidate rows as fp32, and their scores.
        private var candidateRows: [Float] = []
        private var candidateScores: [Float] = []

        init(pack: DraftPack, topK: Int) {
            self.scan = pack
            self.topK = topK
        }

        /// The best-scoring token for the head's state `x`, given its centroid logits `cl`.
        mutating func bestToken(x: [Float], centroidLogits cl: [Float]) -> Int32 {
            let perCentroid = scan.index.perCentroid
            let hidden = scan.shape("embed_perm_f16")[1]
            var topV = [Float](repeating: -.greatestFiniteMagnitude, count: topK)
            var topI = [Int](repeating: 0, count: topK)
            for c in 0..<scan.index.centroids where cl[c] > topV[topK - 1] {
                var j = topK - 1
                while j > 0 && cl[c] > topV[j - 1] {
                    topV[j] = topV[j - 1]
                    topI[j] = topI[j - 1]
                    j -= 1
                }
                topV[j] = cl[c]
                topI[j] = c
            }
            // A centroid's candidates are contiguous rows, so each is one fp16 -> fp32 conversion and one
            // matrix-vector product. Scoring row by row through `DraftScan.dot` spent most of the scan
            // in the runtime availability check `SIMD8<Float>(SIMD8<Float16>)` makes on every call.
            let table = scan.f16("embed_perm_f16")
            let block = perCentroid * hidden
            if candidateRows.count != block { candidateRows = [Float](repeating: 0, count: block) }
            if candidateScores.count != perCentroid { candidateScores = [Float](repeating: 0, count: perCentroid) }
            var best = -Float.greatestFiniteMagnitude
            var bestSlot = 0
            x.withUnsafeBufferPointer { xp in
                candidateRows.withUnsafeMutableBufferPointer { rows in
                    candidateScores.withUnsafeMutableBufferPointer { scores in
                        for c in topI {
                            var src = vImage_Buffer(
                                data: UnsafeMutableRawPointer(mutating: table + c * block), height: 1,
                                width: vImagePixelCount(block), rowBytes: block * 2)
                            var dst = vImage_Buffer(
                                data: rows.baseAddress, height: 1, width: vImagePixelCount(block), rowBytes: block * 4)
                            guard vImageConvert_Planar16FtoPlanarF(&src, &dst, 0) == kvImageNoError else {
                                // Unconverted rows would score stale data, so this centroid is scored row by row.
                                for j in 0..<perCentroid {
                                    let slot = c * perCentroid + j
                                    let score = DraftScan.dot(x, table + slot * hidden, n: hidden)
                                    if score > best {
                                        best = score
                                        bestSlot = slot
                                    }
                                }
                                continue
                            }
                            vDSP_mmul(
                                rows.baseAddress!, 1, xp.baseAddress!, 1, scores.baseAddress!, 1,
                                vDSP_Length(perCentroid), 1, vDSP_Length(hidden))
                            var score: Float = 0
                            var j: vDSP_Length = 0
                            vDSP_maxvi(scores.baseAddress!, 1, &score, &j, vDSP_Length(perCentroid))
                            if score > best {
                                best = score
                                bestSlot = c * perCentroid + Int(j)
                            }
                        }
                    }
                }
            }
            return scan.i32("token_ordering_i32")[bestSlot]
        }
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
