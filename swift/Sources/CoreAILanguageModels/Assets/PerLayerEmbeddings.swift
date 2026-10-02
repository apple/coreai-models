// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Loads an externalized INT8 Per-Layer Embeddings (PLE) table and gathers
/// per-token rows to feed the `ple_embeddings` graph input.
///
/// The table (one INT8 row of `numLayers * perLayerDim` values per vocabulary
/// token) is multiple gigabytes, so the export writes it to a sidecar rather than
/// into the graph: a single-tensor safetensors file holding `embed_tokens_per_layer`,
/// shaped `[vocabSize, rowWidth]`. It is mapped, not read; the graph dequantizes the
/// gathered rows with the scale and zero point baked in at export time.
struct PerLayerEmbeddings: Sendable {
    private let tensor: SafeTensorsReader
    /// Number of vocabulary rows.
    let vocabSize: Int
    /// INT8 elements per token row (`numLayers * perLayerDim`).
    let rowWidth: Int

    private static let tensorName = "embed_tokens_per_layer"

    enum PLEError: Error, CustomStringConvertible {
        case invalidTable(String)

        var description: String {
            switch self {
            case .invalidTable(let message): return "PLE table invalid: \(message)"
            }
        }
    }

    init(contentsOf url: URL) throws {
        let tensor = try SafeTensorsReader(url: url, singleTensor: Self.tensorName)
        guard tensor.dtype == "I8" else {
            throw PLEError.invalidTable("expected INT8 (I8) data, got \(tensor.dtype)")
        }
        guard tensor.shape.count == 2 else {
            throw PLEError.invalidTable("expected a 2-D table, got shape \(tensor.shape)")
        }
        // Every row must lie inside the mapping, or a valid token id could index past
        // it (SIGBUS) during gather.
        let (elements, overflow) = tensor.shape[0].multipliedReportingOverflow(by: tensor.shape[1])
        guard !overflow, elements == tensor.byteCount else {
            throw PLEError.invalidTable(
                "shape \(tensor.shape) does not match \(tensor.byteCount) bytes of INT8 data")
        }
        self.tensor = tensor
        self.vocabSize = tensor.shape[0]
        self.rowWidth = tensor.shape[1]
    }

    /// Copies the PLE rows for `tokenIDs` into `dest`, a buffer holding
    /// `batchSize * rowWidth` INT8 values laid out row-major (token-major).
    ///
    /// Rows for token ids outside the table, and for padding slots beyond
    /// `tokenIDs.count`, are left as whatever `dest` already contains (callers pass a
    /// zeroed buffer).
    func gather(
        tokenIDs: some Collection<Int32>, batchSize: Int, into dest: UnsafeMutableBufferPointer<Int8>
    ) {
        precondition(dest.count >= batchSize * rowWidth, "PLE destination buffer too small")
        tensor.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            let src = base.advanced(by: tensor.dataStart).assumingMemoryBound(to: Int8.self)
            for (i, tokenID) in tokenIDs.prefix(batchSize).enumerated() {
                let token = Int(tokenID)
                guard token >= 0, token < vocabSize else { continue }
                let srcRow = src.advanced(by: token * rowWidth)
                let dstRow = dest.baseAddress!.advanced(by: i * rowWidth)
                dstRow.update(from: srcRow, count: rowWidth)
            }
        }
    }
}
