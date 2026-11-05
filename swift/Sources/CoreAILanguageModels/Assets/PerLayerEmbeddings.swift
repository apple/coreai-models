// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Loads an externalized Per-Layer Embeddings (PLE) table and gathers per-token rows to feed the
/// `ple_embeddings` graph input.
///
/// The table (one row of `numLayers * perLayerDim` values per vocabulary token) is multiple
/// gigabytes, so the export writes it to a sidecar rather than into the graph. The file is mapped,
/// not read. It comes in one of two formats, told apart by the `dtype` entry of the safetensors
/// `__metadata__`:
///
/// - INT8 (`embed_tokens_per_layer`, shaped `[vocabSize, rowWidth]`): the graph dequantizes the
///   gathered rows with the scale and zero point baked in at export time, so a gather copies the
///   bytes as they are.
/// - Native-quantized (`dtype` of `native_quantized`, the QAT checkpoints): packed low-bit rows
///   (`embed_tokens_per_layer`, `uint8`, `[vocabSize, packedRowWidth]`) and a per-row, per-block
///   float16 scale (`embed_tokens_per_layer_scale`, `[vocabSize, numBlocks]`). The graph never sees
///   the quantized table: a gather unpacks and dequantizes the rows here and feeds the graph a
///   float16 `ple_embeddings` input.
struct PerLayerEmbeddings: Sendable {
    private let tensor: SafeTensorsReader
    /// The per-block scales of a native-quantized table; nil for an INT8 one.
    private let scales: SafeTensorsReader?
    /// Number of vocabulary rows.
    let vocabSize: Int
    /// Elements per token row (`numLayers * perLayerDim`), after dequantization.
    let rowWidth: Int
    /// Whether the table is native-quantized rather than INT8.
    let isNativeQuantized: Bool
    /// Bits per packed element; 8 for an INT8 table.
    private let nbits: Int
    /// Elements per scale block; 0 for an INT8 table.
    private let blockSize: Int
    /// Scales per row; 0 for an INT8 table.
    private let numBlocks: Int
    /// Bytes per row as stored.
    private let packedRowWidth: Int

    private static let tensorName = "embed_tokens_per_layer"
    private static let scaleName = "embed_tokens_per_layer_scale"
    private static let nativeQuantizedDType = "native_quantized"

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
        guard tensor.shape.count == 2 else {
            throw PLEError.invalidTable("expected a 2-D table, got shape \(tensor.shape)")
        }
        self.tensor = tensor
        self.vocabSize = tensor.shape[0]

        if tensor.metadata["dtype"] == Self.nativeQuantizedDType {
            guard tensor.dtype == "U8" else {
                throw PLEError.invalidTable("expected packed uint8 (U8) data, got \(tensor.dtype)")
            }
            guard
                let nbitsText = tensor.metadata["nbits"], let nbits = Int(nbitsText),
                [1, 2, 4, 8].contains(nbits),
                let widthText = tensor.metadata["ple_total_dim"], let rowWidth = Int(widthText), rowWidth > 0
            else {
                throw PLEError.invalidTable(
                    "a native-quantized table needs nbits (1, 2, 4 or 8) and a positive ple_total_dim in its metadata")
            }
            let scales = try SafeTensorsReader(url: url, singleTensor: Self.scaleName)
            guard scales.dtype == "F16", scales.shape.count == 2, scales.shape[0] == tensor.shape[0] else {
                throw PLEError.invalidTable(
                    "expected one float16 (F16) scale row per vocabulary row, got \(scales.dtype) \(scales.shape)")
            }
            let numBlocks = scales.shape[1]
            guard rowWidth % numBlocks == 0 else {
                throw PLEError.invalidTable("\(numBlocks) scale blocks do not divide a row of \(rowWidth) values")
            }
            let valuesPerByte = 8 / nbits
            let packedRowWidth = (rowWidth + valuesPerByte - 1) / valuesPerByte
            // Every row and every scale row must lie inside the mapping, or a valid token id could
            // index past it (SIGBUS) during gather.
            let (packedBytes, packedOverflow) = tensor.shape[0].multipliedReportingOverflow(by: packedRowWidth)
            let (scaleElements, scaleOverflow) = scales.shape[0].multipliedReportingOverflow(by: numBlocks)
            guard
                tensor.shape[1] == packedRowWidth, !packedOverflow, packedBytes == tensor.byteCount,
                !scaleOverflow, scaleElements * MemoryLayout<Float16>.size == scales.byteCount
            else {
                throw PLEError.invalidTable(
                    "shapes \(tensor.shape) and \(scales.shape) do not match \(tensor.byteCount) and "
                        + "\(scales.byteCount) bytes for \(nbits)-bit rows of \(rowWidth) values")
            }
            self.scales = scales
            self.isNativeQuantized = true
            self.nbits = nbits
            self.rowWidth = rowWidth
            self.numBlocks = numBlocks
            self.blockSize = rowWidth / numBlocks
            self.packedRowWidth = packedRowWidth
        } else {
            guard tensor.dtype == "I8" else {
                throw PLEError.invalidTable("expected INT8 (I8) data, got \(tensor.dtype)")
            }
            // Every row must lie inside the mapping, or a valid token id could index past
            // it (SIGBUS) during gather.
            let (elements, overflow) = tensor.shape[0].multipliedReportingOverflow(by: tensor.shape[1])
            guard !overflow, elements == tensor.byteCount else {
                throw PLEError.invalidTable(
                    "shape \(tensor.shape) does not match \(tensor.byteCount) bytes of INT8 data")
            }
            self.scales = nil
            self.isNativeQuantized = false
            self.nbits = 8
            self.rowWidth = tensor.shape[1]
            self.numBlocks = 0
            self.blockSize = 0
            self.packedRowWidth = tensor.shape[1]
        }
    }

    /// Copies the INT8 PLE rows for `tokenIDs` into `dest`, a buffer holding
    /// `batchSize * rowWidth` INT8 values laid out row-major (token-major). For an INT8 table only.
    ///
    /// Rows for token ids outside the table, and for padding slots beyond
    /// `tokenIDs.count`, are left as whatever `dest` already contains (callers pass a
    /// zeroed buffer).
    func gather(
        tokenIDs: some Collection<Int32>, batchSize: Int, into dest: UnsafeMutableBufferPointer<Int8>
    ) {
        precondition(!isNativeQuantized, "the INT8 gather needs an INT8 table")
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

    /// Unpacks and dequantizes the PLE rows for `tokenIDs` into `dest`, a buffer holding
    /// `batchSize * rowWidth` float16 values laid out row-major, ready to feed `ple_embeddings`. For a
    /// native-quantized table only.
    ///
    /// The packing is the checkpoint's own: `8 / nbits` values per byte, low-order bits first. Each
    /// value is stored as an unsigned field holding the signed value plus `2^(nbits - 1)` (offset
    /// binary, not two's complement), so the stored bytes are unsigned and the decoded values are
    /// signed; for 2 bits, fields 0...3 decode to -2...1. A value times its block's scale is the
    /// dequantized element.
    ///
    /// Rows for token ids outside the table, and for padding slots beyond `tokenIDs.count`, are left
    /// as whatever `dest` already contains (callers pass a zeroed buffer).
    func gather(
        tokenIDs: some Collection<Int32>, batchSize: Int, into dest: UnsafeMutableBufferPointer<Float16>
    ) {
        guard let scales else { preconditionFailure("the float16 gather needs a native-quantized table") }
        precondition(dest.count >= batchSize * rowWidth, "PLE destination buffer too small")
        let valuesPerByte = 8 / nbits
        // Built in `Int` before narrowing: `UInt8(1) << 8` would mask to 0, not trap.
        let mask = UInt8((1 << nbits) - 1)
        let bias = Int32(1 << (nbits - 1))

        tensor.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            scales.data.withUnsafeBytes { (scaleRaw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress, let scaleBase = scaleRaw.baseAddress else { return }
                let packed = base.advanced(by: tensor.dataStart).assumingMemoryBound(to: UInt8.self)
                let blockScales = scaleBase.advanced(by: scales.dataStart).assumingMemoryBound(to: Float16.self)
                for (i, tokenID) in tokenIDs.prefix(batchSize).enumerated() {
                    let token = Int(tokenID)
                    guard token >= 0, token < vocabSize else { continue }
                    let packedRow = packed.advanced(by: token * packedRowWidth)
                    let scaleRow = blockScales.advanced(by: token * numBlocks)
                    let dstRow = dest.baseAddress!.advanced(by: i * rowWidth)
                    for e in 0..<rowWidth {
                        let raw = (packedRow[e / valuesPerByte] >> ((e % valuesPerByte) * nbits)) & mask
                        dstRow[e] = Float16(Int32(raw) - bias) * scaleRow[e / blockSize]
                    }
                }
            }
        }
    }
}
