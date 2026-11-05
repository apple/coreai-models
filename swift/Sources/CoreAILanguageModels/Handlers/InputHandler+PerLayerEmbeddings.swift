// Per-Layer Embeddings input for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Gathers Per-Layer Embedding rows into the `ple_embeddings` graph input.
///
/// Models with per-layer embeddings carry one row of `numLayers · perLayerDim` values
/// per vocabulary token. That table is multiple gigabytes, so it is externalized into a
/// sidecar artifact instead of baked into the graph (see ``PerLayerEmbeddings``); each
/// step the runner copies (and, for native-quantized checkpoints, dequantizes) the rows
/// for the current batch's tokens into the input. The legacy per-tensor INT8 format
/// keeps the graph doing the dequantization (baked-in scale/zero-point); the
/// native-quantized format dequantizes here in Swift and feeds the graph a plain float
/// input instead -- see ``PerLayerEmbeddings`` for why.
struct PerLayerEmbeddingsInputHandler: StaticInputHandler {
    static let inputName = "ple_embeddings"

    let inputNames: [String] = [inputName]

    private let table: PerLayerEmbeddings
    private let descriptors: BucketedInputDescriptors

    init(table: PerLayerEmbeddings, descriptors: BucketedInputDescriptors) throws {
        for descriptor in descriptors.descriptors {
            try Self.validate(table: table, rowWidth: descriptor.shape.last, scalarType: descriptor.scalarType)
        }
        self.table = table
        self.descriptors = descriptors
    }

    /// Throws unless a graph input with this row width and scalar type can take the table's rows:
    /// float16 for a native-quantized table, INT8 otherwise.
    static func validate(table: PerLayerEmbeddings, rowWidth: Int?, scalarType: NDArray.ScalarType) throws {
        guard rowWidth == table.rowWidth else {
            throw InferenceRuntimeError.invalidState(
                "PLE row width mismatch: graph expects \(rowWidth ?? -1), table has \(table.rowWidth)")
        }
        let expected: NDArray.ScalarType = table.isNativeQuantized ? .float16 : .int8
        guard scalarType == expected else {
            throw InferenceRuntimeError.invalidState(
                "PLE type mismatch: graph expects \(scalarType), "
                    + "\(table.isNativeQuantized ? "native-quantized" : "INT8") table needs \(expected)")
        }
    }

    func registerBuffers(into buffers: inout InputBuffers) {
        descriptors.registerBuffers(name: Self.inputName, into: &buffers)
    }

    func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let key = StaticBucketKey(batchSize: context.batchSize, contextBucket: context.contextBucket)
        let descriptor = try descriptors.require(key, input: Self.inputName)

        buffers.ensureCapacity(name: Self.inputName, descriptor: descriptor)

        let elementCount = descriptor.shape.reduce(1, *)
        let tokenIDs = context.tokens
        let batchSize = context.batchSize
        let table = self.table

        let span = InstrumentsProfiler.beginPLEGather()
        do {
            try buffers.withMutableBuffer(Self.inputName) { array in
                if table.isNativeQuantized {
                    let view = array.mutableView(as: Float16.self)
                    guard view.isContiguous else {
                        throw InferenceRuntimeError.invalidState(
                            "ple_embeddings array has non-contiguous layout")
                    }
                    view.withUnsafeMutablePointer { ptr, _, _ in
                        ptr.update(repeating: 0, count: elementCount)
                        let buffer = UnsafeMutableBufferPointer(start: ptr, count: elementCount)
                        table.gather(tokenIDs: tokenIDs, batchSize: batchSize, into: buffer)
                    }
                } else {
                    let view = array.mutableView(as: Int8.self)
                    // The flat row-major gather assumes a contiguous buffer. The
                    // ple_embeddings input is exported without interleave so it is, but
                    // verify rather than silently write to wrong offsets.
                    guard view.isContiguous else {
                        throw InferenceRuntimeError.invalidState(
                            "ple_embeddings array has non-contiguous layout")
                    }
                    view.withUnsafeMutablePointer { ptr, _, _ in
                        // Zero first: a partial final batch leaves the padding slots
                        // holding the previous step's rows otherwise.
                        ptr.update(repeating: 0, count: elementCount)
                        let buffer = UnsafeMutableBufferPointer(start: ptr, count: elementCount)
                        table.gather(tokenIDs: tokenIDs, batchSize: batchSize, into: buffer)
                    }
                }
            }
        } catch {
            span.end()
            throw error
        }
        span.end()
    }
}
