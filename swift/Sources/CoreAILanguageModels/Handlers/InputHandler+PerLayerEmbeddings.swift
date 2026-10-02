// Per-Layer Embeddings input for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Gathers INT8 Per-Layer Embedding rows from the ``PerLayerEmbeddings`` sidecar
/// into the `ple_embeddings` graph input.
struct PerLayerEmbeddingsInputHandler: StaticInputHandler {
    static let inputName = "ple_embeddings"

    let inputNames: [String] = [inputName]

    private let table: PerLayerEmbeddings
    private let descriptors: BucketedInputDescriptors

    init(table: PerLayerEmbeddings, descriptors: BucketedInputDescriptors) throws {
        for descriptor in descriptors.descriptors where descriptor.shape.last != table.rowWidth {
            throw InferenceRuntimeError.invalidState(
                "PLE row width mismatch: graph expects \(descriptor.shape.last ?? -1), "
                    + "table has \(table.rowWidth)")
        }
        self.table = table
        self.descriptors = descriptors
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
        } catch {
            span.end()
            throw error
        }
        span.end()
    }
}
