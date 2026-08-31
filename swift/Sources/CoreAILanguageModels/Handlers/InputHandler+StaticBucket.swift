// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

// MARK: - Static Bucket Input Filler

/// Zero-allocation input filler for static-shape (ANE) engines.
///
/// Pre-allocates all input NDArrays per bucket (graph variant) at init time.
/// Within a bucket, every call to `fill()` swaps in the right pre-allocated
/// buffer and fills in-place — zero per-step allocation.
///
/// Fills position_ids (UInt16), causal_mask (Float16), and step (Int32).
/// Inputs NOT managed (handled by the engine directly):
/// - `embedding_table` (constant, passed through)
/// - `transformer_input` (produced by gather function)
public struct StaticBucketInputFiller: StaticInputHandler {
    public let inputNames: [String]

    private let positionIdsName: String
    private let causalMaskName: String?
    private let stepName: String?

    private let bucketDescriptors: [BucketKey: BucketDescriptors]

    public struct BucketKey: Hashable, Sendable {
        public let batchSize: Int
        public let contextBucket: Int
    }

    public struct BucketDescriptors: Sendable {
        public let positionIds: NDArrayDescriptor
        public let causalMask: NDArrayDescriptor?
        public let step: NDArrayDescriptor?
    }

    public init(
        positionIdsName: String,
        causalMaskName: String? = nil,
        stepName: String? = nil,
        bucketDescriptors: [BucketKey: BucketDescriptors]
    ) {
        self.positionIdsName = positionIdsName
        self.causalMaskName = causalMaskName
        self.stepName = stepName
        self.bucketDescriptors = bucketDescriptors

        var names = [positionIdsName]
        if let m = causalMaskName { names.append(m) }
        if let s = stepName { names.append(s) }
        self.inputNames = names
    }

    public func registerBuffers(into buffers: inout InputBuffers) {
        for (_, descs) in bucketDescriptors {
            buffers.preAllocate(name: positionIdsName, descriptor: descs.positionIds)
            if let maskName = causalMaskName, let maskDesc = descs.causalMask {
                buffers.preAllocate(name: maskName, descriptor: maskDesc)
            }
            if let sName = stepName, let sDesc = descs.step {
                buffers.preAllocate(name: sName, descriptor: sDesc)
            }
        }
    }

    public func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let batchSize = context.batchSize
        let alignedStep = context.alignedStep
        let tokensInBatch = context.tokens.count
        let contextLength = context.contextBucket

        // TODO: resolve bucket at graph-selection time to skip per-step lookup
        let key = BucketKey(batchSize: batchSize, contextBucket: contextLength)
        guard let descs = bucketDescriptors[key] else {
            throw InferenceRuntimeError.invalidState(
                "No pre-allocated bucket for (batch=\(batchSize), ctx=\(contextLength)). "
                    + "Available: \(bucketDescriptors.keys.sorted { ($0.contextBucket, $0.batchSize) < ($1.contextBucket, $1.batchSize) })"
            )
        }

        // Position IDs: UInt16 ascending from alignedStep
        buffers.ensureCapacity(name: positionIdsName, descriptor: descs.positionIds)
        try buffers.withMutableBuffer(positionIdsName) { array in
            fillNDArray(&array, as: UInt16.self, count: batchSize) { i in
                UInt16(alignedStep + i)
            }
        }

        // Causal mask: [1, ctx, 1, batch] — lower-triangular
        if let maskName = causalMaskName, let maskDesc = descs.causalMask {
            buffers.ensureCapacity(name: maskName, descriptor: maskDesc)
            try buffers.withMutableBuffer(maskName) { array in
                array.mutableView(as: Float16.self)
                    .withUnsafeMutablePointer { ptr, shape, strides in
                        for ctx in 0..<shape[1] {
                            for query in 0..<shape[3] {
                                let offset = ctx &* strides[1] &+ query &* strides[3]
                                ptr[offset] = causalMaskSentinel
                            }
                        }
                        for query in 0..<tokensInBatch {
                            let queryPos = alignedStep + query
                            let upperBound = min(queryPos, shape[1] &- 1)
                            for ctx in 0...upperBound {
                                let offset = ctx &* strides[1] &+ query &* strides[3]
                                ptr[offset] = 0
                            }
                        }
                    }
            }
        }

        // Step scalar: Int32
        if let sName = stepName, let sDesc = descs.step {
            buffers.ensureCapacity(name: sName, descriptor: sDesc)
            try buffers.withMutableBuffer(sName) { array in
                fillNDArray(&array, as: Int32.self, count: 1) { _ in Int32(alignedStep) }
            }
        }
    }
}
