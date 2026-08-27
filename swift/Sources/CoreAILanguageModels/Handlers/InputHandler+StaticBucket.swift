// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

// MARK: - Static Bucket Input Filler

/// Zero-allocation input filler for static-shape (ANE) engines.
///
/// Fills position_ids (UInt16), causal_mask (Float16), and step (Int32) into
/// engine-owned `InputBuffers` in-place. Buffers are registered at the
/// max-bucket descriptor size; `ensureCapacity` handles per-graph shape selection.
///
/// Conforms to `StaticInputHandler` — the engine owns the buffers, this handler
/// is stateless and can be stored as `let`.
public struct StaticBucketInputFiller: StaticInputHandler {
    public let inputNames: [String]

    private let positionIdsName: String
    private let causalMaskName: String?
    private let stepName: String?

    private let positionIdsDescriptor: NDArrayDescriptor
    private let causalMaskDescriptor: NDArrayDescriptor?
    private let stepDescriptor: NDArrayDescriptor?

    /// Create from descriptors extracted from the model's function descriptors.
    ///
    /// - Parameters:
    ///   - positionIdsName: Input name for position IDs
    ///   - positionIdsDescriptor: Descriptor from the largest-batch function
    ///   - causalMaskName: Input name for causal mask (nil if model doesn't declare it)
    ///   - causalMaskDescriptor: Descriptor from the largest-context function
    ///   - stepName: Input name for step scalar (nil if model doesn't declare it)
    ///   - stepDescriptor: Descriptor for the step input
    public init(
        positionIdsName: String,
        positionIdsDescriptor: NDArrayDescriptor,
        causalMaskName: String? = nil,
        causalMaskDescriptor: NDArrayDescriptor? = nil,
        stepName: String? = nil,
        stepDescriptor: NDArrayDescriptor? = nil
    ) {
        self.positionIdsName = positionIdsName
        self.positionIdsDescriptor = positionIdsDescriptor
        self.causalMaskName = causalMaskName
        self.causalMaskDescriptor = causalMaskDescriptor
        self.stepName = stepName
        self.stepDescriptor = stepDescriptor

        var names = [positionIdsName]
        if let m = causalMaskName { names.append(m) }
        if let s = stepName { names.append(s) }
        self.inputNames = names
    }

    public func registerBuffers(into buffers: inout InputBuffers) {
        buffers.register(name: positionIdsName, descriptor: positionIdsDescriptor)
        if let maskName = causalMaskName, let maskDesc = causalMaskDescriptor {
            buffers.register(name: maskName, descriptor: maskDesc)
        }
        if let sName = stepName, let sDesc = stepDescriptor {
            buffers.register(name: sName, descriptor: sDesc)
        }
    }

    public func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let batchSize = context.batchSize
        let alignedStep = context.alignedStep
        let tokensInBatch = context.tokens.count
        let contextLength = context.contextBucket

        guard contextLength > 0 else {
            throw InferenceRuntimeError.invalidState(
                "StaticBucketInputFiller requires contextBucket > 0 in InputContext")
        }

        // Position IDs: UInt16 ascending from alignedStep
        buffers.ensureCapacity(name: positionIdsName, shape: [1, batchSize])
        buffers.withBuffer(positionIdsName) { array in
            fillNDArray(&array, as: UInt16.self, count: batchSize) { i in
                UInt16(alignedStep + i)
            }
        }

        // Causal mask: [1, ctx, 1, batch] — lower-triangular
        if let maskName = causalMaskName {
            buffers.ensureCapacity(name: maskName, shape: [1, contextLength, 1, batchSize])
            buffers.withBuffer(maskName) { array in
                var view = array.mutableView(as: Float16.self)
                view.withUnsafeMutablePointer { ptr, shape, strides in
                    for ctx in 0..<shape[1] {
                        for query in 0..<shape[3] {
                            let offset = ctx &* strides[1] &+ query &* strides[3]
                            ptr[offset] = Float16(-40000)
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
        if let sName = stepName {
            buffers.ensureCapacity(name: sName, shape: [1])
            buffers.withBuffer(sName) { array in
                fillNDArray(&array, as: Int32.self, count: 1) { _ in Int32(alignedStep) }
            }
        }
    }
}
