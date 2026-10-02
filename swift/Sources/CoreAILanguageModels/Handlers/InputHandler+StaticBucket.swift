// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

// MARK: - Bucket Identity

/// Identifies one static-shape graph variant: a (query length, context bucket) pair.
///
/// A static-shape asset ships a ladder of programs specialized on both axes, and
/// every per-step input is shaped for the pair the engine is about to run.
public struct StaticBucketKey: Hashable, Sendable, Comparable {
    public let batchSize: Int
    public let contextBucket: Int

    public init(batchSize: Int, contextBucket: Int) {
        self.batchSize = batchSize
        self.contextBucket = contextBucket
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        (lhs.contextBucket, lhs.batchSize) < (rhs.contextBucket, rhs.batchSize)
    }
}

// MARK: - Per-bucket Descriptors

/// One graph input's `NDArrayDescriptor` in every bucket that declares it.
///
/// Collected once at engine init so handlers can pre-allocate every shape they
/// will ever need and then fill in place — no per-step allocation.
public struct BucketedInputDescriptors: Sendable {
    private let byKey: [StaticBucketKey: NDArrayDescriptor]

    public init(_ byKey: [StaticBucketKey: NDArrayDescriptor]) {
        self.byKey = byKey
    }

    var isEmpty: Bool { byKey.isEmpty }

    /// Every bucket's descriptor, in no particular order.
    var descriptors: Dictionary<StaticBucketKey, NDArrayDescriptor>.Values { byKey.values }

    /// Look up the bucket's descriptor, or throw naming what was available.
    func require(_ key: StaticBucketKey, input name: String) throws -> NDArrayDescriptor {
        guard let descriptor = byKey[key] else {
            throw InferenceRuntimeError.invalidState(
                "No pre-allocated '\(name)' buffer for (batch=\(key.batchSize), "
                    + "ctx=\(key.contextBucket)). Available: \(byKey.keys.sorted())")
        }
        return descriptor
    }

    /// Pre-allocate a buffer for every bucket's shape.
    func registerBuffers(name: String, into buffers: inout InputBuffers) {
        for (_, descriptor) in byKey {
            buffers.preAllocate(name: name, descriptor: descriptor)
        }
    }

    /// Collect one input's descriptor from each bucket's function descriptor.
    /// Buckets whose function doesn't declare the input are omitted.
    static func collect(
        _ inputName: String,
        from functions: [(key: StaticBucketKey, descriptor: InferenceFunctionDescriptor)]
    ) -> BucketedInputDescriptors {
        var byKey: [StaticBucketKey: NDArrayDescriptor] = [:]
        for (key, descriptor) in functions {
            guard case .ndArray(let inputDescriptor) = descriptor.inputDescriptor(of: inputName) else {
                continue
            }
            byKey[key] = inputDescriptor
        }
        return BucketedInputDescriptors(byKey)
    }
}

// MARK: - Load-time Coverage Check

enum StaticInputCoverage {
    /// Verify that the engine's handlers together produce every input the model
    /// declares. Called at init so an unhandled input fails loudly at load time
    /// rather than silently feeding an unwritten buffer into the graph.
    ///
    /// - Parameters:
    ///   - handlers: Every handler the engine will run each step.
    ///   - descriptor: The model function declaring the required inputs.
    ///   - ignoring: Inputs the engine supplies itself (constant pass-throughs
    ///     such as `embedding_table`, or outputs of another graph such as
    ///     `transformer_input`).
    static func verify(
        handlers: [any StaticInputHandler],
        descriptor: InferenceFunctionDescriptor,
        ignoring: Set<String> = []
    ) throws {
        let produced = handlers.reduce(into: Set<String>()) { $0.formUnion($1.inputNames) }
        let declared = Set(descriptor.inputNames).subtracting(ignoring)
        let uncovered = declared.subtracting(produced)
        guard uncovered.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "No input handler produces required input(s): \(uncovered.sorted()). "
                    + "Produced: \(produced.sorted()), ignored: \(ignoring.sorted())")
        }
    }
}

// MARK: - Static Bucket Input Filler

/// Zero-allocation filler for the inputs almost every static-shape model takes.
///
/// Pre-allocates each input's NDArray for every bucket at init. Within a bucket,
/// `fill()` swaps in the right pre-allocated buffer and writes in place.
///
/// Produces `position_ids` (UInt16), `causal_mask` (Float16), and a step scalar
/// (Int32). Each is optional: a model that gathers RoPE in-graph takes
/// `position_ids`, while one that takes precomputed `rope_cos`/`rope_sin` rows
/// has no position input at all (see ``DualRoPEInputHandler``).
///
/// Inputs NOT managed here, because the engine supplies them directly:
/// - `embedding_table` (constant pass-through)
/// - `transformer_input` (produced by the gather function)
public struct StaticBucketInputFiller: StaticInputHandler {
    public let inputNames: [String]

    private let positionIdsName: String?
    private let causalMaskName: String?
    private let stepName: String?

    private let positionIds: BucketedInputDescriptors
    private let causalMask: BucketedInputDescriptors
    private let step: BucketedInputDescriptors

    public init(
        positionIdsName: String? = nil,
        causalMaskName: String? = nil,
        stepName: String? = nil,
        positionIds: BucketedInputDescriptors = .init([:]),
        causalMask: BucketedInputDescriptors = .init([:]),
        step: BucketedInputDescriptors = .init([:])
    ) {
        self.positionIdsName = positionIdsName
        self.causalMaskName = causalMaskName
        self.stepName = stepName
        self.positionIds = positionIds
        self.causalMask = causalMask
        self.step = step
        self.inputNames = [positionIdsName, causalMaskName, stepName].compactMap { $0 }
    }

    public func registerBuffers(into buffers: inout InputBuffers) {
        if let name = positionIdsName { positionIds.registerBuffers(name: name, into: &buffers) }
        if let name = causalMaskName { causalMask.registerBuffers(name: name, into: &buffers) }
        if let name = stepName { step.registerBuffers(name: name, into: &buffers) }
    }

    public func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let batchSize = context.batchSize
        let alignedStep = context.alignedStep
        let tokensInBatch = context.tokens.count
        let key = StaticBucketKey(batchSize: batchSize, contextBucket: context.contextBucket)

        // Position IDs: UInt16 ascending from alignedStep
        if let name = positionIdsName {
            let descriptor = try positionIds.require(key, input: name)
            // position_ids is UInt16, so this path serves models up to 65_535
            // tokens; larger contexts export precomputed rope_cos/rope_sin rows
            // and declare no position input at all.
            guard alignedStep + batchSize - 1 <= Int(UInt16.max) else {
                throw InferenceRuntimeError.invalidState(
                    "position_ids overflow at \(alignedStep + batchSize - 1); models beyond "
                        + "65_535 tokens must export precomputed RoPE rows")
            }
            buffers.ensureCapacity(name: name, descriptor: descriptor)
            try buffers.withMutableBuffer(name) { array in
                fillNDArray(&array, as: UInt16.self, count: batchSize) { i in
                    UInt16(alignedStep + i)
                }
            }
        }

        // Causal mask: [1, ctx, 1, batch] — lower-triangular
        if let name = causalMaskName {
            let descriptor = try causalMask.require(key, input: name)
            buffers.ensureCapacity(name: name, descriptor: descriptor)
            try buffers.withMutableBuffer(name) { array in
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

        // Step scalar: the absolute flat write offset into the cache.
        if let name = stepName {
            let descriptor = try step.require(key, input: name)
            buffers.ensureCapacity(name: name, descriptor: descriptor)
            try buffers.withMutableBuffer(name) { array in
                fillNDArray(&array, as: Int32.self, count: 1) { _ in Int32(alignedStep) }
            }
        }
    }
}
