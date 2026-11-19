// Gated-DeltaNet (hybrid linear-attention) inputs for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Supplies the per-step inputs a Gated-DeltaNet decoder needs alongside its two states.
///
/// Models that interleave linear-attention (DeltaNet) layers with full attention carry two
/// mutable states beyond the KV cache — a recurrent state and a causal-conv window. Both are
/// context-independent and classified `fixed`, so the engine binds them with no extra code.
/// This handler fills three inputs:
///
/// - `ssm_update_flag` — per batch slot, 1 where the slot holds a real, not-yet-processed
///   token and 0 elsewhere. The recurrence multiplies its state update by this, so padding
///   slots and re-sent prefix tokens leave the state untouched.
/// - `conv_step` — the conv ring's write offset, `alignedStep % ringWidth`.
/// - `conv_seed_selector` — the one-hot that reads this block's whole conv window back out
///   of the ring.
public struct DeltaNetInputHandler: StaticInputHandler {
    public static let updateFlagInputName = "ssm_update_flag"
    public static let convStepInputName = "conv_step"
    public static let seedSelectorInputName = "conv_seed_selector"

    public let inputNames: [String]

    private let updateFlagDescriptors: BucketedInputDescriptors
    private let convStepDescriptors: BucketedInputDescriptors
    private let seedSelectorDescriptors: BucketedInputDescriptors

    public init(
        updateFlagDescriptors: BucketedInputDescriptors,
        convStepDescriptors: BucketedInputDescriptors,
        seedSelectorDescriptors: BucketedInputDescriptors
    ) {
        self.updateFlagDescriptors = updateFlagDescriptors
        self.convStepDescriptors = convStepDescriptors
        self.seedSelectorDescriptors = seedSelectorDescriptors

        var names: [String] = []
        if !updateFlagDescriptors.isEmpty { names.append(Self.updateFlagInputName) }
        if !convStepDescriptors.isEmpty { names.append(Self.convStepInputName) }
        if !seedSelectorDescriptors.isEmpty { names.append(Self.seedSelectorInputName) }
        self.inputNames = names
    }

    /// Throws unless the bundle's states rewind `.startOnly`.
    ///
    /// The recurrent and conv states must be declared `fixed` in metadata.json; otherwise
    /// the engine treats them as rewindable and reuses prefixes over stale state.
    public static func validate(rollback: RollbackCapability) throws {
        guard rollback == .startOnly else {
            throw InferenceRuntimeError.invalidState(
                "The graph has gated DeltaNet inputs but metadata.json declares no "
                    + "`fixed` states. Re-export the bundle so its state classification "
                    + "marks the conv and recurrent states `fixed`.")
        }
    }

    /// Checks every bucket's DeltaNet inputs once, when the engine is built, so a malformed
    /// bundle fails at load rather than on its first step.
    public static func validate(
        updateFlag: BucketedInputDescriptors,
        convStep: BucketedInputDescriptors,
        seedSelector: BucketedInputDescriptors
    ) throws {
        try validate(
            updateFlagShapes: updateFlag.shapesByKey,
            convStepShapes: convStep.shapesByKey,
            seedSelectorShapes: seedSelector.shapesByKey)
    }

    /// Shape-level checks behind ``validate(updateFlag:convStep:seedSelector:)``. Internal for
    /// testing.
    static func validate(
        updateFlagShapes: [StaticBucketKey: [Int]],
        convStepShapes: [StaticBucketKey: [Int]],
        seedSelectorShapes: [StaticBucketKey: [Int]]
    ) throws {
        for (key, shape) in updateFlagShapes where shape.reduce(1, *) != key.batchSize {
            // The pooled buffer is not zeroed, so the flag must be exactly batchSize wide.
            throw InferenceRuntimeError.invalidState(
                "\(updateFlagInputName) holds \(shape.reduce(1, *)) elements in bucket \(key), "
                    + "expected \(key.batchSize)")
        }
        // conv_step is derived from the ring width, which only the selector's shape carries.
        for key in convStepShapes.keys where seedSelectorShapes[key] == nil {
            throw InferenceRuntimeError.invalidState(
                "\(convStepInputName) is declared without \(seedSelectorInputName) in bucket \(key)")
        }
        for (key, shape) in seedSelectorShapes {
            _ = try ringGeometry(selectorShape: shape, batchSize: key.batchSize)
        }
    }

    public func registerBuffers(into buffers: inout InputBuffers) {
        updateFlagDescriptors.registerBuffers(name: Self.updateFlagInputName, into: &buffers)
        convStepDescriptors.registerBuffers(name: Self.convStepInputName, into: &buffers)
        seedSelectorDescriptors.registerBuffers(name: Self.seedSelectorInputName, into: &buffers)
    }

    // MARK: - Index arithmetic
    //
    // Internal for testing.

    /// True when the slot holds a real token at a position not yet processed.
    static func isNewToken(
        slot: Int, tokensInBatch: Int, alignedStep: Int, processedTokenCount: Int
    ) -> Bool {
        slot < tokensInBatch && alignedStep + slot >= processedTokenCount
    }

    /// The conv ring's write offset for a block starting at `alignedStep`.
    ///
    /// `ringWidth` is validated against the selector's descriptor in `fill`, so it is positive.
    static func convStep(alignedStep: Int, ringWidth: Int) -> Int {
        alignedStep % ringWidth
    }

    /// The ring width and kernel window from the selector's shape,
    /// `(1, ringWidth, kernelWindow + batchSize)`, or throws when it cannot address a block.
    static func ringGeometry(
        selectorShape shape: [Int], batchSize: Int
    ) throws -> (ringWidth: Int, kernelWindow: Int) {
        guard shape.count == 3, shape[0] == 1 else {
            throw InferenceRuntimeError.invalidState(
                "\(seedSelectorInputName) has shape \(shape), expected "
                    + "(1, ringWidth, kernelWindow + batchSize)")
        }
        let ringWidth = shape[1]
        let kernelWindow = shape[2] - batchSize
        guard kernelWindow > 0, ringWidth >= kernelWindow + batchSize else {
            throw InferenceRuntimeError.invalidState(
                "\(seedSelectorInputName) \(shape) cannot address a \(batchSize)-wide block: "
                    + "ring \(ringWidth), window \(kernelWindow)")
        }
        return (ringWidth, kernelWindow)
    }

    /// Overwrites `array` (the selector, `(1, ringWidth, kernelWindow + batchSize)`) with the
    /// one-hot for a block at `alignedStep`. The pooled buffer is reused across steps, so every
    /// entry is cleared first.
    static func writeSeedSelector(
        _ array: inout NDArray, alignedStep: Int, kernelWindow: Int, ringWidth: Int
    ) {
        array.mutableView(as: Float16.self)
            .withUnsafeMutablePointer { ptr, shape, strides in
                for row in 0..<shape[1] {
                    for col in 0..<shape[2] {
                        ptr[row &* strides[1] &+ col &* strides[2]] = 0
                    }
                }
                for col in 0..<shape[2] {
                    let row = selectorRow(
                        column: col, alignedStep: alignedStep,
                        kernelWindow: kernelWindow, ringWidth: ringWidth)
                    ptr[row &* strides[1] &+ col &* strides[2]] = 1.0
                }
            }
    }

    /// Ring row for selector column `column`. Biased so negative positions map to the ring's
    /// tail.
    static func selectorRow(
        column: Int, alignedStep: Int, kernelWindow: Int, ringWidth: Int
    ) -> Int {
        let position = alignedStep - kernelWindow + column
        return ((position % ringWidth) + ringWidth) % ringWidth
    }

    public func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let key = StaticBucketKey(batchSize: context.batchSize, contextBucket: context.contextBucket)
        let alignedStep = context.alignedStep
        let batchSize = context.batchSize

        // Shapes were checked at load by `validate(updateFlag:convStep:seedSelector:)`.
        if !updateFlagDescriptors.isEmpty {
            let descriptor = try updateFlagDescriptors.require(key, input: Self.updateFlagInputName)
            buffers.ensureCapacity(name: Self.updateFlagInputName, descriptor: descriptor)
            let tokensInBatch = context.tokens.count
            let processedTokenCount = context.processedTokenCount
            try buffers.withMutableBuffer(Self.updateFlagInputName) { array in
                fillNDArray(&array, as: Float16.self, count: batchSize) { slot in
                    Self.isNewToken(
                        slot: slot, tokensInBatch: tokensInBatch, alignedStep: alignedStep,
                        processedTokenCount: processedTokenCount) ? 1.0 : 0.0
                }
            }
        }

        // Ring geometry comes from the selector's shape (1, ringWidth, kernelWindow + batchSize);
        // conv_step requires it.
        guard !seedSelectorDescriptors.isEmpty || !convStepDescriptors.isEmpty else { return }
        let seedSelector = try seedSelectorDescriptors.require(
            key, input: Self.seedSelectorInputName)
        let (ringWidth, kernelWindow) = try Self.ringGeometry(
            selectorShape: seedSelector.shape, batchSize: batchSize)

        if !convStepDescriptors.isEmpty {
            let descriptor = try convStepDescriptors.require(key, input: Self.convStepInputName)
            buffers.ensureCapacity(name: Self.convStepInputName, descriptor: descriptor)
            try buffers.withMutableBuffer(Self.convStepInputName) { array in
                fillNDArray(&array, as: Int32.self, count: 1) { _ in
                    Int32(Self.convStep(alignedStep: alignedStep, ringWidth: ringWidth))
                }
            }
        }

        buffers.ensureCapacity(name: Self.seedSelectorInputName, descriptor: seedSelector)
        try buffers.withMutableBuffer(Self.seedSelectorInputName) { array in
            Self.writeSeedSelector(
                &array, alignedStep: alignedStep, kernelWindow: kernelWindow, ringWidth: ringWidth)
        }
    }
}
