// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Static-shape input handlers
//
// Concrete `SyncInputHandler`s (Sukru's shared protocol, #147) for static-shape graphs.
// Each reads the per-step `InputContext` — including `descriptors` (the running graph's
// input descriptors) so it can size its own buffer to the active bucket. Embedding gather
// is produced by the engine directly (it needs the model's gather graph), so there is no
// handler for `transformer_input`.

/// `position_ids` — absolute UInt16 positions across the aligned block.
final class PositionIdsProvider: SyncInputHandler {
    private let name: String
    var inputNames: [String] { [name] }
    init(name: String) { self.name = name }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors[name] else { return [:] }
        guard d.scalarType.byteSize == 2 else {
            throw InferenceRuntimeError.invalidState(
                "position_ids '\(name)' expects a 2-byte int type, got \(d.scalarType)")
        }
        var pos = NDArray(descriptor: d)
        var view = pos.mutableView(as: UInt16.self)
        guard var span = view.contiguousElements else {
            throw InferenceRuntimeError.invalidState("position_ids non-contiguous")
        }
        for i in 0..<ctx.batchSize { span[i] = UInt16(truncatingIfNeeded: ctx.alignedStep + i) }
        return [name: pos]
    }
}

/// `causal_mask` — `-40000` fill, unmask `0...min(queryPos, ctx-1)`.
final class CausalMaskProvider: SyncInputHandler {
    private let name: String
    var inputNames: [String] { [name] }
    init(name: String) { self.name = name }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors[name] else { return [:] }
        guard d.shape.count >= 4 else {
            throw InferenceRuntimeError.invalidState(
                "causal_mask '\(name)' expects a rank-4 shape (1, ctx, 1, q), got \(d.shape)")
        }
        var mask = NDArray(descriptor: d)
        var view = mask.mutableView(as: LogitsScalarType.self)
        StaticMaskFill.fill(view, tokensInBatch: ctx.tokens.count, alignedStep: ctx.alignedStep)
        return [name: mask]
    }
}

/// `in_step` — Int32 scalar block-start / KV write offset.
final class StepProvider: SyncInputHandler {
    private let name: String
    var inputNames: [String] { [name] }
    init(name: String) { self.name = name }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors[name] else { return [:] }
        guard d.scalarType.byteSize == 4 else {
            throw InferenceRuntimeError.invalidState(
                "step '\(name)' expects a 4-byte int type, got \(d.scalarType)")
        }
        var step = NDArray(descriptor: d)
        var view = step.mutableView(as: Int32.self)
        guard var span = view.contiguousElements else {
            throw InferenceRuntimeError.invalidState("step non-contiguous")
        }
        span[0] = Int32(ctx.alignedStep)
        return [name: step]
    }
}

/// Shared causal-mask fill (stride-aware). Callers validate the mask is rank-4 (1, ctx, 1, q).
enum StaticMaskFill {
    static func fill(
        _ view: consuming NDArray.MutableView<LogitsScalarType>,
        tokensInBatch: Int,
        alignedStep: Int
    ) {
        view.withUnsafeMutablePointer { ptr, shape, strides in
            precondition(shape.count >= 4, "causal mask fill expects a rank-4 shape (1, ctx, 1, q)")
            for context in 0..<shape[1] {
                for query in 0..<shape[3] {
                    ptr[context &* strides[1] &+ query &* strides[3]] = LogitsScalarType(-40000.0)
                }
            }
            for query in 0..<tokensInBatch {
                let upperBound = min(alignedStep + query, shape[1] &- 1)
                for context in 0...upperBound {
                    ptr[context &* strides[1] &+ query &* strides[3]] = 0
                }
            }
        }
    }
}
