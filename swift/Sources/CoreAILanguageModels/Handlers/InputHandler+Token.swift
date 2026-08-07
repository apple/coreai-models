// Synchronous token input handler for Sequential and StaticShape engines.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Standard input handler for text LLMs: `input_ids` (Int32) + `position_ids` (Int32).
///
/// Pre-allocates the `input_ids` NDArray and reuses it when batch size is unchanged.
public struct TokenInputHandler: SyncInputHandler {
    public let inputNames: [String]

    private let inputIdsName: String
    private let positionIdsName: String
    private let inputIdsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor

    private var inputIdsArray: NDArray
    private var cachedBatchSize: Int

    public init(
        inputIdsName: String,
        positionIdsName: String,
        inputIdsDescriptor: NDArrayDescriptor,
        positionIdsDescriptor: NDArrayDescriptor
    ) {
        self.inputIdsName = inputIdsName
        self.positionIdsName = positionIdsName
        self.inputIdsDescriptor = inputIdsDescriptor
        self.positionIdsDescriptor = positionIdsDescriptor
        self.inputNames = [inputIdsName, positionIdsName]

        let initDesc = inputIdsDescriptor.resolvingDynamicDimensions([1, 1])
        self.inputIdsArray = NDArray(descriptor: initDesc)
        self.cachedBatchSize = 1
    }

    public mutating func prepare(_ context: InputContext) async throws -> [String: NDArray] {
        let tokens = context.tokens
        let batchSize = tokens.count
        precondition(batchSize > 0, "TokenInputHandler: empty token batch")

        if cachedBatchSize != batchSize {
            let resolved = inputIdsDescriptor.resolvingDynamicDimensions([1, batchSize])
            inputIdsArray = NDArray(descriptor: resolved)
            cachedBatchSize = batchSize
        }
        fillNDArray(&inputIdsArray, as: Int32.self, with: tokens)

        let totalPositions = context.processedTokenCount + batchSize
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions([1, totalPositions])
        var positionIds = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIds, as: Int32.self, count: totalPositions) { Int32($0) }

        return [
            inputIdsName: inputIdsArray,
            positionIdsName: positionIds,
        ]
    }
}

/// Wraps a base input handler and appends model-specific extra inputs.
///
/// Use for any input that needs per-step computation beyond standard token/position IDs.
public struct CompositeInputHandler<Base: SyncInputHandler>: SyncInputHandler {
    public var inputNames: [String] {
        base.inputNames + extras.map(\.name)
    }

    private var base: Base
    private let extras: [ExtraInput]

    public struct ExtraInput: Sendable {
        public let name: String
        public let prepare: @Sendable (InputContext) throws -> NDArray

        public init(name: String, prepare: @Sendable @escaping (InputContext) throws -> NDArray) {
            self.name = name
            self.prepare = prepare
        }
    }

    public init(base: Base, extras: [ExtraInput]) {
        self.base = base
        self.extras = extras
    }

    public mutating func prepare(_ context: InputContext) async throws -> [String: NDArray] {
        var inputs = try await base.prepare(context)
        for extra in extras {
            inputs[extra.name] = try extra.prepare(context)
        }
        return inputs
    }
}
