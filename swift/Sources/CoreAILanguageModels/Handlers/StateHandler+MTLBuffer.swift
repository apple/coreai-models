// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Metal

/// Fixed-size MTLBuffer state for non-truncatable persistent states.
/// Allocated once at init, zero-initialized, never grows.
public struct FixedMTLBufferState {
    public let stateNames: [String]
    public var stateCount: Int { bindings.count }

    private var bindings:
        [(name: String, buffer: MTLBuffer, scalarType: NDArray.ScalarType, shape: [Int], strides: [Int])]

    public init(
        states: [(name: String, descriptor: NDArrayDescriptor)],
        device: MTLDevice
    ) throws {
        var bindings: [(String, MTLBuffer, NDArray.ScalarType, [Int], [Int])] = []
        for (name, desc) in states {
            guard !desc.shape.contains(where: { $0 < 0 }) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "FixedMTLBufferState '\(name)' has dynamic shape \(desc.shape)")
            }
            let resolved = desc.resolvingDynamicDimensions(desc.shape)
            let strides = resolved.preferredStrides
            let byteCount = resolved.minimumByteCount
            guard let buffer = device.makeBuffer(length: max(byteCount, 64), options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed("\(name) (\(byteCount) bytes)")
            }
            memset(buffer.contents(), 0, buffer.length)
            bindings.append((name, buffer, desc.scalarType, desc.shape, strides))
        }
        self.bindings = bindings
        self.stateNames = states.map(\.name)
    }

    /// Access a state binding by index for AsyncMutableValue construction.
    /// The engine builds AsyncMutableValue views from these in the same scope as encode().
    /// Note: MTLBuffer is a reference type — the returned buffer is shared, not copied.
    public subscript(stateIndex index: Int) -> (
        name: String, buffer: MTLBuffer, scalarType: NDArray.ScalarType, shape: [Int], strides: [Int]
    ) {
        get { bindings[index] }
    }

    /// Insert all states into AsyncMutableViews for pipelined encoding.
    public func insertAll(into views: inout InferenceFunction.AsyncMutableViews) {
        for (name, buffer, scalarType, shape, strides) in bindings {
            var value = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: buffer, byteOffset: 0,
                scalarType: scalarType, shape: shape, strides: strides)
            views.insert(&value, for: name)
        }
    }

    /// Zero all state buffers. Caller must ensure no in-flight GPU work references these.
    public mutating func reset() {
        for (_, buffer, _, _, _) in bindings {
            memset(buffer.contents(), 0, buffer.length)
        }
    }
}
