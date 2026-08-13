// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Metal

/// Fixed-size MTLBuffer state for non-truncatable persistent states (pipelined engine).
/// Allocated once at init, zero-initialized, never grows.
public final class FixedMTLBufferState {
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
            guard let buffer = device.makeBuffer(length: max(byteCount, 64), options: .storageModeShared)
            else {
                throw InferenceRuntimeError.bufferAllocationFailed("\(name) (\(byteCount) bytes)")
            }
            memset(buffer.contents(), 0, buffer.length)
            bindings.append((name, buffer, desc.scalarType, desc.shape, strides))
        }
        self.bindings = bindings
        self.stateNames = states.map(\.name)
    }

    /// Insert all managed states into async mutable views. MTLBuffer is a reference
    /// type (no COW). Uses _overrideLifetime for disjoint element access in the loop.
    @_lifetime(views: borrow self)
    public func bind(into views: inout InferenceFunction.AsyncMutableViews) {
        for binding in bindings {
            var value = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: binding.buffer, byteOffset: 0,
                scalarType: binding.scalarType, shape: binding.shape, strides: binding.strides)
            views.insert(&value, for: binding.name)
            views = unsafe _overrideLifetime(consume views, borrowing: self)
        }
    }

    /// Zero all state buffers. Caller must ensure no in-flight GPU work references these.
    public func reset() {
        for (_, buffer, _, _, _) in bindings {
            memset(buffer.contents(), 0, buffer.length)
        }
    }
}
