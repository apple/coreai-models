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

    public convenience init(
        states: [(name: String, descriptor: NDArrayDescriptor)],
        device: MTLDevice
    ) throws {
        try self.init(
            layouts: states.map { name, desc in
                guard !desc.shape.contains(where: { $0 < 0 }) else {
                    throw InferenceRuntimeError.invalidOutputType(
                        "FixedMTLBufferState '\(name)' has dynamic shape \(desc.shape)")
                }
                let resolved = desc.resolvingDynamicDimensions(desc.shape)
                return (name, desc.scalarType, desc.shape, resolved.preferredStrides, resolved.minimumByteCount)
            }, device: device)
    }

    /// States of the given shapes, laid out densely, for a caller with no model descriptor to hand.
    package convenience init(
        dense states: [(name: String, scalarType: NDArray.ScalarType, shape: [Int])], device: MTLDevice
    ) throws {
        try self.init(
            layouts: states.map { name, type, shape in
                let strides = shape.indices.map { shape[($0 + 1)...].reduce(1, *) }
                return (name, type, shape, strides, shape.reduce(1, *) * type.byteSize)
            }, device: device)
    }

    private init(
        layouts: [(name: String, scalarType: NDArray.ScalarType, shape: [Int], strides: [Int], byteCount: Int)],
        device: MTLDevice
    ) throws {
        var bindings: [(String, MTLBuffer, NDArray.ScalarType, [Int], [Int])] = []
        for (name, scalarType, shape, strides, byteCount) in layouts {
            guard let buffer = device.makeBuffer(length: max(byteCount, 64), options: .storageModeShared)
            else {
                throw InferenceRuntimeError.bufferAllocationFailed("\(name) (\(byteCount) bytes)")
            }
            memset(buffer.contents(), 0, buffer.length)
            bindings.append((name, buffer, scalarType, shape, strides))
        }
        self.bindings = bindings
        self.stateNames = layouts.map(\.name)
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

    /// Bytes the state buffers take together, and so a copy of them all.
    package var byteCount: Int { bindings.reduce(0) { $0 + $1.buffer.length } }

    /// Each state's name and the bytes a snapshot of it takes, in order.
    package var snapshotLayout: [(name: String, bytes: Int)] {
        bindings.map { ($0.name, $0.buffer.length) }
    }

    /// Queues a copy of every state buffer, in order, into new buffers on `commandBuffer`.
    ///
    /// Encoding doesn't wait for the GPU. Read the copies only once `commandBuffer` completes: they
    /// then hold the states as the work queued before it left them. Nil when the blit encoder or a
    /// copy can't be made.
    package func encodeSnapshot(on commandBuffer: MTLCommandBuffer) -> [MTLBuffer]? {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return nil }
        var copies: [MTLBuffer] = []
        for binding in bindings {
            guard
                let copy = binding.buffer.device.makeBuffer(
                    length: binding.buffer.length, options: .storageModeShared)
            else {
                blit.endEncoding()
                return nil
            }
            copies.append(copy)
        }
        for (binding, copy) in zip(bindings, copies) {
            blit.copy(
                from: binding.buffer, sourceOffset: 0, to: copy, destinationOffset: 0,
                size: binding.buffer.length)
        }
        blit.endEncoding()
        return copies
    }

    /// Overwrites every state buffer with a copy of the same states, one blob per state, in order, as
    /// ``encodeSnapshot(on:)`` makes.
    /// Caller must ensure no in-flight GPU work references these.
    package func restore(_ snapshot: [Data]) {
        precondition(snapshot.count == bindings.count, "snapshot is of other states")
        for (blob, binding) in zip(snapshot, bindings) {
            precondition(blob.count == binding.buffer.length, "snapshot is of other states")
            blob.withUnsafeBytes { raw in
                if let source = raw.baseAddress {
                    binding.buffer.contents().copyMemory(from: source, byteCount: blob.count)
                }
            }
        }
    }
}
