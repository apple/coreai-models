// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Darwin

// MARK: - Fixed NDArray State

/// Fixed-size state for non-truncatable persistent states.
/// Allocated at full size on init, zero-initialized. No capacity management needed.
public final class FixedNDArrayState: SyncStateHandler {
    public let stateNames: [String]
    public let supportsTruncation: Bool = false
    public let currentCapacity: Int = .max
    public var stateCount: Int { arrays.count }

    private var arrays: [String: NDArray]

    public init(states: [(name: String, descriptor: NDArrayDescriptor)]) {
        var arrays: [String: NDArray] = [:]
        for (name, desc) in states {
            var array = NDArray(descriptor: desc)
            zeroFillNDArray(&array)
            arrays[name] = array
        }
        self.arrays = arrays
        self.stateNames = states.map(\.name)
    }

    public func ensureCapacity(forContextLength contextLength: Int) throws -> Bool {
        false
    }

    public subscript(stateIndex index: Int) -> (name: String, array: NDArray) {
        get { (stateNames[index], arrays[stateNames[index]]!) }
        set { arrays[stateNames[index]] = newValue.array }
    }

    @_lifetime(views: borrow self)
    public func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    public func reset() {
        for name in stateNames {
            zeroFillNDArray(&arrays[name]!)
        }
    }

    public func truncate(to tokenCount: Int) {
        preconditionFailure("truncate(to:) called on non-truncatable FixedNDArrayState")
    }
}

// MARK: - Growing NDArray State

/// Dynamically-growing KV cache state. Starts small and doubles capacity.
public final class GrowingNDArrayState: SyncStateHandler {
    public let stateNames: [String]
    public let supportsTruncation: Bool = true
    public private(set) var currentCapacity: Int
    public var stateCount: Int { arrays.count }

    private var arrays: [String: NDArray]
    private let descriptors: [NDArrayDescriptor]
    private let maxCapacity: Int
    private let sequenceDimIndex: Int

    public init(
        states: [(name: String, descriptor: NDArrayDescriptor)],
        initialCapacity: Int,
        maxCapacity: Int
    ) {
        self.maxCapacity = maxCapacity
        self.descriptors = states.map(\.descriptor)
        self.stateNames = states.map(\.name)

        let firstDesc = states[0].descriptor
        self.sequenceDimIndex = firstDesc.shape.firstIndex(where: { $0 < 0 }) ?? max(0, firstDesc.shape.count - 2)

        let capacity = min(initialCapacity, maxCapacity)
        self.currentCapacity = capacity

        var arrays: [String: NDArray] = [:]
        for (name, desc) in states {
            let resolved = desc.resolvingDynamicDimensions(
                desc.shape.map { $0 < 0 ? capacity : $0 })
            arrays[name] = NDArray(descriptor: resolved)
        }
        self.arrays = arrays
    }

    public func ensureCapacity(forContextLength contextLength: Int) throws -> Bool {
        guard contextLength > currentCapacity else { return false }
        guard contextLength <= maxCapacity else {
            throw InferenceRuntimeError.invalidState(
                "Context length \(contextLength) exceeds maximum \(maxCapacity)")
        }

        var newCapacity = max(currentCapacity, 1)
        while newCapacity < contextLength {
            newCapacity = min(newCapacity * 2, maxCapacity)
        }

        for (i, name) in stateNames.enumerated() {
            let desc = descriptors[i]
            let newShape = desc.shape.map { $0 < 0 ? newCapacity : $0 }
            let resolvedDesc = desc.resolvingDynamicDimensions(newShape)
            var newArray = NDArray(descriptor: resolvedDesc)
            _ = newArray.mutableRawView()
            copyCache(from: arrays[name]!, to: &newArray, sequenceDim: sequenceDimIndex)
            arrays[name] = newArray
        }

        currentCapacity = newCapacity
        return true
    }

    public subscript(stateIndex index: Int) -> (name: String, array: NDArray) {
        get { (stateNames[index], arrays[stateNames[index]]!) }
        set { arrays[stateNames[index]] = newValue.array }
    }

    @_lifetime(views: borrow self)
    public func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    public func reset() {
        for name in stateNames {
            zeroFillNDArray(&arrays[name]!)
        }
    }

    public func truncate(to tokenCount: Int) {}

    // MARK: - Private

    private func copyCache(from source: NDArray, to destination: inout NDArray, sequenceDim: Int) {
        let srcShape = source.shape
        let dstShape = destination.shape
        guard let headDim = srcShape.last else { return }

        let numBlocks = srcShape[..<sequenceDim].reduce(1, *)
        let oldSeqLen = srcShape[sequenceDim]
        let copyElements = oldSeqLen * headDim
        let srcBlockStride = srcShape[sequenceDim...].reduce(1, *)
        let dstBlockStride = dstShape[sequenceDim...].reduce(1, *)

        switch source.scalarType {
        case .float16, .bfloat16:
            source.view(as: Float16.self).withUnsafePointer { srcPtr, _, _ in
                var dstView = destination.mutableView(as: Float16.self)
                dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                    for block in 0..<numBlocks {
                        dstPtr.advanced(by: block * dstBlockStride).update(
                            from: srcPtr.advanced(by: block * srcBlockStride), count: copyElements)
                    }
                }
            }
        case .float32:
            source.view(as: Float.self).withUnsafePointer { srcPtr, _, _ in
                var dstView = destination.mutableView(as: Float.self)
                dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                    for block in 0..<numBlocks {
                        dstPtr.advanced(by: block * dstBlockStride).update(
                            from: srcPtr.advanced(by: block * srcBlockStride), count: copyElements)
                    }
                }
            }
        default:
            preconditionFailure("Unsupported scalar type for state copy: \(source.scalarType)")
        }
    }
}

// MARK: - Shared Utilities

func zeroFillNDArray(_ array: inout NDArray) {
    let count = array.shape.reduce(1, *)
    switch array.scalarType {
    case .float16, .bfloat16:
        var view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            memset(ptr, 0, count * MemoryLayout<Float16>.size)
        }
    case .float32:
        var view = array.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            memset(ptr, 0, count * MemoryLayout<Float>.size)
        }
    default:
        preconditionFailure("Unsupported scalar type for state: \(array.scalarType)")
    }
}
