// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation

// MARK: - View Construction Helpers

/// Resolve strides from an NDArrayDescriptor for a given concrete shape.
///
/// Uses `NDArrayDescriptor.resolvingDynamicDimensions().preferredStrides` to get
/// framework-blessed strides that respect hardware alignment constraints.
public func resolvedStrides(descriptor: NDArrayDescriptor, shape: [Int]) throws -> [Int] {
    let resolved = descriptor.resolvingDynamicDimensions(shape)
    return resolved.preferredStrides
}

// MARK: - Span helpers

/// Product of the elements of a Span<Int> — used to compute the flat
/// capacity from an NDArray shape. `Span` doesn't conform to `Sequence`
/// (non-escapable by design), so `.reduce` isn't available.
extension Span where Element == Int {
    var product: Int {
        var result = 1
        for i in 0..<count {
            result *= self[i]
        }
        return result
    }
}

// MARK: - NDArray Fill / Read Helpers

/// Fill an NDArray from a collection of elements.
public func fillNDArray<T: BitwiseCopyable>(
    _ array: inout NDArray, as type: T.Type, with elements: some Collection<T>
) {
    var view = array.mutableView(as: type)
    view.copyElements(fromContentsOf: elements)
}

/// Fill an NDArray using a closure that maps index → value.
///
/// - Precondition: `count` must not exceed the number of elements in the
///   array (derived from the shape).
public func fillNDArray<T: BitwiseCopyable>(
    _ array: inout NDArray, as type: T.Type, count: Int, using generator: (Int) -> T
) {
    let view = array.mutableView(as: type)
    view.withUnsafeMutablePointer { ptr, shape, _ in
        let capacity = shape.product
        precondition(count <= capacity, "fillNDArray: count \(count) exceeds array capacity \(capacity)")
        for i in 0..<count {
            ptr[i] = generator(i)
        }
    }
}

/// Read elements from an NDArray into a new Array.
///
/// - Precondition: `count` must not exceed the number of elements in the
///   array (derived from the shape).
public func readNDArray<T: BitwiseCopyable>(
    _ array: NDArray, as type: T.Type, count: Int
) -> [T] {
    array.view(as: type).withUnsafePointer { ptr, shape, _ in
        let capacity = shape.product
        precondition(count <= capacity, "readNDArray: count \(count) exceeds array capacity \(capacity)")
        var result = [T]()
        result.reserveCapacity(count)
        result.append(contentsOf: UnsafeBufferPointer(start: ptr, count: count))
        return result
    }
}

// MARK: - Flatten Helpers

/// Flatten an NDArray output into `[Float]`, branching on its own scalar type.
/// Output dtype can differ from the model's input dtype, so always inspect the array
/// rather than threading an `isFloat16` flag from input descriptors.
public func flattenAsFloat(_ array: NDArray) -> [Float] {
    switch array.scalarType {
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    case .float16:
        return flattenNDArray(array, as: Float16.self)
    #endif
    case .float32:
        return flattenNDArray(array, as: Float.self)
    default:
        preconditionFailure("flattenAsFloat: unsupported scalar type \(array.scalarType)")
    }
}

/// Flatten an NDArray to a `[Float]` in row-major order, converting from `T`.
///
/// Fast path skips per-element stride arithmetic when the array is already
/// row-major contiguous (the common case for Core AI outputs).
public func flattenNDArray<T: BinaryFloatingPoint & BitwiseCopyable>(
    _ array: NDArray, as type: T.Type
) -> [Float] {
    let outerShape = array.shape
    let rank = outerShape.count
    let total = outerShape.reduce(1, *)
    var result = [Float](repeating: 0, count: total)
    array.view(as: type).withUnsafePointer { ptr, shape, strides in
        // Fast path: row-major contiguous layout — avoids per-element stride arithmetic.
        var expectedStride = 1
        var isContiguous = true
        for d in (0..<rank).reversed() {
            if strides[d] != expectedStride {
                isContiguous = false
                break
            }
            expectedStride *= shape[d]
        }
        if isContiguous {
            for i in 0..<total { result[i] = Float(ptr[i]) }
            return
        }
        // Slow path: non-contiguous strides.
        var indices = [Int](repeating: 0, count: rank)
        for i in 0..<total {
            var offset = 0
            for d in 0..<rank { offset += indices[d] * strides[d] }
            result[i] = Float(ptr[offset])
            var dim = rank - 1
            while dim >= 0 {
                indices[dim] += 1
                if indices[dim] < shape[dim] { break }
                indices[dim] = 0
                dim -= 1
            }
        }
    }
    return result
}
