// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// Partial reads of a graph output, for the transducer's per-step hot loop. Deliberately here
// rather than alongside `flattenAsFloat` in CoreAIShared: the decode loop is the only caller,
// and shared modules carry API for every dependent library to maintain.

/// Elements `elementRange` of `array` as `[Float]`, in row-major order.
///
/// Lets a chunked decoder convert only the frames it reads. A streaming hop's encoder output
/// also holds left and right context the loop never indexes — at the default geometry, 12
/// frames of 151 — so flattening it whole converts an order of magnitude more than is used.
func floatElements(_ array: NDArray, in elementRange: Range<Int>) -> [Float] {
    var result = [Float](repeating: 0, count: elementRange.count)
    forEachFloatElement(array, in: elementRange) { result[$0] = $1 }
    return result
}

/// Index of the largest value in `elementRange`, relative to `elementRange.lowerBound`.
///
/// Scans in place, because the alternative — flatten to `[Float]`, then scan — allocates and
/// converts a whole vocab row per emitted symbol (32 KB for Parakeet's 8,198 logits).
func argmaxFloat(_ array: NDArray, in elementRange: Range<Int>) -> Int {
    var scan = FloatArgmax()
    forEachFloatElement(array, in: elementRange) { scan.offer($0, $1) }
    return scan.best
}

/// Running argmax over values offered in order. Ties go to the lowest index, and offering
/// nothing — or only `-infinity` — yields 0.
///
/// Kept as a separate type so the rule lives in one place: `argmaxFloat` is the only caller,
/// and the decoder's tests drive it through that same function.
private struct FloatArgmax {
    private(set) var best = 0
    private var bestValue = -Float.infinity

    @inline(__always)
    mutating func offer(_ index: Int, _ value: Float) {
        if value > bestValue {
            bestValue = value
            best = index
        }
    }
}

/// Visit logical row-major elements `elementRange` of `array` as `Float`, in order. `visit`
/// receives the offset within the range, not the absolute index.
///
/// Output dtype can differ from the model's input dtype, so this branches on the array's own
/// scalar type rather than threading a flag from the input descriptors.
@inline(__always)
private func forEachFloatElement(
    _ array: NDArray, in elementRange: Range<Int>, _ visit: (Int, Float) -> Void
) {
    switch array.scalarType {
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    case .float16:
        forEachElement(array, as: Float16.self, in: elementRange, visit)
    #endif
    case .float32:
        forEachElement(array, as: Float.self, in: elementRange, visit)
    default:
        preconditionFailure("forEachFloatElement: unsupported scalar type \(array.scalarType)")
    }
}

@inline(__always)
private func forEachElement<T: BinaryFloatingPoint & BitwiseCopyable>(
    _ array: NDArray, as type: T.Type, in elementRange: Range<Int>,
    _ visit: (Int, Float) -> Void
) {
    let total = array.shape.reduce(1, *)
    precondition(
        elementRange.lowerBound >= 0 && elementRange.upperBound <= total,
        "element range \(elementRange) exceeds element count \(total)")
    if elementRange.isEmpty { return }

    array.view(as: type).withUnsafePointer { ptr, shape, strides in
        if isContiguousRowMajor(shape: shape, strides: strides) {
            for i in 0..<elementRange.count { visit(i, Float(ptr[elementRange.lowerBound + i])) }
            return
        }
        // Seed the index walk at the range's own start rather than counting up from zero.
        let rank = shape.count
        var indices = [Int](repeating: 0, count: rank)
        var remainder = elementRange.lowerBound
        for d in (0..<rank).reversed() {
            indices[d] = remainder % shape[d]
            remainder /= shape[d]
        }
        for i in 0..<elementRange.count {
            var offset = 0
            for d in 0..<rank { offset += indices[d] * strides[d] }
            visit(i, Float(ptr[offset]))
            var dim = rank - 1
            while dim >= 0 {
                indices[dim] += 1
                if indices[dim] < shape[dim] { break }
                indices[dim] = 0
                dim -= 1
            }
        }
    }
}
