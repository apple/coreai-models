// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAIShared

// MARK: - Argmax

/// Exercised through an `NDArray` rather than a `[Float]`, so these cover the scalar-type
/// dispatch and stride walk as well as the scan itself. Parakeet's TDT decoder is the original
/// caller — it scans a `[1, 1, N]` joint output — and `SpeechTests` pins the vocab and duration
/// ranges it derives from these results.
@Suite("argmaxFloat")
struct ArgmaxFloatTests {
    /// A `[1, 1, values.count]` row, the shape a joint or classifier head emits.
    private func row(
        _ values: [Float], scalarType: NDArray.ScalarType = .float32
    ) -> NDArray {
        var array = NDArray(shape: [1, 1, values.count], scalarType: scalarType)
        fillFloatNDArray(&array, with: values)
        return array
    }

    @Test("Returns the index of the largest value")
    func returnsLargest() {
        #expect(argmaxFloat(row([1, 5, 3]), in: 0..<3) == 1)
    }

    @Test("Ties go to the lowest index")
    func tiesGoLow() {
        // Documented contract, and it differs from CoreAISpeech's WhisperDecoder, whose
        // `indices.max(by:)` returns the *last* maximal element. Pinned so a future cleanup
        // does not unify the two on the assumption that they already agree.
        #expect(argmaxFloat(row([2, 2, 1]), in: 0..<3) == 0)
    }

    @Test("An all-negative-infinity range returns zero")
    func allNegativeInfinityReturnsZero() {
        #expect(argmaxFloat(row([-.infinity, -.infinity]), in: 0..<2) == 0)
    }

    @Test("Indices are relative to the range lower bound")
    func indicesAreRelative() {
        // Callers index their own side tables with the result, so an absolute index would read
        // the wrong entry or run off the end.
        #expect(argmaxFloat(row([9, 9, 0, 7]), in: 2..<4) == 1)
    }

    @Test("A single-element range returns zero")
    func singleElementRange() {
        #expect(argmaxFloat(row([4, 8, 2]), in: 1..<2) == 0)
    }

    /// The scan converts as it reads, so an f16 row — what a `--dtype float16` bundle emits —
    /// must order identically to f32.
    @Test("An f16 row scans the same as f32")
    func float16RowMatches() {
        let values: [Float] = [1, 5, 3, 5, 2]
        #expect(argmaxFloat(row(values, scalarType: .float16), in: 0..<5) == 1)
        #expect(argmaxFloat(row(values, scalarType: .float16), in: 2..<5) == 1)
    }
}

// MARK: - Partial reads

/// `floatElements` converts only the elements a caller reads, leaving the rest of the tensor
/// unconverted — a streaming speech hop skips its window's left and right context this way.
/// Getting the range arithmetic wrong hands back the wrong region, so the offsets are pinned.
@Suite("floatElements")
struct FloatElementsTests {
    /// A `[1, outer, inner]` output, the shape a sliced sequence output takes.
    private func output(
        outer: Int, inner: Int, scalarType: NDArray.ScalarType = .float32
    ) -> NDArray {
        var array = NDArray(shape: [1, outer, inner], scalarType: scalarType)
        fillFloatNDArray(&array, with: (0..<(outer * inner)).map { Float($0) })
        return array
    }

    @Test("Converts exactly the requested range, in row-major order")
    func convertsRequestedRange() {
        let array = output(outer: 4, inner: 3)
        #expect(floatElements(array, in: 0..<3) == [0, 1, 2])
        // Row 2 of an inner-3 output: one step's worth of a per-row loop.
        #expect(floatElements(array, in: 6..<9) == [6, 7, 8])
        #expect(floatElements(array, in: 0..<12).count == 12)
    }

    @Test("An empty range converts to nothing")
    func emptyRange() {
        #expect(floatElements(output(outer: 2, inner: 3), in: 3..<3).isEmpty)
    }

    /// The usual case at runtime: a `--dtype float16` bundle's output, converted up.
    @Test("An f16 output converts to the same values as f32")
    func float16Matches() {
        let expected: [Float] = [3, 4, 5]
        #expect(floatElements(output(outer: 4, inner: 3), in: 3..<6) == expected)
        #expect(
            floatElements(output(outer: 4, inner: 3, scalarType: .float16), in: 3..<6) == expected)
    }
}
