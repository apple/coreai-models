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

@Suite("BFloat16 Flatten")
struct BFloat16FlattenTests {
    @Test("flattenBFloat16NDArray converts known values correctly")
    func knownValues() {
        // BFloat16 for 1.0 = 0x3F80, for -2.0 = 0xC000, for 0.5 = 0x3F00
        let bf16Values: [UInt16] = [0x3F80, 0xC000, 0x3F00]
        let expected: [Float] = [1.0, -2.0, 0.5]

        var array = NDArray(shape: [3], scalarType: .bfloat16)
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            let dst = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0..<3 { dst[i] = bf16Values[i] }
        }

        let result = flattenBFloat16NDArray(array)
        #expect(result == expected)
    }

    @Test("flattenAsFloat dispatches bfloat16 correctly")
    func flattenAsFloatBF16() {
        var array = NDArray(shape: [2], scalarType: .bfloat16)
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            let dst = ptr.assumingMemoryBound(to: UInt16.self)
            dst[0] = 0x4040  // 3.0 in bf16
            dst[1] = 0x4080  // 4.0 in bf16
        }

        let result = flattenAsFloat(array)
        #expect(result == [3.0, 4.0])
    }
}

// MARK: - Stride-aware fill and read

/// `fillNDArray` and `readNDArray` index by stride so they survive alignment padding, which
/// appears when the innermost dimension is small: FLUX.2's `img_ids` is `[1, 4096, 4]` fp32,
/// a 16-byte row that the framework pads to a 64-byte stride.
///
/// This matters because writing such a buffer linearly — as `CoreAIDiffusionModelFunction`
/// once did — scatters values across the padding. Each token then reads a *later* token's
/// coordinates and only the first quarter of the grid gets written at all, which rendered as
/// an image tiled 4x across its top quarter with the rest blank.
///
/// These round-trips exercise the stride walk whenever the buffer is padded. They pass on a
/// densely packed buffer too, so they pin the contract rather than the padding itself — the
/// bypass that caused the original defect can only be caught at the pipeline level, against a
/// real asset whose descriptor reports padded strides.
@Suite("Stride-aware fill and read")
struct StrideAwareFillReadTests {
    private func elementCount(_ array: NDArray) -> Int {
        array.view(as: Float.self).withUnsafePointer { _, shape, _ in
            (0..<shape.count).reduce(1) { $0 * shape[$1] }
        }
    }

    /// FLUX.2 `img_ids`: one row per image token as [T, H, W, L].
    @Test("A [1, 4096, 4] position-ID buffer round-trips exactly")
    func imageIdsRoundTrip() {
        let side = 64
        let axisCount = 4
        var ids = [Float](repeating: 0, count: side * side * axisCount)
        for h in 0..<side {
            for w in 0..<side {
                let index = h * side + w
                ids[index * axisCount + 1] = Float(h)
                ids[index * axisCount + 2] = Float(w)
            }
        }

        var array = NDArray(shape: [1, side * side, axisCount], scalarType: .float32)
        fillNDArray(&array, as: Float.self, count: ids.count) { ids[$0] }

        // Exact equality, not a tolerance: these are integer coordinates, and the failure
        // mode being pinned is displacement rather than drift.
        #expect(readNDArray(array, as: Float.self, count: ids.count) == ids)
    }

    /// FLUX.2 `txt_ids`: sequence index on the last axis, spatial axes unused.
    @Test("A [1, 512, 4] position-ID buffer round-trips exactly")
    func textIdsRoundTrip() {
        let textSeqLen = 512
        let axisCount = 4
        var ids = [Float](repeating: 0, count: textSeqLen * axisCount)
        for s in 0..<textSeqLen {
            ids[s * axisCount + (axisCount - 1)] = Float(s)
        }

        var array = NDArray(shape: [1, textSeqLen, axisCount], scalarType: .float32)
        fillNDArray(&array, as: Float.self, count: ids.count) { ids[$0] }
        #expect(readNDArray(array, as: Float.self, count: ids.count) == ids)
    }

    /// A distinct value per element, so any displacement shows up rather than cancelling
    /// against a repeated coordinate.
    @Test("Every element of a small-innermost-dimension buffer is placed distinctly")
    func distinctValuesSurviveRoundTrip() {
        var array = NDArray(shape: [1, 1024, 4], scalarType: .float32)
        let count = elementCount(array)
        let values = (0..<count).map { Float($0) }
        fillNDArray(&array, as: Float.self, count: count) { values[$0] }
        #expect(readNDArray(array, as: Float.self, count: count) == values)
    }

    /// A large innermost dimension needs no padding, so this is the layout the pre-computed
    /// RoPE tables used — and why the defect stayed hidden until an id-shaped input appeared.
    @Test("A [4608, 128] table round-trips exactly")
    func denseTableRoundTrip() {
        var array = NDArray(shape: [4608, 128], scalarType: .float32)
        let count = elementCount(array)
        let values = (0..<count).map { Float($0 % 97) }
        fillNDArray(&array, as: Float.self, count: count) { values[$0] }
        #expect(readNDArray(array, as: Float.self, count: count) == values)
    }
}

// MARK: - copyIntoNDArray / region fill

/// Both of `copyIntoNDArray`'s paths need pinning: the contiguous `memcpy` and the stride
/// walk that runs when the array is padded for hardware alignment.
@Suite("copyIntoNDArray")
struct CopyIntoNDArrayTests {
    @Test("Writes at an offset without disturbing its neighbours")
    func writesAtOffset() {
        var array = NDArray(shape: [4, 8], scalarType: .float32)
        fillNDArray(&array, as: Float.self, with: [Float](repeating: -1, count: 32))

        copyIntoNDArray(
            &array, as: Float.self, elementOffset: 8, from: (0..<8).map { Float($0) })

        let read = readNDArray(array, as: Float.self, count: 32)
        #expect(Array(read[0..<8]) == [Float](repeating: -1, count: 8))
        #expect(Array(read[8..<16]) == (0..<8).map { Float($0) })
        #expect(Array(read[16..<32]) == [Float](repeating: -1, count: 16))
    }

    @Test("Round-trips a slot layout the memory bank would use")
    func slotLayout() {
        // Same shape family as `spatial_memory`: [slots, tokens, 1, dim]. Each slot is
        // written independently and must come back independently.
        let slots = 3
        let tokens = 5
        let dim = 4
        let perSlot = tokens * dim
        var array = NDArray(shape: [slots, tokens, 1, dim], scalarType: .float32)
        for slot in 0..<slots {
            let payload = (0..<perSlot).map { Float(slot * 100 + $0) }
            copyIntoNDArray(
                &array, as: Float.self, elementOffset: slot * perSlot, from: payload)
        }
        let read = readNDArray(array, as: Float.self, count: slots * perSlot)
        for slot in 0..<slots {
            let got = Array(read[(slot * perSlot)..<((slot + 1) * perSlot)])
            #expect(got == (0..<perSlot).map { Float(slot * 100 + $0) }, "slot \(slot)")
        }
    }

    @Test("Survives a small innermost dimension, where padding is likely")
    func smallInnermostDimension() {
        // A 4-element innermost dimension is where the framework's preferred strides are
        // most likely to be padded, exercising the non-contiguous fallback.
        var array = NDArray(shape: [64, 4], scalarType: .float32)
        let payload = (0..<256).map { Float($0) * 0.5 }
        copyIntoNDArray(&array, as: Float.self, elementOffset: 0, from: payload)
        #expect(readNDArray(array, as: Float.self, count: 256) == payload)
    }

    @Test("An empty source is a no-op")
    func emptySource() {
        var array = NDArray(shape: [2, 4], scalarType: .float32)
        fillNDArray(&array, as: Float.self, with: [Float](repeating: 7, count: 8))
        copyIntoNDArray(&array, as: Float.self, elementOffset: 4, from: [Float]())
        #expect(readNDArray(array, as: Float.self, count: 8) == [Float](repeating: 7, count: 8))
    }

    @Test("Region fill zeroes exactly the requested range")
    func clearsRange() {
        // This is how the packer wipes slots that were valid on the previous call and are
        // not on this one. Over-clearing would destroy live memory in a lower slot.
        var array = NDArray(shape: [4, 4], scalarType: .float32)
        fillNDArray(&array, as: Float.self, with: (0..<16).map { Float($0 + 1) })
        fillNDArray(&array, as: Float.self, elementRange: 4..<12, with: 0)

        let read = readNDArray(array, as: Float.self, count: 16)
        #expect(Array(read[0..<4]) == [1, 2, 3, 4])
        #expect(Array(read[4..<12]) == [Float](repeating: 0, count: 8))
        #expect(Array(read[12..<16]) == [13, 14, 15, 16])
    }

    @Test("Region fill writes a non-zero value through the strided path")
    func fillsNonZeroWhenPadded() {
        // Explicit strides so the padded branch is guaranteed to run; a non-zero value catches
        // a fill that writes zeros regardless of `value`.
        var array = NDArray(shape: [64, 4], scalarType: .float32, strides: [8, 1])
        #expect(array.strides == [8, 1], "the padded layout must survive construction")
        fillNDArray(&array, as: Float.self, count: 256) { _ in -1 }
        fillNDArray(&array, as: Float.self, elementRange: 6..<250, with: 3.5)

        let read = readNDArray(array, as: Float.self, count: 256)
        #expect(Array(read[0..<6]) == [Float](repeating: -1, count: 6))
        #expect(Array(read[6..<250]) == [Float](repeating: 3.5, count: 244))
        #expect(Array(read[250..<256]) == [Float](repeating: -1, count: 6))
    }

    @Test("Copying at an offset survives a padded layout")
    func copiesWhenPadded() {
        // The `copyIntoNDArray` counterpart: same padded layout, pinning its stride walk.
        var array = NDArray(shape: [64, 4], scalarType: .float32, strides: [8, 1])
        fillNDArray(&array, as: Float.self, count: 256) { _ in -1 }
        let payload = (0..<100).map { Float($0) * 0.25 }
        copyIntoNDArray(&array, as: Float.self, elementOffset: 13, from: payload)

        let read = readNDArray(array, as: Float.self, count: 256)
        #expect(Array(read[0..<13]) == [Float](repeating: -1, count: 13))
        #expect(Array(read[13..<113]) == payload)
        #expect(Array(read[113..<256]) == [Float](repeating: -1, count: 143))
    }

    @Test("Filling an empty range does nothing")
    func clearEmptyRange() {
        var array = NDArray(shape: [2, 2], scalarType: .float32)
        fillNDArray(&array, as: Float.self, with: [1, 2, 3, 4] as [Float])
        fillNDArray(&array, as: Float.self, elementRange: 2..<2, with: 0)
        #expect(readNDArray(array, as: Float.self, count: 4) == [1, 2, 3, 4] as [Float])
    }
}
