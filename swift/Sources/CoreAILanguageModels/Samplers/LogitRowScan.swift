// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Greedy and top-k selection over one logit row widened to `Float`, used by the speculative
/// verify paths.
///
/// - Ties: the lowest index wins, as plain greedy sampling's `vDSP_maxvi` does on every row tried.
/// - NaN: a NaN entry ranks as negative infinity. It never beats a number, and ties with other
///   negative-infinity entries break toward the lowest index. An empty row selects index 0.
///   `vDSP_maxvi` returns an arbitrary index when the row holds a NaN, so rows are checked first.
/// - Final-logit soft cap: with a cap, the result is that of `LogitSoftcapProcessor.apply`
///   followed by the same selection, including the ties the cap's rounding to `LogitsScalarType`
///   creates, but only entries that can still tie or beat the maximum after capping are capped.
enum LogitRowScan {
    /// Index of the first maximum of `row` under the NaN rule above.
    static func argmax(_ row: UnsafeBufferPointer<Float>) -> Int {
        guard let base = row.baseAddress, !row.isEmpty else { return 0 }
        if hasNaNOrOpposingInfinities(row) { return scalarArgmax(row) }
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(base, 1, &maxValue, &maxIndex, vDSP_Length(row.count))
        return Int(maxIndex)
    }

    /// Whether `row` may hold a NaN. The sum of a row is NaN when it does, and also when it holds
    /// both infinities; callers take the exact scalar path in either case.
    static func hasNaNOrOpposingInfinities(_ row: UnsafeBufferPointer<Float>) -> Bool {
        guard let base = row.baseAddress, !row.isEmpty else { return false }
        var sum: Float = 0
        vDSP_sve(base, 1, &sum, vDSP_Length(row.count))
        return sum.isNaN
    }

    private static func scalarArgmax(_ row: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        var bestValue = -Float.infinity
        for i in 0..<row.count where row[i] > bestValue {
            bestValue = row[i]
            best = i
        }
        return best
    }

    /// The `k` largest entries of `row` (of the capped row when `cap` is set), highest first,
    /// with the capped values. `row` is scratch: its contents are unspecified afterwards.
    ///
    /// With a cap, a candidate set is built from the raw ranking. Capping is
    /// `f(x) = round(c * tanh(x / c))` with `round` to `LogitsScalarType`, which is non-decreasing
    /// in `x`. Let `x_k` be the k-th largest raw entry. The real-valued capped
    /// `c * tanh(x / c)` is non-decreasing and every capped value has magnitude below `c`, so
    /// the rounding grid is at most `ulp(c)` wide. An entry `x` with
    /// `c * tanh(x / c) < c * tanh(x_k / c) - 2 * ulp(c) - slack` therefore rounds strictly below
    /// `f(x_k)`, and below at least `k` entries, so it is not in the top `k`. `slack` covers the
    /// error of the `Float` arithmetic in `apply`. Every entry at or above the threshold
    /// `c * atanh(...)` is capped exactly as `apply` does and ranked by (capped value, index).
    static func topK(
        _ row: UnsafeMutableBufferPointer<Float>, k: Int, cap: Float?
    ) -> (indices: [Int32], values: [Float]) {
        let count = row.count
        let wanted = min(k, count)
        guard wanted > 0, let base = row.baseAddress else { return ([], []) }
        if hasNaNOrOpposingInfinities(UnsafeBufferPointer(row)) {
            return reference(UnsafeBufferPointer(row), k: wanted, cap: cap)
        }

        // Raw top-k by repeated maximum, knocking each found entry out of the scratch row.
        var foundIndex: [Int] = []
        var foundValue: [Float] = []
        foundIndex.reserveCapacity(wanted)
        foundValue.reserveCapacity(wanted)
        for _ in 0..<wanted {
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(base, 1, &maxValue, &maxIndex, vDSP_Length(count))
            if maxValue == -.infinity {
                // Only negative infinities remain, and a knocked-out entry looks like one. Fill
                // with the lowest indices not yet taken.
                let taken = Set(foundIndex)
                var next = 0
                while foundIndex.count < wanted {
                    while taken.contains(next) { next += 1 }
                    foundIndex.append(next)
                    foundValue.append(-.infinity)
                    next += 1
                }
                break
            }
            foundIndex.append(Int(maxIndex))
            foundValue.append(maxValue)
            base[Int(maxIndex)] = -.infinity
        }

        guard let cap, cap > 0 else {
            return (foundIndex.map(Int32.init), foundValue)
        }

        let threshold = rawThreshold(forKthLargest: foundValue[wanted - 1], cap: cap)
        for (i, v) in zip(foundIndex, foundValue) { base[i] = v }
        guard let threshold else {
            return reference(UnsafeBufferPointer(row), k: wanted, cap: cap)
        }

        // Candidates: the raw top-k, plus every other entry at or above the threshold.
        var candidateIndex = foundIndex
        let isFound = Set(foundIndex)
        let blockLength = 2048
        var blockStart = 0
        while blockStart < count {
            let length = min(blockLength, count - blockStart)
            var blockMax: Float = 0
            vDSP_maxv(base + blockStart, 1, &blockMax, vDSP_Length(length))
            if blockMax >= threshold {
                for i in blockStart..<(blockStart + length)
                where base[i] >= threshold && !isFound.contains(i) {
                    candidateIndex.append(i)
                }
            }
            blockStart += length
        }

        var capped = candidateIndex.map { LogitsScalarType(base[$0]) }
        LogitSoftcapProcessor.apply(to: &capped, cap: cap)
        let order = capped.indices.sorted { a, b in
            let (va, vb) = (Float(capped[a]), Float(capped[b]))
            return va != vb ? va > vb : candidateIndex[a] < candidateIndex[b]
        }.prefix(wanted)
        return (order.map { Int32(candidateIndex[$0]) }, order.map { Float(capped[$0]) })
    }

    /// Smallest raw value that can still reach the top `k` after capping, given the k-th largest
    /// raw value; `nil` when no entry can be excluded.
    private static func rawThreshold(forKthLargest kth: Float, cap: Float) -> Float? {
        let c = Double(cap)
        let cappedKth = c * tanh(Double(kth) / c)
        // Capped values have magnitude below `c`, so the rounding grid is at most this wide.
        let spacing = Double(LogitsScalarType(cap).ulp)
        let target = cappedKth - (2 * spacing + c * 1e-5)
        guard target > -c else { return nil }
        let raw = c * atanh(target / c)
        guard raw.isFinite else { return nil }
        return Float(raw - abs(raw) * 1e-6 - 1e-6)
    }

    /// The exact result by capping the whole row: used when the row holds a NaN, or when no
    /// entry can be excluded.
    static func reference(
        _ row: UnsafeBufferPointer<Float>, k: Int, cap: Float?
    ) -> (indices: [Int32], values: [Float]) {
        var capped = row.map { LogitsScalarType($0) }
        if let cap { LogitSoftcapProcessor.apply(to: &capped, cap: cap) }
        let values = capped.map { v -> Float in v.isNaN ? -.infinity : Float(v) }
        let order = values.indices.sorted { a, b in
            values[a] != values[b] ? values[a] > values[b] : a < b
        }.prefix(k)
        return (order.map { Int32($0) }, order.map { values[$0] })
    }
}
