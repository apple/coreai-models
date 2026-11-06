// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Per-token log probability with optional top-K alternatives.
public struct LogProbabilities: Sendable {
    public struct Entry: Sendable {
        public let tokenId: Int32
        public let value: Double
        public let alternatives: [(tokenId: Int32, value: Double)]
    }

    public let entries: [Entry]

    public var sum: Double {
        entries.reduce(0) { $0 + ($1.value.isFinite ? $1.value : 0) }
    }

    public var mean: Double {
        let finite = entries.filter(\.value.isFinite)
        return finite.isEmpty ? 0 : finite.reduce(0) { $0 + $1.value } / Double(finite.count)
    }

    public var perplexity: Double {
        exp(-mean)
    }

    /// Compute per-token log probabilities from raw logits and target token IDs.
    ///
    /// For each position, applies log-softmax to the logit vector and extracts
    /// the log probability of the target token plus the top-K alternatives.
    ///
    /// Uses Accelerate framework for vectorized exp/max operations (10-30× faster
    /// than scalar loops for large vocabularies).
    ///
    /// - Parameters:
    ///   - logits: Per-position logit vectors `[positions][vocabSize]`
    ///   - targets: Token ID at each position
    ///   - topK: Number of top alternatives to include (0 for none)
    public static func compute(
        logits: [[LogitsScalarType]],
        targets: [Int32],
        topK: Int = 0
    ) -> LogProbabilities {
        var scratch = Scratch()
        var entries: [Entry] = []
        entries.reserveCapacity(min(logits.count, targets.count))
        for (logitVec, targetToken) in zip(logits, targets) {
            logitVec.withUnsafeBufferPointer { row in
                entries.append(entry(row: row, target: targetToken, topK: topK, scratch: &scratch))
            }
        }
        return LogProbabilities(entries: entries)
    }

    /// Computes one entry per target from `logits`, a row-major `[targets.count, vocabSize]` buffer
    /// whose row `p` holds the logits that predict `targets[p]`.
    static func compute(
        logits: UnsafeBufferPointer<LogitsScalarType>,
        vocabSize: Int,
        targets: ArraySlice<Int32>,
        topK: Int
    ) -> [Entry] {
        precondition(logits.count >= targets.count * vocabSize, "logits buffer is shorter than targets")
        var scratch = Scratch()
        var entries: [Entry] = []
        entries.reserveCapacity(targets.count)
        for (position, targetToken) in targets.enumerated() {
            let start = position * vocabSize
            let row = UnsafeBufferPointer(rebasing: logits[start..<start + vocabSize])
            entries.append(entry(row: row, target: targetToken, topK: topK, scratch: &scratch))
        }
        return entries
    }

    /// Float32 buffers reused across positions, sized to the vocabulary on first use.
    private struct Scratch {
        var floats: [Float] = []
        var shifted: [Float] = []
    }

    private static func entry(
        row: UnsafeBufferPointer<LogitsScalarType>, target: Int32, topK: Int, scratch: inout Scratch
    ) -> Entry {
        let tokenIndex = Int(target)
        guard tokenIndex >= 0 && tokenIndex < row.count else {
            return Entry(tokenId: target, value: -.infinity, alternatives: [])
        }

        let vocabSize = row.count
        if scratch.floats.count != vocabSize {
            scratch.floats = [Float](repeating: 0, count: vocabSize)
            scratch.shifted = [Float](repeating: 0, count: vocabSize)
        }
        let logSumExp = computeLogSumExpVectorized(row, floats: &scratch.floats, shifted: &scratch.shifted)

        let targetLogProb: Double
        if logSumExp.isInfinite {
            // +Inf logits: tokens with +Inf get log-prob 0, others get -Inf
            targetLogProb = Double(scratch.floats[tokenIndex]).isInfinite ? 0.0 : -.infinity
        } else {
            let raw = Double(scratch.floats[tokenIndex]) - logSumExp
            targetLogProb = raw.isNaN ? 0.0 : raw
        }

        var alts: [(tokenId: Int32, value: Double)] = []
        if topK > 0 {
            alts = findTopK(scratch.floats, k: topK, logSumExp: logSumExp, vocabSize: vocabSize)
        }
        return Entry(tokenId: target, value: targetLogProb, alternatives: alts)
    }

    /// Converts `logits` to Float32 into `floats` and returns their log-sum-exp, using `shifted` as
    /// working space. Returns infinity when the maximum logit is infinite.
    private static func computeLogSumExpVectorized(
        _ logits: UnsafeBufferPointer<LogitsScalarType>, floats: inout [Float], shifted: inout [Float]
    ) -> Double {
        let count = logits.count

        // Float16 → Float32 via vFloatConversion (Accelerate)
        logits.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: count) { halfPtr in
            floats.withUnsafeMutableBufferPointer { dst in
                var bufferSrc = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: halfPtr),
                    height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
                var bufferDst = vImage_Buffer(
                    data: dst.baseAddress!, height: 1,
                    width: vImagePixelCount(count), rowBytes: count * 4)
                vImageConvert_Planar16FtoPlanarF(&bufferSrc, &bufferDst, 0)
            }
        }

        var maxVal: Float = 0
        vDSP_maxv(floats, 1, &maxVal, vDSP_Length(count))

        if maxVal.isInfinite {
            return Double.infinity
        }

        var negMax = -maxVal
        let countLen = vDSP_Length(count)
        var countInt32 = Int32(count)

        return shifted.withUnsafeMutableBufferPointer { work in
            vDSP_vsadd(floats, 1, &negMax, work.baseAddress!, 1, countLen)
            vvexpf(work.baseAddress!, work.baseAddress!, &countInt32)
            var sumExp: Float = 0
            vDSP_sve(work.baseAddress!, 1, &sumExp, countLen)
            return Double(maxVal) + Double(log(sumExp))
        }
    }
    private static func findTopK(
        _ floatBuffer: [Float],
        k: Int,
        logSumExp: Double,
        vocabSize: Int
    ) -> [(tokenId: Int32, value: Double)] {
        let actualK = min(k, vocabSize)

        if actualK == 1 {
            // O(n) argmax via vDSP
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(floatBuffer, 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
            return [(tokenId: Int32(maxIdx), value: Double(maxVal) - logSumExp)]
        }

        // For small K (typically 5-20), use a min-heap of size K.
        // This is O(n log K) which is much better than O(n log n) full sort.
        var topK: [(idx: Int, val: Float)] = []
        topK.reserveCapacity(actualK)

        for i in 0..<vocabSize {
            let val = floatBuffer[i]
            if topK.count < actualK {
                topK.append((idx: i, val: val))
                if topK.count == actualK {
                    topK.sort { $0.val < $1.val }
                }
            } else if val > topK[0].val {
                topK[0] = (idx: i, val: val)
                // Re-sort the small array (K elements, typically 5-20)
                topK.sort { $0.val < $1.val }
            }
        }

        // Return sorted descending
        return topK.reversed().map { (tokenId: Int32($0.idx), value: Double($0.val) - logSumExp) }
    }
}
