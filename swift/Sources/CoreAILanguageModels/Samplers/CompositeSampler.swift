// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation
import os.signpost

/// Unified CPU sampler supporting greedy, temperature, topK, topP, and minP sampling.
///
/// This sampler uses Float32 internally for all non-greedy sampling to ensure
/// numerical precision. The Float16 → Float32 conversion happens once at entry,
/// avoiding repeated conversions during softmax and probability calculations.
///
/// - Note: This implementation prioritizes correctness and precision over raw performance.
///         Engines should prefer GPU based samplers when possible.
///
/// ## Supported Configurations
/// - **Greedy** (temperature=0): Argmax, no conversion needed
/// - **Temperature only**: Scale logits, softmax, multinomial sample
/// - **TopK**: Keep K highest probability tokens, sample from them
/// - **TopP (nucleus)**: Keep tokens until cumulative probability >= P
/// - **MinP**: Keep tokens whose probability >= minP × max probability
/// - **Combined**: Apply minP first, then topP (broader filter), then topK (hard limit)
///
/// ## Algorithm Order
/// ```
/// logits → [temperature scaling] → [minP filter] → [topP filter] → [topK filter] → [softmax] → [sample]
/// ```
public struct CompositeSampler {
    // MARK: - Public API (Float16 input - backward compatible)

    /// Samples the next token from Float16 logits.
    ///
    /// For greedy sampling (temperature=0), operates directly on Float16.
    /// For non-greedy sampling, converts to Float32 internally for precision.
    ///
    /// - Parameters:
    ///   - logits: Mutable array of Float16 logits. May be modified during sampling.
    ///   - config: Sampling configuration with temperature, topK, and topP settings.
    /// - Returns: The sampled token ID.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    public static func sample(from logits: inout [Float16], config: SamplingConfiguration) -> Int32 {
        var rng = SystemRandomNumberGenerator()
        return sample(from: &logits, config: config, using: &rng)
    }
    #endif

    /// Samples from Float16 logits using a caller-supplied RNG.
    ///
    /// `inout` so state advances across calls. Use this for deterministic / reproducible
    /// sampling (tests, evals). For non-deterministic use, call the no-RNG overload.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    public static func sample(
        from logits: inout [Float16],
        config: SamplingConfiguration,
        using rng: inout some RandomNumberGenerator
    ) -> Int32 {
        if config.temperature == 0 {
            return greedySampleFloat16(&logits)
        }
        var logitsF32 = convertFloat16ToFloat32(&logits)
        return sampleFloat32(&logitsF32, config: config, using: &rng)
    }
    #endif

    // MARK: - Public API (Float32 input - for engines that output Float32)

    /// Samples the next token from Float32 logits.
    ///
    /// Use this overload when the engine provides Float32 logits directly,
    /// or when converting from BFloat16 at the engine boundary.
    ///
    /// - Parameters:
    ///   - logits: Mutable array of Float logits. May be modified during sampling.
    ///   - config: Sampling configuration with temperature, topK, and topP settings.
    /// - Returns: The sampled token ID.
    public static func sample(from logits: inout [Float], config: SamplingConfiguration) -> Int32 {
        var rng = SystemRandomNumberGenerator()
        return sample(from: &logits, config: config, using: &rng)
    }

    /// Samples from Float32 logits using a caller-supplied RNG. See the Float16 overload.
    public static func sample(
        from logits: inout [Float],
        config: SamplingConfiguration,
        using rng: inout some RandomNumberGenerator
    ) -> Int32 {
        if config.temperature == 0 {
            return greedySampleFloat32(&logits)
        }
        return sampleFloat32(&logits, config: config, using: &rng)
    }

    // MARK: - Validation

    /// Returns `true` when every logit is non-finite (−∞ or NaN), meaning no
    /// valid token can be sampled. Callers should check this before sampling
    /// when an external mask (e.g. grammar constraint) may eliminate all tokens.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    public static func allMasked(_ logits: [Float16]) -> Bool {
        for v in logits where v.isFinite { return false }
        return true
    }
    #endif

    /// Float32 overload.
    public static func allMasked(_ logits: [Float]) -> Bool {
        for v in logits where v.isFinite { return false }
        return true
    }

    // MARK: - Greedy Sampling (Argmax)

    /// Greedy sampling on Float16 logits - returns index of maximum value.
    /// Converts to Float32 via vImage and uses vDSP_maxvi (faster than the
    /// pure-Swift `logits.enumerated().max(by:)` for vocab-sized arrays).
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    private static func greedySampleFloat16(_ logits: inout [Float16]) -> Int32 {
        let span = InstrumentsProfiler.beginSample(strategy: "greedy", temperature: 0)

        let count = logits.count
        guard count > 0 else {
            span.end()
            return 0
        }

        let logitsF32 = convertFloat16ToFloat32(&logits)

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(logitsF32, 1, &maxVal, &maxIdx, vDSP_Length(count))

        span.end()
        return Int32(maxIdx)
    }
    #endif

    /// Greedy sampling on Float32 logits - returns index of maximum value via vDSP_maxvi.
    private static func greedySampleFloat32(_ logits: inout [Float]) -> Int32 {
        let span = InstrumentsProfiler.beginSample(strategy: "greedy", temperature: 0)

        let count = logits.count
        guard count > 0 else {
            span.end()
            return 0
        }

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(logits, 1, &maxVal, &maxIdx, vDSP_Length(count))

        span.end()
        return Int32(maxIdx)
    }

    // MARK: - Composite Sampling (Float32 internal)

    /// Full sampling pipeline in Float32. Two paths:
    ///
    /// **Fast path** — no topK/topP/minP: vectorized softmax + multinomial over the full vocab.
    /// **Slow path** — topK, topP, and/or minP set: identify the active subset (typically 5-500 tokens),
    /// compact those logits into a small array, softmax + sample over just those K. Skips the
    /// V-sized softmax and the V-sized multinomial scan entirely.
    ///
    /// Guided generation works through either path: GrammarEngine pre-masks logits with
    /// `-Float.infinity` (or `-Float16.greatestFiniteMagnitude` for fp16). Those positions
    /// have exp() = 0 and never enter the active subset.
    private static func sampleFloat32(
        _ logits: inout [Float],
        config: SamplingConfiguration,
        using rng: inout some RandomNumberGenerator
    ) -> Int32 {
        let vocabSize = logits.count
        let strategyName = describeStrategy(config)

        let span = InstrumentsProfiler.beginSample(strategy: strategyName, temperature: config.temperature)

        var temp = Float(config.temperature)
        vDSP_vsdiv(logits, 1, &temp, &logits, 1, vDSP_Length(vocabSize))

        let needsTopP = (config.topP.map { $0 < 1.0 } ?? false)
        let needsTopK = (config.topK != nil)
        let needsMinP = (config.minP != nil)

        if !needsTopP && !needsTopK && !needsMinP {
            // Fast path: vectorized softmax + multinomial over the full vocab.
            softmaxVectorized(&logits)
            let token = Int32(multinomialSample(logits, using: &rng))
            span.end()
            return token
        }

        // Slow path: compact-then-sample.
        let activeIndices = selectActiveIndices(
            logits: logits,
            topK: config.topK,
            topP: config.topP.map { Float($0) },
            minP: config.minP.map { Float($0) })

        // Degenerate: empty active set (e.g. all logits -.infinity). Fall back to index 0.
        guard !activeIndices.isEmpty else {
            span.end()
            return 0
        }

        var activeLogits = activeIndices.map { logits[$0] }
        softmaxVectorized(&activeLogits)
        let localIdx = multinomialSample(activeLogits, using: &rng)
        let token = Int32(activeIndices[localIdx])
        span.end()
        return token
    }

    /// Vectorized softmax using vDSP. Operates in-place on the entire buffer.
    /// Numerically stable: subtracts max before exp. Handles `-.infinity` positions
    /// naturally (`exp(-inf) = 0`).
    static func softmaxVectorized(_ logits: inout [Float]) {
        let n = vDSP_Length(logits.count)

        var maxLogit: Float = 0
        vDSP_maxv(logits, 1, &maxLogit, n)

        var negMax = -maxLogit
        vDSP_vsadd(logits, 1, &negMax, &logits, 1, n)

        var elementCount = Int32(logits.count)
        vvexpf(&logits, logits, &elementCount)

        var sumExp: Float = 0
        vDSP_sve(logits, 1, &sumExp, n)

        var invSum = 1.0 / max(sumExp, .leastNormalMagnitude)
        vDSP_vsmul(logits, 1, &invSum, &logits, 1, n)
    }

    /// Multinomial sample from a normalized probability distribution via inverse CDF scan.
    /// Positions with probability `0` never advance the cumulative sum and are never selected.
    static func multinomialSample(
        _ probs: [Float],
        using rng: inout some RandomNumberGenerator
    ) -> Int {
        let r = Float.random(in: 0..<1, using: &rng)
        var cum: Float = 0
        for (i, p) in probs.enumerated() {
            cum += p
            if r < cum { return i }
        }
        return probs.count - 1
    }

    // MARK: - Active-subset selection (slow path)

    /// Identify the indices kept after minP / topK / topP filtering, sorted descending by logit.
    ///
    /// When `topK` is set: partial sort via min-heap of size K (O(V log K)) — avoids the
    /// O(V log V) full sort and the V-sized [Int] allocation.
    ///
    /// When only `topP` or `minP` is set: O(V) passes find the max logit and, for topP, the
    /// logit bucket where the nucleus ends, so only that small prefix is sorted
    /// (see `candidateIndicesAboveCutoff`). This avoids the O(V log V) full-vocabulary sort,
    /// which dominates sampling time for large vocabularies.
    ///
    /// `minP` is applied first in logit space: tokens with logit < max_logit + log(minP)
    /// are excluded. This is equivalent to excluding tokens with probability < minP × max_prob.
    ///
    /// `topP` is applied within the partial-sort window. When topK is also set, this is
    /// numerically equivalent to global topP because top-K already captures essentially
    /// all probability mass for realistic distributions.
    private static func selectActiveIndices(
        logits: [Float],
        topK: Int?,
        topP: Float?,
        minP: Float?
    ) -> [Int] {
        let vocabSize = logits.count

        let sortedIndices: [Int]
        // Softmax denominator over the candidates, when the selection already computed it.
        var candidateSumExp: Float?
        if let k = topK {
            sortedIndices = topIndicesByLogit(logits, k: min(k, vocabSize))
        } else {
            (sortedIndices, candidateSumExp) = candidateIndicesAboveCutoff(
                logits, topP: topP, minP: minP)
        }

        guard !sortedIndices.isEmpty else { return [] }

        let maxLogit = logits[sortedIndices[0]]
        // All-masked degenerate input.
        guard maxLogit > -.infinity else { return [] }

        // Apply minP in logit space: keep tokens where logit >= maxLogit + log(minP).
        // This is equivalent to P(token) >= minP * P(best) in probability space.
        var filtered = sortedIndices
        if let m = minP {
            let threshold = maxLogit + logf(m)
            filtered = filtered.filter { logits[$0] >= threshold }
            if filtered.isEmpty { return [sortedIndices[0]] }
        }

        guard let p = topP else { return filtered }

        // Compute exp(logit - max) per candidate (only K entries, not V).
        var expValues = [Float](repeating: 0, count: filtered.count)
        var sumExp: Float = 0
        for (i, idx) in filtered.enumerated() {
            let e = expf(logits[idx] - maxLogit)
            expValues[i] = e
            sumExp += e
        }
        // `filtered` may be a trimmed prefix; the denominator must cover every candidate.
        if let total = candidateSumExp { sumExp = total }
        let invSumExp = 1.0 / max(sumExp, .leastNormalMagnitude)

        var cumProb: Float = 0
        var cutoff = filtered.count
        for i in 0..<filtered.count {
            cumProb += expValues[i] * invSumExp
            if cumProb >= p {
                cutoff = i + 1
                break
            }
        }

        return Array(filtered.prefix(cutoff))
    }

    /// Logit distance below the maximum beyond which a token is ignored when `topP` is set.
    /// `exp(-20)` is about 2e-9 of the best token's probability, so dropping everything below
    /// it cannot change the nucleus in any realistic distribution.
    private static let topPLogitWindow: Float = 20

    /// Number of logit buckets used to locate the end of the nucleus.
    private static let nucleusBucketCount = 2048

    /// Returns the tokens that can survive minP / topP, sorted descending by logit, plus the
    /// softmax denominator (relative to the max logit) over all candidates when topP is set.
    ///
    /// Candidates are tokens with logit >= `max - 20` (topP) and >= `max + log(minP)` (minP);
    /// the tighter bound wins. With minP alone the candidates are sorted directly.
    ///
    /// With topP the candidate set can still be large (a flat distribution), so the end of the
    /// nucleus is located without sorting: candidates are bucketed by logit, per-bucket softmax
    /// mass is accumulated from the top bucket down until it reaches `topP`, and only tokens in
    /// those buckets (plus one for rounding slack) are returned. The caller's exact cumulative
    /// scan over the sorted prefix then picks the precise cutoff, so the result is independent
    /// of the distribution's shape.
    ///
    /// Indices are collected in ascending order and the sort is stable, so ties resolve the same
    /// way as sorting the whole vocabulary. -inf logits never pass the cutoff.
    private static func candidateIndicesAboveCutoff(
        _ logits: [Float],
        topP: Float?,
        minP: Float?
    ) -> (indices: [Int], sumExp: Float?) {
        guard !logits.isEmpty else { return ([], nil) }

        var maxLogit: Float = 0
        vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(logits.count))
        // All-masked degenerate input.
        guard maxLogit > -.infinity else { return ([], nil) }

        var cutoff = -Float.infinity
        if topP != nil { cutoff = maxLogit - topPLogitWindow }
        if let m = minP { cutoff = max(cutoff, maxLogit + logf(m)) }

        guard let p = topP else {
            var candidates: [Int] = []
            for (i, v) in logits.enumerated() where v >= cutoff {
                candidates.append(i)
            }
            return (candidates.sorted { logits[$0] > logits[$1] }, nil)
        }

        let bucketCount = nucleusBucketCount
        // Bucket 0 holds the logits closest to the max. A zero span (minP == 1) maps to bucket 0.
        let span = maxLogit - cutoff
        let scale: Float = span > 0 ? Float(bucketCount) / span : 0
        @inline(__always) func bucket(_ v: Float) -> Int {
            min(bucketCount - 1, Int((maxLogit - v) * scale))
        }

        var mass = [Double](repeating: 0, count: bucketCount)
        var total: Double = 0
        for v in logits where v >= cutoff {
            let e = Double(expf(v - maxLogit))
            mass[bucket(v)] += e
            total += e
        }

        // First bucket where the cumulative mass reaches topP, plus one bucket of slack.
        let target = Double(p) * total
        var cum: Double = 0
        var lastBucket = bucketCount - 1
        for b in 0..<bucketCount {
            cum += mass[b]
            if cum >= target {
                lastBucket = min(b + 1, bucketCount - 1)
                break
            }
        }

        var candidates: [Int] = []
        for (i, v) in logits.enumerated() where v >= cutoff && bucket(v) <= lastBucket {
            candidates.append(i)
        }
        return (candidates.sorted { logits[$0] > logits[$1] }, Float(total))
    }

    /// Returns the K indices with the largest logit values, sorted descending by logit.
    /// O(V log K) via a min-heap of size K storing (logit, originalIndex) pairs.
    private static func topIndicesByLogit(_ logits: [Float], k: Int) -> [Int] {
        precondition(k > 0)
        let n = logits.count
        let kEff = min(k, n)

        var heap: [(Float, Int)] = []
        heap.reserveCapacity(kEff)

        for i in 0..<n {
            let v = logits[i]
            if heap.count < kEff {
                heap.append((v, i))
                siftUpByLogit(&heap, index: heap.count - 1)
            } else if v > heap[0].0 {
                heap[0] = (v, i)
                siftDownByLogit(&heap, index: 0)
            }
        }

        return heap.sorted { $0.0 > $1.0 }.map { $0.1 }
    }

    private static func siftUpByLogit(_ heap: inout [(Float, Int)], index: Int) {
        var i = index
        while i > 0 {
            let parent = (i - 1) / 2
            if heap[i].0 < heap[parent].0 {
                heap.swapAt(i, parent)
                i = parent
            } else {
                break
            }
        }
    }

    private static func siftDownByLogit(_ heap: inout [(Float, Int)], index: Int) {
        let n = heap.count
        var i = index
        while true {
            let left = 2 * i + 1
            let right = 2 * i + 2
            var smallest = i
            if left < n && heap[left].0 < heap[smallest].0 { smallest = left }
            if right < n && heap[right].0 < heap[smallest].0 { smallest = right }
            if smallest == i { break }
            heap.swapAt(i, smallest)
            i = smallest
        }
    }

    // MARK: - Helpers

    /// Vectorized Float16 → Float32 conversion via vImage's Planar16F → PlanarF.
    /// Centralizes the conversion used by both the temperature entry path and the
    /// fp16 greedy path.
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    private static func convertFloat16ToFloat32(_ logits: inout [Float16]) -> [Float] {
        let count = logits.count
        var logitsF32 = [Float](repeating: 0, count: count)
        logits.withUnsafeMutableBufferPointer { src in
            logitsF32.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(
                    data: src.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float16>.stride
                )
                var dstBuf = vImage_Buffer(
                    data: dst.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.stride
                )
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }
        return logitsF32
    }
    #endif

    /// Describe the sampling strategy for profiling.
    private static func describeStrategy(_ config: SamplingConfiguration) -> String {
        var parts: [String] = []

        if config.temperature > 0 {
            parts.append("temp")
        }
        if config.minP != nil {
            parts.append("minP")
        }
        if config.topP != nil {
            parts.append("topP")
        }
        if config.topK != nil {
            parts.append("topK")
        }

        return parts.isEmpty ? "temperature" : parts.joined(separator: "+")
    }
}
