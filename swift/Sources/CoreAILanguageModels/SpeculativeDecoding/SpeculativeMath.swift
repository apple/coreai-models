// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Sampling / selection primitives shared by the speculative drafter and decoder.
enum SpeculativeMath {
    /// Index of the largest logit.
    static func argmax(_ logits: [LogitsScalarType]) -> Int32 {
        var best = 0
        var bestValue = logits[0]
        for i in 1..<logits.count where logits[i] > bestValue {
            bestValue = logits[i]
            best = i
        }
        return Int32(best)
    }

    /// Numerically stable temperature-scaled softmax: `softmax(logits / temperature)`.
    static func temperatureSoftmax(_ logits: [LogitsScalarType], temperature: Double) -> [Float] {
        let n = logits.count
        var maxLogit = -Float.greatestFiniteMagnitude
        for i in 0..<n {
            let v = Float(logits[i])
            if v > maxLogit { maxLogit = v }
        }
        let invT = Float(1.0 / temperature)
        var probs = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for i in 0..<n {
            let e = expf((Float(logits[i]) - maxLogit) * invT)
            probs[i] = e
            sum += e
        }
        if sum > 0 {
            let inv = 1.0 / sum
            for i in 0..<n { probs[i] *= inv }
        }
        return probs
    }

    /// Inverse-CDF sample from a normalized probability vector.
    static func sampleFromProbs(_ probs: [Float], using rng: inout SystemRandomNumberGenerator) -> Int32 {
        let u = Float(Double.random(in: 0..<1, using: &rng))
        var cumulative: Float = 0
        for i in 0..<probs.count {
            cumulative += probs[i]
            if u < cumulative { return Int32(i) }
        }
        return Int32(probs.count - 1)
    }

    /// Sample from the normalized residual `norm(relu(p − q))` used on rejection.
    static func sampleResidual(
        p: [Float], q: [Float], using rng: inout SystemRandomNumberGenerator
    ) -> Int32 {
        let n = p.count
        var residual = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for i in 0..<n {
            let r = p[i] - q[i]
            if r > 0 {
                residual[i] = r
                sum += r
            }
        }
        // Degenerate residual (p ⪯ q everywhere): fall back to sampling from p.
        guard sum > 0 else { return sampleFromProbs(p, using: &rng) }
        let u = Float(Double.random(in: 0..<1, using: &rng)) * sum
        var cumulative: Float = 0
        for i in 0..<n {
            cumulative += residual[i]
            if u < cumulative { return Int32(i) }
        }
        return Int32(n - 1)
    }
}
