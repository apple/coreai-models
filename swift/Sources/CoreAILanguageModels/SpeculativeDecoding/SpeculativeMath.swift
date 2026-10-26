// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import Foundation

/// Selection and probability primitives shared by the speculative drafters and decoder.
///
/// Argmax, softmax and the multinomial draw are `CompositeSampler`'s, so the drafter and the target
/// select tokens the same way; the accept/reject step and its residual distribution (Chen et al., 2023,
/// https://arxiv.org/abs/2302.01318) are the only parts with no sampler equivalent. Every function that draws takes the random
/// number generator it draws from.
enum SpeculativeMath {
    /// Index of the largest logit, 0 for an empty array.
    static func argmax(_ logits: [LogitsScalarType]) -> Int32 {
        var logits = logits
        return CompositeSampler.sample(from: &logits, config: .greedy)
    }

    /// Numerically stable temperature-scaled softmax: `softmax(logits / temperature)`. Empty in, empty out.
    ///
    /// A NaN logit counts as negative infinity and takes no mass. When no logit is greater than
    /// negative infinity the row has no mass to share, which a plain softmax turns into NaN; the
    /// result is then uniform, so the draws and the accept/reject ratios that follow stay valid.
    ///
    /// A temperature that is not positive (zero, negative or NaN) is the greedy limit: the result is
    /// one-hot at the first maximum, rather than the NaN a division by zero would produce.
    static func temperatureSoftmax(_ logits: [LogitsScalarType], temperature: Double) -> [Float] {
        guard !logits.isEmpty else { return [] }
        var probs = logits.map { value -> Float in
            let f = Float(value)
            return f.isNaN ? -.infinity : f
        }
        var largest: Float = 0
        vDSP_maxv(probs, 1, &largest, vDSP_Length(probs.count))
        guard largest > -.infinity else {
            return [Float](repeating: 1 / Float(probs.count), count: probs.count)
        }
        // A positive Double below Float's range is zero as a Float, so test the Float.
        var t = Float(temperature)
        guard t > 0 else {
            var best = 0
            for i in 1..<probs.count where probs[i] > probs[best] { best = i }
            var oneHot = [Float](repeating: 0, count: probs.count)
            oneHot[best] = 1
            return oneHot
        }
        vDSP_vsdiv(probs, 1, &t, &probs, 1, vDSP_Length(probs.count))
        CompositeSampler.softmaxVectorized(&probs)
        return probs
    }

    /// Inverse-CDF sample from a normalized probability vector, 0 for an empty one.
    static func sampleFromProbs(_ probs: [Float], using rng: inout some RandomNumberGenerator) -> Int32 {
        guard !probs.isEmpty else { return 0 }
        return Int32(CompositeSampler.multinomialSample(probs, using: &rng))
    }

    /// Sample from the normalized residual `norm(relu(p − q))` used on rejection.
    static func sampleResidual(
        p: [Float], q: [Float], using rng: inout some RandomNumberGenerator
    ) -> Int32 {
        var residual = [Float](repeating: 0, count: p.count)
        var sum: Float = 0
        for i in 0..<p.count {
            let r = p[i] - (i < q.count ? q[i] : 0)
            if r > 0 {
                residual[i] = r
                sum += r
            }
        }
        // Degenerate residual (p ⪯ q everywhere): fall back to sampling from p.
        guard sum > 0 else { return sampleFromProbs(p, using: &rng) }
        var inv = 1 / sum
        vDSP_vsmul(residual, 1, &inv, &residual, 1, vDSP_Length(residual.count))
        return sampleFromProbs(residual, using: &rng)
    }
}
