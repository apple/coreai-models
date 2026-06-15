// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Test coverage for `CompositeSampler` — pinning current behavior before the slow-path
/// refactor (replace `[Bool]` mask with `-Float.infinity`, compact-then-sample after topK,
/// partial sort, etc.).
///
/// Most tests use a seeded `Xoshiro256StarStar` so multinomial sampling is deterministic.
/// `@testable import` reaches `CompositeSampler.sample(from:config:using:)` — the
/// production no-RNG overload is unchanged.
@Suite("CompositeSampler", .serialized)
struct CompositeSamplerTests {
    // MARK: - Greedy (deterministic — no RNG needed)

    @Test("Greedy Float32 returns the argmax")
    func greedyFloat32ReturnsArgmax() {
        var logits: [Float] = [0.1, -0.5, 3.2, 1.7, 0.0, 2.9]
        let token = CompositeSampler.sample(from: &logits, config: .greedy)
        #expect(token == 2)
    }

    @Test("Greedy Float16 returns the argmax")
    func greedyFloat16ReturnsArgmax() {
        var logits: [Float16] = [0.1, -0.5, 3.2, 1.7, 0.0, 2.9]
        let token = CompositeSampler.sample(from: &logits, config: .greedy)
        #expect(token == 2)
    }

    @Test("Greedy tie-breaking returns first occurrence (vDSP_maxvi behavior)")
    func greedyTieBreaksToFirst() {
        var logits: [Float] = [1.0, 3.0, 3.0, 3.0, 2.0]
        let token = CompositeSampler.sample(from: &logits, config: .greedy)
        #expect(token == 1)
    }

    @Test("Greedy on single-element logits returns 0")
    func greedySingleElement() {
        var logits: [Float] = [42.0]
        let token = CompositeSampler.sample(from: &logits, config: .greedy)
        #expect(token == 0)
    }

    // MARK: - Fast path (no topK/topP) — mask via -inf

    @Test("Fast path: Float32 -inf positions are never sampled")
    func fastPathFloat32MaskRespected() {
        let vocab = 256
        let allowed: Set<Int32> = [3, 17, 42, 100, 200]
        var rng = Xoshiro256StarStar(seed: 0xC0FFEE)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<10_000 {
            var logits = makeMaskedLogits(vocab: vocab, allowed: allowed, fill: 1.0, sentinel: -.infinity)
            let token = CompositeSampler.sample(from: &logits, config: .init(temperature: 1.0), using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "Masked token \(i) sampled \(counts[i]) times — must be 0")
        }
        let totalActive = counts.reduce(0, +)
        #expect(totalActive == 10_000)
    }

    @Test("Fast path: Float16 -greatestFiniteMagnitude positions are never sampled")
    func fastPathFloat16MaskRespected() {
        let vocab = 256
        let allowed: Set<Int32> = [3, 17, 42, 100, 200]
        let sentinel = -Float16.greatestFiniteMagnitude
        var rng = Xoshiro256StarStar(seed: 0xBADF00D)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<10_000 {
            var logits = [Float16](repeating: sentinel, count: vocab)
            for i in allowed { logits[Int(i)] = 1.0 }
            let token = CompositeSampler.sample(from: &logits, config: .init(temperature: 1.0), using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "Float16 masked token \(i) sampled \(counts[i]) times — must be 0")
        }
    }

    @Test("Fast path: uniform logits produce roughly uniform sample frequencies")
    func fastPathUniformDistribution() {
        let vocab = 10
        var rng = Xoshiro256StarStar(seed: 0xDEAD_BEEF)

        var counts = [Int](repeating: 0, count: vocab)
        let iterations = 10_000
        for _ in 0..<iterations {
            var logits = [Float](repeating: 1.0, count: vocab)
            let token = CompositeSampler.sample(from: &logits, config: .init(temperature: 1.0), using: &rng)
            counts[Int(token)] += 1
        }

        let expected = Double(iterations) / Double(vocab)
        let sigma = (expected * (1.0 - 1.0 / Double(vocab))).squareRoot()
        for (i, c) in counts.enumerated() {
            let deviation = abs(Double(c) - expected)
            #expect(deviation < 5 * sigma, "Token \(i) count \(c) deviates >5σ from expected \(expected)")
        }
    }

    // MARK: - Slow path (topK / topP)

    @Test("topK=K limits the sampled set to at most K distinct tokens")
    func topKLimitsSampledSet() {
        let vocab = 100
        let k = 5
        var rng = Xoshiro256StarStar(seed: 0xABCD)

        var rawLogits = [Float](repeating: 0, count: vocab)
        for i in 0..<vocab { rawLogits[i] = Float(i) }

        var sampledIndices = Set<Int32>()
        for _ in 0..<2_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topK: k), using: &rng)
            sampledIndices.insert(token)
        }

        #expect(sampledIndices.count <= k)
        // The top K logits are indices [vocab-k, ..., vocab-1] = [95, 96, 97, 98, 99]
        for idx in sampledIndices {
            #expect(idx >= Int32(vocab - k), "Sampled \(idx) outside top-K set")
        }
    }

    @Test("topP=0.9 limits the sampled set to a minimal cumulative-probability prefix")
    func topPLimitsByProbabilityMass() {
        let vocab = 100
        var rng = Xoshiro256StarStar(seed: 0x1234_5678)

        // Sharply peaked distribution: token 99 dominates after softmax with temperature=1.
        var rawLogits = [Float](repeating: -5.0, count: vocab)
        rawLogits[99] = 5.0
        rawLogits[98] = 4.5
        rawLogits[97] = 4.0

        var sampledIndices = Set<Int32>()
        for _ in 0..<2_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topP: 0.9), using: &rng)
            sampledIndices.insert(token)
        }

        // With this distribution, topP=0.9 should keep only the few peaked tokens.
        #expect(sampledIndices.isSubset(of: [97, 98, 99]))
    }

    @Test("Combined topK + topP intersect correctly")
    func topKAndTopPCombined() {
        let vocab = 50
        var rng = Xoshiro256StarStar(seed: 0xFEED_FACE)

        var rawLogits = [Float](repeating: 0, count: vocab)
        for i in 0..<vocab { rawLogits[i] = Float(i) * 0.1 }

        var sampledIndices = Set<Int32>()
        for _ in 0..<2_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topK: 10, topP: 0.95), using: &rng)
            sampledIndices.insert(token)
        }

        for idx in sampledIndices {
            #expect(idx >= Int32(vocab - 10), "Combined filter sampled \(idx) outside top-10")
        }
    }

    // MARK: - MinP sampling

    @Test("minP limits the sampled set to tokens with sufficient relative probability")
    func minPLimitsSampledSet() {
        let vocab = 100
        var rng = Xoshiro256StarStar(seed: 0xAAAA_BBBB)

        // Token 99 has logit 10.0, tokens 98-95 have logit ~9.5-8.0
        // Everything else is -5.0 (very low relative probability)
        var rawLogits = [Float](repeating: -5.0, count: vocab)
        rawLogits[99] = 10.0
        rawLogits[98] = 9.5
        rawLogits[97] = 9.0
        rawLogits[96] = 8.5
        rawLogits[95] = 8.0

        var sampledIndices = Set<Int32>()
        for _ in 0..<2_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, minP: 0.1), using: &rng)
            sampledIndices.insert(token)
        }

        // With minP=0.1, only tokens whose probability is >= 10% of the max token's probability
        // should be sampled. The -5.0 tokens should be far below this threshold.
        for idx in sampledIndices {
            #expect(idx >= 95, "minP=0.1 sampled low-probability token \(idx)")
        }
    }

    @Test("minP=1.0 keeps only the most probable token (equivalent to greedy)")
    func minPOneIsGreedy() {
        let vocab = 50
        var rng = Xoshiro256StarStar(seed: 0xCCCC_DDDD)

        var rawLogits = [Float](repeating: 0, count: vocab)
        rawLogits[42] = 5.0
        rawLogits[10] = 4.9

        var allSame = true
        for _ in 0..<100 {
            var logits = rawLogits
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, minP: 1.0), using: &rng)
            if token != 42 {
                allSame = false
                break
            }
        }
        #expect(allSame, "minP=1.0 should always pick the top token")
    }

    @Test("minP + topK combined: both filters apply")
    func minPWithTopK() {
        let vocab = 100
        var rng = Xoshiro256StarStar(seed: 0x1111_2222)

        // Spread: top 5 tokens have high logits, next 5 have medium, rest are low
        var rawLogits = [Float](repeating: -10.0, count: vocab)
        for i in 95..<100 { rawLogits[i] = 5.0 }  // high
        for i in 90..<95 { rawLogits[i] = 2.0 }  // medium

        var sampledIndices = Set<Int32>()
        for _ in 0..<2_000 {
            var logits = rawLogits
            // topK=10 would include indices 90-99, but minP=0.3 should exclude
            // the medium ones since exp(2-5)/exp(0) = exp(-3) ≈ 0.05 < 0.3
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topK: 10, minP: 0.3), using: &rng)
            sampledIndices.insert(token)
        }

        // Only the top 5 (indices 95-99) should survive both filters
        for idx in sampledIndices {
            #expect(idx >= 95, "minP+topK: sampled \(idx) which should have been filtered")
        }
    }

    @Test("minP with guided generation mask: masked tokens never sampled")
    func minPWithGGMask() {
        let vocab = 256
        let allowed: Set<Int32> = [10, 20, 30, 40, 50]
        var rng = Xoshiro256StarStar(seed: 0x3333_4444)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<5_000 {
            var logits = makeMaskedLogits(vocab: vocab, allowed: allowed, fill: 2.0, sentinel: -.infinity)
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, minP: 0.05), using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "GG+minP: masked \(i) sampled \(counts[i]) times")
        }
    }

    // MARK: - Guided generation (mask + topK/topP)

    @Test("GG path: -inf + topK=50, masked positions never sampled")
    func ggMaskWithTopK() {
        let vocab = 256
        let allowed: Set<Int32> = [5, 50, 100, 150, 250]
        var rng = Xoshiro256StarStar(seed: 0xD00D)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<5_000 {
            var logits = makeMaskedLogits(vocab: vocab, allowed: allowed, fill: 2.0, sentinel: -.infinity)
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topK: 50), using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "GG+topK: masked \(i) sampled \(counts[i]) times")
        }
    }

    @Test("GG path: -inf + topP=0.9, masked positions never sampled")
    func ggMaskWithTopP() {
        let vocab = 256
        let allowed: Set<Int32> = [5, 50, 100, 150, 250]
        var rng = Xoshiro256StarStar(seed: 0xCAFE)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<5_000 {
            var logits = makeMaskedLogits(vocab: vocab, allowed: allowed, fill: 2.0, sentinel: -.infinity)
            let token = CompositeSampler.sample(
                from: &logits, config: .init(temperature: 1.0, topP: 0.9), using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "GG+topP: masked \(i) sampled \(counts[i]) times")
        }
    }

    @Test("GG path: -inf + topK + topP combined, masked positions never sampled")
    func ggMaskWithTopKAndTopP() {
        let vocab = 256
        let allowed: Set<Int32> = [5, 50, 100, 150, 250]
        var rng = Xoshiro256StarStar(seed: 0x5A5A_5A5A)

        var counts = [Int](repeating: 0, count: vocab)
        for _ in 0..<5_000 {
            var logits = makeMaskedLogits(vocab: vocab, allowed: allowed, fill: 2.0, sentinel: -.infinity)
            let token = CompositeSampler.sample(
                from: &logits,
                config: .init(temperature: 1.0, topK: 50, topP: 0.9),
                using: &rng)
            counts[Int(token)] += 1
        }

        for i in 0..<vocab where !allowed.contains(Int32(i)) {
            #expect(counts[i] == 0, "GG+topK+topP: masked \(i) sampled \(counts[i]) times")
        }
    }

    @Test("GG path: single-allowed-token mask always returns that token")
    func ggSingleAllowedToken() {
        let vocab = 256
        let allowedIndex: Int32 = 137
        var rng = Xoshiro256StarStar(seed: 0x9999)

        let configs: [SamplingConfiguration] = [
            .init(temperature: 0.1),
            .init(temperature: 1.0),
            .init(temperature: 1.0, topK: 50),
            .init(temperature: 1.0, topP: 0.9),
            .init(temperature: 1.0, topK: 50, topP: 0.9),
        ]
        for config in configs {
            for _ in 0..<100 {
                var logits = [Float](repeating: -.infinity, count: vocab)
                logits[Int(allowedIndex)] = 5.0
                let token = CompositeSampler.sample(from: &logits, config: config, using: &rng)
                #expect(
                    token == allowedIndex,
                    "Single-allowed should always return \(allowedIndex), got \(token) with config \(config)")
            }
        }
    }

    // MARK: - Numerical edge cases

    @Test("Saturated logits: one position 50 above the rest is overwhelmingly sampled")
    func saturatedLogitsDominate() {
        let vocab = 100
        let dominant: Int32 = 73
        var rng = Xoshiro256StarStar(seed: 0xAA55)

        var rawLogits = [Float](repeating: 0.0, count: vocab)
        rawLogits[Int(dominant)] = 50.0

        var hits = 0
        for _ in 0..<5_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(from: &logits, config: .init(temperature: 1.0), using: &rng)
            if token == dominant { hits += 1 }
        }
        #expect(hits == 5_000, "Saturated logit should be sampled every time, got \(hits)/5000")
    }

    @Test("Very low temperature concentrates probability on argmax")
    func lowTemperatureNearGreedy() {
        let vocab = 50
        var rng = Xoshiro256StarStar(seed: 0xB00B)

        var rawLogits = [Float](repeating: 0.0, count: vocab)
        rawLogits[42] = 1.0
        rawLogits[10] = 0.5

        var hitsArgmax = 0
        for _ in 0..<5_000 {
            var logits = rawLogits
            let token = CompositeSampler.sample(from: &logits, config: .init(temperature: 0.01), using: &rng)
            if token == 42 { hitsArgmax += 1 }
        }
        #expect(hitsArgmax >= 4_950, "Low temp should pick argmax >99% of the time, got \(hitsArgmax)/5000")
    }

    @Test("Float16 round-trip preserves greedy argmax across the vImage convert")
    func float16ArgmaxParity() {
        let vocab = 100
        var rawLogits16 = [Float16](repeating: 0, count: vocab)
        for i in 0..<vocab { rawLogits16[i] = Float16(Float(i) * 0.123) }

        var logits16 = rawLogits16
        let token16 = CompositeSampler.sample(from: &logits16, config: .greedy)

        var logits32 = rawLogits16.map { Float($0) }
        let token32 = CompositeSampler.sample(from: &logits32, config: .greedy)

        #expect(token16 == token32, "Float16 (\(token16)) and Float32 (\(token32)) greedy must agree")
    }

    // MARK: - Same-seed reproducibility (snapshot for the upcoming refactor)

    @Test("Same seed produces same sequence — slow-path snapshot")
    func sameSeedReproducible() {
        let logits1 = makeSnapshotLogits()
        let logits2 = makeSnapshotLogits()
        let config = SamplingConfiguration(temperature: 0.7, topK: 50, topP: 0.9)

        var rng1 = Xoshiro256StarStar(seed: 0x1234)
        var rng2 = Xoshiro256StarStar(seed: 0x1234)

        for _ in 0..<200 {
            var l1 = logits1
            var l2 = logits2
            let t1 = CompositeSampler.sample(from: &l1, config: config, using: &rng1)
            let t2 = CompositeSampler.sample(from: &l2, config: config, using: &rng2)
            #expect(t1 == t2)
        }
    }

    // MARK: - Test helpers

    /// Build a Float32 logits array of `vocab` length where `allowed` indices hold `fill`
    /// and all others hold `sentinel`. Used to simulate guided-generation pre-masking.
    private func makeMaskedLogits(
        vocab: Int,
        allowed: Set<Int32>,
        fill: Float,
        sentinel: Float
    ) -> [Float] {
        var logits = [Float](repeating: sentinel, count: vocab)
        for idx in allowed { logits[Int(idx)] = fill }
        return logits
    }

    /// Build a representative non-trivial logits array used as the snapshot baseline.
    private func makeSnapshotLogits() -> [Float] {
        var logits = [Float](repeating: 0, count: 500)
        for i in 0..<500 {
            logits[i] = Float((i * 7919) % 100) * 0.05 - 1.0
        }
        return logits
    }
}

// MARK: - Seeded RNG (Xoshiro256**)

/// A small, fast, well-tested PRNG for reproducible test sampling.
/// Reference: https://prng.di.unimi.it/xoshiro256starstar.c
///
/// Tests use this so that multinomial sampling is deterministic across runs and
/// across refactors. Production code uses `SystemRandomNumberGenerator`.
struct Xoshiro256StarStar: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // Seed via SplitMix64 to avoid weak Xoshiro initial states.
        var z = seed &+ 0x9E37_79B9_7F4A_7C15
        let splitmix = { () -> UInt64 in
            z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z &>> 31)
        }
        self.state = (splitmix(), splitmix(), splitmix(), splitmix())
    }

    mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 &<< 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
        (x &<< k) | (x &>> (64 &- k))
    }
}
