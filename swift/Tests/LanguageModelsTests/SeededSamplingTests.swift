// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Coverage for seeded, reproducible sampling: the `SeededRandomNumberGenerator` and the
/// `SamplingConfiguration.seed` path through `fallbackSampler`. All model-free and
/// hardware-free — pure CPU sampling over hand-built logits.
@Suite("Seeded sampling", .serialized)
struct SeededSamplingTests {
    // MARK: - SeededRandomNumberGenerator

    @Test("Same seed reproduces the same stream")
    func sameSeedSameStream() {
        var a = SeededRandomNumberGenerator(seed: 12345)
        var b = SeededRandomNumberGenerator(seed: 12345)
        for _ in 0..<32 {
            #expect(a.next() == b.next())
        }
    }

    @Test("Different seeds diverge immediately")
    func differentSeedsDiverge() {
        var a = SeededRandomNumberGenerator(seed: 1)
        var b = SeededRandomNumberGenerator(seed: 2)
        // SplitMix64 mixes hard, so adjacent seeds differ on the first draw.
        #expect(a.next() != b.next())
    }

    // MARK: - Reproducible token selection

    /// A flat distribution: equal logits so softmax is uniform and the RNG alone
    /// decides the token. This makes the seed's effect observable.
    private func flatLogits(_ count: Int) -> [LogitsScalarType] {
        [LogitsScalarType](repeating: 1, count: count)
    }

    @Test("Same seed and step reproduce the same token")
    func sameSeedSameStepSameToken() {
        let config = SamplingConfiguration(temperature: 1.0, seed: 777)
        var first = flatLogits(16)
        var second = flatLogits(16)
        let a = config.fallbackSampler(from: &first, step: 3)
        let b = config.fallbackSampler(from: &second, step: 3)
        #expect(a == b)
    }

    @Test("A seeded run reproduces its full token sequence")
    func seededSequenceIsReproducible() {
        let config = SamplingConfiguration(temperature: 1.0, seed: 999)

        func run() -> [Int32] {
            (0..<24).map { step in
                var logits = flatLogits(32)
                return config.fallbackSampler(from: &logits, step: step)
            }
        }

        #expect(run() == run())
    }

    @Test("The seed actually influences the sampled token")
    func seedInfluencesSelection() {
        // Over many seeds on a uniform 4-way distribution, seeing only one token would be
        // 4 * (1/4)^N — vanishing for N=128. So >1 distinct token is effectively certain,
        // and proves the seed feeds the sampler rather than being ignored.
        var tokens: Set<Int32> = []
        for seed in 0..<UInt64(128) {
            let config = SamplingConfiguration(temperature: 1.0, seed: seed)
            var logits = flatLogits(4)
            tokens.insert(config.fallbackSampler(from: &logits, step: 0))
        }
        #expect(tokens.count > 1)
        #expect(tokens.allSatisfy { $0 >= 0 && $0 < 4 })
    }

    @Test("Different steps under one seed advance the generator")
    func differentStepsDiffer() {
        // Same seed, different step -> a different derived generator. Over a flat
        // distribution the per-step tokens should not all collapse to one value.
        let config = SamplingConfiguration(temperature: 1.0, seed: 55)
        var tokens: Set<Int32> = []
        for step in 0..<64 {
            var logits = flatLogits(4)
            tokens.insert(config.fallbackSampler(from: &logits, step: step))
        }
        #expect(tokens.count > 1)
    }

    // MARK: - Greedy ignores the seed

    @Test("Greedy is argmax regardless of seed")
    func greedyIgnoresSeed() {
        var logits: [LogitsScalarType] = [0.1, 3.0, 0.2, 1.5]
        let s1 = SamplingConfiguration(temperature: 0, seed: 1).fallbackSampler(from: &logits, step: 0)
        var logits2: [LogitsScalarType] = [0.1, 3.0, 0.2, 1.5]
        let s2 = SamplingConfiguration(temperature: 0, seed: 2).fallbackSampler(from: &logits2, step: 9)
        #expect(s1 == 1)
        #expect(s2 == 1)
    }

    // MARK: - Unseeded path stays live

    @Test("Nil seed still returns an in-range token")
    func nilSeedIsLive() {
        let config = SamplingConfiguration(temperature: 1.0)
        #expect(config.seed == nil)
        var logits = flatLogits(8)
        let token = config.fallbackSampler(from: &logits, step: 0)
        #expect(token >= 0 && token < 8)
    }

    // MARK: - Normalization preserves the seed

    @Test("normalized() carries the seed through")
    func normalizedPreservesSeed() {
        let config = SamplingConfiguration(temperature: 0.8, topP: 1.0, seed: 4242)
        #expect(config.normalized().seed == 4242)
    }

    // MARK: - sampleToken (used by the constrained-decoding path)

    @Test("sampleToken is reproducible for the same seed and step")
    func sampleTokenReproducible() {
        let config = SamplingConfiguration(temperature: 1.0, seed: 321)
        var a = flatLogits(16)
        var b = flatLogits(16)
        #expect(config.sampleToken(from: &a, step: 4) == config.sampleToken(from: &b, step: 4))
    }

    @Test("sampleToken with nil seed still returns an in-range token")
    func sampleTokenNilSeedLive() {
        let config = SamplingConfiguration(temperature: 1.0)
        var logits = flatLogits(8)
        let token = config.sampleToken(from: &logits, step: 0)
        #expect(token >= 0 && token < 8)
    }

    // MARK: - fallbackSampler(from:tokenHistory:step:) — the static-shape engine path

    /// The static-shape/ANE engine samples through `fallbackSampler(from:tokenHistory:step:)`.
    /// Before the fix it always passed `step: 0`, so `seed &+ step` never advanced and every
    /// token drew the same quantile. With the step threaded through, consecutive steps derive
    /// distinct generators and do not collapse to a single token on a flat distribution.
    @Test("Threading step advances the RNG on the tokenHistory sampler")
    func tokenHistorySamplerAdvancesWithStep() {
        let config = SamplingConfiguration(temperature: 1.0, seed: 88)
        let history: [Int32] = []
        var tokens: Set<Int32> = []
        for step in 0..<64 {
            var logits = flatLogits(4)
            tokens.insert(config.fallbackSampler(from: &logits, tokenHistory: history, step: step))
        }
        #expect(tokens.count > 1)
    }

    /// A frozen step (the pre-fix behavior) collapses to one token — the bug this fix removes.
    @Test("A frozen step collapses to a single quantile")
    func frozenStepCollapses() {
        let config = SamplingConfiguration(temperature: 1.0, seed: 88)
        let history: [Int32] = []
        var tokens: Set<Int32> = []
        for _ in 0..<64 {
            var logits = flatLogits(4)
            tokens.insert(config.fallbackSampler(from: &logits, tokenHistory: history, step: 0))
        }
        #expect(tokens.count == 1)
    }

    // MARK: - GPU pipelined engine rejects seeded requests

    /// The GPU pipelined engine samples on-GPU and cannot honor the seed, so it must reject
    /// seeded requests loudly rather than silently ignoring reproducibility.
    @Test("Pipelined engine guard throws when a seed is set")
    func pipelinedGuardThrowsOnSeed() {
        let seeded = SamplingConfiguration(temperature: 1.0, seed: 7)
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAIPipelinedEngine.rejectSeedIfUnsupported(seeded)
        }
    }

    @Test("Pipelined engine guard is a no-op without a seed")
    func pipelinedGuardAllowsUnseeded() throws {
        let unseeded = SamplingConfiguration(temperature: 1.0)
        try CoreAIPipelinedEngine.rejectSeedIfUnsupported(unseeded)
    }
}
