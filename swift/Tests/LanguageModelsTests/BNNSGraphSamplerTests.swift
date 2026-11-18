// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// `BNNSGraphSampler` samples from `ReferenceSampler`'s distribution but walks the kept tokens in
/// vocabulary order, so the tests compare distributions, not tokens, except where the graph
/// defers to `ReferenceSampler` or the result is deterministic (greedy).
@Suite("BNNSGraphSampler", .serialized)
struct BNNSGraphSamplerTests {
    static let stochasticConfigs: [SamplingConfiguration] = [
        .init(temperature: 0.8),
        .init(temperature: 0.8, topK: 10),
        .init(temperature: 0.8, topP: 0.9),
        .init(temperature: 0.8, minP: 0.05),
        .init(temperature: 0.8, topK: 10, topP: 0.9),
        .init(temperature: 0.8, topK: 50, topP: 0.9, minP: 0.05),
        .init(temperature: 1.3, topP: 0.5, minP: 0.1),
    ]

    @Test("Greedy returns ReferenceSampler's token", arguments: [151_936, 32_000, 7])
    func greedyMatchesReference(vocab: Int) {
        var rng = Xoshiro256StarStar(seed: 7)
        for _ in 0..<20 {
            let row = Self.logits(vocab: vocab, scale: 4, using: &rng)
            var a = row
            var b = row
            #expect(
                BNNSGraphSampler.sample(from: &a, config: .greedy) == ReferenceSampler.sample(from: &b, config: .greedy)
            )
        }
    }

    @Test("Samples from ReferenceSampler's distribution", arguments: stochasticConfigs)
    func sameDistribution(config: SamplingConfiguration) {
        let vocab = 1000
        var logitRNG = Xoshiro256StarStar(seed: 11)
        let row = Self.logits(vocab: vocab, scale: 6, using: &logitRNG)
        let expected = Self.referenceDistribution(row, config: config)
        let draws = 20_000
        var counts = [Int](repeating: 0, count: vocab)
        var rng = Xoshiro256StarStar(seed: 1)
        for _ in 0..<draws {
            var a = row
            counts[Int(BNNSGraphSampler.sample(from: &a, config: config, using: &rng))] += 1
        }
        for i in 0..<vocab where expected[i] == 0 {
            #expect(counts[i] == 0, "token \(i) drawn \(counts[i]) times; ReferenceSampler never keeps it")
        }
        let totalVariation =
            zip(counts, expected)
            .map { abs(Double($0) / Double(draws) - $1) }.reduce(0, +) / 2
        #expect(totalVariation < 0.03, "total variation \(totalVariation)")
    }

    @Test("A row with NaN or +infinity goes to ReferenceSampler")
    func nonFiniteRowsDefer() {
        for bad in [LogitsScalarType.nan, .infinity] {
            var rng = Xoshiro256StarStar(seed: 3)
            var row = Self.logits(vocab: 500, scale: 3, using: &rng)
            row[123] = bad

            var a = row
            var b = row
            #expect(
                BNNSGraphSampler.sample(from: &a, config: .greedy) == ReferenceSampler.sample(from: &b, config: .greedy)
            )

            for config in Self.stochasticConfigs {
                #expect(Self.defersToReference(row, config: config))
            }
        }
    }

    @Test("An all-masked row goes to ReferenceSampler")
    func allMaskedRowDefers() {
        let row = [LogitsScalarType](repeating: -.infinity, count: 300)
        for config in Self.stochasticConfigs {
            #expect(Self.defersToReference(row, config: config))
        }
    }

    @Test("topP alone goes to ReferenceSampler when the nucleus is wider than the window")
    func wideNucleusDefers() {
        var rng = Xoshiro256StarStar(seed: 5)
        // Nearly flat logits: a 0.95 nucleus needs far more than 128 tokens.
        let row = Self.logits(vocab: 2000, scale: 0.01, using: &rng)
        #expect(Self.defersToReference(row, config: .init(temperature: 1, topP: 0.95)))
    }

    @Test("Grammar-masked tokens are never drawn")
    func maskedTokensNeverDrawn() {
        let vocab = 4096
        let allowed: Set<Int> = [5, 900, 1700, 3000, 4095]
        var row = [LogitsScalarType](repeating: -.infinity, count: vocab)
        for i in allowed { row[i] = LogitsScalarType(Float(i % 7)) }
        var rng = Xoshiro256StarStar(seed: 9)
        for config in Self.stochasticConfigs {
            for _ in 0..<500 {
                var a = row
                #expect(allowed.contains(Int(BNNSGraphSampler.sample(from: &a, config: config, using: &rng))))
            }
        }
    }

    @Test("Graphs are cached per configuration and vocabulary size")
    func graphsAreCached() throws {
        let key = BNNSGraphSampler.GraphKey(.init(temperature: 0.7, topK: 40), vocabSize: 777)
        let first = try #require(BNNSGraphSampler.graph(for: key))
        let second = try #require(BNNSGraphSampler.graph(for: key))
        #expect(first === second)
        let other = BNNSGraphSampler.GraphKey(.init(temperature: 0.7, topK: 40), vocabSize: 778)
        let otherGraph = try #require(BNNSGraphSampler.graph(for: other))
        #expect(otherGraph !== first)
        // Repetition penalty and the seed don't change the graph.
        #expect(
            BNNSGraphSampler.GraphKey(.init(temperature: 0.7, topK: 40, repetitionPenalty: 1.2), vocabSize: 777) == key)
    }

    @Test("Every configuration compiles for a prime vocabulary size", arguments: stochasticConfigs)
    func primeVocabularyCompiles(config: SamplingConfiguration) {
        #expect(BNNSGraphSampler.graph(for: .init(config, vocabSize: 997)) != nil)
    }

    @Test("A prime vocabulary size samples from ReferenceSampler's distribution")
    func primeVocabularyDistribution() {
        var logitRNG = Xoshiro256StarStar(seed: 12)
        let row = Self.logits(vocab: 997, scale: 6, using: &logitRNG)
        let config = SamplingConfiguration(temperature: 0.8)
        let expected = Self.referenceDistribution(row, config: config)
        let draws = 20_000
        var counts = [Int](repeating: 0, count: row.count)
        var rng = Xoshiro256StarStar(seed: 3)
        for _ in 0..<draws {
            var a = row
            counts[Int(BNNSGraphSampler.sample(from: &a, config: config, using: &rng))] += 1
        }
        let totalVariation =
            zip(counts, expected)
            .map { abs(Double($0) / Double(draws) - $1) }.reduce(0, +) / 2
        #expect(totalVariation < 0.03, "total variation \(totalVariation)")
    }

    @Test("The graph cache drops the oldest graph beyond its limit")
    func cacheEvictsOldest() throws {
        let keys = (0...BNNSGraphSampler.cacheLimit).map {
            BNNSGraphSampler.GraphKey(.init(temperature: 0.5 + Double($0) / 1000, topK: 8), vocabSize: 64)
        }
        let first = try #require(BNNSGraphSampler.graph(for: keys[0]))
        for key in keys.dropFirst() {
            _ = try #require(BNNSGraphSampler.graph(for: key))
        }
        let recompiled = try #require(BNNSGraphSampler.graph(for: keys[0]))
        #expect(recompiled !== first)
    }

    @Test("Block split covers the vocabulary", arguments: [151_936, 262_144, 32_000, 7, 1])
    func blockSplit(vocab: Int) {
        let (blocks, blockSize) = BNNSGraphSampler.blockSplit(vocab)
        #expect(blocks * blockSize == vocab)
        #expect(blocks <= blockSize)
    }

    // MARK: - Helpers

    /// Logits with a few large entries over a noise floor, rounded to the logit dtype.
    static func logits(vocab: Int, scale: Float, using rng: inout some RandomNumberGenerator) -> [LogitsScalarType] {
        (0..<vocab).map { _ in
            let u = Float.random(in: 0..<1, using: &rng)
            return LogitsScalarType(scale * (u * u * u * 8 - 2))
        }
    }

    /// ReferenceSampler's distribution, written out from its documented order: z = logits /
    /// temperature; topK; minP (z >= max + log(minP)); topP (the shortest z-descending prefix
    /// whose share reaches topP); then probabilities in proportion to exp(z - max).
    ///
    /// Without topK, ReferenceSampler's topP also ignores tokens below max - (ln V + 18); together
    /// they carry under 1.5e-8 of the mass, so they are kept here. Rows are finite: +infinity
    /// rows go to ReferenceSampler and are covered by the defer tests.
    static func referenceDistribution(_ row: [LogitsScalarType], config: SamplingConfiguration) -> [Double] {
        let z = row.map { Float($0) / Float(config.temperature) }
        var kept = z.indices.sorted { z[$0] > z[$1] }
        if let topK = config.topK { kept = Array(kept.prefix(topK)) }
        let maxZ = z[kept[0]]
        if let minP = config.minP { kept = kept.filter { z[$0] >= maxZ + logf(Float(minP)) } }
        let weight = { (i: Int) in Double(expf(z[i] - maxZ)) }
        if let topP = config.topP, topP < 1 {
            let total = kept.map(weight).reduce(0, +)
            var running = 0.0
            var cut = kept.count
            for (n, i) in kept.enumerated() {
                running += weight(i) / total
                if running >= topP {
                    cut = n + 1
                    break
                }
            }
            kept = Array(kept.prefix(cut))
        }
        let total = kept.map(weight).reduce(0, +)
        var probabilities = [Double](repeating: 0, count: row.count)
        for i in kept { probabilities[i] = weight(i) / total }
        return probabilities
    }

    /// True when BNNSGraphSampler returns what ReferenceSampler returns after the one random
    /// number BNNSGraphSampler draws before deferring.
    static func defersToReference(_ row: [LogitsScalarType], config: SamplingConfiguration) -> Bool {
        for seed in UInt64(0)..<8 {
            var bnnsRNG = Xoshiro256StarStar(seed: seed)
            var referenceRNG = Xoshiro256StarStar(seed: seed)
            _ = Float.random(in: 0..<1, using: &referenceRNG)
            var a = row
            var b = row
            if BNNSGraphSampler.sample(from: &a, config: config, using: &bnnsRNG)
                != ReferenceSampler.sample(from: &b, config: config, using: &referenceRNG)
            {
                return false
            }
        }
        return true
    }
}
