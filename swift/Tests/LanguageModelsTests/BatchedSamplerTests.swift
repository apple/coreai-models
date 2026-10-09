// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

/// Each row samples with its own ``SamplingConfiguration`` and step, so greedy and seeded rows can
/// share a batch.
@Suite("Batched Sampler Tests")
struct BatchedSamplerTests {
    /// A logit row of length `vocab` with a single clear argmax at `peak`.
    private func peakedRow(vocab: Int, peak: Int) -> [LogitsScalarType] {
        var row = [LogitsScalarType](repeating: -10, count: vocab)
        row[peak] = 10
        return row
    }

    private let noHistory = ArraySlice<Int32>()

    @Test("N=2, both greedy: each row samples its own argmax")
    func batchedGreedyPerRow() {
        let rows = [peakedRow(vocab: 32, peak: 3), peakedRow(vocab: 32, peak: 7)]
        let tokens = BatchedSampler.sample(
            rows: rows,
            configurations: [.greedy, .greedy],
            histories: [noHistory, noHistory],
            steps: [nil, nil])
        #expect(tokens == [3, 7])
    }

    @Test("N=2, greedy and seeded temperature rows: greedy is deterministic, seeded is reproducible")
    func batchedMixedSampling() {
        let rows = [peakedRow(vocab: 64, peak: 2), peakedRow(vocab: 64, peak: 50)]
        let configs: [SamplingConfiguration] = [
            .greedy,
            SamplingConfiguration(temperature: 0.7, seed: 42),
        ]
        let histories = [noHistory, noHistory]
        let steps: [Int?] = [nil, 5]

        let first = BatchedSampler.sample(
            rows: rows, configurations: configs, histories: histories, steps: steps)
        let second = BatchedSampler.sample(
            rows: rows, configurations: configs, histories: histories, steps: steps)

        // Greedy row 0 returns its argmax.
        #expect(first[0] == 2)
        #expect(second[0] == 2)
        // Seeded row 1 repeats across identical calls and stays in the vocab range.
        #expect(first[1] == second[1])
        #expect(first[1] >= 0 && first[1] < 64)
    }
}
