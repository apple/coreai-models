// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

/// Batched (N>1) per-row sampling: each row carries its own ``SamplingConfiguration`` and
/// generation step, so one batch can mix a greedy sequence with a temperature-sampled one.
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

    @Test("N=2, mixed greedy + temperature: greedy row deterministic, seeded row reproducible")
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

        // Row 0 is greedy -> argmax, deterministic and independent of row 1.
        #expect(first[0] == 2)
        #expect(second[0] == 2)
        // Row 1 is seeded -> reproducible across identical calls, and a valid token id.
        #expect(first[1] == second[1])
        #expect(first[1] >= 0 && first[1] < 64)
    }

    @Test("Per-row isolation: changing one row's logits leaves the other row's token unchanged")
    func batchedRowIsolation() {
        let base = BatchedSampler.sample(
            rows: [peakedRow(vocab: 16, peak: 1), peakedRow(vocab: 16, peak: 9)],
            configurations: [.greedy, .greedy],
            histories: [noHistory, noHistory],
            steps: [nil, nil])
        let swapped = BatchedSampler.sample(
            rows: [peakedRow(vocab: 16, peak: 4), peakedRow(vocab: 16, peak: 9)],
            configurations: [.greedy, .greedy],
            histories: [noHistory, noHistory],
            steps: [nil, nil])
        #expect(base[0] == 1)
        #expect(swapped[0] == 4)
        #expect(base[1] == swapped[1])  // row 1 unchanged (== 9)
    }
}
