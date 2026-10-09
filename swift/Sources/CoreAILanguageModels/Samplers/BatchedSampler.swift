// Batched per-row CPU sampler for N>1 decoding.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Samples one token per row from a batch of per-row logit vectors.
///
/// Each row carries its own ``SamplingConfiguration`` and generation step, so a single batch can
/// mix a greedy sequence with a temperature-sampled one — the configuration decides argmax vs.
/// temperature/top-k/top-p/min-p per row, and the `(seed, step)` pair makes seeded rows
/// reproducible and independent of one another.
///
/// CPU, per-row, and substrate-free: it consumes already-sliced `[vocab]` rows (e.g. the per-row
/// slices of a `[B, vocab]` logits output), so it is independent of how those logits were produced
/// (dense GPU graph, paged engine, …). Greedy-only batches may instead be sampled on-device by a
/// specialized argmax path; anything heterogeneous routes here.
public struct BatchedSampler {
    /// Samples the next token for every row.
    ///
    /// - Parameters:
    ///   - rows: Per-row logit vectors, each `vocab` long, in row order.
    ///   - configurations: Per-row sampling configuration (one per row).
    ///   - histories: Per-row recent-token history for repetition penalty (empty slice if unused).
    ///   - steps: Per-row generation step index, required for a row whose configuration sets a
    ///     `seed` (each step derives its own generator from `(seed, step)`); pass `nil` otherwise.
    /// - Returns: One sampled token id per row, in row order.
    public static func sample(
        rows: [[LogitsScalarType]],
        configurations: [SamplingConfiguration],
        histories: [ArraySlice<Int32>],
        steps: [Int?]
    ) -> [Int32] {
        precondition(
            rows.count == configurations.count
                && rows.count == histories.count
                && rows.count == steps.count,
            "BatchedSampler: rows (\(rows.count)), configurations (\(configurations.count)), "
                + "histories (\(histories.count)) and steps (\(steps.count)) must have equal counts")

        var tokens = [Int32]()
        tokens.reserveCapacity(rows.count)
        for row in rows.indices {
            var logits = rows[row]
            tokens.append(
                configurations[row].normalized().fallbackSampler(
                    from: &logits, tokenHistory: histories[row], step: steps[row]))
        }
        return tokens
    }
}
