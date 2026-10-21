// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Samples one token per row, each row with its own ``SamplingConfiguration`` and step.
///
/// Rows are independent, so one batch can mix greedy and seeded temperature rows. A seeded row's
/// draw depends only on its own `(seed, step)`. Each configuration is normalized before sampling.
struct BatchedSampler {
    /// Samples the next token for every row.
    ///
    /// - Parameters:
    ///   - rows: Logits for each row, each `[vocab]`, such as the rows of a `[N, vocab]` output.
    ///   - configurations: Sampling configuration for each row.
    ///   - histories: Recent tokens for each row's repetition penalty. Pass an empty slice when unused.
    ///   - steps: Generation step for each row. Required for rows whose configuration sets `seed`,
    ///     ignored otherwise.
    /// - Precondition: All four arrays have the same count.
    /// - Returns: One token id per row, in row order.
    static func sample(
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
