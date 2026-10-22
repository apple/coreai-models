// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Dispatch
import Synchronization

extension BatchScheduler {
    /// Scores `ScoreRequest`s in padded batches on `decodeStep`. Each batch runs on a new session.
    struct Scorer {
        /// Decode step the scoring passes run on.
        let decodeStep: any BatchedDecodeStep
        /// Maximum number of requests per scoring batch.
        let maxConcurrent: Int
        /// Maximum number of positions per scoring forward pass. Each pass returns at most
        /// `rows * chunkLength * vocab` logits.
        let chunkLength = 128

        /// Counts the leading entries of `lengths`, prompt lengths in queue order, that form the next
        /// scoring batch. Takes up to `limit` entries and stops before one that would make more than a
        /// third of the batch's positions padding. Returns 0 when `lengths` is empty. With `limit` 4
        /// and lengths `[7, 40, 130, 300]`, successive batches are `[7]`, `[40]` and `[130, 300]`.
        static func batchCount(_ lengths: [Int], limit: Int) -> Int {
            guard let first = lengths.first else { return 0 }
            var count = 1
            var total = first
            var longest = first
            while count < min(limit, lengths.count) {
                let next = lengths[count]
                let padded = (count + 1) * max(longest, next)
                if 2 * padded > 3 * (total + next) { break }
                count += 1
                total += next
                longest = max(longest, next)
            }
            return count
        }

        /// Scores `requests` in one padded batch on a new session. Returns one `LogProbabilities` per
        /// request, in request order. At most `decodeStep.maxBatchCapacity` requests.
        ///
        /// Short rows repeat their last token. For prompts of lengths 3 and 5 with a chunk length of 4,
        /// the rows are
        /// ```
        /// chunk 0: [a0 a1 a2 a2] [b0 b1 b2 b3]    positions 0..<4
        /// chunk 1: [a2]          [b4]             position 4, row a is padding only
        /// ```
        func scoreBatch(_ requests: [ScoreRequest]) async throws -> [LogProbabilities] {
            // Static graphs take exactly `capacity` rows; dynamic graphs (`Int.max`) take one per request.
            let capacity = decodeStep.maxBatchCapacity
            let rows = capacity == Int.max ? requests.count : capacity
            let session = try decodeStep.makeSessionState(batchSize: rows)
            let longest = requests.map(\.tokens.count).max() ?? 0
            var entries = Array(repeating: [LogProbabilities.Entry](), count: requests.count)

            var start = 0
            while start < longest {
                let q = min(chunkLength, longest - start)
                let tokensPerRow = (0..<rows).map { row -> [Int32] in
                    guard row < requests.count else { return Array(repeating: 0, count: q) }
                    let tokens = requests[row].tokens
                    return (start..<start + q).map { tokens[min($0, tokens.count - 1)] }
                }
                let logits = try await decodeStep.allPositionLogits(
                    tokensPerRow: tokensPerRow, startPosition: start, session: session)
                let vocab = logits.count / (rows * q)
                let chunkEntries = Mutex([[LogProbabilities.Entry]](repeating: [], count: requests.count))
                let chunkStart = start
                logits.withUnsafeBufferPointer { buffer in
                    DispatchQueue.concurrentPerform(iterations: requests.count) { row in
                        let request = requests[row]
                        let real = min(q, request.tokens.count - chunkStart)
                        guard real > 0 else { return }
                        let rowStart = row * q * vocab
                        let rowEntries = LogProbabilities.compute(
                            logits: UnsafeBufferPointer(rebasing: buffer[rowStart..<rowStart + real * vocab]),
                            vocabSize: vocab,
                            targets: request.targets[chunkStart..<chunkStart + real],
                            topK: request.topK)
                        chunkEntries.withLock { $0[row] = rowEntries }
                    }
                }
                for (row, rowEntries) in chunkEntries.withLock({ $0 }).enumerated() { entries[row] += rowEntries }
                start += q
            }
            return entries.map { LogProbabilities(entries: $0) }
        }
    }
}

/// A scoring request for `BatchScheduler.score(_:)`: prompt tokens and the target token to score at
/// each position.
///
/// For a prompt `[t0, t1, t2]` the usual targets are `[t1, t2, x]`. `x` can be any token, and the
/// caller ignores the last position's entry in the result.
package struct ScoreRequest: Sendable {
    /// Prompt tokens, cannot be empty.
    let tokens: [Int32]
    /// Token scored at each position of `tokens`, same length as `tokens`. `targets[p]` is scored
    /// against the logits at position `p`.
    let targets: [Int32]
    /// Number of top alternatives reported per position.
    let topK: Int

    package init(tokens: [Int32], targets: [Int32], topK: Int) {
        self.tokens = tokens
        self.targets = targets
        self.topK = topK
    }
}
