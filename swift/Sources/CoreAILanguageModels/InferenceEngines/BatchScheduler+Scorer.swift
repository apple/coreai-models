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
        /// Positions the KV cache holds. A padded chunk never runs past it.
        let maxPositions: Int
        /// Maximum number of positions per scoring forward pass. Each pass returns at most
        /// `rows * chunkLength * vocab` logits.
        let chunkLength = 128
        /// Shortest padded chunk. The last chunk of a batch pads to 16, 32, 64 or 128 positions, so
        /// scoring reuses a few input shapes instead of compiling one per prompt length.
        static let shortestChunk = 16

        /// Positions in the chunk at `start` for a batch whose longest prompt has `longest` tokens: a
        /// full `chunkLength`, or the remaining positions rounded up to a power of two (at least
        /// `shortestChunk`), unless that would run past `maxPositions`. With `chunkLength` 128 and
        /// `longest` 300, the chunks at 0, 128 and 256 take 128, 128 and 64 positions.
        static func paddedChunk(start: Int, longest: Int, chunkLength: Int, maxPositions: Int) -> Int {
            let remaining = min(chunkLength, longest - start)
            var padded = shortestChunk
            while padded < remaining { padded *= 2 }
            let fitting = min(padded, chunkLength, maxPositions - start)
            return max(remaining, fitting)
        }

        /// Rows in a scoring pass over `requests` requests: the graph's fixed `capacity` for a static
        /// graph, otherwise `requests` rounded up to a power of two, at most `maxConcurrent`.
        static func paddedRows(requests: Int, capacity: Int, maxConcurrent: Int) -> Int {
            guard capacity == Int.max else { return capacity }
            var rows = 1
            while rows < requests { rows *= 2 }
            return max(requests, min(rows, maxConcurrent))
        }

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
        /// Short rows repeat their last token, and the last chunk pads to `paddedChunk`. For prompts of
        /// lengths 3 and 5 with a chunk length of 4 and a shortest chunk of 2, the rows are
        /// ```
        /// chunk 0: [a0 a1 a2 a2] [b0 b1 b2 b3]    positions 0..<4
        /// chunk 1: [a2 a2]       [b4 b4]          positions 4..<6, only b4 is read
        /// ```
        func scoreBatch(_ requests: [ScoreRequest]) async throws -> [LogProbabilities] {
            let rows = Self.paddedRows(
                requests: requests.count, capacity: decodeStep.maxBatchCapacity, maxConcurrent: maxConcurrent)
            let session = try decodeStep.makeSessionState(batchSize: rows)
            let longest = requests.map(\.tokens.count).max() ?? 0
            var entries = Array(repeating: [LogProbabilities.Entry](), count: requests.count)

            var start = 0
            while start < longest {
                let q = Self.paddedChunk(
                    start: start, longest: longest, chunkLength: chunkLength, maxPositions: maxPositions)
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
