// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// One request's reply from ``BatchScheduler/generateBatch(_:)``, and when it came.
package struct BatchGenerationResult: Sendable {
    /// The reply, without the end-of-sequence token that ended it.
    package let tokens: [Int32]
    package let stopReason: StopReason
    package let promptTokenCount: Int
    /// Submission to first token, queueing included.
    package let timeToFirstToken: Double
    /// First token of the reply to its last, what the request's own rate divides by.
    package let decodeSeconds: Double

    package init(
        tokens: [Int32], stopReason: StopReason, promptTokenCount: Int, timeToFirstToken: Double,
        decodeSeconds: Double
    ) {
        self.tokens = tokens
        self.stopReason = stopReason
        self.promptTokenCount = promptTokenCount
        self.timeToFirstToken = timeToFirstToken
        self.decodeSeconds = decodeSeconds
    }

    /// `row` as a reply, timed from `started`. The end-of-sequence token that ended it is not part
    /// of the reply, so it leaves both the tokens and the span. An EOS-only reply has no tokens, and
    /// its EOS still marks its time to first token.
    init(_ row: RowResult, promptTokenCount: Int, started: SuspendingClock.Instant) {
        var tokens = row.tokens
        var times = row.tokenTimes
        if row.stopReason == .eos, !tokens.isEmpty {
            tokens.removeLast()
            times.removeLast()
        }
        let first = times.first ?? row.tokenTimes.first ?? started
        let last = times.last ?? first
        self.init(
            tokens: tokens, stopReason: row.stopReason, promptTokenCount: promptTokenCount,
            timeToFirstToken: (first - started).inSeconds, decodeSeconds: (last - first).inSeconds)
    }

    /// Decode rate for this request alone, or nil for a reply of fewer than two tokens, which has no
    /// interval to divide by.
    package var tokensPerSecond: Double? {
        guard tokens.count > 1, decodeSeconds > 0 else { return nil }
        return Double(tokens.count - 1) / decodeSeconds
    }
}

/// Whole-batch numbers from ``BatchScheduler/generateBatch(_:)``, computed once for every tool that
/// reports them.
///
/// Rates across requests divide by the time the requests actually spanned, never sum each request's
/// own rate: with more requests than rows they run in turns, and a sum would count turns that never
/// overlapped.
package struct BatchGenerationMetrics: Sendable {
    package let requestCount: Int
    /// Wall clock for the whole batch, queueing included.
    package let wallSeconds: Double
    package let generatedTokens: Int
    /// Prompt tokens over prefill time for the first `rows` requests to start, the ones admitted
    /// before any request queued: their prompt tokens over the time until the last of them produced
    /// its first token.
    package let promptTokensPerSecond: Double
    /// Tokens decoded after each request's first, over the span from the earliest first token to the
    /// latest last one. With more requests than rows, that span also covers the later requests'
    /// prefills.
    package let decodeTokensPerSecond: Double
    /// Each request's own decode rate, averaged over the replies long enough to have one: what one
    /// caller sees. Zero when none is.
    package let meanRequestTokensPerSecond: Double
    package let meanTimeToFirstToken: Double

    /// Every generated token over the batch's wall clock, prefill and queueing in.
    package var endToEndTokensPerSecond: Double {
        wallSeconds > 0 ? Double(generatedTokens) / wallSeconds : 0
    }

    /// - Parameter rows: Requests the scheduler decodes at once, which bounds the first wave.
    package init(results: [BatchGenerationResult], wallSeconds: Double, rows: Int) {
        self.requestCount = results.count
        self.wallSeconds = wallSeconds
        self.generatedTokens = results.reduce(0) { $0 + $1.tokens.count }
        let firstWave = results.sorted { $0.timeToFirstToken < $1.timeToFirstToken }.prefix(Swift.max(1, rows))
        let firstWaveEnd = firstWave.map(\.timeToFirstToken).max() ?? 0
        let firstWavePrompt = firstWave.reduce(0) { $0 + $1.promptTokenCount }
        self.promptTokensPerSecond = firstWaveEnd > 0 ? Double(firstWavePrompt) / firstWaveEnd : 0
        let decodeStart = results.map(\.timeToFirstToken).min() ?? 0
        let decodeEnd = results.map { $0.timeToFirstToken + $0.decodeSeconds }.max() ?? 0
        let decoded = results.reduce(0) { $0 + Swift.max(0, $1.tokens.count - 1) }
        self.decodeTokensPerSecond = decodeEnd > decodeStart ? Double(decoded) / (decodeEnd - decodeStart) : 0
        let rates = results.compactMap(\.tokensPerSecond)
        self.meanRequestTokensPerSecond = rates.isEmpty ? 0 : rates.reduce(0, +) / Double(rates.count)
        self.meanTimeToFirstToken =
            results.isEmpty ? 0 : results.map(\.timeToFirstToken).reduce(0, +) / Double(results.count)
    }
}

extension BatchScheduler {
    /// Generates a fixed set of requests at once, through the same loop that serves requests arriving
    /// over time, and times each one.
    ///
    /// The requests are queued in one step (see ``run(_:)``): all or none, admitted in request order,
    /// up to `maxConcurrent` at once with the rest waiting their turn.
    ///
    /// - Returns: One result per request in request order, and whole-batch numbers.
    package func generateBatch(
        _ requests: [BatchRequest]
    ) async throws -> (results: [BatchGenerationResult], metrics: BatchGenerationMetrics) {
        guard !requests.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("generateBatch called with no requests")
        }
        let started = SuspendingClock.now
        let rows = try await run(requests)
        let results = zip(rows, requests).map { row, request in
            BatchGenerationResult(row, promptTokenCount: request.promptTokens.count, started: started)
        }
        return (
            results,
            BatchGenerationMetrics(
                results: results, wallSeconds: (SuspendingClock.now - started).inSeconds, rows: maxConcurrent)
        )
    }
}
