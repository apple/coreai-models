// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// One request's reply from ``RequestSink/generateBatch(_:)``, and when it came.
package struct BatchGenerationResult: Sendable {
    /// The reply, without the end-of-sequence token that ended it.
    package let tokens: [Int32]
    package let stopReason: StopReason
    package let promptTokenCount: Int
    /// Submission to first token, queueing included.
    package let timeToFirstToken: Double
    /// First token to last, what the request's own rate divides by.
    package let decodeSeconds: Double

    /// Decode rate for this request alone; zero for a single token, which has no interval.
    package var tokensPerSecond: Double {
        guard tokens.count > 1, decodeSeconds > 0 else { return 0 }
        return Double(tokens.count - 1) / decodeSeconds
    }
}

/// Whole-batch numbers from ``RequestSink/generateBatch(_:)``, computed once for every tool that
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
    /// Every prompt token over the time until the last request's first token, when the last prefill
    /// finished.
    package let promptTokensPerSecond: Double
    /// Tokens decoded after each request's first, over the span from the earliest first token to the
    /// latest last one.
    package let decodeTokensPerSecond: Double
    /// Each request's own decode rate, averaged: what one caller sees.
    package let meanRequestTokensPerSecond: Double
    package let meanTimeToFirstToken: Double

    /// Every generated token over the batch's wall clock, prefill and queueing in.
    package var endToEndTokensPerSecond: Double {
        wallSeconds > 0 ? Double(generatedTokens) / wallSeconds : 0
    }

    package init(results: [BatchGenerationResult], wallSeconds: Double) {
        self.requestCount = results.count
        self.wallSeconds = wallSeconds
        self.generatedTokens = results.reduce(0) { $0 + $1.tokens.count }
        let lastFirstToken = results.map(\.timeToFirstToken).max() ?? 0
        let promptTotal = results.reduce(0) { $0 + $1.promptTokenCount }
        self.promptTokensPerSecond = lastFirstToken > 0 ? Double(promptTotal) / lastFirstToken : 0
        let decodeStart = results.map(\.timeToFirstToken).min() ?? 0
        let decodeEnd = results.map { $0.timeToFirstToken + $0.decodeSeconds }.max() ?? 0
        let decoded = results.reduce(0) { $0 + Swift.max(0, $1.tokens.count - 1) }
        self.decodeTokensPerSecond = decodeEnd > decodeStart ? Double(decoded) / (decodeEnd - decodeStart) : 0
        let rates = results.map(\.tokensPerSecond)
        self.meanRequestTokensPerSecond = rates.isEmpty ? 0 : rates.reduce(0, +) / Double(rates.count)
        self.meanTimeToFirstToken =
            results.isEmpty ? 0 : results.map(\.timeToFirstToken).reduce(0, +) / Double(results.count)
    }
}

extension RequestSink {
    /// Generates a fixed set of requests at once, through the same loop that serves requests arriving
    /// over time, and times each one.
    ///
    /// Requests are submitted in order, so admission order is request order; up to `maxConcurrent`
    /// run at once and the rest wait their turn.
    ///
    /// - Returns: One result per request in request order, and whole-batch numbers.
    package func generateBatch(
        _ requests: [BatchRequest]
    ) async throws -> (results: [BatchGenerationResult], metrics: BatchGenerationMetrics) {
        guard !requests.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("generateBatch called with no requests")
        }
        let started = SuspendingClock.now
        let streams = try requests.map { try submit($0) }
        let results = try await withThrowingTaskGroup(of: (Int, BatchGenerationResult).self) { group in
            for (index, stream) in streams.enumerated() {
                let promptTokens = requests[index].promptTokens.count
                group.addTask {
                    var tokens: [Int32] = []
                    var reason = StopReason.maxTokens
                    var first: SuspendingClock.Instant?
                    var last = started
                    for try await event in stream {
                        switch event {
                        case .token(let token, _):
                            tokens.append(token)
                            last = .now
                            if first == nil { first = last }
                        case .finished(let stopReason):
                            reason = stopReason
                        }
                    }
                    // The end-of-sequence token is delivered like any other; it is not part of the reply.
                    if reason == .eos, !tokens.isEmpty { tokens.removeLast() }
                    let firstAt = first ?? last
                    return (
                        index,
                        BatchGenerationResult(
                            tokens: tokens, stopReason: reason, promptTokenCount: promptTokens,
                            timeToFirstToken: (firstAt - started).inSeconds,
                            decodeSeconds: (last - firstAt).inSeconds)
                    )
                }
            }
            var results = [BatchGenerationResult?](repeating: nil, count: requests.count)
            for try await (index, result) in group { results[index] = result }
            return results.map { $0! }
        }
        return (
            results,
            BatchGenerationMetrics(results: results, wallSeconds: (SuspendingClock.now - started).inSeconds)
        )
    }
}
