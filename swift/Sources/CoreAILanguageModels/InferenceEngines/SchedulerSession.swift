// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// An `InferenceEngine` that represents one request inside a `RequestSink` such as `BatchScheduler`.
///
/// Create one per request. Decoding strategies and the server see an ordinary engine while several
/// requests share one batched forward. `generate` submits a `BatchRequest` and returns the sink's
/// per-request event stream as a `GenerationSequence`.
///
/// Differences from a single-request engine:
/// - `reset(to:)` does nothing. The sink owns the shared KV cache, so there is no per-request
///   history to rewind.
/// - `supportsLogits` is false. The sink samples internally, so guided generation and
///   loglikelihood scoring must use the fallback lane (see `BatchServingRouter`).
/// - The sink stops a row only on `stopTokenIds` or `maxTokens`. Multi-token stop sequences and
///   detokenization stay in the per-request `DecodingStrategy`.
///
/// ```swift
/// let engine = SchedulerSession(sink: scheduler, config: modelConfig, stopTokenIds: eosIds)
/// let sequence = try await engine.generate(
///     with: promptTokens, samplingConfiguration: sampling, inferenceOptions: options)
/// for try await output in sequence { /* output.tokenId */ }
/// ```
package final class SchedulerSession: InferenceEngine, @unchecked Sendable {
    package typealias ConfigType = ModelConfig

    package let config: ModelConfig
    private let sink: any RequestSink
    /// Token ids that end the row inside the sink, usually EOS plus any additional EOS ids.
    private let stopTokenIds: Set<Int32>

    package init(sink: any RequestSink, config: ModelConfig, stopTokenIds: Set<Int32>) {
        self.sink = sink
        self.config = config
        self.stopTokenIds = stopTokenIds
    }

    package func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        // Cap `maxTokens` at the room left in the context window after the prompt. Decoding past
        // the window would make the shared batched forward throw, and a failed forward fails every
        // resident row, so each row must retire with `.maxTokens` before that point.
        let contextRoom = Swift.max(1, config.maxContextLength - input.count)
        let requested = inferenceOptions.maxTokens ?? contextRoom
        let stream = try sink.submit(
            BatchRequest(
                promptTokens: input.map { Int32($0) },
                sampling: samplingConfiguration,
                eosTokenIds: stopTokenIds,
                maxTokens: Swift.max(1, Swift.min(requested, contextRoom))))
        return GenerationSequence(events: stream)
    }

    /// Does nothing. The sink reclaims this request's state, and resetting the shared engine would
    /// destroy the other rows.
    package func reset(to tokenIndex: Int) async throws {}

    /// Always false because the sink samples internally and returns no per-step logits.
    package var supportsLogits: Bool { false }

    /// Always false. Each session is one request, and the sink runs several at once.
    package var isBusy: Bool { false }

    package var maxContextLength: Int { config.maxContextLength }
}

// MARK: - Output sequence

extension SchedulerSession {
    /// The sink's per-request event stream, presented as an engine output sequence.
    package struct GenerationSequence: InferenceOutputSequence {
        package typealias Element = InferenceOutput
        package typealias Failure = Error

        let events: AsyncThrowingStream<BatchSinkEvent, Error>
        let stopReasonStore = StopReasonStore()

        package var stopReason: StopReason? { stopReasonStore.stopReason }

        package func setStopReason(_ reason: StopReason) { stopReasonStore.set(reason) }

        package func makeAsyncIterator() -> Iterator {
            Iterator(upstream: events.makeAsyncIterator(), stopReasonStore: stopReasonStore)
        }
    }
}

extension SchedulerSession.GenerationSequence {
    package final class Iterator: AsyncIteratorProtocol {
        package typealias Element = InferenceOutput
        package typealias Failure = Error

        private var upstream: AsyncThrowingStream<BatchSinkEvent, Error>.AsyncIterator
        private let stopReasonStore: StopReasonStore

        init(
            upstream: AsyncThrowingStream<BatchSinkEvent, Error>.AsyncIterator,
            stopReasonStore: StopReasonStore
        ) {
            self.upstream = upstream
            self.stopReasonStore = stopReasonStore
        }

        package func next() async throws -> InferenceOutput? {
            while let event = try await upstream.next() {
                switch event {
                case .token(let id, let logits):
                    return InferenceOutput(tokenId: id, logits: logits)
                case .finished(let reason):
                    // Record the sink's reason unless the consumer already set one. The
                    // `.maxTokens` default below only covers a stream that ends without `.finished`.
                    stopReasonStore.setIfUnset(reason)
                }
            }
            stopReasonStore.setIfUnset(.maxTokens)
            return nil
        }
    }
}
