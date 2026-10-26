// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// An `InferenceEngine` that runs one request on a `RequestSink`, such as `BatchScheduler`.
///
/// Create one per request. The sink ends the request only on `stopTokenIds` or `maxTokens`. Match
/// multi-token stop sequences in the caller's `DecodingStrategy`.
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
    /// Token ids that end the request inside the sink.
    private let stopTokenIds: Set<Int32>

    package init(sink: any RequestSink, config: ModelConfig, stopTokenIds: Set<Int32>) {
        self.sink = sink
        self.config = config
        self.stopTokenIds = stopTokenIds
    }

    /// Submits `input` to the sink and returns the request's output sequence.
    ///
    /// The request's token limit is `inferenceOptions.maxTokens` capped at the context room left after
    /// the prompt, and defaults to that room. At least 1. Throws when the sink rejects the request,
    /// for example with `InferenceRuntimeError.queueFull`.
    package func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        let contextRoom = Swift.max(1, config.maxContextLength - input.count)
        let requested = inferenceOptions.maxTokens ?? contextRoom
        let request = BatchRequest(
            promptTokens: input.map { Int32($0) },
            sampling: samplingConfiguration,
            eosTokenIds: stopTokenIds,
            maxTokens: Swift.max(1, Swift.min(requested, contextRoom)))
        return GenerationSequence(events: try sink.submit(request), cancellation: request.cancellation)
    }

    /// Does nothing.
    package func reset(to tokenIndex: Int) async throws {}

    /// Always false.
    package var supportsLogits: Bool { false }

    /// Always false.
    package var isBusy: Bool { false }

    package var maxContextLength: Int { config.maxContextLength }
}

// MARK: - Output sequence

extension SchedulerSession {
    /// The sink's event stream for one request, as an `InferenceOutputSequence`.
    ///
    /// `stopReason` is the sink's `.finished` reason, or `.maxTokens` when the stream ends without
    /// one. `setStopReason(_:)` overrides it and cancels the request, so the stream ends within one
    /// decode step.
    package struct GenerationSequence: InferenceOutputSequence {
        package typealias Element = InferenceOutput
        package typealias Failure = Error

        let events: AsyncThrowingStream<BatchSinkEvent, Error>
        let cancellation: RequestCancellation
        let stopReasonStore = StopReasonStore()

        package var stopReason: StopReason? { stopReasonStore.stopReason }

        package func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
            cancellation.cancel()
        }

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
                    stopReasonStore.setIfUnset(reason)
                }
            }
            stopReasonStore.setIfUnset(.maxTokens)
            return nil
        }
    }
}
