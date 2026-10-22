// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// An `InferenceEngine` that runs one request on a `CoreAISequentialEngine` with its own
/// `GenerationSessionState`. Used for `.fallbackBatchOne` lane requests, such as guided generation.
///
/// Create one per request and drive it from one task at a time. The KV cache, token history and
/// `lastPrefixHitCount` belong to this engine's session, separate from `base`'s. When `base` also
/// backs a `BatchScheduler`, hold the scheduler's `ExclusiveEngineLease` while the request runs.
package final class IsolatedSessionEngine: InferenceEngine, @unchecked Sendable {
    package typealias ConfigType = ModelConfig

    private let base: CoreAISequentialEngine
    /// Session for the request, created on first use. `reset(to:)` replaces it.
    private var sessionState: GenerationSessionState?

    package init(base: CoreAISequentialEngine) {
        self.base = base
    }

    private func ensureSession() throws -> GenerationSessionState {
        if let sessionState { return sessionState }
        let fresh = try base.makeSessionState()
        sessionState = fresh
        return fresh
    }

    package var config: ModelConfig { base.config }

    package func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> CoreAISequentialEngine.GenerationSequence {
        try await base.generate(
            with: input,
            sessionState: try ensureSession(),
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions)
    }

    /// Replaces this request's session with a fresh one. Ignores `tokenIndex`.
    package func reset(to tokenIndex: Int) async throws {
        sessionState = try base.makeSessionState()
    }

    package var supportsLogits: Bool { base.supportsLogits }

    /// True while a generation on this request's session is in flight.
    package var isBusy: Bool { sessionState?.tokenBox.isBusy ?? false }

    package var lastPrefixHitCount: Int { sessionState?.lastPrefixHitCount ?? 0 }

    package func cancel() async throws {
        sessionState?.tokenBox.cancelActive()
    }

    package var maxContextLength: Int { base.config.maxContextLength }
}
