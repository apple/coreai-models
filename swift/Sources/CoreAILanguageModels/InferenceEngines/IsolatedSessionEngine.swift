// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// An `InferenceEngine` that runs one fallback-lane request, such as guided generation, on a
/// `CoreAISequentialEngine` through its own `GenerationSessionState`.
///
/// Create one per request. The KV cache, token history and prefix-hit count belong to this session
/// rather than the engine's default session, so other sessions keep their state. It does not make
/// concurrent use of the engine safe: next to a `BatchScheduler`, hold an `ExclusiveEngineLease`
/// while the request runs.
///
/// The class is `@unchecked Sendable` because a single request owns it, so no two tasks drive it at
/// once.
package final class IsolatedSessionEngine: InferenceEngine, @unchecked Sendable {
    package typealias ConfigType = ModelConfig

    private let base: CoreAISequentialEngine
    /// Created on first use so `init` cannot throw. `reset(to:)` replaces it with a fresh one.
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

    /// Replaces this request's session with a fresh one, whatever `tokenIndex` is. Guided generation
    /// resets to 0 before generating, so this lane never needs a partial rewind.
    package func reset(to tokenIndex: Int) async throws {
        sessionState = try base.makeSessionState()
    }

    package var supportsLogits: Bool { base.supportsLogits }

    /// This session's busy state. The base engine's flag would reflect other sessions too.
    package var isBusy: Bool { sessionState?.tokenBox.isBusy ?? false }

    package var lastPrefixHitCount: Int { sessionState?.lastPrefixHitCount ?? 0 }

    package func cancel() async throws {
        sessionState?.tokenBox.cancelActive()
    }

    package var maxContextLength: Int { base.config.maxContextLength }
}
