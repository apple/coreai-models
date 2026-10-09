// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - Idempotent Engine

/// An inference engine that keeps a request's KV cache and history in a
/// caller-owned `GenerationSessionState` instead of on the engine, so one engine
/// can serve independent sessions. Detected with `engine is any IdempotentEngine`
/// (like `ConstrainedGenerationCapable`).
///
/// Example:
///
///     let state = try engine.makeSessionState()
///     let output = try await engine.generate(
///         with: promptTokens, sessionState: state,
///         samplingConfiguration: sampling, inferenceOptions: options)
///     // Continue the same conversation on `state` to reuse its prefix:
///     let next = try await engine.generate(
///         with: followUpTokens, sessionState: state,
///         samplingConfiguration: sampling, inferenceOptions: options)
///     // Or call `makeSessionState()` again for an independent conversation.
///
/// Caveats:
/// - A session is single-owner: only one active generation may drive a given state
///   at a time (enforced by the session's own `GenerationTokenBox`; see `GenerationSessionState`).
/// - Per-request serialization state is per-session: each `GenerationSessionState` carries its
///   own `GenerationTokenBox` and `lastPrefixHitCount`, so distinct sessions may be driven
///   independently without cancelling or clobbering each other. The engine's
///   `lastPrefixHitCount` getter mirrors the most-recently-driven session's value for the
///   single-valued `InferenceEngine` protocol; a concurrent scheduler reads the per-session value.
package protocol IdempotentEngine: InferenceEngine {
    /// Mint a fresh, independent per-request state (fresh KV cache + empty history).
    func makeSessionState() throws -> GenerationSessionState

    /// Generate over the given session: resolve any reusable prefix from the
    /// session's `history`, run the model, and advance the session's cursor and
    /// history as generation proceeds. Returns the output sequence, which yields
    /// each generated token and, when `inferenceOptions` requests it, its logits.
    func generate(
        with input: [TokenId],
        sessionState: GenerationSessionState,
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> OutputSequence
}
