// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

// MARK: - Generation Session State

/// Single-owner container for the per-request mutable state of an idempotent engine.
///
/// Holds everything a `CoreAISequentialEngine` generation mutates on a per-request
/// basis — the KV cache, optional persistent (hybrid) states, the processed-token
/// cursor, the implicit prefix-cache history, the in-flight generation token, and the
/// last prefix-hit count. Relocating this state off the engine lets one engine serve
/// independent sessions: a fresh session per conversation, or a reused one for prefix
/// reuse. Each session carries its own `GenerationTokenBox`, so a generation on one
/// session never cancels another's, and `generate(with:sessionState:…)` is idempotent
/// with respect to the engine.
///
/// Not thread-safe — a session is owned by exactly one active generation at a time.
/// Its `GenerationTokenBox` enforces single-active-generation *for that session*, so only
/// one task ever mutates a given session; concurrent access to one session is a
/// programming error, but distinct sessions may be driven independently.
///
/// It is deliberately not `Sendable`: it holds non-`Sendable`, NDArray-backed
/// state (`SyncStateHandler` / `FixedNDArrayState`) and is confined to the engine's
/// single isolation domain, so it never crosses a concurrency boundary (region
/// isolation covers the `generate()` async call). If a future scheduler ever drives
/// sessions from another isolation domain, revisit its concurrency story then rather
/// than reaching for `@unchecked` now.
///
/// It is a reference type on purpose: `processedTokenCount` and `history` are mutated
/// by the generation `Iterator` and observed by the engine's `generate()` shim across
/// calls. A value type would give the iterator its own copy and break prefix reuse.
package final class GenerationSessionState {
    /// KV cache — grows dynamically as tokens are processed.
    let kvCache: any SyncStateHandler

    /// Optional persistent fixed-shape states for hybrid models.
    let additionalStates: FixedNDArrayState?

    /// Whether the session carries non-truncatable (recurrent) state.
    /// Fixed by the model/handlers at creation time — never varies per request.
    let hasNonTruncatableStates: Bool

    /// Tokens processed so far in this session (incremental-inference cursor).
    var processedTokenCount: Int = 0

    /// Token history backing implicit prefix caching.
    var history = TokenHistory()

    /// Thread-safe holder for this session's in-flight `GenerationToken`. Per-session so
    /// independent sessions (a concurrent scheduler, or a batch=1 fallback lane running beside
    /// the batched lane) do not cancel or release each other's generations. A new generation on
    /// this session supersedes only this session's prior one.
    let tokenBox = GenerationTokenBox()

    /// Prefix-cache hit count of this session's most recent generation. Per-session so each
    /// session reports its own reuse; the engine mirrors the last-driven session's value for the
    /// single-valued `InferenceEngine.lastPrefixHitCount` protocol getter.
    var lastPrefixHitCount: Int = 0

    /// Per-session single-path (fallback-lane) scratch, relocated off the engine so independent
    /// sessions never share it (the batched lane does not use these — it builds inputs + output
    /// locally in `decodeBatch`). Both are lazily populated by the engine's single-path.
    ///
    /// Reusable logits output buffer + the batch size it was sized for, reused across this session's
    /// decode steps (perf) without crossing sessions.
    var logitsScratch: NDArray?
    var logitsScratchBatchSize: Int = 0

    /// This session's own input handler (owns the reusable input-id buffer). Freshly built per
    /// session by `makeSessionState`, so concurrent sessions never share the handler's scratch.
    var inputHandler: (any SyncInputHandler)?

    init(
        kvCache: any SyncStateHandler,
        additionalStates: FixedNDArrayState?,
        hasNonTruncatableStates: Bool
    ) {
        self.kvCache = kvCache
        self.additionalStates = additionalStates
        self.hasNonTruncatableStates = hasNonTruncatableStates
    }
}
