// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - Generation Session State

/// Single-owner container for the per-request mutable state of an idempotent engine.
///
/// Holds everything a `CoreAISequentialEngine` generation mutates on a per-request
/// basis — the KV cache, optional persistent (hybrid) states, the processed-token
/// cursor, and the implicit prefix-cache history. Relocating this state off the
/// engine lets one engine serve independent sessions sequentially: a fresh session
/// per conversation, or a reused one for prefix reuse. The engine's single
/// `GenerationTokenBox` serializes generation, so only one session is driven at a
/// time; each carries its own cache and history, so `generate(with:sessionState:…)`
/// is idempotent with respect to the engine.
///
/// Not thread-safe — a session is owned by exactly one active generation at a time.
/// The engine's `GenerationTokenBox` enforces single-active-generation, so only one
/// task ever mutates a given session; concurrent access is a programming error.
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
