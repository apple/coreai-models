// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

// MARK: - Generation Session State

/// Single-owner container for the per-request mutable state of an idempotent engine.
///
/// It holds the KV cache, any hybrid states, the processed-token cursor, the prefix-cache history,
/// the in-flight generation token, and the last prefix-hit count. Use a new session for each
/// conversation, or reuse one to keep its prefix.
///
/// Not thread-safe. Only one generation drives a session at a time: starting a generation cancels
/// the previous one on the same session through its own `GenerationTokenBox`. Other sessions are
/// not affected.
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

    /// This session's in-flight generation. A new generation cancels only this session's previous one.
    let tokenBox = GenerationTokenBox()

    /// Prefix-cache hit count of this session's last generation. The engine-level
    /// `lastPrefixHitCount` is copied from whichever session ran last.
    var lastPrefixHitCount: Int = 0

    /// Reusable logits output for the single-row path, reallocated when the step's token count
    /// changes. `decodeBatch` does not use it.
    var logitsScratch: NDArray?
    /// Token count `logitsScratch` was sized for.
    var logitsScratchBatchSize: Int = 0

    /// This session's own input handler, so sessions never share input buffers.
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

    /// Bytes held by this session's KV cache and persistent states at their current capacity.
    var estimatedStateBytes: Int { kvCache.estimatedBytes + (additionalStates?.estimatedBytes ?? 0) }

    /// Bytes held by this session's reusable logits buffer; 0 until it is first allocated.
    var estimatedLogitsScratchBytes: Int { logitsScratch.map(estimatedArrayBytes) ?? 0 }
}
