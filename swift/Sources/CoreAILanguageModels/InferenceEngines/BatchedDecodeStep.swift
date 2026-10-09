// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - Batched Decode Step

/// A capability for driving one batched forward step: advance `N` rows together through a
/// batch-capable graph and return each row's last-token logits. Sampling, EOS/stop handling, and
/// admission live above this seam (in the scheduler), so the step itself is a pure forward.
///
/// The decode step (`q == 1`) is **per-row**: each row may sit at its own cache cursor (ragged /
/// continuous batching via a shared write at `max(cursor)` + a runner blit). Batched
/// *prefill* of a newly admitted row happens through `prefillRow`, which writes that one row's prompt
/// into its own slab.
package protocol BatchedDecodeStep {
    /// Upper bound on concurrent rows the loaded graph can run: `N` for a static `batch=N` graph,
    /// `Int.max` for a dynamic-batch graph, `1` for a pinned single-batch graph. The scheduler sizes
    /// a cohort against this. (The concrete engine also exposes `declaredBatchSize` /
    /// `supportsBatching` for the server's auto-detection — those are read off the concrete type, not
    /// this seam.)
    var maxBatchCapacity: Int { get }

    /// One batched forward: advance `tokensPerRow` (N rows × q columns) at the per-row cache
    /// `startPositions` against `session`'s batch=N KV cache, returning each row's last-token logits
    /// (`[N][vocab]`). Decode (`q == 1`) allows ragged `startPositions`; batched prefill (`q > 1`)
    /// is equal-length.
    func decodeBatch(
        tokensPerRow: [[Int32]],
        startPositions: [Int],
        session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]]

    /// Prefill one row's prompt into batch slab `row` of `session`'s KV cache (cursor 0 → prompt
    /// length), returning that row's last-prompt-token logits (`[vocab]`). Used by the continuous
    /// scheduler to admit a request mid-flight without disturbing the other rows' slabs.
    func prefillRow(
        row: Int,
        promptTokens: [Int32],
        session: GenerationSessionState
    ) async throws -> [LogitsScalarType]
}
