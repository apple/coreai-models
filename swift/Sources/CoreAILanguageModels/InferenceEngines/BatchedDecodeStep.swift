// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - Batched Decode Step

/// Batched forward passes over the `N` rows of a batch KV cache: admitting a row, decoding all rows,
/// and reading logits at every position. Sampling, stop handling and admission order stay with the
/// caller.
///
/// The caller tracks each row's cursor. Neither method advances `session.processedTokenCount`.
///
/// ```swift
/// let session = try engine.makeSessionState(batchSize: 2)
/// let firstA = try await engine.prefillRow(row: 0, promptTokens: promptA, session: session)
/// let firstB = try await engine.prefillRow(row: 1, promptTokens: promptB, session: session)
/// // Sample nextA and nextB, then decode with each row at its own cursor.
/// let logits = try await engine.decodeBatch(
///     tokensPerRow: [[nextA], [nextB]],
///     startPositions: [promptA.count, promptB.count],
///     session: session)
/// ```
protocol BatchedDecodeStep {
    /// Most rows one step can take: `N` for a static `batch=N` graph, 1 for a single-batch graph, and
    /// `Int.max` for a dynamic-batch graph, whose traced maximum is only enforced at run time.
    var maxBatchCapacity: Int { get }

    /// Runs `tokensPerRow` (`[N][q]`) and returns each row's last-token logits as `[N][vocab]`.
    ///
    /// `startPositions[i]` is row `i`'s cache cursor. A decode step (`q == 1`) may give each row its
    /// own cursor, which needs an `attn_mask` input and a `RegionCopyable` KV cache. Every row then
    /// writes at slot `max(startPositions)`, and shorter rows are copied down to their own cursor
    /// afterwards. A prefill step (`q > 1`) needs the same cursor for every row.
    func decodeBatch(
        tokensPerRow: [[Int32]],
        startPositions: [Int],
        session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]]

    /// Prefills `promptTokens` into row `row` of `session`'s KV cache, at positions
    /// `[0, promptTokens.count)`, and returns the last prompt token's logits as `[vocab]`. The other
    /// rows are left untouched, so a request can join a running batch.
    func prefillRow(
        row: Int,
        promptTokens: [Int32],
        session: GenerationSessionState
    ) async throws -> [LogitsScalarType]

    /// Runs `tokensPerRow` (`[N][q]`, every row at `startPosition`) and returns the logits at every
    /// position, flat row-major `[N, q, vocab]`.
    func allPositionLogits(
        tokensPerRow: [[Int32]],
        startPosition: Int,
        session: GenerationSessionState
    ) async throws -> [LogitsScalarType]

    /// Creates a session whose KV cache holds `batchSize` rows, at most `maxBatchCapacity`.
    func makeSessionState(batchSize: Int) throws -> GenerationSessionState
}
