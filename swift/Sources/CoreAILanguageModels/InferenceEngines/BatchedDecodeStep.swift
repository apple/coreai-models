// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - Batched Decode Step

/// Batched forward passes over the `N` rows of a batch KV cache: writing a row's prompt, decoding all
/// rows, and reading logits at every position. Sampling, stop handling and admission order stay with
/// the caller.
///
/// The caller tracks each row's cursor. `decodeBatch` and `commitRow` don't advance
/// `session.processedTokenCount`; `prefillPiece` advances its scratch session's.
///
/// ```swift
/// let session = try engine.makeSessionState(batchSize: 2)
/// // Row 0's prompt, all but its last token, through a scratch session, then into the row.
/// let scratch = try engine.makeSessionState(batchSize: 1)
/// try await engine.prefillPiece(promptA.dropLast(), scratch: scratch)
/// try engine.commitRow(0, from: scratch, count: promptA.count - 1, session: session)
/// // Row 0's last prompt token is its first decode step, at its own cursor; row 1 is free.
/// let logits = try await engine.decodeBatch(
///     tokensPerRow: [[promptA.last!], [0]],
///     startPositions: [promptA.count - 1, 0],
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

    /// Writes `tokens` into `scratch`, a batch-1 session from ``makeSessionState(batchSize:)``, at its
    /// next positions, reading no logits.
    func prefillPiece(_ tokens: ArraySlice<Int32>, scratch: GenerationSessionState) async throws

    /// Copies positions `[0, count)` of `scratch` into row `row` of `session`'s KV cache. The other
    /// rows are left untouched, so a request can join a running batch.
    func commitRow(
        _ row: Int, from scratch: GenerationSessionState, count: Int, session: GenerationSessionState
    ) throws

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
