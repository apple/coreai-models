// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - KV storage for a batch scheduler

/// The KV cache behind a ``BatchScheduler``: a set of rows, each prefilled with one request's prompt in
/// chunks, and a decode step that runs over the active rows.
///
/// Apart from ``estimatedBytes`` and the chunk sizes, members are only called from the scheduler's
/// pump, one call at a time.
protocol SchedulerKV: AnyObject, Sendable {
    /// Number of rows, which is the most requests that can decode at once.
    var rowCount: Int { get }

    /// Estimated bytes used by the KV cache and the logits scratch, based on their shapes.
    var estimatedBytes: (states: Int, logitsScratch: Int) { get }

    /// Maximum number of tokens one ``prefill(row:tokens:)`` call writes while other rows are decoding.
    /// Prompts are prefilled in chunks that alternate with decode steps, so a long prompt doesn't stall
    /// the other rows.
    var prefillChunk: Int { get }

    /// Maximum chunk size when no row is decoding, at least ``prefillChunk``. With no stream to keep
    /// moving, larger chunks get the new request started sooner.
    var idlePrefillChunk: Int { get }

    /// Reserves `row` for a request with `prompt`, without running anything. Everything but the last
    /// prompt token is then written with ``prefill(row:tokens:)``. The last token is processed by the
    /// row's first decode step, whose logits give the first sample.
    func reserve(row: Int, prompt: [Int32]) throws

    /// Writes `tokens`, the next chunk of `row`'s prompt. Nothing is sampled.
    func prefill(row: Int, tokens: ArraySlice<Int32>) async throws

    /// Runs one decode step, in which each entry in `rows` writes its token at its position. Returns one
    /// result per entry, in the same order.
    func decode(_ rows: [SchedulerKVRow]) async throws -> [SchedulerKVStep]

    /// Releases whatever `row` holds, once its request has finished.
    func retire(row: Int)
}

/// An active row in a decode step: the token it writes, and the position it writes it at.
struct SchedulerKVRow {
    let row: Int
    let token: Int32
    let cursor: Int
}

/// The result of a decode step for one row.
enum SchedulerKVStep {
    /// Logits after processing the row's token.
    case logits([LogitsScalarType])
    /// The row has no positions left, so its request finishes as if it had reached `maxTokens`.
    case rowFull
}

// MARK: - Contiguous batch session

/// A ``SchedulerKV`` backed by a contiguous batch session, driven through ``BatchedDecodeStep``.
///
/// Each prompt is prefilled in chunks into its own batch-1 scratch session, created on the first chunk,
/// and copied into its row once complete, before the row's first decode step. It can't be written into
/// the row directly, because every decode step writes every row.
///
/// A decode step is a single ``BatchedDecodeStep/decodeBatch(tokensPerRow:startPositions:session:)``
/// over all rows, each at its own position. Rows that are free or still prefilling get token 0 at
/// position 0, and their logits are discarded. A row whose next token would go past `maxPositions`
/// reports ``SchedulerKVStep/rowFull``.
final class ContiguousSchedulerKV: SchedulerKV, @unchecked Sendable {
    let step: any BatchedDecodeStep
    let session: GenerationSessionState
    let rowCount: Int
    let prefillChunk: Int
    let idlePrefillChunk: Int
    /// Number of positions in a row, covering both the prompt and the generated tokens.
    let maxPositions: Int

    /// A row that's still prefilling: how many tokens go into the row, how many have been written so
    /// far, and the scratch session they're written into, created on the first chunk.
    private struct Prefilling {
        let target: Int
        var written = 0
        var scratch: GenerationSessionState?
    }
    private var prefilling: [Int: Prefilling] = [:]

    init(
        step: any BatchedDecodeStep, session: GenerationSessionState, rows: Int, prefillChunk: Int,
        idlePrefillChunk: Int, maxPositions: Int = .max
    ) {
        precondition(prefillChunk >= 1, "prefillChunk must be >= 1")
        self.step = step
        self.session = session
        self.rowCount = rows
        self.prefillChunk = prefillChunk
        self.idlePrefillChunk = max(idlePrefillChunk, prefillChunk)
        self.maxPositions = maxPositions
    }

    var estimatedBytes: (states: Int, logitsScratch: Int) {
        (session.estimatedStateBytes, session.estimatedLogitsScratchBytes)
    }

    /// Everything but the last prompt token goes into the row. A one-token prompt needs no prefill
    /// before its first decode step.
    func reserve(row: Int, prompt: [Int32]) throws {
        guard !prompt.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("a request needs at least one prompt token")
        }
        // The last prompt token is decoded at position `prompt.count - 1`, which must fit in the row.
        guard prompt.count <= maxPositions else {
            throw InferenceRuntimeError.contextLengthExceeded(prompt.count, maxPositions)
        }
        prefilling[row] = prompt.count > 1 ? Prefilling(target: prompt.count - 1) : nil
    }

    /// Writes `tokens` into `row`'s scratch session, then copies it into the row once the whole prompt
    /// is in.
    func prefill(row: Int, tokens: ArraySlice<Int32>) async throws {
        guard var entry = prefilling[row] else {
            throw InferenceRuntimeError.invalidState("row \(row) prefilled with nothing reserved")
        }
        guard entry.written + tokens.count <= entry.target else {
            throw InferenceRuntimeError.invalidState(
                "row \(row) prefilled past its prompt: \(entry.written + tokens.count) of \(entry.target)")
        }
        let scratch = try entry.scratch ?? step.makeSessionState(batchSize: 1)
        entry.scratch = scratch
        try await step.prefillPiece(tokens, scratch: scratch)
        entry.written += tokens.count
        guard entry.written == entry.target else {
            prefilling[row] = entry
            return
        }
        try step.commitRow(row, from: scratch, count: entry.target, session: session)
        prefilling[row] = nil
    }

    func decode(_ rows: [SchedulerKVRow]) async throws -> [SchedulerKVStep] {
        var tokens = [[Int32]](repeating: [0], count: rowCount)
        var cursors = [Int](repeating: 0, count: rowCount)
        var carried: [Int] = []
        for (index, row) in rows.enumerated() {
            guard prefilling[row.row] == nil else {
                throw InferenceRuntimeError.invalidState("row \(row.row) decoded before its prompt was written")
            }
            guard row.cursor < maxPositions else { continue }
            tokens[row.row] = [row.token]
            cursors[row.row] = row.cursor
            carried.append(index)
        }
        var steps = [SchedulerKVStep](repeating: .rowFull, count: rows.count)
        guard !carried.isEmpty else { return steps }
        let logits = try await step.decodeBatch(tokensPerRow: tokens, startPositions: cursors, session: session)
        for index in carried { steps[index] = .logits(logits[rows[index].row]) }
        return steps
    }

    func retire(row: Int) {
        prefilling[row] = nil
    }
}
