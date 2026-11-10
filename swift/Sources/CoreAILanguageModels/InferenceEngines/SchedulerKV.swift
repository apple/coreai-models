// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - The KV a batch scheduler decodes in

/// The KV under a ``BatchScheduler``: rows a request's prompt is written into a piece at a time, and
/// one decode step over the live rows.
///
/// The scheduler's pump calls every member but ``estimatedBytes`` and the chunk sizes, one at a time.
protocol SchedulerKV: AnyObject, Sendable {
    /// Rows, and so the most requests decoded at once.
    var rowCount: Int { get }

    /// Bytes the KV and the logits scratch occupy, estimated from their shapes.
    var estimatedBytes: (states: Int, logitsScratch: Int) { get }

    /// Largest piece of a prompt one ``prefill(row:tokens:)`` call writes while other rows are
    /// decoding. A prompt is written a piece at a time, taking turns with decode steps, so a long
    /// prompt does not stall every other row while it computes.
    var prefillChunk: Int { get }

    /// Largest piece while no row is decoding, at least ``prefillChunk``: there is no row to keep
    /// moving, only the next request to admit.
    var idlePrefillChunk: Int { get }

    /// Takes `row` for a request with `prompt`, computing nothing. Everything but the prompt's last
    /// token is then written with ``prefill(row:tokens:)``; the last is the row's first decode step,
    /// whose logits are the first sample.
    func reserve(row: Int, prompt: [Int32]) throws

    /// Writes `tokens`, the next piece of `row`'s prompt, in one call that samples nothing.
    func prefill(row: Int, tokens: ArraySlice<Int32>) async throws

    /// One decode step: each of `rows` writes its token at its cursor. One step result per entry,
    /// in order.
    func decode(_ rows: [SchedulerKVRow]) async throws -> [SchedulerKVStep]

    /// Gives back whatever `row` held, once its request has ended.
    func retire(row: Int)
}

/// One live row in a decode step: the token it writes, and where.
struct SchedulerKVRow {
    let row: Int
    let token: Int32
    let cursor: Int
}

/// What a decode step produced for one row.
enum SchedulerKVStep {
    /// The logits after the row's token.
    case logits([LogitsScalarType])
    /// The row holds no more positions, so its request ends as if at `maxTokens`.
    case rowFull
}

// MARK: - A contiguous batch session

/// A contiguous batch session as a ``BatchScheduler``'s KV, driven through ``BatchedDecodeStep``.
///
/// A prompt is written a piece at a time into a batch-1 scratch session of its own, made on its first
/// piece, and copied into its row once complete, before the row's first decode step. It cannot go
/// into the row as it is written, since every decode step writes every row.
///
/// A decode step is one ``BatchedDecodeStep/decodeBatch(tokensPerRow:startPositions:session:)`` over
/// every row, each at its own cursor, a free or still-prefilling row with token 0 at cursor 0, its
/// logits dropped. A row whose token would land past `maxPositions` reports
/// ``SchedulerKVStep/rowFull``.
final class ContiguousSchedulerKV: SchedulerKV, @unchecked Sendable {
    let step: any BatchedDecodeStep
    let session: GenerationSessionState
    let rowCount: Int
    let prefillChunk: Int
    let idlePrefillChunk: Int
    /// Positions one row holds, prompt and generation together.
    let maxPositions: Int

    /// A row whose prompt is being written: how many tokens go into the row, how many have, and the
    /// scratch session they are written into, made on the first piece.
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

    /// Everything but the prompt's last token goes into the row; a one-token prompt writes nothing
    /// before its first decode step.
    func reserve(row: Int, prompt: [Int32]) throws {
        guard !prompt.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("a request needs at least one prompt token")
        }
        // The last prompt token is decoded at cursor `prompt.count - 1`, which has to be a position.
        guard prompt.count <= maxPositions else {
            throw InferenceRuntimeError.contextLengthExceeded(prompt.count, maxPositions)
        }
        prefilling[row] = prompt.count > 1 ? Prefilling(target: prompt.count - 1) : nil
    }

    /// Writes `tokens` into `row`'s scratch session, and copies it into the row once the prompt is in.
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
