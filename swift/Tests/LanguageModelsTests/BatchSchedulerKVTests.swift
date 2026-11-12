// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization
import Testing

@testable import CoreAILanguageModels

/// `BatchScheduler` over a ``SchedulerKV``: rows that fill, requests the KV refuses, prompts written in
/// pieces, and failures that give every row back.
@Suite("BatchScheduler over a SchedulerKV", .timeLimit(.minutes(1)))
struct BatchSchedulerKVTests {
    static func oneHot(_ token: Int32) -> [LogitsScalarType] {
        var row = [LogitsScalarType](repeating: 0, count: 64)
        row[Int(token)] = 1
        return row
    }

    static func request(_ prompt: [Int32], maxTokens: Int) -> BatchRequest {
        BatchRequest(promptTokens: prompt, sampling: .greedy, eosTokenIds: [], maxTokens: maxTokens)
    }

    /// Predicts the last token plus one, mod 64, like `BatchSchedulerTests`' count-up step, writing
    /// prompts `chunk` tokens at a time (`idleChunk` while no row decodes) and logging every call.
    final class CountUpKV: SchedulerKV, @unchecked Sendable {
        struct Boom: Error {}

        let rowCount: Int
        let prefillChunk: Int
        let idlePrefillChunk: Int
        /// Longest prompt `reserve` takes.
        let longestPrompt: Int
        /// A row decoding at a cursor at or past this is full.
        let rowLimit: Int
        /// A prefill whose piece starts with this token throws `Boom`.
        let failPrefillFor: Int32?
        /// A decode step at or past this many steps throws `Boom`.
        let failDecodeAfter: Int?
        private let state = Mutex((log: [String](), decodes: 0))

        init(
            rows: Int, chunk: Int = 8, idleChunk: Int? = nil, longestPrompt: Int = .max, rowLimit: Int = .max,
            failPrefillFor: Int32? = nil, failDecodeAfter: Int? = nil
        ) {
            self.rowCount = rows
            self.prefillChunk = chunk
            self.idlePrefillChunk = idleChunk ?? chunk
            self.longestPrompt = longestPrompt
            self.rowLimit = rowLimit
            self.failPrefillFor = failPrefillFor
            self.failDecodeAfter = failDecodeAfter
        }

        var estimatedBytes: (states: Int, logitsScratch: Int) { (0, 0) }
        var log: [String] { state.withLock { $0.log } }

        func reserve(row: Int, prompt: [Int32]) throws {
            guard prompt.count <= longestPrompt else {
                throw InferenceRuntimeError.contextLengthExceeded(prompt.count, longestPrompt)
            }
            state.withLock { $0.log.append("reserve \(row)") }
        }

        func prefill(row: Int, tokens: ArraySlice<Int32>) async throws {
            state.withLock { $0.log.append("prefill \(row) \(Array(tokens))") }
            if let failPrefillFor, tokens.first == failPrefillFor { throw Boom() }
        }

        func decode(_ rows: [SchedulerKVRow]) async throws -> [SchedulerKVStep] {
            try state.withLock { state in
                state.log.append("decode \(rows.map(\.row))")
                if let failDecodeAfter, state.decodes >= failDecodeAfter { throw Boom() }
                state.decodes += 1
                return rows.map { row in
                    row.cursor < rowLimit ? .logits(BatchSchedulerKVTests.oneHot((row.token + 1) % 64)) : .rowFull
                }
            }
        }

        func retire(row: Int) {
            state.withLock { $0.log.append("retire \(row)") }
        }
    }

    @Test("A full row ends its request at the tokens it has, as maxTokens would")
    func rowFullEndsTheRequest() async throws {
        let kv = CountUpKV(rows: 1, rowLimit: 6)
        let result = try await BatchScheduler(kv: kv).run([Self.request([1, 2, 3, 4], maxTokens: 10)])[0]
        // The decodes at cursors 3, 4 and 5 fit; the one at 6 is past the row.
        #expect(result.tokens == [5, 6, 7])
        #expect(result.stopReason == .maxTokens)
        #expect(kv.log.last == "retire 0")
    }

    @Test("A request the KV refuses fails alone, leaves nothing counted, and the next request runs")
    func refusedRequestFailsAlone() async throws {
        let scheduler = BatchScheduler(kv: CountUpKV(rows: 1, longestPrompt: 4), maxQueued: 0)
        let refused = try scheduler.submit(Self.request([1, 2, 3, 4, 5], maxTokens: 2))
        await #expect(throws: InferenceRuntimeError.self) {
            for try await _ in refused {}
        }
        #expect(scheduler.occupancy == (0, 0))
        let result = try await scheduler.run([Self.request([1, 2], maxTokens: 2)])[0]
        #expect(result.tokens == [3, 4])
    }

    @Test("A failed prompt piece fails only its request and gives its row back")
    func prefillFailureRetiresItsRow() async throws {
        let kv = CountUpKV(rows: 2, failPrefillFor: 50)
        let scheduler = BatchScheduler(kv: kv)
        let failing = try scheduler.submit(Self.request([50, 1], maxTokens: 4))
        let healthy = try scheduler.submit(Self.request([10, 11], maxTokens: 3))
        await #expect(throws: CountUpKV.Boom.self) {
            for try await _ in failing {}
        }
        var tokens: [Int32] = []
        for try await event in healthy {
            if case .token(let token, _) = event { tokens.append(token) }
        }
        #expect(tokens == [12, 13, 14])
        #expect(kv.log.contains("retire 0"))
    }

    @Test("A failed decode step fails every request and gives every row back")
    func decodeFailureRetiresEveryRow() async throws {
        let kv = CountUpKV(rows: 2, failDecodeAfter: 1)
        let scheduler = BatchScheduler(kv: kv)
        let streams = try [Int32(1), 20].map { try scheduler.submit(Self.request([$0], maxTokens: 4)) }
        for stream in streams {
            await #expect(throws: CountUpKV.Boom.self) {
                for try await _ in stream {}
            }
        }
        #expect(Set(kv.log.filter { $0.hasPrefix("retire") }) == ["retire 0", "retire 1"])
        #expect(scheduler.occupancy == (0, 0))
    }

    @Test("Scoring needs a contiguous batch session")
    func noScoringWithoutAScorer() async throws {
        let scheduler = BatchScheduler(kv: CountUpKV(rows: 1))
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await scheduler.score([ScoreRequest(tokens: [1, 2], targets: [2, 3], topK: 1)])
        }
    }

    @Test("A prompt is written a chunk at a time, and its last token is decoded")
    func promptInChunks() async throws {
        let kv = CountUpKV(rows: 1, chunk: 2)
        let result = try await BatchScheduler(kv: kv).run([Self.request([1, 2, 3, 4, 5, 6], maxTokens: 2)])[0]
        #expect(result.tokens == [7, 8])
        #expect(
            kv.log == [
                "reserve 0", "prefill 0 [1, 2]", "prefill 0 [3, 4]", "prefill 0 [5]", "decode [0]", "decode [0]",
                "retire 0",
            ])
    }

    @Test("Prompt pieces and decode steps take turns, so neither starves the other")
    func prefillAndDecodeAlternate() async throws {
        let kv = CountUpKV(rows: 2, chunk: 2)
        let results = try await BatchScheduler(kv: kv).run([
            Self.request([1, 2], maxTokens: 4), Self.request([10, 11, 12, 13, 14, 15, 16], maxTokens: 1),
        ])
        #expect(results.map(\.tokens) == [[3, 4, 5, 6], [17]])
        let calls = kv.log.filter { !$0.hasPrefix("reserve") }
        // Row 0's one-token prompt piece, then row 1's pieces each followed by a decode of row 0.
        #expect(
            calls.prefix(7) == [
                "prefill 0 [1]", "decode [0]", "prefill 1 [10, 11]", "decode [0]", "prefill 1 [12, 13]", "decode [0]",
                "prefill 1 [14, 15]",
            ])
    }

    @Test("Prompts are written oldest first, whichever row they hold")
    func oldestPromptFirst() async throws {
        let kv = CountUpKV(rows: 2, chunk: 2)
        // The first request ends after one token and frees row 0 while row 1's prompt is being written;
        // the third request takes row 0, and row 1's older prompt is still written first.
        _ = try await BatchScheduler(kv: kv).run([
            Self.request([1], maxTokens: 1), Self.request([10, 11, 12, 13, 14, 15, 16], maxTokens: 1),
            Self.request([20, 21, 22], maxTokens: 1),
        ])
        let pieces = kv.log.filter { $0.hasPrefix("prefill") }
        #expect(
            pieces == ["prefill 1 [10, 11]", "prefill 1 [12, 13]", "prefill 1 [14, 15]", "prefill 0 [20, 21]"])
    }

    @Test("While no row is decoding, a prompt is written in idle-sized pieces")
    func idlePieces() async throws {
        let kv = CountUpKV(rows: 2, chunk: 2, idleChunk: 4)
        let results = try await BatchScheduler(kv: kv).run([
            Self.request([1, 2, 3, 4, 5, 6, 7, 8, 9], maxTokens: 2),
            Self.request([30, 31, 32, 33, 34, 35], maxTokens: 1),
        ])
        #expect(results.map(\.tokens) == [[10, 11], [36]])
        // Nothing decodes while row 0's prompt goes in, so its pieces are 4 tokens. Row 1's prompt then
        // takes turns with row 0's decode steps in pieces of 2, and its rest is 4 again once row 0 ends.
        let calls = kv.log.filter { !$0.hasPrefix("reserve") && !$0.hasPrefix("retire") }
        #expect(
            calls == [
                "prefill 0 [1, 2, 3, 4]", "prefill 0 [5, 6, 7, 8]", "decode [0]", "prefill 1 [30, 31]", "decode [0]",
                "prefill 1 [32, 33, 34]", "decode [1]",
            ])
    }

    // MARK: - A contiguous batch session written in pieces

    /// A `BatchedDecodeStep` that counts up and logs every call, so the order of scratch writes,
    /// commits and decode steps can be checked.
    final class ChunkingStep: BatchedDecodeStep, @unchecked Sendable {
        let maxBatchCapacity: Int
        private let calls = Mutex([String]())
        var log: [String] { calls.withLock { $0 } }

        init(capacity: Int) { self.maxBatchCapacity = capacity }

        func decodeBatch(
            tokensPerRow: [[Int32]], startPositions: [Int], session: GenerationSessionState
        ) async throws -> [[LogitsScalarType]] {
            calls.withLock { $0.append("decode \(tokensPerRow.map { $0[0] }) at \(startPositions)") }
            return tokensPerRow.map { BatchSchedulerKVTests.oneHot(($0[0] + 1) % 64) }
        }

        func allPositionLogits(
            tokensPerRow: [[Int32]], startPosition: Int, session: GenerationSessionState
        ) async throws -> [LogitsScalarType] { [] }

        func makeSessionState(batchSize: Int) throws -> GenerationSessionState {
            calls.withLock { $0.append("session \(batchSize)") }
            return GenerationSessionState(
                kvCache: MockStateHandler(names: ["k"], shape: [1, 1, 1, 1, 1]),
                additionalStates: nil, hasNonTruncatableStates: false)
        }

        func prefillPiece(_ tokens: ArraySlice<Int32>, scratch: GenerationSessionState) async throws {
            calls.withLock { $0.append("piece \(Array(tokens))") }
        }

        func commitRow(
            _ row: Int, from scratch: GenerationSessionState, count: Int, session: GenerationSessionState
        ) throws {
            calls.withLock { $0.append("commit \(row) \(count)") }
        }
    }

    private func contiguousScheduler(_ step: ChunkingStep, rows: Int, chunk: Int = 2, maxPositions: Int = .max)
        throws -> BatchScheduler
    {
        BatchScheduler(
            decodeStep: step, session: try step.makeSessionState(batchSize: rows), maxConcurrent: rows,
            maxPositions: maxPositions, prefillChunk: chunk)
    }

    @Test("A contiguous session writes a prompt into scratch in pieces, commits it, then decodes its last token")
    func contiguousChunkedPrefill() async throws {
        let step = ChunkingStep(capacity: 1)
        let result = try await contiguousScheduler(step, rows: 1).run([Self.request([1, 2, 3, 4, 5, 6], maxTokens: 2)])
        #expect(result[0].tokens == [7, 8])
        // Five tokens go into the row, then the sixth is the first decode step, at cursor 5.
        #expect(
            step.log == [
                "session 1", "session 1", "piece [1, 2]", "piece [3, 4]", "piece [5]", "commit 0 5",
                "decode [6] at [5]", "decode [7] at [6]",
            ])
    }

    @Test("Chunked and whole prefill give the same replies")
    func chunkedMatchesWhole() async throws {
        let requests = [
            Self.request([1, 2, 3, 4, 5, 6, 7], maxTokens: 3), Self.request([9], maxTokens: 4),
            Self.request([20, 21, 22], maxTokens: 2),
        ]
        let chunked = try await contiguousScheduler(ChunkingStep(capacity: 2), rows: 2).run(requests)
        let whole = try await contiguousScheduler(ChunkingStep(capacity: 2), rows: 2, chunk: .max).run(requests)
        #expect(chunked.map(\.tokens) == whole.map(\.tokens))
        #expect(chunked.map(\.stopReason) == whole.map(\.stopReason))
    }

    @Test("A row that is still being written decodes nothing; a one-token prompt needs no scratch")
    func rowsWaitForTheirPrompt() async throws {
        let step = ChunkingStep(capacity: 2)
        let results = try await contiguousScheduler(step, rows: 2).run([
            Self.request([30], maxTokens: 3), Self.request([1, 2, 3, 4, 5], maxTokens: 1),
        ])
        #expect(results.map(\.tokens) == [[31, 32, 33], [6]])
        // Row 1 joins decode steps only after its commit; until then it decodes token 0 at cursor 0,
        // as a free row does.
        let commit = try #require(step.log.firstIndex(of: "commit 1 4"))
        let before = step.log[..<commit].filter { $0.hasPrefix("decode") }
        #expect(!before.isEmpty)
        #expect(before.allSatisfy { $0.contains(", 0] at [") && $0.hasSuffix(", 0]") })
        #expect(!step.log.contains { $0.hasPrefix("commit 0") })
        // The session, and one scratch for row 1's prompt.
        #expect(step.log.filter { $0.hasPrefix("session") } == ["session 2", "session 1"])
    }

    @Test("A prompt as long as the row fits and samples one token; one longer fails alone")
    func promptAtTheRowLimit() async throws {
        let scheduler = try contiguousScheduler(ChunkingStep(capacity: 1), rows: 1, maxPositions: 4)
        let fits = try await scheduler.run([Self.request([1, 2, 3, 4], maxTokens: 5)])[0]
        #expect(fits.tokens == [5])
        #expect(fits.stopReason == .maxTokens)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await scheduler.run([Self.request([1, 2, 3, 4, 5], maxTokens: 5)])
        }
    }

    @Test("A contiguous KV refuses to decode a row whose prompt is still being written")
    func decodeBeforeCommitThrows() async throws {
        let step = ChunkingStep(capacity: 1)
        let kv = ContiguousSchedulerKV(
            step: step, session: try step.makeSessionState(batchSize: 1), rows: 1, prefillChunk: 2,
            idlePrefillChunk: 2)
        try kv.reserve(row: 0, prompt: [1, 2, 3])
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await kv.decode([SchedulerKVRow(row: 0, token: 3, cursor: 2)])
        }
        kv.retire(row: 0)
        try kv.reserve(row: 0, prompt: [7])
        let steps = try await kv.decode([SchedulerKVRow(row: 0, token: 7, cursor: 0)])
        guard case .logits(let logits) = steps[0] else {
            Issue.record("expected logits")
            return
        }
        #expect(logits == Self.oneHot(8))
    }

    @Test("A contiguous row past its positions ends as maxTokens")
    func contiguousRowFull() async throws {
        let step = ChunkingStep(capacity: 1)
        let result = try await contiguousScheduler(step, rows: 1, maxPositions: 6).run([
            Self.request([1, 2, 3, 4], maxTokens: 10)
        ])[0]
        // The decodes at cursors 3, 4 and 5 fit; the one at 6 is past the row.
        #expect(result.tokens == [5, 6, 7])
        #expect(result.stopReason == .maxTokens)
    }
}
