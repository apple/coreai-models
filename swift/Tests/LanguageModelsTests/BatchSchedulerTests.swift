// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization
import Testing

@testable import CoreAILanguageModels

/// Makes logits over a 64-token vocabulary with the maximum at `token`.
private func oneHot(_ token: Int32) -> [LogitsScalarType] {
    var row = [LogitsScalarType](repeating: 0, count: 64)
    row[Int(token)] = 1
    return row
}

/// Makes a session state backed by a one-element mock KV cache.
private func dummySession() -> GenerationSessionState {
    GenerationSessionState(
        kvCache: MockStateHandler(names: ["k"], shape: [1, 1, 1, 1, 1]),
        additionalStates: nil,
        hasNonTruncatableStates: false)
}

/// Makes flattened count-up logits, `row.count * 64` values per row, where position `p` peaks at
/// that row's token at `p` plus one, mod 64.
private func countUpAllPositions(_ tokensPerRow: [[Int32]]) -> [LogitsScalarType] {
    tokensPerRow.flatMap { row in row.flatMap { oneHot(($0 + 1) % 64) } }
}

/// Collects a request stream's tokens and its `.finished` reason.
private func drain(
    _ stream: AsyncThrowingStream<BatchSinkEvent, Error>
) async throws -> (tokens: [Int32], reason: StopReason?) {
    var tokens: [Int32] = []
    var reason: StopReason?
    for try await event in stream {
        switch event {
        case .token(let token, _): tokens.append(token)
        case .finished(let stopReason): reason = stopReason
        }
    }
    return (tokens, reason)
}

/// Mock `BatchedDecodeStep` that predicts the last input token plus one, mod 64. A request with
/// prompt `[..., k]` and `maxTokens: m` generates `k+1` through `k+m`.
///
/// Records the forward count, each decode's `startPositions`, each scoring forward's row count, and
/// the peak number of overlapping forward calls. `forwardDelay` is a sleep added to every forward call.
private final class CountUpDecodeStep: BatchedDecodeStep, @unchecked Sendable {
    struct Boom: Error {}

    let maxBatchCapacity: Int
    /// `prefillRow` throws `Boom` for a prompt that starts with this token.
    private let failPrefillFor: Int32?
    /// When true, every `decodeBatch` throws `Boom`.
    private let failDecode: Bool
    private let forwardDelay: Duration?

    private struct Record {
        var startPositions: [[Int]] = []
        var inFlight = 0
        var maxInFlight = 0
        var forwards = 0
        var scoringRows: [Int] = []
        var scoringPositions: [Int] = []
    }
    private let record = Mutex(Record())

    var observedStartPositions: [[Int]] { record.withLock { $0.startPositions } }
    var maxInFlight: Int { record.withLock { $0.maxInFlight } }
    /// Row count of each scoring forward, in call order.
    var scoringRows: [Int] { record.withLock { $0.scoringRows } }
    var scoringPositions: [Int] { record.withLock { $0.scoringPositions } }
    /// Number of prefill, decode and scoring forward calls so far.
    var forwards: Int { record.withLock { $0.forwards } }

    init(
        capacity: Int, failPrefillFor: Int32? = nil, failDecode: Bool = false,
        forwardDelay: Duration? = nil
    ) {
        self.maxBatchCapacity = capacity
        self.failPrefillFor = failPrefillFor
        self.failDecode = failDecode
        self.forwardDelay = forwardDelay
    }

    private func enter() async {
        record.withLock {
            $0.forwards += 1
            $0.inFlight += 1
            $0.maxInFlight = max($0.maxInFlight, $0.inFlight)
        }
        if let forwardDelay { try? await Task.sleep(for: forwardDelay) }
    }

    private func leave() { record.withLock { $0.inFlight -= 1 } }

    func prefillRow(
        row: Int, promptTokens: [Int32], session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        await enter()
        defer { leave() }
        if let failPrefillFor, promptTokens.first == failPrefillFor { throw Boom() }
        return oneHot(((promptTokens.last ?? 0) + 1) % 64)
    }

    func decodeBatch(
        tokensPerRow: [[Int32]], startPositions: [Int], session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]] {
        await enter()
        defer { leave() }
        record.withLock { $0.startPositions.append(startPositions) }
        if failDecode { throw Boom() }
        return tokensPerRow.map { oneHot((($0.last ?? 0) + 1) % 64) }
    }

    func allPositionLogits(
        tokensPerRow: [[Int32]], startPosition: Int, session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        await enter()
        defer { leave() }
        record.withLock {
            $0.scoringRows.append(tokensPerRow.count)
            $0.scoringPositions.append(tokensPerRow.first?.count ?? 0)
        }
        return countUpAllPositions(tokensPerRow)
    }

    func makeSessionState(batchSize: Int) throws -> GenerationSessionState { dummySession() }
}

/// Mock count-up `BatchedDecodeStep` whose `prefillRow` blocks until `release()`.
private final class GatedPrefillStep: BatchedDecodeStep, @unchecked Sendable {
    let maxBatchCapacity: Int

    private struct Gate {
        var released = false
        var releaseWaiter: CheckedContinuation<Void, Never>?
    }
    private let gate = Mutex(Gate())

    init(capacity: Int) { maxBatchCapacity = capacity }

    /// Unblocks the waiting `prefillRow` call. Later calls return without blocking.
    func release() {
        let waiter: CheckedContinuation<Void, Never>? = gate.withLock { g in
            g.released = true
            defer { g.releaseWaiter = nil }
            return g.releaseWaiter
        }
        waiter?.resume()
    }

    func prefillRow(
        row: Int, promptTokens: [Int32], session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            gate.withLock { g in
                if g.released { cont.resume() } else { g.releaseWaiter = cont }
            }
        }
        return oneHot((promptTokens.last ?? 0) + 1)
    }

    func decodeBatch(
        tokensPerRow: [[Int32]], startPositions: [Int], session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]] {
        tokensPerRow.map { oneHot(($0.last ?? 0) + 1) }
    }

    func allPositionLogits(
        tokensPerRow: [[Int32]], startPosition: Int, session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        countUpAllPositions(tokensPerRow)
    }

    func makeSessionState(batchSize: Int) throws -> GenerationSessionState { dummySession() }
}

@Suite("BatchScheduler")
struct BatchSchedulerTests {
    private func makeScheduler(
        _ step: any BatchedDecodeStep, maxConcurrent: Int, maxQueued: Int? = nil
    ) -> BatchScheduler {
        BatchScheduler(
            decodeStep: step,
            session: dummySession(),
            maxConcurrent: maxConcurrent,
            maxQueued: maxQueued)
    }

    private func req(
        prompt: [Int32], maxTokens: Int, eos: Set<Int32> = [], stops: [[Int32]] = []
    ) -> BatchRequest {
        BatchRequest(
            promptTokens: prompt,
            sampling: .greedy,
            eosTokenIds: eos,
            maxTokens: maxTokens,
            stopSequences: StopSequences(sequences: stops),
            decodeStopText: { $0.map(String.init).joined(separator: ",") })
    }

    @Test("Rows retire independently on EOS, max tokens and stop sequence")
    func independentRetirement() async throws {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 4), maxConcurrent: 4)
        let results = try await scheduler.run([
            req(prompt: [9], maxTokens: 100, eos: [12]),
            req(prompt: [20], maxTokens: 5),
            req(prompt: [5], maxTokens: 100, stops: [[8]]),
        ])
        #expect(results.map(\.tokens) == [[10, 11, 12], [21, 22, 23, 24, 25], [6, 7, 8]])
        #expect(results.map(\.stopReason) == [.eos, .maxTokens, .stopSequence("8")])
    }

    @Test("Rows decode at their own cursors, and a retired row's slot admits the next request")
    func raggedCursorsAndSlotReuse() async throws {
        let mock = CountUpDecodeStep(capacity: 2)
        let scheduler = makeScheduler(mock, maxConcurrent: 2)
        let results = try await scheduler.run([
            req(prompt: [1], maxTokens: 2),
            req(prompt: [1, 2, 3], maxTokens: 3),
            req(prompt: [20], maxTokens: 1),
        ])
        #expect(results.map(\.tokens) == [[2, 3], [4, 5, 6], [21]])
        #expect(mock.observedStartPositions.first == [1, 3])
    }

    @Test("Forwards from run and submit never overlap")
    func singlePump() async throws {
        let mock = CountUpDecodeStep(capacity: 4, forwardDelay: .milliseconds(5))
        let scheduler = makeScheduler(mock, maxConcurrent: 4)
        let stream = try scheduler.submit(req(prompt: [1, 2], maxTokens: 6))
        async let ran = scheduler.run([req(prompt: [1, 2], maxTokens: 6)])
        _ = try await drain(stream)
        _ = try await ran
        #expect(mock.maxInFlight == 1)
    }

    @Test(
        "A burst of submits fills the free rows plus maxQueued, then throws queueFull",
        arguments: [(1, 3), (0, 2)] as [(Int, Int)])
    func backpressure(maxQueued: Int, accepted: Int) async throws {
        let mock = GatedPrefillStep(capacity: 2)
        let scheduler = makeScheduler(mock, maxConcurrent: 2, maxQueued: maxQueued)
        let streams = try (0..<accepted).map { _ in try scheduler.submit(req(prompt: [0], maxTokens: 1)) }
        #expect(throws: InferenceRuntimeError.self) {
            _ = try scheduler.submit(req(prompt: [0], maxTokens: 1))
        }

        mock.release()
        for stream in streams { _ = try await drain(stream) }
    }

    @Test("An exclusive lease pauses the pump until it is released")
    func exclusiveLeasePausesPump() async throws {
        let step = CountUpDecodeStep(capacity: 2)
        let scheduler = makeScheduler(step, maxConcurrent: 2)
        let lease = await scheduler.acquireExclusiveAccess()
        let stream = try scheduler.submit(req(prompt: [5], maxTokens: 3))

        try await Task.sleep(for: .milliseconds(50))
        #expect(step.forwards == 0)

        lease.release()
        let (tokens, reason) = try await drain(stream)
        #expect(tokens == [6, 7, 8])
        #expect(reason == .maxTokens)
    }

    @Test("Empty prompts and mismatched scoring targets are rejected")
    func invalidRequests() async {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 2), maxConcurrent: 2)
        #expect(throws: InferenceRuntimeError.self) {
            _ = try scheduler.submit(req(prompt: [], maxTokens: 1))
        }
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await scheduler.score([ScoreRequest(tokens: [1, 2], targets: [2], topK: 1)])
        }
    }

    @Test("A prefill failure fails only its own request")
    func prefillFailureIsolated() async throws {
        let scheduler = makeScheduler(
            CountUpDecodeStep(capacity: 2, failPrefillFor: 50), maxConcurrent: 2)
        let failing = try scheduler.submit(req(prompt: [50], maxTokens: 4))
        let healthy = try scheduler.submit(req(prompt: [5], maxTokens: 3))

        await #expect(throws: CountUpDecodeStep.Boom.self) { _ = try await drain(failing) }
        let (tokens, reason) = try await drain(healthy)
        #expect(tokens == [6, 7, 8])
        #expect(reason == .maxTokens)
    }

    @Test("A failed shared decode also fails the queued requests")
    func decodeFailureDrainsQueue() async throws {
        let scheduler = makeScheduler(
            CountUpDecodeStep(capacity: 1, failDecode: true), maxConcurrent: 1)
        let streams = try [Int32(1), 2, 3].map {
            try scheduler.submit(req(prompt: [$0], maxTokens: 4))
        }
        // With one slot, requests 2 and 3 are still queued when request 1's decode throws.
        for stream in streams {
            await #expect(throws: CountUpDecodeStep.Boom.self) { _ = try await drain(stream) }
        }
    }

    @Test("Dropped streams free their row, and a dropped queued request is never decoded")
    func droppedStreamsFreeRows() async throws {
        let mock = CountUpDecodeStep(capacity: 1, forwardDelay: .milliseconds(1))
        let scheduler = makeScheduler(mock, maxConcurrent: 1)
        do {
            var reading = try scheduler.submit(req(prompt: [1], maxTokens: 1000)).makeAsyncIterator()
            _ = try await reading.next()
            _ = try scheduler.submit(req(prompt: [2], maxTokens: 1000))
        }
        let (tokens, _) = try await drain(try scheduler.submit(req(prompt: [5], maxTokens: 2)))
        #expect(tokens == [6, 7])
        #expect(mock.forwards < 100)
    }

    @Test("setStopReason on the output sequence ends the request within one decode step")
    func setStopReasonCancels() async throws {
        let mock = CountUpDecodeStep(capacity: 1, forwardDelay: .milliseconds(1))
        let scheduler = makeScheduler(mock, maxConcurrent: 1)
        let request = req(prompt: [1], maxTokens: 1000)
        let sequence = SchedulerSession.GenerationSequence(
            events: try scheduler.submit(request), cancellation: request.cancellation)
        var forwardsAtStop = 0
        for try await _ in sequence where forwardsAtStop == 0 {
            forwardsAtStop = mock.forwards
            sequence.setStopReason(.stopSequence("4"))
        }
        #expect(mock.forwards - forwardsAtStop <= 1)
        #expect(sequence.stopReason == .stopSequence("4"))
    }

    // MARK: - Scoring

    /// Makes a top-2 score request for `tokens` whose target at each position is the next token,
    /// and 63 at the last position.
    private func scoreRequest(_ tokens: [Int32]) -> ScoreRequest {
        ScoreRequest(tokens: tokens, targets: tokens.dropFirst() + [63], topK: 2)
    }

    @Test(
        "Batched scoring matches scoring each request alone, across padding and chunks",
        arguments: [
            (2, [2, 2, 2]),  // A static batch=2 graph pads every forward to two rows.
            (Int.max, [2, 1, 1]),  // A dynamic-batch graph uses one row per request.
        ] as [(Int, [Int])])
    func scoringBatchInvariance(capacity: Int, expectedRows: [Int]) async throws {
        let step = CountUpDecodeStep(capacity: capacity)
        let scheduler = makeScheduler(step, maxConcurrent: 2)
        // Three requests, at most two per batch. The 200-token one spans two 128-position chunks.
        let prompts: [[Int32]] = [[1, 2, 3], [7, 8, 9, 10, 11], (0..<200).map { Int32($0 % 60) }]
        let together = try await scheduler.score(prompts.map(scoreRequest))
        #expect(step.scoringRows == expectedRows)

        // Argmax targets score highest, and each entry reports topK alternatives.
        let first = together[0].entries
        #expect(first.map(\.tokenId) == [2, 3, 63])
        #expect(first[0].value == first[1].value && first[2].value < first[0].value)
        #expect(first[0].alternatives.count == 2)

        for (prompt, batched) in zip(prompts, together) {
            let alone = try await scheduler.score([scoreRequest(prompt)])[0]
            #expect(batched.entries.map(\.tokenId) == alone.entries.map(\.tokenId))
            #expect(batched.entries.map(\.value) == alone.entries.map(\.value))
        }
    }

    @Test("Scoring pads the last chunk and the rows to a few fixed shapes, with unchanged scores")
    func scoringPadsShapes() async throws {
        let step = CountUpDecodeStep(capacity: Int.max)
        let scheduler = makeScheduler(step, maxConcurrent: 4)
        // One batch of three short prompts: 3 rows pad to 4, 6 positions pad to 16.
        let prompts: [[Int32]] = [[1, 2, 3], [7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]]
        let together = try await scheduler.score(prompts.map(scoreRequest))
        #expect(step.scoringRows == [4])
        #expect(step.scoringPositions == [16])
        for (prompt, batched) in zip(prompts, together) {
            let alone = try await scheduler.score([scoreRequest(prompt)])[0]
            #expect(batched.entries.count == prompt.count)
            #expect(batched.entries.map(\.tokenId) == alone.entries.map(\.tokenId))
            #expect(batched.entries.map(\.value) == alone.entries.map(\.value))
        }
    }

    @Test(
        "The last scoring chunk pads to a power of two, at least 16, within the context",
        arguments: [
            (0, 7, Int.max, 16),
            (0, 17, Int.max, 32),
            (0, 300, Int.max, 128),
            (256, 300, Int.max, 64),
            (256, 300, 300, 44),  // Padding to 64 would run past a 300-position context.
            (128, 256, Int.max, 128),
        ] as [(Int, Int, Int, Int)])
    func scoringChunkLength(start: Int, longest: Int, maxPositions: Int, expected: Int) {
        let chunk = BatchScheduler.Scorer.paddedChunk(
            start: start, longest: longest, chunkLength: 128, maxPositions: maxPositions)
        #expect(chunk == expected)
    }

    @Test(
        "Scoring rows: a static graph's capacity, else a power of two up to maxConcurrent",
        arguments: [
            (1, 4, 4, 4),
            (3, Int.max, 8, 4),
            (3, Int.max, 3, 3),
            (5, Int.max, 6, 6),
            (1, Int.max, 8, 1),
        ] as [(Int, Int, Int, Int)])
    func scoringRowCount(requests: Int, capacity: Int, maxConcurrent: Int, expected: Int) {
        let rows = BatchScheduler.Scorer.paddedRows(
            requests: requests, capacity: capacity, maxConcurrent: maxConcurrent)
        #expect(rows == expected)
    }

    @Test("Scoring runs between decode steps while a request streams")
    func scoringBesideDecode() async throws {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 2), maxConcurrent: 2)
        let stream = try scheduler.submit(req(prompt: [5], maxTokens: 6))
        let scores = try await scheduler.score([scoreRequest([1, 2, 3])])
        let (tokens, _) = try await drain(stream)
        #expect(tokens == [6, 7, 8, 9, 10, 11])
        #expect(scores[0].entries.count == 3)
    }

    @Test(
        "Scoring batches stop before a third of the positions would be padding",
        arguments: [
            ([7, 40, 130, 300], 1),
            ([130, 300], 2),
            ([64, 64, 64, 64, 64], 4),
            ([], 0),
        ] as [([Int], Int)])
    func scoringBatchCount(lengths: [Int], expected: Int) {
        #expect(BatchScheduler.Scorer.batchCount(lengths, limit: 4) == expected)
    }
}

@Suite("SchedulerSession.GenerationSequence")
struct SchedulerSessionSequenceTests {
    @Test(
        "Maps tokens and takes the stop reason from .finished, else .maxTokens",
        arguments: [StopReason.eos, nil])
    func mapsEvents(finished: StopReason?) async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: BatchSinkEvent.self)
        continuation.yield(.token(10, logits: nil))
        continuation.yield(.token(20, logits: nil))
        if let finished { continuation.yield(.finished(finished)) }
        continuation.finish()

        let sequence = SchedulerSession.GenerationSequence(events: stream, cancellation: RequestCancellation())
        var tokens: [Int32] = []
        for try await output in sequence { tokens.append(output.tokenId) }
        #expect(tokens == [10, 20])
        #expect(sequence.stopReason == (finished ?? .maxTokens))
    }
}
