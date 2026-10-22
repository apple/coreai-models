// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization
import Testing

@testable import CoreAILanguageModels

/// Logits over a 64-token vocabulary that greedy sampling resolves to `token`.
private func oneHot(_ token: Int32) -> [LogitsScalarType] {
    var row = [LogitsScalarType](repeating: 0, count: 64)
    row[Int(token)] = 1
    return row
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

/// A `BatchedDecodeStep` that predicts the input token plus one, so a request with prompt `[..., k]`
/// and `maxTokens: m` generates `k+1` through `k+m` whatever its slot or admission time.
///
/// It records each decode's `startPositions` and the peak number of overlapping forward calls.
/// `forwardDelay` keeps each call in flight longer so that an overlap is reliably observed.
private final class CountUpDecodeStep: BatchedDecodeStep, @unchecked Sendable {
    struct Boom: Error {}

    let maxBatchCapacity: Int
    /// `prefillRow` throws for prompts that start with this token.
    private let failPrefillFor: Int32?
    /// Every `decodeBatch` throws.
    private let failDecode: Bool
    private let forwardDelay: Duration?

    private struct Record {
        var startPositions: [[Int]] = []
        var inFlight = 0
        var maxInFlight = 0
        var forwards = 0
    }
    private let record = Mutex(Record())

    var observedStartPositions: [[Int]] { record.withLock { $0.startPositions } }
    var maxInFlight: Int { record.withLock { $0.maxInFlight } }
    /// Prefill and decode calls so far.
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
        return oneHot((promptTokens.last ?? 0) + 1)
    }

    func decodeBatch(
        tokensPerRow: [[Int32]], startPositions: [Int], session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]] {
        await enter()
        defer { leave() }
        record.withLock { $0.startPositions.append(startPositions) }
        if failDecode { throw Boom() }
        return tokensPerRow.map { oneHot(($0.last ?? 0) + 1) }
    }
}

/// A count-up `BatchedDecodeStep` whose `prefillRow` blocks until `release()`, so a test can keep a
/// row occupying its slot while it fills the queue.
private final class GatedPrefillStep: BatchedDecodeStep, @unchecked Sendable {
    let maxBatchCapacity: Int

    private struct Gate {
        var entered = false
        var released = false
        var enteredWaiter: CheckedContinuation<Void, Never>?
        var releaseWaiter: CheckedContinuation<Void, Never>?
    }
    private let gate = Mutex(Gate())

    init(capacity: Int) { maxBatchCapacity = capacity }

    /// Returns once a `prefillRow` call has started, which means its row holds a slot.
    func waitEntered() async {
        await withCheckedContinuation { cont in
            gate.withLock { g in
                if g.entered { cont.resume() } else { g.enteredWaiter = cont }
            }
        }
    }

    /// Unblocks the waiting `prefillRow`. Later calls no longer block.
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
        let enteredWaiter: CheckedContinuation<Void, Never>? = gate.withLock { g in
            g.entered = true
            defer { g.enteredWaiter = nil }
            return g.enteredWaiter
        }
        enteredWaiter?.resume()
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
}

@Suite("BatchScheduler")
struct BatchSchedulerTests {
    private func makeScheduler(
        _ step: any BatchedDecodeStep, maxConcurrent: Int, maxQueued: Int? = nil
    ) -> BatchScheduler {
        BatchScheduler(
            decodeStep: step,
            session: GenerationSessionState(
                kvCache: MockStateHandler(names: ["k"], shape: [1, 1, 1, 1, 1]),
                additionalStates: nil,
                hasNonTruncatableStates: false),
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

    @Test("Submit streams the tokens, then .finished")
    func streamingSubmit() async throws {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 2), maxConcurrent: 2)
        let (tokens, reason) = try await drain(
            scheduler.submit(req(prompt: [5], maxTokens: 100, eos: [8])))
        // The EOS token itself is streamed.
        #expect(tokens == [6, 7, 8])
        #expect(reason == .eos)
    }

    @Test("A retired row's slot admits the next queued request")
    func slotReuse() async throws {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 2), maxConcurrent: 2)
        let results = try await scheduler.run([
            req(prompt: [0], maxTokens: 2),
            req(prompt: [10], maxTokens: 3),
            req(prompt: [20], maxTokens: 1),
            req(prompt: [30], maxTokens: 4),
        ])
        #expect(results.map(\.tokens) == [[1, 2], [11, 12, 13], [21], [31, 32, 33, 34]])
        #expect(results.allSatisfy { $0.stopReason == .maxTokens })
    }

    @Test("Rows decode at their own cursors in one forward")
    func raggedCursors() async throws {
        let mock = CountUpDecodeStep(capacity: 2)
        let scheduler = makeScheduler(mock, maxConcurrent: 2)
        let results = try await scheduler.run([
            req(prompt: [1], maxTokens: 3),
            req(prompt: [1, 2, 3], maxTokens: 3),
        ])
        #expect(results.map(\.tokens) == [[2, 3, 4], [4, 5, 6]])
        // Each cursor starts at the row's prompt length.
        #expect(mock.observedStartPositions == [[1, 3], [2, 4]])
    }

    @Test("Run and submit share one pump, so forwards never overlap")
    func singlePump() async throws {
        let mock = CountUpDecodeStep(capacity: 4, forwardDelay: .milliseconds(5))
        let scheduler = makeScheduler(mock, maxConcurrent: 4)
        let stream = try scheduler.submit(req(prompt: [1, 2], maxTokens: 6))
        async let ran = scheduler.run([req(prompt: [1, 2], maxTokens: 6)])
        _ = try await drain(stream)
        _ = try await ran
        #expect(mock.maxInFlight == 1)
    }

    @Test("Submit throws queueFull past free rows plus maxQueued", arguments: [false, true])
    func backpressure(waitForAdmission: Bool) async throws {
        // Queued plus admitted requests may total maxConcurrent + maxQueued, here 3. The limit is
        // the same whether the first request is still queued (a burst) or already holds a row.
        let mock = GatedPrefillStep(capacity: 2)
        let scheduler = makeScheduler(mock, maxConcurrent: 2, maxQueued: 1)
        var streams = [try scheduler.submit(req(prompt: [0], maxTokens: 1))]
        if waitForAdmission { await mock.waitEntered() }
        for _ in 0..<2 { streams.append(try scheduler.submit(req(prompt: [0], maxTokens: 1))) }
        #expect(throws: InferenceRuntimeError.self) {
            _ = try scheduler.submit(req(prompt: [0], maxTokens: 1))
        }

        mock.release()
        for stream in streams { _ = try await drain(stream) }
    }

    @Test("maxQueued 0 accepts only as many requests as there are free rows")
    func noWaiting() async throws {
        let mock = GatedPrefillStep(capacity: 2)
        let scheduler = makeScheduler(mock, maxConcurrent: 2, maxQueued: 0)
        let streams = try (0..<2).map { _ in try scheduler.submit(req(prompt: [0], maxTokens: 1)) }
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

    @Test("AsyncFIFOLock grants waiters in arrival order")
    func fifoLockOrder() async {
        let lock = AsyncFIFOLock()
        let order = Mutex<[Int]>([])
        await lock.lock()
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<3 {
                group.addTask {
                    await lock.lock()
                    order.withLock { $0.append(i) }
                    lock.unlock()
                }
                // Let each waiter enqueue before starting the next one.
                try? await Task.sleep(for: .milliseconds(20))
            }
            lock.unlock()
        }
        #expect(order.withLock { $0 } == [0, 1, 2])
    }

    @Test("An empty prompt is rejected at submit")
    func emptyPromptRejected() {
        let scheduler = makeScheduler(CountUpDecodeStep(capacity: 2), maxConcurrent: 2)
        #expect(throws: InferenceRuntimeError.self) {
            _ = try scheduler.submit(req(prompt: [], maxTokens: 1))
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
        // With one slot, requests 2 and 3 wait in the queue while request 1's decode throws. If
        // the queue were not drained, iterating their streams would hang.
        for stream in streams {
            await #expect(throws: CountUpDecodeStep.Boom.self) { _ = try await drain(stream) }
        }
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

        let sequence = SchedulerSession.GenerationSequence(events: stream)
        var tokens: [Int32] = []
        for try await output in sequence { tokens.append(output.tokenId) }
        #expect(tokens == [10, 20])
        #expect(sequence.stopReason == (finished ?? .maxTokens))
    }
}
