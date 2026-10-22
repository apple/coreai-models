// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// One request submitted to a `RequestSink`.
package struct BatchRequest: Sendable {
    /// Must not be empty; `submit(_:)` throws `invalidArgument` for an empty prompt.
    let promptTokens: [Int32]
    let sampling: SamplingConfiguration
    let eosTokenIds: Set<Int32>
    let maxTokens: Int
    let stopSequences: StopSequences
    /// Turns a matched stop sequence's tokens into the text reported in `StopReason.stopSequence`,
    /// so the scheduler needs no tokenizer. The default returns an empty string.
    let decodeStopText: @Sendable ([Int32]) -> String

    init(
        promptTokens: [Int32],
        sampling: SamplingConfiguration,
        eosTokenIds: Set<Int32>,
        maxTokens: Int,
        stopSequences: StopSequences = StopSequences(sequences: []),
        decodeStopText: @escaping @Sendable ([Int32]) -> String = { _ in "" }
    ) {
        self.promptTokens = promptTokens
        self.sampling = sampling
        self.eosTokenIds = eosTokenIds
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
        self.decodeStopText = decodeStopText
    }
}

/// The tokens one request generated in `BatchScheduler.run(_:)` and why it stopped.
struct RowResult: Sendable {
    let tokens: [Int32]
    let stopReason: StopReason
}

/// One event on a request's stream from a `RequestSink`. A stream yields `.token` events and then
/// one `.finished`, or ends by throwing.
package enum BatchSinkEvent: Sendable {
    /// A sampled token. `BatchScheduler` always sends nil `logits` because it samples internally.
    case token(Int32, logits: [LogitsScalarType]?)
    /// Why the request stopped. Nothing follows it.
    case finished(StopReason)
}

/// A shared batched backend that gives each submitted request its own event stream.
///
/// The server does not call `submit(_:)` directly. It wraps the sink in one `SchedulerSession` per
/// request, which submits on `generate` and presents the stream as an ordinary `InferenceEngine`.
///
/// ```swift
/// let engine = SchedulerSession(sink: scheduler, config: modelConfig, stopTokenIds: eosIds)
/// let sequence = try await engine.generate(
///     with: promptTokens, samplingConfiguration: sampling, inferenceOptions: options)
/// ```
package protocol RequestSink: Sendable {
    /// The most requests that decode in the same batch.
    var maxConcurrent: Int { get }

    /// Queues `request` and returns its event stream. When the sink rejects the request, for example
    /// with `InferenceRuntimeError.queueFull`, this throws before any stream exists.
    func submit(_ request: BatchRequest) throws -> AsyncThrowingStream<BatchSinkEvent, Error>
}

/// Runs up to `maxConcurrent` requests through one shared batch-N session and admits new requests
/// while others are still decoding.
///
/// Slot `i` is row `i` of the batch KV cache. A slot goes through these phases:
/// - Admission: a queued request takes a free slot and is prefilled into its row with
///   `BatchedDecodeStep.prefillRow`.
/// - Decode: each step runs `decodeBatch` over all `N` rows, every live row at its own cursor, so a
///   short request admitted late runs beside an older long one.
/// - Retirement: `BatchSessionAdapter` ends the row on EOS, a stop sequence or `maxTokens`. The next
///   pump iteration refills the slot from the queue while the other rows keep decoding.
///
/// The class is `@unchecked Sendable` because the queue and row count sit behind a `Mutex`, and at
/// most one pump task runs at a time. That task alone drives the session.
package final class BatchScheduler: RequestSink, @unchecked Sendable {
    private let decodeStep: any BatchedDecodeStep
    private let session: GenerationSessionState
    package let maxConcurrent: Int
    /// How many requests may wait beyond the free rows before `submit(_:)` throws. `nil` means
    /// unbounded.
    private let maxQueued: Int?

    /// Whether the server can skip its single-slot permit for the batched lane. This scheduler queues
    /// and bounds requests itself, so that holds whenever it has more than one row.
    package var supportsContinuousBatching: Bool { maxConcurrent > 1 }

    private struct Pending {
        let request: BatchRequest
        let continuation: AsyncThrowingStream<BatchSinkEvent, Error>.Continuation
    }
    private struct State {
        var queue: [Pending] = []
        /// Rows held by admitted requests.
        var occupiedRows = 0
        var pumping = false
    }
    private let state = Mutex(State())
    /// Held by the pump for each step and by an `ExclusiveEngineLease` for its whole lifetime, so
    /// the engine never runs work from two tasks at once.
    private let engineLock = AsyncFIFOLock()

    /// Creates a scheduler with a new `maxConcurrent`-row session on `engine`.
    /// `engine.supportsBatching` must be true, and `maxConcurrent` must not exceed the graph's batch
    /// capacity or a precondition fails.
    package convenience init(engine: CoreAISequentialEngine, maxConcurrent: Int, maxQueued: Int? = nil) throws {
        self.init(
            decodeStep: engine, session: try engine.makeSessionState(batchSize: maxConcurrent),
            maxConcurrent: maxConcurrent, maxQueued: maxQueued)
    }

    init(
        decodeStep: any BatchedDecodeStep, session: GenerationSessionState, maxConcurrent: Int,
        maxQueued: Int? = nil
    ) {
        precondition(maxConcurrent >= 1, "maxConcurrent must be >= 1")
        precondition(
            maxConcurrent <= decodeStep.maxBatchCapacity,
            "maxConcurrent \(maxConcurrent) exceeds the graph's batch capacity \(decodeStep.maxBatchCapacity)")
        precondition((maxQueued ?? 0) >= 0, "maxQueued must be >= 0")
        self.decodeStep = decodeStep
        self.session = session
        self.maxConcurrent = maxConcurrent
        self.maxQueued = maxQueued
    }

    // MARK: - RequestSink

    /// Queues `request` and returns its event stream.
    ///
    /// Throws `InferenceRuntimeError.queueFull`, which the server maps to HTTP 429, when
    /// `queue.count + 1 > freeRows + maxQueued`. With `maxConcurrent == 2`, `maxQueued == 1` and both
    /// rows busy, one request may wait and the next one throws. With `maxQueued == nil` it never throws.
    ///
    /// `queue.count` includes requests the pump has not admitted yet, so free rows absorb a burst
    /// before `maxQueued` applies. In the same configuration with both rows free, a burst of four
    /// requests accepts three and the fourth throws.
    package func submit(_ request: BatchRequest) throws -> AsyncThrowingStream<BatchSinkEvent, Error> {
        try enqueue([request])[0]
    }

    /// Enqueues `requests` atomically and starts the pump if none is running. Returns one stream per
    /// request, in order. `submit(_:)` and `run(_:)` both go through here, so only the pump ever
    /// drives `session`.
    private func enqueue(
        _ requests: [BatchRequest]
    ) throws -> [AsyncThrowingStream<BatchSinkEvent, Error>] {
        if requests.contains(where: { $0.promptTokens.isEmpty }) {
            throw InferenceRuntimeError.invalidArgument("a batched request needs a non-empty prompt")
        }
        var streams: [AsyncThrowingStream<BatchSinkEvent, Error>] = []
        var pending: [Pending] = []
        streams.reserveCapacity(requests.count)
        pending.reserveCapacity(requests.count)
        for request in requests {
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: BatchSinkEvent.self)
            streams.append(stream)
            pending.append(Pending(request: request, continuation: continuation))
        }
        let shouldStart = try state.withLock { s -> Bool in
            let freeRows = maxConcurrent - s.occupiedRows
            if let maxQueued, s.queue.count + pending.count > freeRows + maxQueued {
                throw InferenceRuntimeError.queueFull(depth: maxQueued)
            }
            s.queue.append(contentsOf: pending)
            if s.pumping { return false }
            s.pumping = true
            return true
        }
        if shouldStart { Task { await self.pump() } }
        return streams
    }

    /// A request that holds a slot. `cursor` is the next KV write position in its row.
    private struct ActiveRow {
        let continuation: AsyncThrowingStream<BatchSinkEvent, Error>.Continuation
        var adapter: BatchSessionAdapter
        var lastToken: Int32
        var cursor: Int
    }

    /// The single pump. Each iteration does the following:
    /// 1. Admit queued requests into free slots.
    /// 2. If no slot is live and the queue is empty, clear `pumping` and return.
    /// 3. Run one `decodeBatch` over all `N` slots with `tokens` `[N][1]` and `cursors` `[N]`. A free
    ///    slot gets token 0 at cursor 0 and its logits are dropped.
    /// 4. Advance each live cursor, then sample, emit or retire each live row.
    ///
    /// The step always spans `maxConcurrent` rows, so a lone request still pays for a full batch.
    /// If `decodeBatch` throws, every resident row and every queued request ends with that error.
    private func pump() async {
        var active: [ActiveRow?] = Array(repeating: nil, count: maxConcurrent)
        do {
            while true {
                await engineLock.lock()
                defer { engineLock.unlock() }
                await admit(into: &active)

                let liveSlots = active.indices.filter { active[$0] != nil }
                if liveSlots.isEmpty {
                    // Recheck the queue under the lock. A request enqueued after `admit` saw
                    // `pumping == true` and did not start a pump, so this one must keep going.
                    let more = state.withLock { s -> Bool in
                        if s.queue.isEmpty {
                            s.pumping = false
                            return false
                        }
                        return true
                    }
                    if more { continue }
                    return
                }

                let tokens = active.map { [$0?.lastToken ?? 0] }
                let cursors = active.map { $0?.cursor ?? 0 }
                let logits = try await decodeStep.decodeBatch(
                    tokensPerRow: tokens, startPositions: cursors, session: session)
                for slot in active.indices where active[slot] != nil { active[slot]!.cursor += 1 }
                for slot in liveSlots { stepSlot(slot, logits: logits[slot], active: &active) }
            }
        } catch {
            for row in active where row != nil { row!.continuation.finish(throwing: error) }
            // Drain the queue under the same lock that clears `pumping`. A queued request left
            // behind would wait forever on a pump that has stopped.
            let orphaned = state.withLock { s -> [Pending] in
                let waiting = s.queue
                s.queue.removeAll()
                s.occupiedRows = 0
                s.pumping = false
                return waiting
            }
            for pending in orphaned { pending.continuation.finish(throwing: error) }
        }
    }

    /// Fills free slots from the queue. Each admitted request is prefilled into its row and its first
    /// token is sampled at once, so a request can also retire here.
    ///
    /// A prefill error fails only that request's stream and leaves its slot free.
    private func admit(into active: inout [ActiveRow?]) async {
        for slot in active.indices where active[slot] == nil {
            let pending: Pending? = state.withLock { s in
                guard !s.queue.isEmpty else { return nil }
                s.occupiedRows += 1
                return s.queue.removeFirst()
            }
            guard let pending else { break }
            do {
                let adapter = BatchSessionAdapter(request: pending.request)
                let logits = try await decodeStep.prefillRow(
                    row: slot, promptTokens: pending.request.promptTokens, session: session)
                active[slot] = ActiveRow(
                    continuation: pending.continuation, adapter: adapter, lastToken: 0,
                    cursor: pending.request.promptTokens.count)
                stepSlot(slot, logits: logits, active: &active)
            } catch {
                state.withLock { $0.occupiedRows -= 1 }
                pending.continuation.finish(throwing: error)
            }
        }
    }

    /// Samples one token for `slot`, emits it, and retires the row if the adapter says it is done.
    private func stepSlot(_ slot: Int, logits: [LogitsScalarType], active: inout [ActiveRow?]) {
        guard active[slot] != nil else { return }
        let token = BatchedSampler.sample(
            rows: [logits],
            configurations: [active[slot]!.adapter.request.sampling],
            histories: [ArraySlice(active[slot]!.adapter.generated)],
            steps: [active[slot]!.adapter.step])[0]
        let stillLive = active[slot]!.adapter.accept(token)
        active[slot]!.continuation.yield(.token(token, logits: nil))
        if stillLive {
            active[slot]!.lastToken = token
        } else {
            active[slot]!.continuation.yield(.finished(active[slot]!.adapter.stopReason ?? .maxTokens))
            active[slot]!.continuation.finish()
            active[slot] = nil
            state.withLock { $0.occupiedRows -= 1 }
        }
    }

    // MARK: - Exclusive engine access

    /// Waits until the pump finishes its current step, then gives the caller sole use of the engine
    /// until the lease is released. Resident rows stay in their slots and resume afterwards.
    ///
    /// A request that runs on its own session of the same engine, such as one that needs per-step
    /// logits, holds a lease for as long as it runs:
    ///
    /// ```swift
    /// let lease = await scheduler.acquireExclusiveAccess()
    /// defer { lease.release() }
    /// // Generate on an IsolatedSessionEngine over the same CoreAISequentialEngine.
    /// ```
    package func acquireExclusiveAccess() async -> ExclusiveEngineLease {
        await engineLock.lock()
        return ExclusiveEngineLease(lock: engineLock)
    }

    // MARK: - Closed batch

    /// Submits `requests` together and waits for all of them. Returns one `RowResult` per request, in
    /// order. It shares the pump with `submit(_:)`, so it is safe to call while other requests stream.
    func run(_ requests: [BatchRequest]) async throws -> [RowResult] {
        let streams = try enqueue(requests)
        return try await withThrowingTaskGroup(of: (Int, RowResult).self) { group in
            for (i, stream) in streams.enumerated() {
                group.addTask {
                    var tokens: [Int32] = []
                    var reason = StopReason.maxTokens
                    for try await event in stream {
                        switch event {
                        case .token(let token, _): tokens.append(token)
                        case .finished(let stopReason): reason = stopReason
                        }
                    }
                    return (i, RowResult(tokens: tokens, stopReason: reason))
                }
            }
            var results = Array(
                repeating: RowResult(tokens: [], stopReason: .maxTokens), count: streams.count)
            for try await (i, result) in group { results[i] = result }
            return results
        }
    }
}

/// Sole use of a `BatchScheduler`'s engine, from `acquireExclusiveAccess()`. Releasing it more than
/// once is harmless, and it releases on deinit, so a dropped lease cannot stall the batch.
package final class ExclusiveEngineLease: Sendable {
    private let lock: AsyncFIFOLock
    private let released = Mutex(false)

    init(lock: AsyncFIFOLock) { self.lock = lock }

    package func release() {
        let first = released.withLock { wasReleased -> Bool in
            defer { wasReleased = true }
            return !wasReleased
        }
        if first { lock.unlock() }
    }

    deinit { release() }
}

/// An async lock that grants waiters in arrival order. `unlock()` hands the lock straight to the
/// next waiter, so a task that unlocks and locks again goes to the back of the line.
final class AsyncFIFOLock: Sendable {
    private struct State {
        var held = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Mutex(State())

    func lock() async {
        await withCheckedContinuation { continuation in
            let granted = state.withLock { s -> Bool in
                if !s.held {
                    s.held = true
                    return true
                }
                s.waiters.append(continuation)
                return false
            }
            if granted { continuation.resume() }
        }
    }

    func unlock() {
        let next = state.withLock { s -> CheckedContinuation<Void, Never>? in
            if s.waiters.isEmpty {
                s.held = false
                return nil
            }
            return s.waiters.removeFirst()
        }
        next?.resume()
    }
}
