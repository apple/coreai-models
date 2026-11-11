// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Synchronization

/// A `RequestSink` that decodes up to `maxConcurrent` requests in one shared batch session and
/// admits new requests as rows free up. Safe to call from any task.
///
/// `submit(_:)` queues a request and returns its event stream. Wrap the scheduler in a
/// `SchedulerSession`, one per request, to use it as an `InferenceEngine`. `score(_:)` runs scoring
/// requests as their own padded batch between steps.
///
/// A, B and C hold rows 0 to 2 of a batch session with `N` = 4 rows. A and B are decoding when C
/// joins, so C's prompt chunks alternate with their decode steps:
///
/// ```
///   iteration   1   2   3   4   5   6   7
///   row 0  A        a       a       a   a     a = token sampled by a decode step
///   row 1  B        b       b       b   b
///   row 2  C    p       p       p   c   c     p = chunk of C's prompt
///   row 3       .   .   .   .   .   .   .     free row
/// ```
///
/// Slot `i` is row `i` of the batch KV cache, and `N` is `maxConcurrent`. A slot goes through these
/// phases:
/// - Admission: a queued request takes the next free slot.
/// - Prefill: the prompt up to its last token is prefilled in chunks, oldest request first. While
///   other rows are decoding, chunks of at most ``SchedulerKV/prefillChunk`` tokens alternate with
///   decode steps; otherwise chunks are up to ``SchedulerKV/idlePrefillChunk`` tokens.
/// - Decode: each step advances every prefilled row by one token, at its own position. A row's first
///   step processes its last prompt token and samples its first token (C at iteration 6 above).
/// - Retirement: a row finishes on EOS, a stop sequence, `maxTokens`, running out of positions, or
///   cancellation, and its slot takes the next queued request.
///
/// A request is cancelled by `BatchRequest.cancellation` or by dropping its stream, and leaves at the
/// next step.
package final class BatchScheduler: RequestSink, @unchecked Sendable {
    private let kv: any SchedulerKV
    package let maxConcurrent: Int
    /// Requests allowed to wait beyond the free rows, or `nil` for no limit. At least 0.
    private let maxQueued: Int?

    /// True when more than one request can decode at a time (`maxConcurrent > 1`).
    package var supportsContinuousBatching: Bool { maxConcurrent > 1 }

    /// Requests holding a row, and requests waiting for one, at the time of the call.
    package var occupancy: (active: Int, queued: Int) {
        state.withLock { ($0.occupiedRows, $0.queue.count) }
    }

    /// Bytes held by the shared batch session: its KV cache and persistent states for all rows, and its
    /// logits buffer, estimated from their shapes and element sizes. The batch-1 scratch session of a
    /// row that is still prefilling isn't included; it holds up to that prompt's KV until the prompt
    /// is copied into its row.
    package var estimatedSessionBytes: (states: Int, logitsScratch: Int) { kv.estimatedBytes }

    private struct Pending {
        let request: BatchRequest
        let continuation: AsyncThrowingStream<BatchSinkEvent, Error>.Continuation
    }
    private struct State {
        var queue: [Pending] = []
        /// Rows held by admitted requests.
        var occupiedRows = 0
        /// Scoring jobs waiting for the pump. Each `score(_:)` call appends its jobs shortest prompt
        /// first.
        var scoring: [ScoringJob] = []
        var pumping = false

        /// Removes cancelled requests from the queue and finishes their streams.
        mutating func dropCancelledRequests() {
            queue.removeAll { pending in
                guard pending.request.cancellation.isCancelled else { return false }
                pending.continuation.finish()
                return true
            }
        }
    }
    private let state = Mutex(State())
    /// Lock held by the pump during each iteration, and by an `ExclusiveEngineLease` until it is
    /// released.
    private let engineLock = AsyncFIFOLock()
    /// Runs the scoring batches. Only set when the KV is a contiguous batch session.
    private let scorer: Scorer?

    private struct ScoringJob {
        let request: ScoreRequest
        let continuation: AsyncThrowingStream<LogProbabilities, Error>.Continuation
    }

    /// Creates a scheduler over `engine` with a new `maxConcurrent`-row session.
    ///
    /// - Parameters:
    ///   - maxConcurrent: Rows in the batch, from 1 to the graph's batch capacity. Above 1,
    ///     `engine.supportsBatching` must be true.
    ///   - maxQueued: Requests allowed to wait beyond the free rows, at least 0. `nil` means no limit.
    ///   - prefillChunk: Maximum number of prompt tokens prefilled between two decode steps, at least 1.
    ///     Smaller values keep the pauses in running streams shorter; larger values give a new request
    ///     its first token sooner. When no row is decoding, the engine's prefill chunk size is used.
    ///
    /// Throws if the session can't be created, or if the model can't prefill a single row of it
    /// (vision-language models, assets without a prefill graph, and models with conv, recurrent or
    /// sliding-window state). Out-of-range values fail a precondition.
    package convenience init(
        engine: CoreAISequentialEngine, maxConcurrent: Int, maxQueued: Int? = nil,
        prefillChunk: Int = BatchScheduler.defaultPrefillChunk
    ) throws {
        precondition(
            maxConcurrent == 1 || engine.supportsBatching,
            "maxConcurrent \(maxConcurrent) needs a graph with an attn_mask input and a batch dimension above 1")
        let session = try engine.makeSessionState(batchSize: maxConcurrent)
        try engine.requireRowWritableCache(session)
        self.init(
            decodeStep: engine, session: session, maxConcurrent: maxConcurrent, maxQueued: maxQueued,
            maxPositions: engine.config.maxContextLength, prefillChunk: prefillChunk,
            idlePrefillChunk: engine.config.prefillChunkSize)
    }

    /// Default number of prompt tokens prefilled between two decode steps.
    package static let defaultPrefillChunk = 512

    convenience init(
        decodeStep: any BatchedDecodeStep, session: GenerationSessionState, maxConcurrent: Int,
        maxQueued: Int? = nil, maxPositions: Int = .max, prefillChunk: Int = BatchScheduler.defaultPrefillChunk,
        idlePrefillChunk: Int? = nil
    ) {
        precondition(maxConcurrent >= 1, "maxConcurrent must be >= 1")
        precondition(
            maxConcurrent <= decodeStep.maxBatchCapacity,
            "maxConcurrent \(maxConcurrent) exceeds the graph's batch capacity \(decodeStep.maxBatchCapacity)")
        self.init(
            kv: ContiguousSchedulerKV(
                step: decodeStep, session: session, rows: maxConcurrent, prefillChunk: prefillChunk,
                idlePrefillChunk: idlePrefillChunk ?? prefillChunk, maxPositions: maxPositions),
            maxQueued: maxQueued,
            scorer: Scorer(decodeStep: decodeStep, maxConcurrent: maxConcurrent, maxPositions: maxPositions))
    }

    /// Creates a scheduler over the rows of `kv`. Scoring needs all-position logits from a contiguous
    /// batch session, so it's only available when `scorer` is passed.
    init(kv: any SchedulerKV, maxQueued: Int? = nil, scorer: Scorer? = nil) {
        precondition(kv.rowCount >= 1, "a scheduler needs a row")
        precondition(kv.prefillChunk >= 1, "prefill chunks must be >= 1 token")
        precondition(kv.idlePrefillChunk >= kv.prefillChunk, "idlePrefillChunk must be >= prefillChunk")
        precondition((maxQueued ?? 0) >= 0, "maxQueued must be >= 0")
        self.kv = kv
        self.maxConcurrent = kv.rowCount
        self.maxQueued = maxQueued
        self.scorer = scorer
    }

    // MARK: - RequestSink

    /// Queues `request` and returns its event stream.
    ///
    /// Throws `invalidArgument` when the prompt is empty, and `InferenceRuntimeError.queueFull` when
    /// admitted plus queued requests would exceed `maxConcurrent + maxQueued`. With
    /// `maxConcurrent == 2` and `maxQueued == 1`, a request throws while three others are admitted or
    /// queued. With `maxQueued == nil`, it throws only for an empty prompt.
    package func submit(_ request: BatchRequest) throws -> AsyncThrowingStream<BatchSinkEvent, Error> {
        try enqueue([request])[0]
    }

    /// Enqueues `requests` as one unit and starts the pump if none is running. Returns one stream per
    /// request, in order. Throws `invalidArgument` when a prompt is empty and `queueFull` when the
    /// queue cannot take all of `requests`. A throw enqueues nothing.
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
            continuation.onTermination = { [cancellation = request.cancellation] termination in
                if case .cancelled = termination { cancellation.cancel() }
            }
            streams.append(stream)
            pending.append(Pending(request: request, continuation: continuation))
        }
        let shouldStart = try state.withLock { s -> Bool in
            s.dropCancelledRequests()
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
    ///
    /// `pendingPrompt` is the part of the prompt that hasn't been prefilled yet, which never includes
    /// the last token. That one is `lastToken`, which the row's first decode step processes and samples
    /// from.
    private struct ActiveRow {
        let continuation: AsyncThrowingStream<BatchSinkEvent, Error>.Continuation
        var adapter: BatchSessionAdapter
        var lastToken: Int32
        var cursor: Int
        var pendingPrompt: ArraySlice<Int32> = []
        /// Order of admission, so the oldest prompt is prefilled first.
        var admission = 0
    }

    /// Scheduling loop, run by at most one task at a time. Each iteration does the following:
    /// 1. Run every waiting scoring job.
    /// 2. Admit queued requests into free slots, and retire cancelled rows.
    /// 3. If no slot is live, loop again when requests or scoring jobs are waiting, otherwise clear
    ///    `pumping` and return.
    /// 4. Prefill the next chunk of the oldest prompt that's still prefilling, unless the previous
    ///    iteration also prefilled and a row is waiting to decode. This makes prefill chunks and decode
    ///    steps alternate.
    /// 5. Otherwise, run one decode step over the rows that have finished prefilling, then sample,
    ///    emit or retire each of them.
    ///
    /// If a decode step throws, every active row, queued request and queued scoring job fails with
    /// that error, and every row is released back to the KV.
    private func pump() async {
        var active: [ActiveRow?] = Array(repeating: nil, count: maxConcurrent)
        // Prefill chunks and decode steps alternate so that neither starves the other.
        var lastWasPrefill = false
        var admissions = 0
        do {
            while true {
                await engineLock.lock()
                defer { engineLock.unlock() }
                await runPendingScoring()
                admit(into: &active, admissions: &admissions)
                for slot in active.indices where active[slot]?.adapter.request.cancellation.isCancelled == true {
                    retire(slot, active: &active)
                }

                let liveSlots = active.indices.filter { active[$0] != nil }
                if liveSlots.isEmpty {
                    // Requests and scoring jobs enqueued while `pumping` is set start no pump.
                    let more = state.withLock { s -> Bool in
                        if s.queue.isEmpty && s.scoring.isEmpty {
                            s.pumping = false
                            return false
                        }
                        return true
                    }
                    if more { continue }
                    return
                }

                let prefilling = liveSlots.filter { !active[$0]!.pendingPrompt.isEmpty }
                let decoding = liveSlots.filter { active[$0]!.pendingPrompt.isEmpty }
                if let slot = prefilling.min(by: { active[$0]!.admission < active[$1]!.admission }),
                    decoding.isEmpty || !lastWasPrefill
                {
                    lastWasPrefill = true
                    // When no row is decoding there's no stream to keep moving, so use larger chunks.
                    let piece = active[slot]!.pendingPrompt.prefix(
                        decoding.isEmpty ? kv.idlePrefillChunk : kv.prefillChunk)
                    do {
                        try await kv.prefill(row: slot, tokens: piece)
                        active[slot]!.pendingPrompt.removeFirst(piece.count)
                    } catch {
                        retire(slot, active: &active, throwing: error)
                    }
                    continue
                }
                lastWasPrefill = false

                let steps = try await kv.decode(
                    decoding.map {
                        SchedulerKVRow(row: $0, token: active[$0]!.lastToken, cursor: active[$0]!.cursor)
                    })
                guard steps.count == decoding.count else {
                    throw InferenceRuntimeError.invalidState(
                        "a decode step of \(decoding.count) rows returned \(steps.count) results")
                }
                for (slot, step) in zip(decoding, steps) {
                    switch step {
                    case .logits(let logits):
                        active[slot]!.cursor += 1
                        stepSlot(slot, logits: logits, active: &active)
                    case .rowFull:
                        CLILogger.log(
                            "row \(slot) has no positions left at \(active[slot]!.cursor); ending its request as "
                                + "maxTokens", component: "BatchScheduler")
                        active[slot]!.continuation.yield(.finished(.maxTokens))
                        retire(slot, active: &active)
                    }
                }
            }
        } catch {
            let resident = active.compactMap { $0?.continuation }
            for slot in active.indices where active[slot] != nil { kv.retire(row: slot) }
            let (orphaned, orphanedScoring) = state.withLock { s -> ([Pending], [ScoringJob]) in
                let waiting = (s.queue, s.scoring)
                s.queue.removeAll()
                s.scoring.removeAll()
                s.occupiedRows = 0
                s.pumping = false
                return waiting
            }
            for continuation in resident { continuation.finish(throwing: error) }
            for pending in orphaned { pending.continuation.finish(throwing: error) }
            for job in orphanedScoring { job.continuation.finish(throwing: error) }
        }
    }

    /// Assigns queued requests to free slots. Each prompt is prefilled later, between decode steps. If
    /// reserving a row fails, only that request's stream fails.
    private func admit(into active: inout [ActiveRow?], admissions: inout Int) {
        var free = active.indices.filter { active[$0] == nil }
        guard !free.isEmpty else { return }
        let taken = state.withLock { s -> [Pending] in
            s.dropCancelledRequests()
            let taken = Array(s.queue.prefix(free.count))
            s.queue.removeFirst(taken.count)
            s.occupiedRows += taken.count
            return taken
        }
        for pending in taken {
            let slot = free.removeFirst()
            let prompt = pending.request.promptTokens
            do {
                try kv.reserve(row: slot, prompt: prompt)
                active[slot] = ActiveRow(
                    continuation: pending.continuation, adapter: BatchSessionAdapter(request: pending.request),
                    lastToken: prompt[prompt.count - 1], cursor: prompt.count - 1,
                    pendingPrompt: prompt[..<(prompt.count - 1)], admission: admissions)
                admissions += 1
            } catch {
                state.withLock { $0.occupiedRows -= 1 }
                pending.continuation.finish(throwing: error)
                free.insert(slot, at: 0)
            }
        }
    }

    /// Samples one token for `slot`, emits it, and retires the row when its request is done.
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
            retire(slot, active: &active)
        }
    }

    /// Releases `slot`'s row, then finishes its stream, throwing `error` if one is given. The row is
    /// released first, so anyone who sees the stream end also sees the row as free.
    private func retire(_ slot: Int, active: inout [ActiveRow?], throwing error: Error? = nil) {
        let continuation = active[slot]!.continuation
        active[slot] = nil
        state.withLock { $0.occupiedRows -= 1 }
        kv.retire(row: slot)
        if let error { continuation.finish(throwing: error) } else { continuation.finish() }
    }

    // MARK: - Scoring

    /// Scores each request's targets. Returns one `LogProbabilities` per request, in request order,
    /// with one entry per position: the target's log-probability and the top `topK` alternatives.
    ///
    /// Throws `invalidArgument` when a request has no tokens or its `targets` and `tokens` counts
    /// differ. Rethrows the error of a failed scoring batch or decode step.
    ///
    /// Requests run between decode steps, grouped by prompt length into batches of up to
    /// `maxConcurrent` requests, each batch on a new session. Decode rows pause during a scoring
    /// batch and keep their KV state. A failed scoring batch fails only its own requests.
    package func score(_ requests: [ScoreRequest]) async throws -> [LogProbabilities] {
        guard scorer != nil else {
            throw InferenceRuntimeError.invalidArgument(
                "scoring needs every position's logits, which only a contiguous batch session gives")
        }
        if requests.contains(where: { $0.tokens.isEmpty || $0.targets.count != $0.tokens.count }) {
            throw InferenceRuntimeError.invalidArgument(
                "a scoring request needs a non-empty prompt and one target per token")
        }
        let order = requests.indices.sorted { requests[$0].tokens.count < requests[$1].tokens.count }
        var jobs: [ScoringJob] = []
        var streams = [AsyncThrowingStream<LogProbabilities, Error>?](repeating: nil, count: requests.count)
        for index in order {
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: LogProbabilities.self)
            jobs.append(ScoringJob(request: requests[index], continuation: continuation))
            streams[index] = stream
        }
        let shouldStart = state.withLock { s -> Bool in
            s.scoring.append(contentsOf: jobs)
            if s.pumping { return false }
            s.pumping = true
            return true
        }
        if shouldStart { Task { await self.pump() } }

        var results: [LogProbabilities] = []
        results.reserveCapacity(streams.count)
        for stream in streams {
            var iterator = stream!.makeAsyncIterator()
            guard let result = try await iterator.next() else {
                throw InferenceRuntimeError.invalidState("scoring ended without a result")
            }
            results.append(result)
        }
        return results
    }

    /// Runs every waiting scoring job, in batches of up to `maxConcurrent` sized by
    /// `Scorer.batchCount(_:limit:)`. Call only while holding `engineLock`.
    private func runPendingScoring() async {
        // `score(_:)` never queues a job when there's no scorer.
        guard let scorer else { return }
        while true {
            let jobs = state.withLock { s -> [ScoringJob] in
                let count = Scorer.batchCount(
                    s.scoring.map(\.request.tokens.count), limit: scorer.maxConcurrent)
                let taken = Array(s.scoring.prefix(count))
                s.scoring.removeFirst(taken.count)
                return taken
            }
            if jobs.isEmpty { return }
            do {
                let results = try await scorer.scoreBatch(jobs.map(\.request))
                for (job, result) in zip(jobs, results) {
                    job.continuation.yield(result)
                    job.continuation.finish()
                }
            } catch {
                for job in jobs { job.continuation.finish(throwing: error) }
            }
        }
    }

    // MARK: - Exclusive engine access

    /// Waits behind the current scheduling step and any earlier lease, then returns a lease that
    /// gives the caller sole use of the engine until `release()`. Resident rows keep their slots and
    /// resume decoding after release.
    ///
    /// A request that runs directly on the scheduler's engine holds a lease for as long as it runs:
    ///
    /// ```swift
    /// let lease = await scheduler.acquireExclusiveAccess()
    /// defer { lease.release() }
    /// let sequence = try await engine.generate(
    ///     with: promptTokens, samplingConfiguration: sampling, inferenceOptions: options)
    /// ```
    package func acquireExclusiveAccess() async -> ExclusiveEngineLease {
        await engineLock.lock()
        return ExclusiveEngineLease(lock: engineLock)
    }

    // MARK: - Closed batch

    /// Submits `requests` together, waits for all of them, and returns one `RowResult` per request, in
    /// request order. Can be called while other requests stream. Throws `invalidArgument` when a
    /// prompt is empty, `queueFull` when the queue cannot take all of `requests`, and any error that
    /// ends a request's stream.
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

/// A batched backend shared by many requests, each with its own event stream. Wrap it in a
/// `SchedulerSession`, one per request, to use it as an `InferenceEngine`.
package protocol RequestSink: Sendable {
    /// Maximum number of requests decoded in one batch. At least 1.
    var maxConcurrent: Int { get }

    /// Queues `request` and returns its event stream. Throws when the sink rejects the request, such
    /// as `InferenceRuntimeError.queueFull` when its queue is full.
    func submit(_ request: BatchRequest) throws -> AsyncThrowingStream<BatchSinkEvent, Error>
}

/// A generation request for a `RequestSink`: prompt, sampling and stop conditions.
package struct BatchRequest: Sendable {
    /// List of prompt tokens, cannot be empty.
    let promptTokens: [Int32]
    let sampling: SamplingConfiguration
    let eosTokenIds: Set<Int32>
    let maxTokens: Int
    let stopSequences: StopSequences
    /// Converts a matched stop sequence's tokens to the text reported in `StopReason.stopSequence`.
    /// Defaults to an empty string.
    let decodeStopText: @Sendable ([Int32]) -> String
    /// Cancels the request when set. Dropping the request's stream also cancels it.
    let cancellation = RequestCancellation()

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

/// A flag that cancels a `BatchRequest`. Safe to set from any task.
final class RequestCancellation: Sendable {
    private let cancelled = Atomic<Bool>(false)

    var isCancelled: Bool { cancelled.load(ordering: .relaxed) }

    func cancel() { cancelled.store(true, ordering: .relaxed) }
}

/// An event on a request's stream from a `RequestSink`. A stream yields `.token` events, then one
/// `.finished`, or ends by throwing.
package enum BatchSinkEvent: Sendable {
    /// A sampled token and, when the sink provides them, its logits. `BatchScheduler` always sends
    /// `nil` logits.
    case token(Int32, logits: [LogitsScalarType]?)
    /// The reason the request stopped. Always the last event.
    case finished(StopReason)
}

/// A lease on sole use of a `BatchScheduler`'s engine, returned by `acquireExclusiveAccess()`.
/// Calls to `release()` after the first do nothing. Deinit releases a lease that is still held.
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

/// An async lock granted to waiters in arrival order. `unlock()` hands the lock directly to the next
/// waiter, and a task that unlocks and locks again queues behind the existing waiters.
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

/// Generated tokens and stop reason for one request from `BatchScheduler.run(_:)`.
struct RowResult: Sendable {
    let tokens: [Int32]
    let stopReason: StopReason
}
