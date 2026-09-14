// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Testing

/// Unit tests for the async `RequestQueue` semaphore that replaced the binary
/// tryAcquire/release busy-gate in the llm-server.
struct RequestQueueTests {
    // MARK: - Helpers

    /// Poll `condition` until it becomes true or `timeout` elapses. Used to wait
    /// for waiters to enqueue without relying on fixed sleeps.
    private func waitUntil(
        timeout: Duration = .seconds(2),
        _ condition: @Sendable () -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(2))
        }
        Issue.record("waitUntil timed out")
    }

    /// Records the order in which waiters wake up.
    private actor Recorder {
        private(set) var values: [Int] = []
        func record(_ value: Int) { values.append(value) }
    }

    // MARK: - Tests

    @Test("First acquire succeeds immediately and marks the queue active")
    func acquireImmediate() async throws {
        let queue = RequestQueue(maxDepth: 4)
        #expect(queue.isActive == false)

        let permit = try await queue.acquire()
        #expect(queue.isActive == true)
        #expect(queue.queuedCount == 0)

        permit.release()
        #expect(queue.isActive == false)
    }

    @Test("release with no waiters clears the active flag")
    func releaseWithoutWaiters() async throws {
        let queue = RequestQueue(maxDepth: 4)
        let permit = try await queue.acquire()
        #expect(queue.isActive == true)

        permit.release()
        #expect(queue.isActive == false)
        #expect(queue.queuedCount == 0)
    }

    @Test("A second concurrent acquire waits, then proceeds after release")
    func secondAcquireWaits() async throws {
        let queue = RequestQueue(maxDepth: 4)
        let permit0 = try await queue.acquire()  // holds the single active slot

        let waiter = Task { try await queue.acquire() }
        try await waitUntil { queue.queuedCount == 1 }  // confirm it blocked

        permit0.release()  // hand the slot to the waiter
        let permit1 = try await waiter.value
        #expect(queue.isActive == true)
        #expect(queue.queuedCount == 0)

        permit1.release()
        #expect(queue.isActive == false)
    }

    @Test("Waiters wake in FIFO order")
    func fifoWakeupOrder() async throws {
        let queue = RequestQueue(maxDepth: 8)
        let recorder = Recorder()
        let permit0 = try await queue.acquire()  // active slot held by the test

        // Enqueue three waiters in a deterministic order: launch each only after
        // the previous one is observed in the queue. Each waiter records its id
        // and releases so the next in line wakes.
        var tasks: [Task<Void, Error>] = []
        for index in 0..<3 {
            let task = Task {
                let permit = try await queue.acquire()
                await recorder.record(index)
                permit.release()
            }
            tasks.append(task)
            try await waitUntil { queue.queuedCount == index + 1 }
        }

        permit0.release()  // trigger the cascade: waiter 0 -> 1 -> 2
        for task in tasks { try await task.value }

        #expect(await recorder.values == [0, 1, 2])
        #expect(queue.isActive == false)
    }

    @Test("acquire rejects with queueFull once the queue reaches maxDepth")
    func rejectsAtMaxDepth() async throws {
        let queue = RequestQueue(maxDepth: 2)
        let permit0 = try await queue.acquire()  // active slot

        let w1 = Task { try await queue.acquire() }
        try await waitUntil { queue.queuedCount == 1 }
        let w2 = Task { try await queue.acquire() }
        try await waitUntil { queue.queuedCount == 2 }

        // Queue is now at maxDepth (2 waiters); the next acquire must reject.
        do {
            _ = try await queue.acquire()
            Issue.record("expected queueFull to be thrown")
        } catch let error as ServerError {
            guard case .queueFull(let depth) = error else {
                Issue.record("expected .queueFull, got \(error)")
                return
            }
            #expect(depth == 2)
        }

        // Drain the two queued waiters so no continuation is leaked.
        permit0.release()
        let permit1 = try await w1.value
        permit1.release()
        let permit2 = try await w2.value
        permit2.release()
        #expect(queue.isActive == false)
    }

    @Test("maxDepth 0 rejects the second concurrent request immediately")
    func maxDepthZeroRejects() async throws {
        let queue = RequestQueue(maxDepth: 0)
        let permit0 = try await queue.acquire()  // active slot, no queuing allowed

        await #expect(throws: ServerError.self) {
            _ = try await queue.acquire()
        }
        #expect(queue.queuedCount == 0)

        permit0.release()
        let permit1 = try await queue.acquire()  // slot free again
        #expect(queue.isActive == true)
        permit1.release()
    }

    @Test("Negative maxDepth is clamped to zero")
    func negativeMaxDepthClamped() async throws {
        let queue = RequestQueue(maxDepth: -5)
        #expect(queue.maxDepth == 0)

        let permit0 = try await queue.acquire()
        await #expect(throws: ServerError.self) {
            _ = try await queue.acquire()
        }
        permit0.release()
    }

    @Test("Permit release is idempotent")
    func permitReleaseIdempotent() async throws {
        let queue = RequestQueue(maxDepth: 4)
        let permit = try await queue.acquire()

        permit.release()
        permit.release()  // second release is a no-op
        #expect(queue.isActive == false)

        let permit2 = try await queue.acquire()
        #expect(queue.isActive == true)
        permit2.release()
    }

    @Test(
        "Concurrent releases hand the slot back exactly once",
        .timeLimit(.minutes(1)))
    func concurrentReleasesReturnSlotOnce() async throws {
        let queue = RequestQueue(maxDepth: 4)
        let permit = try await queue.acquire()

        // Enqueue exactly one waiter so a double-return would be observable: if
        // the slot is handed back more than once, the second hand-back would wake
        // this waiter (or corrupt isActive) even though nobody re-acquired.
        let waiter = Task { try await queue.acquire() }
        try await waitUntil { queue.queuedCount == 1 }

        // Race many release() calls on the same permit. The atomic once-guard must
        // let exactly one win, so the slot is handed to the single waiter once.
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask { permit.release() }
            }
        }

        // The single waiter got the slot exactly once; nothing is left queued.
        let permit1 = try await waiter.value
        #expect(queue.isActive == true)
        #expect(queue.queuedCount == 0)

        permit1.release()
        #expect(queue.isActive == false)

        // Slot is usable again — accounting was not corrupted by the race.
        let permit2 = try await queue.acquire()
        #expect(queue.isActive == true)
        permit2.release()
        #expect(queue.isActive == false)
    }

    @Test(
        "Cancelling a queued waiter removes it and frees the slot",
        .timeLimit(.minutes(1)))
    func cancelledWaiterIsRemoved() async throws {
        let queue = RequestQueue(maxDepth: 4)
        let permit0 = try await queue.acquire()  // test holds the active slot

        let waiter = Task { try await queue.acquire() }  // must suspend
        try await waitUntil { queue.queuedCount == 1 }

        waiter.cancel()  // cancel while suspended in acquire()

        // The waiter is removed promptly and its acquire throws CancellationError.
        try await waitUntil { queue.queuedCount == 0 }
        #expect(queue.queuedCount == 0)
        await #expect(throws: CancellationError.self) {
            _ = try await waiter.value
        }

        // The slot was not handed to the cancelled waiter: releasing frees it and
        // a fresh acquire still succeeds.
        permit0.release()
        #expect(queue.isActive == false)
        let permit1 = try await queue.acquire()
        #expect(queue.isActive == true)
        permit1.release()
    }
}
