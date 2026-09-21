// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// An async counting semaphore.
///
/// `drain` exists for stream teardown: a producer parked in `acquire` is suspended on a
/// continuation, which cancellation alone would leave hanging.
actor AsyncPermits {
    private var available: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var drained = false

    init(_ count: Int) {
        self.available = max(1, count)
    }

    func acquire() async {
        if drained { return }
        if available > 0 {
            available -= 1
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        guard !drained else { return }
        if waiters.isEmpty {
            available += 1
        } else {
            waiters.removeFirst().resume()
        }
    }

    /// Resume every parked caller and make later `acquire` calls return immediately.
    func drain() {
        drained = true
        let parked = waiters
        waiters.removeAll()
        for waiter in parked { waiter.resume() }
    }
}

/// Holds a stream iterator so an `unfolding` closure can advance it across calls.
///
/// `@unchecked Sendable` rests on `AsyncThrowingStream(unfolding:)` calling its closure
/// serially, so only one `next` is ever in flight.
private final class IteratorBox<Element: Sendable>: @unchecked Sendable {
    private var iterator: AsyncThrowingStream<Element, Error>.AsyncIterator

    init(_ stream: AsyncThrowingStream<Element, Error>) {
        self.iterator = stream.makeAsyncIterator()
    }

    func next() async throws -> Element? {
        try await iterator.next()
    }
}

/// An `AsyncThrowingStream` that holds its producer to at most `depth` unconsumed elements.
///
/// `AsyncThrowingStream`'s own buffering is unbounded, which for large elements such as
/// decoded video frames lets a slow consumer grow the backlog without limit. Here the
/// producer takes a permit before each yield and the consumer returns one after each pull,
/// so nothing is dropped and the backlog is capped.
///
/// - Parameters:
///   - depth: Maximum unconsumed elements. Clamped to at least 1.
///   - produce: The producer body. Its `yield` argument suspends while the consumer is behind.
public func backpressuredStream<Element: Sendable>(
    depth: Int,
    produce:
        @escaping @Sendable (_ yield: @Sendable @escaping (Element) async -> Void)
        async throws -> Void
) -> AsyncThrowingStream<Element, Error> {
    let permits = AsyncPermits(depth)
    let (inner, continuation) = AsyncThrowingStream<Element, Error>.makeStream()

    let task = Task {
        do {
            try await produce { element in
                await permits.acquire()
                continuation.yield(element)
            }
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }
    continuation.onTermination = { _ in
        task.cancel()
        Task { await permits.drain() }
    }

    let box = IteratorBox(inner)
    return AsyncThrowingStream {
        do {
            guard let element = try await box.next() else {
                await permits.drain()
                return nil
            }
            await permits.release()
            return element
        } catch {
            await permits.drain()
            throw error
        }
    }
}
