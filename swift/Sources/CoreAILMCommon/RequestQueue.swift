// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization

// MARK: - Request Queue

/// Async semaphore gating concurrency at one in-flight request. A second request
/// waits in a FIFO queue instead of getting an immediate 429; it is rejected only
/// once `maxDepth` requests are already waiting.
///
/// `acquire()` returns a `QueuePermit` that frees the slot exactly once. Callers
/// hold the permit for the lifetime of the request and let it release on scope
/// exit (or hand it to a streaming body closure); the permit also releases on
/// deallocation, so the slot is never leaked if a caller drops it.
public final class RequestQueue: Sendable {
    public let maxDepth: Int
    private let state = Mutex<QueueState>(QueueState())

    private struct QueueState {
        var isActive: Bool = false
        // Waiters carry a token so the cancellation handler can find and remove
        // its own continuation (CheckedContinuation is not Equatable).
        var waiters: [(id: UInt64, cont: CheckedContinuation<Void, any Error>)] = []
        var nextID: UInt64 = 0
    }

    public init(maxDepth: Int) {
        self.maxDepth = max(0, maxDepth)
    }

    public var depth: Int {
        state.withLock { $0.waiters.count + ($0.isActive ? 1 : 0) }
    }

    public var queuedCount: Int {
        state.withLock { $0.waiters.count }
    }

    public var isActive: Bool {
        state.withLock { $0.isActive }
    }

    /// Acquire the exclusive slot, waiting in FIFO order if it is held. Throws
    /// `ServerError.queueFull` when `maxDepth` waiters are already queued, or
    /// `CancellationError` if the awaiting task is cancelled while queued.
    public func acquire() async throws -> QueuePermit {
        let needsWait: Bool = state.withLock { s in
            if !s.isActive {
                s.isActive = true
                return false
            }
            return true
        }
        if !needsWait { return QueuePermit(self) }

        // Reserve a token before suspending so the cancellation handler below can
        // match this waiter even if it runs before the continuation is installed.
        let id = state.withLock { s -> UInt64 in
            defer { s.nextID &+= 1 }
            return s.nextID
        }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
                enum Action { case acquired, rejected, queued, cancelled }
                let action: Action = state.withLock { s -> Action in
                    if Task.isCancelled { return .cancelled }
                    if !s.isActive {
                        s.isActive = true
                        return .acquired
                    }
                    if s.waiters.count >= maxDepth {
                        return .rejected
                    }
                    s.waiters.append((id: id, cont: cont))
                    return .queued
                }
                switch action {
                case .acquired:
                    cont.resume()
                case .rejected:
                    cont.resume(throwing: ServerError.queueFull(depth: maxDepth))
                case .cancelled:
                    cont.resume(throwing: CancellationError())
                case .queued:
                    break
                }
            }
        } onCancel: {
            // Remove our still-queued waiter and resume it throwing. The shared
            // Mutex serializes this against release(), so a given waiter is popped
            // by exactly one of the two and resumed exactly once.
            let cont: CheckedContinuation<Void, any Error>? = state.withLock { s in
                guard let idx = s.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return s.waiters.remove(at: idx).cont
            }
            cont?.resume(throwing: CancellationError())
        }
        return QueuePermit(self)
    }

    /// Hand the slot to the next FIFO waiter, or clear the active flag. Internal;
    /// callers release through `QueuePermit`.
    fileprivate func release() {
        let next: CheckedContinuation<Void, any Error>? = state.withLock { s in
            if !s.waiters.isEmpty {
                return s.waiters.removeFirst().cont
            }
            s.isActive = false
            return nil
        }
        next?.resume()
    }
}

// MARK: - Queue Permit

/// A held `RequestQueue` slot. Releases exactly once — on the first `release()`
/// call or, as a safety net, on deallocation (e.g. if a streaming response body
/// is dropped by the server before its writer closure ever runs).
public final class QueuePermit: Sendable {
    private let queue: RequestQueue
    private let released = Atomic<Bool>(false)

    fileprivate init(_ queue: RequestQueue) {
        self.queue = queue
    }

    public func release() {
        // Flip false -> true atomically; exactly one caller wins the exchange and
        // hands the slot back. `.relaxed` suffices: the exchange's atomicity (not
        // its ordering) picks the single winner, and queue.release() takes the
        // RequestQueue Mutex, which provides the actual cross-thread handoff.
        let (won, _) = released.compareExchange(
            expected: false, desired: true, ordering: .relaxed)
        if won { queue.release() }
    }

    deinit { release() }
}

// MARK: - Server Errors

public enum ServerError: Error, LocalizedError {
    case badRequest(String)
    case queueFull(depth: Int)

    public var isBadRequest: Bool {
        if case .badRequest = self { return true }
        return false
    }

    public var errorDescription: String? {
        switch self {
        case .badRequest(let msg): return msg
        case .queueFull(let depth): return "Queue full (depth: \(depth))"
        }
    }
}
