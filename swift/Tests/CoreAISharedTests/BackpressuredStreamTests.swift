// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIShared

@Suite("backpressuredStream")
struct BackpressuredStreamTests {
    /// Records how far the producer has run and whether it ever unwound, so a test can read
    /// both while the producer is parked. `finished` is what distinguishes a producer that
    /// was resumed and cancelled from one still parked on a permit.
    private actor Progress {
        private(set) var produced = 0
        private(set) var finished = false
        func record() { produced += 1 }
        func finish() { finished = true }
    }

    /// A producer that counts up to `limit`, reporting progress and its own unwind.
    ///
    /// `defer` cannot await, so the unwind is recorded in a `catch` before rethrowing.
    private func countingProducer(
        upTo limit: Int, reporting progress: Progress
    ) -> @Sendable (@Sendable @escaping (Int) async -> Void) async throws -> Void {
        { yield in
            do {
                for value in 0..<limit {
                    try Task.checkCancellation()
                    await progress.record()
                    await yield(value)
                }
            } catch {
                await progress.finish()
                throw error
            }
        }
    }

    @Test("Every element arrives, in order")
    func deliversEverything() async throws {
        let stream = backpressuredStream(depth: 2) { yield in
            for value in 0..<20 { await yield(value) }
        }
        var received: [Int] = []
        for try await value in stream { received.append(value) }
        #expect(received == Array(0..<20))
    }

    @Test("A producer error reaches the consumer")
    func propagatesError() async throws {
        struct Boom: Error {}
        let stream = backpressuredStream(depth: 2) { yield in
            await yield(1)
            throw Boom()
        }
        var received: [Int] = []
        await #expect(throws: Boom.self) {
            for try await value in stream { received.append(value) }
        }
        #expect(received == [1])
    }

    @Test("A depth below one is clamped rather than deadlocking")
    func clampsDepthToOne() async throws {
        // `AsyncPermits.init` clamps with `max(1, count)`. Without it a zero depth would hand
        // out no permits and the first yield would park forever.
        let stream = backpressuredStream(depth: 0) { yield in
            for value in 0..<5 { await yield(value) }
        }
        var received: [Int] = []
        for try await value in stream { received.append(value) }
        #expect(received == Array(0..<5))
    }

    @Test("A producer that yields nothing completes the stream")
    func emptyProducerCompletes() async throws {
        let stream: AsyncThrowingStream<Int, Error> = backpressuredStream(depth: 2) { _ in }
        var received: [Int] = []
        for try await value in stream { received.append(value) }
        #expect(received.isEmpty)
    }
}
