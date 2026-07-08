// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization
import Testing

@testable import CoreAILanguageModels

/// Unit tests for the resource lifecycle (`loadResources`/`unloadResources`,
/// load-once, retry-on-failure, and deferred teardown during an active borrow)
/// using an injected loader — no real engine or weights needed.
@Suite("ModelResources lifecycle")
struct ModelResourcesTests {
    private struct LoadFailure: Error {}

    @Test("Lazy: nothing loads until first use")
    func lazyDefersLoad() {
        let calls = Mutex(0)
        let resources = ModelResources {
            calls.withLock { $0 += 1 }
            return MockEngine()
        }
        #expect(resources.isLoaded == false)
        #expect(calls.withLock { $0 } == 0)
    }

    @Test("load() marks loaded and runs the loader exactly once")
    func loadOnce() async throws {
        let calls = Mutex(0)
        let resources = ModelResources {
            calls.withLock { $0 += 1 }
            return MockEngine()
        }

        let first = try await resources.engine()
        let second = try await resources.engine()

        #expect(resources.isLoaded)
        #expect(calls.withLock { $0 } == 1)
        // Same cached instance returned both times.
        #expect((first as? MockEngine) === (second as? MockEngine))
    }

    @Test("unload() frees the engine; next use reloads")
    func unloadThenReload() async throws {
        let calls = Mutex(0)
        let resources = ModelResources {
            calls.withLock { $0 += 1 }
            return MockEngine()
        }

        _ = try await resources.engine()
        #expect(resources.isLoaded)

        resources.unloadResources()
        #expect(resources.isLoaded == false)

        _ = try await resources.engine()
        #expect(resources.isLoaded)
        #expect(calls.withLock { $0 } == 2)
    }

    @Test("Failures aren't cached — the next call retries")
    func retriesOnFailure() async throws {
        let calls = Mutex(0)
        let resources = ModelResources {
            let attempt = calls.withLock { count -> Int in
                count += 1
                return count
            }
            if attempt == 1 { throw LoadFailure() }
            return MockEngine()
        }

        await #expect(throws: LoadFailure.self) {
            _ = try await resources.engine()
        }
        #expect(resources.isLoaded == false)

        // Second attempt succeeds.
        _ = try await resources.engine()
        #expect(resources.isLoaded)
        #expect(calls.withLock { $0 } == 2)
    }

    @Test("Concurrent callers share a single load")
    func concurrentCallersLoadOnce() async throws {
        let calls = Mutex(0)
        let resources = ModelResources {
            calls.withLock { $0 += 1 }
            // Widen the race window so all callers pile onto the same in-flight load.
            try? await Task.sleep(for: .milliseconds(20))
            return MockEngine()
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<20 {
                group.addTask { _ = try await resources.engine() }
            }
            try await group.waitForAll()
        }

        #expect(calls.withLock { $0 } == 1)
        #expect(resources.isLoaded)
    }

    @Test("unloadResources during an active borrow defers teardown until it finishes")
    func unloadDeferredDuringActiveBorrow() async throws {
        let resources = ModelResources { MockEngine() }
        let inside = Mutex(false)
        let release = Mutex(false)

        // Hold a borrow open until the test releases it.
        async let borrow: Void = resources.withEngine { _ in
            inside.withLock { $0 = true }
            while !release.withLock({ $0 }) { await Task.yield() }
        }

        // Once the body is running, the borrow has been counted.
        while !inside.withLock({ $0 }) { await Task.yield() }

        // Unload requested mid-borrow must be deferred, not applied.
        resources.unloadResources()
        #expect(resources.isLoaded)

        // Let the borrow finish; the deferred teardown then runs.
        release.withLock { $0 = true }
        try await borrow
        #expect(resources.isLoaded == false)
    }
}
