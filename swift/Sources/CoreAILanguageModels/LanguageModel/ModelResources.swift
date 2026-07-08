// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Synchronization

// MARK: - Resource management

/// Owns the load / unload lifecycle of a single inference engine.
final class ModelResources: ResourceManaging {
    private struct State {
        var loaded: (any InferenceEngine)?
        var inFlight: Task<any InferenceEngine, any Error>?
        var activeBorrows: Int = 0
        /// Set when `unloadResources()` is called mid-borrow.
        var unloadPending: Bool = false
        /// Bumped by every teardown; lets an in-flight load detect it was cancelled.
        var generation: Int = 0
    }

    private let state = Mutex(State())
    private let loader: @Sendable () async throws -> any InferenceEngine

    /// Production initializer: loads via `CoreAIRunner` from the configuration.
    init(configuration: CoreAILanguageModel.CoreAIExecutor.Configuration) {
        self.loader = { try await ModelResources.loadEngine(configuration) }
    }

    /// Testable initializer: inject a custom engine loader.
    init(loader: @escaping @Sendable () async throws -> any InferenceEngine) {
        self.loader = loader
    }

    var isLoaded: Bool { state.withLock { $0.loaded != nil } }

    /// `supportsLogits` of the resident engine, or `nil` when nothing is loaded.
    /// Used only for best-effort capability reporting before a load.
    var loadedEngineSupportsLogits: Bool? {
        state.withLock { $0.loaded?.supportsLogits }
    }

    /// Returns the engine, loading it on first use. Concurrent callers share one
    /// load; later callers get the warmed engine instantly.
    func engine() async throws -> any InferenceEngine {
        if let loaded = state.withLock({ $0.loaded }) { return loaded }

        let (task, generation): (Task<any InferenceEngine, any Error>, Int) = state.withLock {
            current in
            if let inFlight = current.inFlight { return (inFlight, current.generation) }
            let task = Task { try await self.loader() }
            current.inFlight = task
            return (task, current.generation)
        }

        do {
            let engine = try await task.value
            state.withLock { current in
                // Only commit if we weren't unloaded mid-flight.
                guard current.generation == generation else { return }
                current.loaded = engine
                current.inFlight = nil
            }
            return engine
        } catch {
            state.withLock { current in
                guard current.generation == generation else { return }
                // Don't cache failures — drop the task so the next caller retries.
                current.inFlight = nil
            }
            throw error
        }
    }

    /// Borrows the engine for the duration of `body`, counting it as an active
    /// use. A concurrent `unloadResources()` defers its teardown until the last
    /// borrow returns, so the engine is never freed mid-generation.
    func withEngine<T>(
        _ body: (any InferenceEngine) async throws -> T
    ) async throws -> T {
        let engine = try await engine()
        state.withLock { $0.activeBorrows += 1 }
        defer {
            let shouldTeardown = state.withLock { current -> Bool in
                current.activeBorrows -= 1
                guard current.activeBorrows == 0, current.unloadPending else { return false }
                current.unloadPending = false
                return true
            }
            if shouldTeardown { teardown() }
        }
        return try await body(engine)
    }

    // MARK: - ResourceManaging

    /// Explicitly loads + warms the engine. No-op if already loaded.
    func loadResources() async throws {
        _ = try await engine()
    }

    func unloadResources() {
        let deferred = state.withLock { current -> Bool in
            guard current.activeBorrows > 0 else { return false }
            current.unloadPending = true
            return true
        }
        if deferred { return }
        teardown()
    }

    /// Drops the cached engine and cancels any in-flight load
    private func teardown() {
        state.withLock { current in
            current.inFlight?.cancel()
            current.inFlight = nil
            current.loaded = nil
            current.generation &+= 1
        }
    }

    // MARK: - Shared registry

    /// Weak box so the registry doesn't keep a `ModelResources` (and its engine)
    /// alive after every owning model/executor has been released.
    private final class WeakBox {
        weak var value: ModelResources?
        init(_ value: ModelResources) { self.value = value }
    }

    private static let registry =
        Mutex<[CoreAILanguageModel.CoreAIExecutor.Configuration: WeakBox]>([:])

    static func shared(
        for configuration: CoreAILanguageModel.CoreAIExecutor.Configuration
    ) -> ModelResources {
        registry.withLock { table in
            if let existing = table[configuration]?.value { return existing }
            let resources = ModelResources(configuration: configuration)
            table[configuration] = WeakBox(resources)
            table = table.filter { $0.value.value != nil }
            return resources
        }
    }

    private static func loadEngine(
        _ configuration: CoreAILanguageModel.CoreAIExecutor.Configuration
    ) async throws -> any InferenceEngine {
        let modelLoadSpan = InstrumentsProfiler.beginModelLoad(name: configuration.modelIdentifier)
        let runner = try CoreAIRunner(
            contentsOf: configuration.url,
            variant: configuration.variant,
            kvCacheStrategy: configuration.kvCacheStrategy
        )
        let engine = try await runner.makeInferenceEngine()
        modelLoadSpan.end()

        try await engine.warmup(queryLength: 1, sampling: nil)
        return engine
    }
}
