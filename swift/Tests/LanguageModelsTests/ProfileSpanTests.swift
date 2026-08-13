// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

@Suite("ProfileSpan")
@MainActor
struct ProfileSpanTests {
    /// Isolated storage for this test suite (not shared with other tests)
    let storage: StatsStorage

    init() {
        // Create isolated instances for test isolation
        self.storage = StatsStorage(forTesting: ())
    }

    // MARK: - Basic Lifecycle Tests

    @Test("Span records stats with correct duration")
    func spanRecordsStats() {
        var span = InstrumentsProfiler.beginPrompt(tokens: 10)
        Thread.sleep(forTimeInterval: 0.01)  // 10ms
        span.end(storingInto: storage)

        let stats = storage.stats(for: .prompt)
        #expect(stats != nil)
        #expect(stats?.count == 1)
        #expect((stats?.totalSeconds ?? 0) > 0.005)  // At least 5ms
        #expect((stats?.totalSeconds ?? 1) < 1.0)  // Upper bound: under 1s
    }

    @Test("Multiple spans aggregate stats correctly")
    func multipleSpansAggregateStats() {
        // Record 3 prompt spans
        for _ in 0..<3 {
            var span = InstrumentsProfiler.beginPrompt(tokens: 10)
            Thread.sleep(forTimeInterval: 0.005)  // 5ms
            span.end(storingInto: storage)
        }

        let stats = storage.stats(for: .prompt)
        #expect(stats != nil)
        #expect(stats?.count == 3)
        #expect((stats?.totalSeconds ?? 0) > 0.010)  // At least 10ms total
        #expect((stats?.avgSeconds ?? 0) > 0.003)  // At least 3ms avg
    }

    @Test("Different categories are tracked separately")
    func differentCategoriesTrackedSeparately() {
        var promptSpan = InstrumentsProfiler.beginPrompt(tokens: 10)
        Thread.sleep(forTimeInterval: 0.005)
        promptSpan.end(storingInto: storage)

        var warmupSpan = InstrumentsProfiler.beginWarmup(step: 999)
        Thread.sleep(forTimeInterval: 0.003)
        warmupSpan.end(storingInto: storage)

        #expect(storage.count(for: .prompt) == 1)
        #expect(storage.count(for: .warmup) == 1)
    }

    // MARK: - Aggregate Statistics Tests

    @Test("Min and max are tracked correctly", .enabled(if: !CIEnvironment.isVM))
    func minMaxTracking() {
        var span1 = InstrumentsProfiler.beginExtend(step: 1)
        Thread.sleep(forTimeInterval: 0.001)  // 1ms
        span1.end(storingInto: storage)

        var span2 = InstrumentsProfiler.beginExtend(step: 2)
        Thread.sleep(forTimeInterval: 0.010)  // 10ms
        span2.end(storingInto: storage)

        var span3 = InstrumentsProfiler.beginExtend(step: 3)
        Thread.sleep(forTimeInterval: 0.005)  // 5ms
        span3.end(storingInto: storage)

        let stats = storage.stats(for: .extend)
        #expect(stats != nil)
        #expect(stats?.count == 3)
        #expect((stats?.minSeconds ?? 1) < 0.003)  // Min < 3ms
        #expect((stats?.maxSeconds ?? 0) > 0.009)  // Max > 9ms
    }

    @Test("Average is calculated correctly")
    func averageCalculation() {
        for _ in 0..<10 {
            var span = InstrumentsProfiler.beginSample()
            Thread.sleep(forTimeInterval: 0.002)  // ~2ms each
            span.end(storingInto: storage)
        }

        let stats = storage.stats(for: .sample)
        #expect(stats != nil)
        #expect(stats?.count == 10)

        // Average should be close to total/count
        let expectedAvg = (stats?.totalSeconds ?? 0) / Double(stats?.count ?? 1)
        let actualAvg = stats?.avgSeconds ?? 0
        #expect(abs(actualAvg - expectedAvg) < 0.00001)
    }

    // MARK: - Factory Method Tests

    @Test("beginPrompt factory method works")
    func promptFactoryMethod() {
        var span = InstrumentsProfiler.beginPrompt(tokens: 100, engine: "MLX")
        span.end(storingInto: storage)

        let count = storage.count(for: .prompt)
        #expect(count == 1)
    }

    @Test("beginExtend factory method works")
    func extendFactoryMethod() {
        var span = InstrumentsProfiler.beginExtend(step: 5, tokens: 1)
        span.end(storingInto: storage)

        let count = storage.count(for: .extend)
        #expect(count == 1)
    }

    @Test("beginWarmup factory method works")
    func warmupFactoryMethod() {
        var span = InstrumentsProfiler.beginWarmup(step: 1)
        span.end(storingInto: storage)

        let count = storage.count(for: .warmup)
        #expect(count == 1)
    }

    @Test("beginSample factory method works")
    func sampleFactoryMethod() {
        var span = InstrumentsProfiler.beginSample(strategy: "greedy", temperature: 0.7)
        span.end(storingInto: storage)

        let count = storage.count(for: .sample)
        #expect(count == 1)
    }

    @Test("beginModelLoad factory method works")
    func modelLoadFactoryMethod() {
        var span = InstrumentsProfiler.beginModelLoad(name: "Qwen2.5")
        span.end(storingInto: storage)

        let count = storage.count(for: .modelLoad)
        #expect(count == 1)
    }

    @Test("beginTokenizerLoad factory method works")
    func tokenizerLoadFactoryMethod() {
        var span = InstrumentsProfiler.beginTokenizerLoad(id: "gpt2")
        span.end(storingInto: storage)

        let count = storage.count(for: .tokenizerLoad)
        #expect(count == 1)
    }

    @Test("beginTokenization factory method works")
    func tokenizationFactoryMethod() {
        var span = InstrumentsProfiler.beginTokenization(inputLength: 256)
        span.end(storingInto: storage)

        let count = storage.count(for: .tokenization)
        #expect(count == 1)
    }

    // MARK: - Thread Safety Tests

    @Test("Concurrent span recording is thread-safe")
    func concurrentSpanRecording() async {
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    var span = InstrumentsProfiler.beginExtend(step: i)
                    try? await Task.sleep(for: .milliseconds(1))
                    await span.end(storingInto: storage)
                }
            }
        }

        let stats = storage.stats(for: .extend)
        #expect(stats?.count == 10)
    }

    // MARK: - Reset Tests

    @Test("Reset clears all stats")
    func resetClearsAllStats() {
        var span1 = InstrumentsProfiler.beginPrompt(tokens: 10)
        span1.end(storingInto: storage)

        var span2 = InstrumentsProfiler.beginExtend(step: 1)
        span2.end(storingInto: storage)

        #expect(storage.count(for: .prompt) == 1)
        #expect(storage.count(for: .extend) == 1)

        storage.reset()

        #expect(storage.count(for: .prompt) == 0)
        #expect(storage.count(for: .extend) == 0)
    }

    // MARK: - Edge Cases

    @Test("Zero duration span works")
    func zeroDurationSpan() {
        var span = InstrumentsProfiler.beginPrompt(tokens: 1)
        // End immediately without sleep
        span.end(storingInto: storage)

        let stats = storage.stats(for: .prompt)
        #expect(stats != nil)
        #expect(stats?.count == 1)
        // Duration should be very small but non-negative
        #expect((stats?.totalSeconds ?? -1) >= 0)
    }

    @Test("Stats for unused category returns nil")
    func statsForUnusedCategory() {
        let stats = storage.stats(for: .warmup)
        #expect(stats == nil)

        let count = storage.count(for: .warmup)
        #expect(count == 0)

        let duration = storage.totalDuration(for: .warmup)
        #expect(duration == 0.0)
    }
}
