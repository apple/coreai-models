// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Mock Clock for Testing

/// A mock clock that allows controlling time in tests.
final class MockClock: TimingClock, @unchecked Sendable {
    private var currentInstant: ContinuousClock.Instant

    init() {
        self.currentInstant = ContinuousClock.now
    }

    var now: ContinuousClock.Instant {
        currentInstant
    }

    /// Advances time by the specified duration.
    func advance(by duration: Duration) {
        currentInstant = currentInstant.advanced(by: duration)
    }
}

@Suite("PerformanceMetrics", .serialized)
@MainActor
struct PerformanceMetricsTests {
    @Test("Reset clears all timing and token counts")
    func resetClearsMetrics() {
        let metrics = PerformanceMetrics.shared
        metrics.reset()

        metrics.startOverallTiming()
        metrics.recordPromptTokens(100)
        metrics.recordGeneratedTokens(50)
        metrics.reset()

        #expect(metrics.totalTime == 0)
        #expect(metrics.generatedTokenCount == 0)
        #expect(metrics.modelLoadTime == 0)
    }

    @Test("Throughput returns zero when time is zero")
    func zeroTimeReturnsZeroThroughput() {
        let metrics = PerformanceMetrics.shared
        metrics.reset()

        metrics.recordPromptTokens(100)
        metrics.recordGeneratedTokens(50)

        #expect(metrics.promptThroughput == 0)
        #expect(metrics.generationThroughput == 0)
    }

    @Test("Mock clock enables deterministic overall timing tests")
    func mockClockDeterministicTiming() {
        let mockClock = MockClock()
        let metrics = PerformanceMetrics(clock: mockClock)

        metrics.startOverallTiming()
        mockClock.advance(by: .seconds(2))
        metrics.endOverallTiming()

        // Exactly 2 seconds because we control the clock
        #expect(metrics.totalTime == 2.0)
    }

    @Test("ProfileSpan populates StatsStorage and PerformanceMetrics can read it")
    func profileSpanPopulatesStatsStorage() {
        // Create isolated storage and profiler for this test
        let storage = StatsStorage(forTesting: ())

        // Use ProfileSpan to record model load timing
        var span = InstrumentsProfiler.beginModelLoad(name: "test-model")
        // Small delay to ensure measurable time
        Thread.sleep(forTimeInterval: 0.01)  // 10ms
        span.end(storingInto: storage)

        // Check the stats directly on the isolated storage
        let stats = storage.stats(for: .modelLoad)
        #expect(stats != nil)
        #expect((stats?.totalSeconds ?? 0) > 0)
        #expect((stats?.totalSeconds ?? 0) >= 0.01)
        #expect((stats?.totalSeconds ?? 1) < 1.0)  // Sanity check
    }

    @Test("Generation throughput calculated from ProfileSpan data")
    func generationThroughputFromProfileSpan() {
        // Create isolated storage and profiler for this test
        let storage = StatsStorage(forTesting: ())

        // Simulate extend spans (generation)
        for step in 0..<10 {
            var span = InstrumentsProfiler.beginExtend(step: step, tokens: step + 1)
            Thread.sleep(forTimeInterval: 0.001)  // 1ms per token
            span.end(storingInto: storage)
        }

        // Check stats directly on isolated storage
        let stats = storage.stats(for: .extend)
        #expect(stats != nil)
        #expect(stats?.count == 10)
        #expect((stats?.totalSeconds ?? 0) > 0)
    }
}
