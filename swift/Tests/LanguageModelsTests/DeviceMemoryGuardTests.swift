// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Metal
import Synchronization
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("DeviceMemoryGuard")
struct DeviceMemoryGuardTests {
    @Test("Reads the device's figures and this process's footprint")
    func snapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let now = DeviceMemoryGuard(device: device).snapshot
        #expect(now.recommended > 0)
        #expect(now.footprint > 0)
        #expect(now.allocated >= 0)
    }

    // MARK: - Decisions, on figures the test sets

    /// Figures a test sets, read by a guard as if from the device. Recommended is 100 unless set.
    private final class Figures: Sendable {
        private let value = Mutex((allocated: 10, footprint: 10, recommended: 100))
        func set(allocated: Int? = nil, footprint: Int? = nil, recommended: Int? = nil) {
            value.withLock {
                if let allocated { $0.allocated = allocated }
                if let footprint { $0.footprint = footprint }
                if let recommended { $0.recommended = recommended }
            }
        }
        var snapshot: DeviceMemoryGuard.Snapshot {
            value.withLock {
                DeviceMemoryGuard.Snapshot(
                    allocated: $0.allocated, recommended: $0.recommended, footprint: $0.footprint)
            }
        }
    }

    private func guarding(_ figures: Figures, armed: Bool = true) -> DeviceMemoryGuard {
        let memoryGuard = DeviceMemoryGuard(tripFraction: 0.98) { figures.snapshot }
        if armed { memoryGuard.arm() }
        return memoryGuard
    }

    @Test("Metal's figure refuses while it is over 98% of the recommendation, and admits again once under")
    func metalRefusesWhileOver() {
        let figures = Figures()
        let memoryGuard = guarding(figures)
        #expect(memoryGuard.refusalReason() == nil)
        figures.set(allocated: 98)
        #expect(memoryGuard.refusalReason()?.contains("Metal has allocated") == true)
        // A KV cache that doubled holds both sizes only until its copy is done.
        figures.set(allocated: 60)
        #expect(memoryGuard.refusalReason() == nil)
    }

    @Test("The footprint refuses while it is at or over the recommendation, and admits again once under")
    func footprintRefusesWhileOver() {
        let figures = Figures()
        let memoryGuard = guarding(figures)
        figures.set(footprint: 100)
        #expect(memoryGuard.refusalReason()?.contains("footprint") == true)
        figures.set(footprint: 99)
        #expect(memoryGuard.refusalReason() == nil)
    }

    @Test("check() reports the refusal from the same measurement as its figures")
    func checkIsOneMeasurement() {
        let figures = Figures()
        let memoryGuard = guarding(figures)
        figures.set(allocated: 99)
        let over = memoryGuard.check()
        #expect(over.snapshot.allocated == 99)
        #expect(over.refusal != nil)
        figures.set(allocated: 20)
        let under = memoryGuard.check()
        #expect(under.snapshot.allocated == 20)
        #expect(under.refusal == nil)
    }

    @Test("A figure already past its line at arm refuses once it grows a margin past where it stood")
    func pastAtArmGetsAHigherLine() {
        let figures = Figures()
        figures.set(allocated: 99, footprint: 120)
        let memoryGuard = guarding(figures)
        // Where they stood at arm: served before, so still served.
        #expect(memoryGuard.refusalReason() == nil)
        // Growth inside the 2% margin.
        figures.set(allocated: 100, footprint: 121)
        #expect(memoryGuard.refusalReason() == nil)
        figures.set(allocated: 101)
        #expect(memoryGuard.refusalReason()?.contains("Metal has allocated") == true)
        figures.set(allocated: 99, footprint: 122)
        #expect(memoryGuard.refusalReason()?.contains("footprint") == true)
    }

    @Test("Nothing refuses before the guard is armed")
    func unarmedRefusesNothing() {
        let figures = Figures()
        figures.set(allocated: 99, footprint: 200)
        #expect(guarding(figures, armed: false).refusalReason() == nil)
    }

    @Test("A device that gives no recommendation never refuses")
    func noRecommendationNeverRefuses() {
        let figures = Figures()
        figures.set(recommended: 0)
        let memoryGuard = guarding(figures)
        figures.set(allocated: 1_000_000, footprint: 1_000_000)
        #expect(memoryGuard.refusalReason() == nil)
    }
}

#endif
