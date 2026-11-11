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
    private let device = MTLCreateSystemDefaultDevice()

    @Test("Reads the device's figures and this process's footprint")
    func snapshot() throws {
        let device = try #require(device)
        let now = DeviceMemoryGuard(device: device).snapshot
        #expect(now.recommended > 0)
        #expect(now.footprint > 0)
        #expect(now.allocated >= 0)
    }

    @Test("Refuses nothing before it is armed")
    func unarmedRefusesNothing() throws {
        let device = try #require(device)
        // A trip fraction of zero would trip on any allocation at all, once armed.
        let memoryGuard = DeviceMemoryGuard(device: device, tripFraction: 0)
        #expect(memoryGuard.refusalReason() == nil)
        #expect(memoryGuard.lastRefusal == nil)
    }

    @Test("A figure already past the line when armed is not refused")
    func pastAtArmIsNotRefused() throws {
        let device = try #require(device)
        // Every process has allocated at least zero, so Metal's figure is past a zero line at arm;
        // the footprint is under the device's recommendation in a test process.
        let memoryGuard = DeviceMemoryGuard(device: device, tripFraction: 0)
        memoryGuard.arm()
        #expect(memoryGuard.refusalReason() == nil)
    }

    // MARK: - Decisions, on figures the test sets

    /// Figures a test sets, read by a guard as if from the device. Recommended is 100 throughout.
    private final class Figures: Sendable {
        private let value = Mutex((allocated: 10, footprint: 10))
        func set(allocated: Int? = nil, footprint: Int? = nil) {
            value.withLock {
                if let allocated { $0.allocated = allocated }
                if let footprint { $0.footprint = footprint }
            }
        }
        var snapshot: DeviceMemoryGuard.Snapshot {
            value.withLock {
                DeviceMemoryGuard.Snapshot(allocated: $0.allocated, recommended: 100, footprint: $0.footprint)
            }
        }
    }

    private func armed(_ figures: Figures) -> DeviceMemoryGuard {
        let memoryGuard = DeviceMemoryGuard(tripFraction: 0.98) { figures.snapshot }
        memoryGuard.arm()
        return memoryGuard
    }

    @Test("Metal's figure crossing the line refuses until a restart, even once it falls back")
    func metalLatches() {
        let figures = Figures()
        let memoryGuard = armed(figures)
        #expect(memoryGuard.refusalReason() == nil)
        figures.set(allocated: 98)
        let reason = memoryGuard.refusalReason()
        #expect(reason?.contains("Metal has allocated") == true)
        figures.set(allocated: 10)
        #expect(memoryGuard.refusalReason() == reason)
        #expect(memoryGuard.lastRefusal == reason)
    }

    @Test("The footprint refuses only while it is over, and admits again once it is back under")
    func footprintRecovers() {
        let figures = Figures()
        let memoryGuard = armed(figures)
        figures.set(footprint: 100)
        let reason = memoryGuard.refusalReason()
        #expect(reason?.contains("footprint") == true)
        // /ready reports the same text the refused request got.
        #expect(memoryGuard.lastRefusal == reason)
        figures.set(footprint: 99)
        #expect(memoryGuard.refusalReason() == nil)
        #expect(memoryGuard.lastRefusal == nil)
    }

    @Test("A figure already past the line at arm never refuses; the other still does")
    func pastAtArmIsIgnored() {
        let figures = Figures()
        figures.set(footprint: 150)
        let memoryGuard = armed(figures)
        #expect(memoryGuard.refusalReason() == nil)
        figures.set(allocated: 99)
        #expect(memoryGuard.refusalReason()?.contains("Metal has allocated") == true)
    }

    @Test("Nothing refuses before the guard is armed")
    func unarmedOnFigures() {
        let figures = Figures()
        figures.set(allocated: 99, footprint: 200)
        let memoryGuard = DeviceMemoryGuard(tripFraction: 0.98) { figures.snapshot }
        #expect(memoryGuard.refusalReason() == nil)
    }

    @Test("Under the line, armed, it admits")
    func underTheLineAdmits() throws {
        let device = try #require(device)
        let memoryGuard = DeviceMemoryGuard(device: device, tripFraction: 1_000)
        memoryGuard.arm()
        #expect(memoryGuard.refusalReason() == nil)
    }
}

#endif
