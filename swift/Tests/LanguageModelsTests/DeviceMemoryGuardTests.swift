// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Metal
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

    @Test("Under the line, armed, it admits")
    func underTheLineAdmits() throws {
        let device = try #require(device)
        let memoryGuard = DeviceMemoryGuard(device: device, tripFraction: 1_000)
        memoryGuard.arm()
        #expect(memoryGuard.refusalReason() == nil)
    }
}

#endif
