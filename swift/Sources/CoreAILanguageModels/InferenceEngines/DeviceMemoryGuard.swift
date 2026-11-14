// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Darwin
import Foundation
import Metal
import Synchronization

/// Watches what this process holds against what the device recommends, and says when a server
/// should stop admitting requests.
///
/// Past the device's recommendation (`recommendedMaxWorkingSetSize`), Metal can drop command buffers
/// out of memory while the calls that issued them return anyway: nothing fails and the GPU computes
/// wrong answers. Two figures are checked, each against its own line, on every request:
/// - Metal's allocated size, which follows the sessions' KV caches as they grow (a 4-row batch
///   session on a 0.6B model went from 140 MB to 4.7 GB over four 3,500-token prompts), against
///   ``defaultTripFraction`` of the recommendation.
/// - What the process holds resident: its physical footprint, plus its memory-mapped files, against the
///   recommendation itself. The footprint leaves out clean file-backed pages, and the asset's weights
///   are memory-mapped: a 3.8 GB 4-bit 7B model served from a 504 MB footprint, with 3.9 GB of mapped
///   file pages beside it.
///
/// Neither figure only grows: a KV cache that doubles holds its old and new buffers together while it
/// copies (the pipelined engine keeps the old ones until the GPU is done with them), and scratch
/// comes and goes. So the guard refuses only while a figure is over its line,
/// and admits again once it is back under. The requests already running finish; they can still take
/// a figure past the line, which is what the margin under the recommendation is for.
///
/// The guard watches only from ``arm()``, called once the model and its sessions are loaded. A
/// figure already past its line then gets a line ``pastArmMargin`` of the recommendation above where
/// it stood: a server that ran that way before still serves, and still refuses once it grows.
package final class DeviceMemoryGuard: Sendable {
    /// Fraction of the device's recommendation, in Metal's allocated size, at which the guard refuses,
    /// leaving room for the requests already running.
    package static let defaultTripFraction = 0.98

    /// Growth past where a figure stood at ``arm()``, as a fraction of the recommendation, at which a
    /// figure that was already past its line refuses.
    package static let pastArmMargin = 0.02

    /// What Metal has allocated on the device and what it recommends, and what this process holds.
    package struct Snapshot: Sendable {
        /// Metal's `currentAllocatedSize`: the asset's buffers, the sessions and the framework's
        /// scratch alike.
        package let allocated: Int
        /// Metal's `recommendedMaxWorkingSetSize`; zero when the device gives none.
        package let recommended: Int
        /// The process's physical footprint, which leaves out clean file-backed pages.
        package let footprint: Int
        /// The process's resident memory-mapped file pages, such as the asset's weights.
        package let mapped: Int

        package init(allocated: Int, recommended: Int, footprint: Int, mapped: Int = 0) {
            self.allocated = allocated
            self.recommended = recommended
            self.footprint = footprint
            self.mapped = mapped
        }

        /// The footprint and the mapped files together: what the process holds resident.
        package var resident: Int { footprint + mapped }

        package var allocatedFraction: Double {
            recommended > 0 ? Double(allocated) / Double(recommended) : 0
        }

        package var description: String {
            "Metal \(Self.gib(allocated)) allocated of \(Self.gib(recommended)) recommended, "
                + "footprint \(Self.gib(footprint)) and mapped files \(Self.gib(mapped))"
        }

        static func gib(_ bytes: Int) -> String {
            String(format: "%.1f GiB", Double(bytes) / 1_073_741_824)
        }
    }

    /// The bytes at which each figure refuses, set by ``arm()``; nil before it.
    private struct Lines {
        var allocated: Int
        var resident: Int
    }

    private struct Watch {
        var lines: Lines?
        /// Whether the last check refused, so each change is logged once.
        var refusing = false
    }

    private let measure: @Sendable () -> Snapshot
    private let tripFraction: Double
    private let watch = Mutex(Watch())

    package convenience init(device: any MTLDevice, tripFraction: Double = DeviceMemoryGuard.defaultTripFraction) {
        self.init(tripFraction: tripFraction) {
            let process = DeviceMemoryGuard.processMemory
            return Snapshot(
                allocated: device.currentAllocatedSize, recommended: Int(device.recommendedMaxWorkingSetSize),
                footprint: process.footprint, mapped: process.mapped)
        }
    }

    /// A guard that reads its figures from `measure`, so its decisions can be tested without a device.
    package init(
        tripFraction: Double = DeviceMemoryGuard.defaultTripFraction, measure: @escaping @Sendable () -> Snapshot
    ) {
        self.measure = measure
        self.tripFraction = tripFraction
    }

    /// Starts watching, against where the process stands now. Does nothing when the device gives no
    /// recommendation, so the guard never refuses.
    package func arm() {
        let start = measure()
        guard start.recommended > 0 else { return }
        let margin = Int(Double(start.recommended) * Self.pastArmMargin)
        let metalLine = Int(Double(start.recommended) * tripFraction)
        let lines = Lines(
            allocated: start.allocated < metalLine ? metalLine : start.allocated + margin,
            resident: start.resident < start.recommended ? start.recommended : start.resident + margin)
        watch.withLock { $0.lines = lines }
    }

    package var snapshot: Snapshot { measure() }

    /// This process's physical footprint and resident memory-mapped file pages (`external`), in bytes;
    /// zeros when they cannot be read.
    package static var processMemory: (footprint: Int, mapped: Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? (Int(info.phys_footprint), Int(info.external)) : (0, 0)
    }

    /// Measures once, and returns the figures with why this process should admit nothing more right
    /// now, or nil while every figure is under its line or the guard is not armed. Logs each change
    /// between refusing and admitting once.
    package func check() -> (snapshot: Snapshot, refusal: String?) {
        let now = measure()
        let (refusal, changed) = watch.withLock { watch -> (String?, Bool) in
            guard let lines = watch.lines else { return (nil, false) }
            let refusal: String?
            if now.allocated >= lines.allocated {
                refusal =
                    "Metal has allocated \(Snapshot.gib(now.allocated)) of the \(Snapshot.gib(now.recommended)) "
                    + "this device recommends, past which the GPU can drop work without an error"
            } else if now.resident >= lines.resident {
                refusal =
                    "this process holds \(Snapshot.gib(now.resident)) (footprint \(Snapshot.gib(now.footprint)), "
                    + "mapped files \(Snapshot.gib(now.mapped))), past the \(Snapshot.gib(now.recommended)) "
                    + "this device recommends"
            } else {
                refusal = nil
            }
            defer { watch.refusing = refusal != nil }
            return (refusal, watch.refusing != (refusal != nil))
        }
        if changed {
            CLILogger.log(
                refusal.map { "\($0). Refusing new requests until it is back under." }
                    ?? "back under the device's recommendation (\(now.description)). Admitting requests again.",
                component: "Memory", level: 0)
        }
        return (now, refusal)
    }

    /// Why this process should admit nothing more right now, or nil. See ``check()``.
    package func refusalReason() -> String? { check().refusal }

    /// Reports where the process stands after `stage` (such as load or warmup): a warning when it is
    /// near or past the device's recommendation, naming `advice` as what would give memory back.
    package func report(after stage: String, advice: String) {
        let now = measure()
        guard now.recommended > 0 else { return }
        if now.allocatedFraction >= 1 || now.resident >= now.recommended {
            CLILogger.log(
                "after \(stage): \(now.description) — past what this device recommends. The GPU may drop "
                    + "work without an error. \(advice)",
                component: "Memory", level: 0)
        } else if now.allocatedFraction >= tripFraction {
            CLILogger.log(
                "after \(stage): \(now.description) — \(Int(now.allocatedFraction * 100))% of this device's "
                    + "recommendation, with no headroom left. \(advice)",
                component: "Memory", level: 0)
        } else {
            CLILogger.log("after \(stage): \(now.description)", component: "Memory")
        }
    }
}
