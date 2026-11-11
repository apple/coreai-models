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
/// Past the device's recommendation, Metal drops command buffers out of memory and the calls that
/// issued them return anyway: nothing fails, the GPU computes wrong answers and the machine swaps.
/// The two figures behave differently, so the guard treats them differently:
/// - Metal's allocated size only grows once the sessions are allocated, and past the line the GPU
///   computes wrong answers. Once it crosses, the guard refuses until a restart.
/// - The physical footprint can rise and fall (a large prompt, the tokenizer, compiler scratch),
///   and past the line the machine swaps. The guard refuses only while it is over, checking again
///   on every call.
///
/// The requests already running finish; new ones are refused.
///
/// The guard watches only from ``arm()``, called once the model and its sessions are loaded. A
/// figure already past the line then is reported, not refused: the process served that way before
/// the guard existed.
package final class DeviceMemoryGuard: Sendable {
    /// Fraction of the device's recommendation, in Metal's allocated size, at which the guard trips,
    /// leaving room for the requests already running.
    package static let defaultTripFraction = 0.98

    /// What Metal has allocated on the device and what it recommends, and what the kernel charges
    /// this process.
    package struct Snapshot: Sendable {
        /// Metal's `currentAllocatedSize`: the asset's buffers, the sessions and the framework's
        /// scratch alike. Past the recommendation the GPU computes wrong answers.
        package let allocated: Int
        /// Metal's `recommendedMaxWorkingSetSize`; zero when the device gives none.
        package let recommended: Int
        /// The process's physical footprint. Past the recommendation the machine swaps.
        package let footprint: Int

        package var allocatedFraction: Double {
            recommended > 0 ? Double(allocated) / Double(recommended) : 0
        }

        package var description: String {
            "Metal \(Self.gb(allocated)) allocated of \(Self.gb(recommended)) recommended, "
                + "footprint \(Self.gb(footprint))"
        }

        static func gb(_ bytes: Int) -> String {
            String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        }
    }

    private struct Watch {
        /// Whether each figure was under the line at ``arm()``; only those can refuse.
        var allocated = false
        var footprint = false
        /// Why Metal's figure tripped the guard, which then refuses until a restart.
        var tripped: String?
        /// Whether the footprint was over the line at the last check, so each crossing is reported once.
        var footprintOver = false
    }

    private let device: any MTLDevice
    private let tripFraction: Double
    private let watch = Mutex(Watch())

    package init(device: any MTLDevice, tripFraction: Double = DeviceMemoryGuard.defaultTripFraction) {
        self.device = device
        self.tripFraction = tripFraction
    }

    /// Starts watching, against where the process stands now.
    package func arm() {
        let start = snapshot
        watch.withLock {
            $0.allocated = start.recommended > 0 && start.allocatedFraction < tripFraction
            $0.footprint = start.recommended > 0 && start.footprint < start.recommended
        }
    }

    package var snapshot: Snapshot { Self.snapshot(of: device) }

    private static func snapshot(of device: any MTLDevice) -> Snapshot {
        Snapshot(
            allocated: device.currentAllocatedSize,
            recommended: Int(device.recommendedMaxWorkingSetSize),
            footprint: physicalFootprint)
    }

    /// What the kernel charges this process, in bytes; zero when it cannot be read.
    package static var physicalFootprint: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    /// Why this process should admit nothing more right now, or nil while it is under the line or the
    /// guard is not armed. Once Metal's figure has crossed, it stays non-nil.
    package func refusalReason() -> String? {
        watch.withLock { watch in
            if let tripped = watch.tripped { return tripped }
            guard watch.allocated || watch.footprint else { return nil }
            let now = snapshot
            if watch.allocated, now.allocatedFraction >= tripFraction {
                let reason =
                    "Metal has allocated \(Snapshot.gb(now.allocated)) of the "
                    + "\(Snapshot.gb(now.recommended)) this device recommends, past which the GPU "
                    + "computes wrong answers without an error; restart to serve more"
                watch.tripped = reason
                CLILogger.report("[Memory] \(reason). Refusing new requests.")
                return reason
            }
            let over = watch.footprint && now.footprint >= now.recommended
            defer { watch.footprintOver = over }
            guard over else {
                if watch.footprintOver {
                    CLILogger.report("[Memory] footprint back under the recommendation. Admitting requests again.")
                }
                return nil
            }
            let reason =
                "this process's footprint is \(Snapshot.gb(now.footprint)), past the "
                + "\(Snapshot.gb(now.recommended)) this device recommends; try again once running requests finish"
            if !watch.footprintOver { CLILogger.report("[Memory] \(reason). Refusing new requests.") }
            return reason
        }
    }

    /// Why the guard is refusing as of its last check, without checking again; nil while it is not.
    package var lastRefusal: String? {
        watch.withLock { watch in
            if let tripped = watch.tripped { return tripped }
            return watch.footprintOver ? "this process's footprint is past what the device recommends" : nil
        }
    }

    /// Reports where the process stands after `stage` (such as load or warmup): a warning when it is
    /// near or past the device's recommendation, naming `advice` as what would give memory back.
    package func report(after stage: String, advice: String) {
        let now = snapshot
        guard now.recommended > 0 else { return }
        if now.allocatedFraction >= 1 || now.footprint >= now.recommended {
            CLILogger.report(
                "[Memory] after \(stage): \(now.description) — past what this device recommends. "
                    + "The GPU may drop work without an error and the machine may swap. \(advice)")
        } else if now.allocatedFraction >= tripFraction {
            CLILogger.report(
                "[Memory] after \(stage): \(now.description) — "
                    + "\(Int(now.allocatedFraction * 100))% of this device's recommendation, with no "
                    + "headroom left. \(advice)")
        } else {
            CLILogger.log("after \(stage): \(now.description)", component: "Memory")
        }
    }
}
