// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Darwin
import Metal

/// Memory readings for the current process, sampled when called.
package enum ProcessMemory {
    private static let device = MTLCreateSystemDefaultDevice()

    /// The process's physical footprint in bytes: the memory the system charges it, and what memory
    /// pressure decisions use. Returns 0 when the kernel query fails.
    package static func physicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// The process's memory in megabytes, with the GPU device's allocated size and recommended
    /// working-set limit when a device is available.
    package static func status() -> ReadyResponse.MemoryStatus {
        let megabyte = 1024.0 * 1024.0
        return ReadyResponse.MemoryStatus(
            physicalFootprintMB: Double(physicalFootprintBytes()) / megabyte,
            gpuAllocatedMB: device.map { Double($0.currentAllocatedSize) / megabyte },
            gpuRecommendedMaxMB: device.map { Double($0.recommendedMaxWorkingSetSize) / megabyte })
    }
}
