// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Darwin
import Foundation

/// The resident bytes of each file this process has mapped, such as model weights and framework code, from one
/// walk of its VM regions.
///
/// The walk skips the dyld shared region (system library code and data). A file mapped more than once as
/// shared memory is counted once.
package struct MappedFiles {
    /// Resident bytes per mapped file, keyed by path.
    package var residentBytes: [String: UInt64] = [:]

    package var total: UInt64 { residentBytes.values.reduce(0, +) }

    /// Resident bytes of mapped files outside `modelDirectories`: code and system data.
    package func shared(modelDirectories: [String]) -> UInt64 {
        residentBytes.reduce(0) { total, file in
            modelDirectories.contains { file.key.hasPrefix($0) } ? total : total + file.value
        }
    }

    /// Resident bytes of mapped files inside `modelDirectories`.
    package func model(modelDirectories: [String]) -> UInt64 {
        total - shared(modelDirectories: modelDirectories)
    }
}

/// `path` with symlinks resolved the way region file names are. That means `realpath`, not
/// `resolvingSymlinksInPath`, which drops the /private that region paths keep. Returns `path` unchanged if
/// it doesn't exist.
package func resolvedPath(_ path: String) -> String {
    guard let resolved = realpath(path, nil) else { return path }
    defer { free(resolved) }
    return String(cString: resolved)
}

/// The null-terminated string, such as a path from `proc_pidpath`, that a system call wrote into `buffer`.
package func nullTerminatedString(_ buffer: [CChar]) -> String {
    buffer.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
}

/// Directories whose mapped files hold the model, each ending in "/": the model bundle, the temporary
/// directory (where the GPU delegate stages weights), and Core AI's specialization cache. Other mapped
/// files are code and system data that would be resident anyway. Peak memory counts the model files only up to
/// what stays resident through `MemoryMeasurement`'s end-of-run check, so a copy being made, or a staged
/// copy the model no longer reads, doesn't count.
package func modelFileDirectories(modelPath: String) -> [String] {
    var directories = [modelPath, NSTemporaryDirectory()]
    if let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
        directories.append(caches.appendingPathComponent("coreai-cache").path)
    }
    return directories.map { resolvedPath($0) + "/" }
}

#if os(macOS)
/// Calls `body` with each of this process's VM regions, in address order, skipping submaps and the dyld
/// shared region (library code and data, not model memory; parts of it are mapped as ordinary regions too,
/// without a file name).
private func forEachRegion(_ body: (mach_vm_address_t, mach_vm_size_t, vm_region_submap_info_data_64_t) -> Void) {
    var address: mach_vm_address_t = 0
    while true {
        var size: mach_vm_size_t = 0
        var depth: natural_t = 0
        var info = vm_region_submap_info_data_64_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_region_submap_info_data_64_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                mach_vm_region_recurse(mach_task_self_, &address, &size, &depth, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { break }
        defer { address += size }
        let inSharedRegion = address >= SHARED_REGION_BASE && address < SHARED_REGION_BASE + SHARED_REGION_SIZE
        guard info.is_submap == 0, !inSharedRegion else { continue }
        body(address, size, info)
    }
}
#endif

/// Drops this process's resident pages of files mapped from `modelDirectories`, writing any unsaved
/// changes back to the files first, as the system would under memory pressure. Pages a driver has wired,
/// such as weights the GPU runs from, stay resident. Does nothing where region file names can't be read
/// (macOS only).
package func evictMappedFiles(in modelDirectories: [String]) {
    #if os(macOS)
    let pid = getpid()
    var pathBuffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
    forEachRegion { address, size, info in
        guard info.external_pager != 0, info.pages_resident > 0,
            proc_regionfilename(pid, address, &pathBuffer, UInt32(pathBuffer.count)) > 0
        else { return }
        let path = nullTerminatedString(pathBuffer)
        guard modelDirectories.contains(where: { path.hasPrefix($0) }) else { return }
        // MS_SYNC writes dirty pages back before MS_INVALIDATE drops them, so no data is lost.
        msync(UnsafeMutableRawPointer(bitPattern: UInt(address)), Int(size), MS_SYNC | MS_INVALIDATE)
    }
    #endif
}

/// Walks this process's VM regions for its mapped files. nil where region file names can't be read, since
/// `proc_regionfilename` is macOS-only.
package func mappedFiles() -> MappedFiles? {
    #if os(macOS)
    var mapped = MappedFiles()
    let pageSize = UInt64(getpagesize())
    let pid = getpid()
    var seenObjects: Set<UInt64> = []
    var pathBuffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
    forEachRegion { address, _, info in
        let shareMode = Int32(info.share_mode)
        let isShared = shareMode == SM_SHARED || shareMode == SM_TRUESHARED || shareMode == SM_SHARED_ALIASED
        if isShared, !seenObjects.insert(info.object_id_full).inserted { return }
        guard info.external_pager != 0 else { return }
        let resident = UInt64(info.pages_resident) * pageSize
        guard resident > 0 else { return }
        let length = proc_regionfilename(pid, address, &pathBuffer, UInt32(pathBuffer.count))
        let path = length > 0 ? nullTerminatedString(pathBuffer) : "(unnamed file)"
        mapped.residentBytes[path, default: 0] += resident
    }
    return mapped
    #else
    return nil
    #endif
}
