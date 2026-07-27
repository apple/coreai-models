// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

extension URL {
    /// Recursively sums the on-disk byte size of the file or directory at this URL.
    ///
    /// For a single file this is its size; for a directory it is the sum of all
    /// contained files' sizes (descending into subdirectories). This reflects
    /// bytes on disk, not resident memory.
    ///
    /// - Returns: Total size in bytes, or `nil` if the path does not exist or
    ///   its attributes cannot be read.
    public func recursiveFileSizeInBytes() -> Int? {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDirectory) else { return nil }

        if !isDirectory.boolValue {
            return (try? fm.attributesOfItem(atPath: path)[.size] as? Int) ?? nil
        }

        guard
            let enumerator = fm.enumerator(
                at: self, includingPropertiesForKeys: [.fileSizeKey], options: [])
        else { return nil }

        var total = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += size
            }
        }
        return total
    }
}
