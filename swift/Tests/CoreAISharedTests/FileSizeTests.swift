// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared

@Suite("URL.recursiveFileSizeInBytes")
struct FileSizeTests {
    /// Creates a unique temporary directory for the test and removes it after.
    private func withTempDirectory(_ body: (URL) throws -> Void) throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appending(
            path: "filesize-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try body(dir)
    }

    @Test("Single file returns its byte size")
    func singleFile() throws {
        try withTempDirectory { dir in
            let file = dir.appending(path: "a.bin")
            let bytes = Data(repeating: 0, count: 1234)
            try bytes.write(to: file)
            #expect(file.recursiveFileSizeInBytes() == 1234)
        }
    }

    @Test("Directory sums all nested files")
    func directorySumsRecursively() throws {
        try withTempDirectory { dir in
            try Data(repeating: 0, count: 100).write(to: dir.appending(path: "a.bin"))
            let sub = dir.appending(path: "sub", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
            try Data(repeating: 0, count: 250).write(to: sub.appending(path: "b.bin"))
            #expect(dir.recursiveFileSizeInBytes() == 350)
        }
    }

    @Test("Missing path returns nil")
    func missingPathReturnsNil() throws {
        let missing = FileManager.default.temporaryDirectory.appending(
            path: "does-not-exist-\(UUID().uuidString)")
        #expect(missing.recursiveFileSizeInBytes() == nil)
    }
}
