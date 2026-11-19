// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import Foundation

/// Command-line options for the pipelined engine's prefix cache, shared by the tools that run one.
///
/// Each tool brings its own default for the memory cache: llm-server keeps one unless told not to,
/// llm-runner only when asked.
package struct PrefixCacheOptions: ParsableArguments {
    @Option(
        name: .customLong("prefix-cache-mb"),
        help: ArgumentHelp(
            "Megabytes of earlier prompts' state to keep in memory, so a prompt that starts the same way "
                + "restores that prefix instead of computing it (default: 1024 in llm-server, 0 in llm-runner; "
                + "0 turns it off; pipelined engine only)")
    )
    package var prefixCacheMB: Int?

    @Option(
        name: .customLong("prefix-cache-disk-mb"),
        help: ArgumentHelp(
            "Megabytes of disk the prefix cache may also use, under ~/Library/Caches/coreai-prefix-cache, so "
                + "it can outgrow --prefix-cache-mb and is kept across restarts. Needs --prefix-cache-mb above "
                + "0; one process per model at a time (default: 0, memory only)")
    )
    package var prefixCacheDiskMB: Int = 0

    package init() {}

    /// The largest value either flag takes: megabytes that still fit in an `Int` of bytes.
    private static let maxMB = Int.max >> 20

    package func validate() throws {
        if let mb = prefixCacheMB, !(0...Self.maxMB).contains(mb) {
            throw ValidationError("--prefix-cache-mb must be between 0 and \(Self.maxMB) (got \(mb))")
        }
        if !(0...Self.maxMB).contains(prefixCacheDiskMB) {
            throw ValidationError(
                "--prefix-cache-disk-mb must be between 0 and \(Self.maxMB) (got \(prefixCacheDiskMB))")
        }
    }

    /// Checks what ``validate()`` can't without the tool's default: that a disk budget comes with a
    /// memory cache, which every chunk goes through.
    package func validate(defaultMB: Int) throws {
        if prefixCacheDiskMB > 0, (prefixCacheMB ?? defaultMB) == 0 {
            throw ValidationError("--prefix-cache-disk-mb needs --prefix-cache-mb above 0")
        }
    }

    /// Bytes of memory for the cache: the flag, or `defaultMB` when it isn't given.
    package func memoryBytes(defaultMB: Int) -> Int { (prefixCacheMB ?? defaultMB) << 20 }

    /// Bytes of disk for the cache, zero for none.
    package var diskBytes: Int { prefixCacheDiskMB << 20 }

    /// The flags given on the command line, for a path that ignores them all to name.
    package var givenFlags: [String] {
        [prefixCacheMB != nil ? "--prefix-cache-mb" : nil, prefixCacheDiskMB > 0 ? "--prefix-cache-disk-mb" : nil]
            .compactMap { $0 }
    }

    /// A warning naming the flags asked for that an engine keeping no prefix cache ignores, or nil when
    /// it keeps one or none was asked for.
    package func ignoredWarning(cacheEnabled: Bool) -> String? {
        let asked = [
            (prefixCacheMB ?? 0) > 0 ? "--prefix-cache-mb \(prefixCacheMB ?? 0)" : nil,
            prefixCacheDiskMB > 0 ? "--prefix-cache-disk-mb \(prefixCacheDiskMB)" : nil,
        ].compactMap { $0 }
        guard !cacheEnabled, !asked.isEmpty else { return nil }
        return "⚠️  \(asked.joined(separator: ", ")): no prefix cache here. Only the pipelined engine keeps one, "
            + "and not for every model (--verbose says why)."
    }
}
