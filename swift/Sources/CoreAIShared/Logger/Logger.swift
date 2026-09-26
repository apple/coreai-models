// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization

/// Centralized logging utility that uses environment variables for verbose control
/// Used across all CLI components: engines, tokenizers, decoding strategies, sampling strategies
public struct CLILogger {
    private static let _level = Atomic<Int>(0)

    public static var level: Int {
        get {
            _level.load(ordering: .acquiring)
        }
        set {
            assert(newValue >= 0, "Log level must be greater than or equal to 0")
            _level.store(newValue, ordering: .releasing)
        }
    }

    /// Performs logging if enabled for the requested level. Diagnostics go to stderr so stdout
    /// stays reserved for program output (for example, `--replay` JSONL results).
    /// - Parameters:
    ///   - message: The message to log.
    ///   - component: The name of the component logging.
    ///   - level: The minimum log level to log at.
    public static func log(_ message: String, component: String? = nil, level: Int = 1) {
        guard isEnabled(at: level) else {
            return
        }

        let line = (component.map { "[\($0)] " } ?? "") + message + "\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    public static func isEnabled(at level: Int) -> Bool {
        Self.level >= level
    }

    public static var isVerbose: Bool {
        Self.level >= 1
    }
}
