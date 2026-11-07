// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared

@Suite("A warning every request would repeat shows once")
struct WarnOnceTests {
    @Test("Only the first warning for a key shows at every level")
    func firstOnly() {
        let key = "test-\(UUID().uuidString)"
        #expect(CLILogger.warnOnce(key, "Warning: first"))
        #expect(!CLILogger.warnOnce(key, "Warning: again"))
        #expect(CLILogger.warnOnce(key + "-other", "Warning: another key"))
    }
}
