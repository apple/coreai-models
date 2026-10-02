// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIDiffusionPipeline

@Suite("Multi-output model function")
struct MultiOutputModelFunctionTests {
    // MARK: - Error surface (no asset needed)

    @Test("expectedSingleOutput description lists all output names and points at predictAllOutputs")
    func expectedSingleOutputDescription() {
        let err = CoreAIDiffusionError.expectedSingleOutput(got: ["hidden_embeds", "pooled_outputs"])
        let msg = err.errorDescription ?? ""
        #expect(msg.contains("2 outputs"))
        #expect(msg.contains("hidden_embeds"))
        #expect(msg.contains("pooled_outputs"))
        #expect(msg.contains("predictAllOutputs"))
    }
}
