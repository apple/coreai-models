// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import Testing

@testable import llm_runner

/// Covers how `llm-runner` parses the raw-token options.
@Suite("LLMRunner arguments")
struct LLMRunnerArgumentTests {
    @Test("--raw-continuation names the ids to score beside --raw-tokens")
    func parsesRawContinuation() throws {
        let runner = try LLMRunner.parse([
            "--raw-tokens", "context.json", "--raw-continuation", "continuation.json", "--save-logits", "out.json",
        ])
        #expect(runner.rawTokens == "context.json")
        #expect(runner.rawContinuation == "continuation.json")
    }
}
