// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import Foundation
import Testing

@testable import llm_runner

/// Covers `--prompt-list`: reading the file, how many prompts run at once, and what it refuses.
@Suite("LLMRunner --prompt-list")
struct PromptListTests {
    private func writeList(_ contents: String) throws -> String {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("prompts-\(UUID()).txt")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    @Test("Each non-blank line is one prompt, trimmed")
    func readsLines() throws {
        let path = try writeList("Name a fruit.\n\n  Name a planet.  \n\nName a colour.")
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try LLMRunner.readPromptList(path) == ["Name a fruit.", "Name a planet.", "Name a colour."])
    }

    @Test("A list with one prompt runs that prompt")
    func readsOneLine() throws {
        let path = try writeList("Only this.\n")
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try LLMRunner.readPromptList(path) == ["Only this."])
    }

    @Test("A list of blank lines is refused")
    func refusesBlankList() throws {
        let path = try writeList("\n   \n\n")
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: ValidationError.self) { try LLMRunner.readPromptList(path) }
    }

    @Test("Width is every prompt, or --concurrent-requests, within the model's batch")
    func width() throws {
        #expect(try LLMRunner.batchWidth(requested: nil, promptCount: 3, maxBatchSize: 8) == 3)
        #expect(try LLMRunner.batchWidth(requested: nil, promptCount: 12, maxBatchSize: 8) == 8)
        #expect(try LLMRunner.batchWidth(requested: 2, promptCount: 5, maxBatchSize: 8) == 2)
        #expect(try LLMRunner.batchWidth(requested: 6, promptCount: 3, maxBatchSize: 8) == 3)
        #expect(throws: ValidationError.self) {
            try LLMRunner.batchWidth(requested: 9, promptCount: 12, maxBatchSize: 8)
        }
    }

    @Test("--prompt-list refuses a single-prompt input, and --concurrent-requests needs it")
    func refusesConflicts() {
        #expect(throws: (any Error).self) {
            try LLMRunner.parse(["--prompt-list", "p.txt", "--prompt", "hi"])
        }
        #expect(throws: (any Error).self) {
            try LLMRunner.parse(["--concurrent-requests", "2", "--prompt", "hi"])
        }
        #expect(throws: (any Error).self) {
            try LLMRunner.parse(["--prompt-list", "p.txt", "--inference-engine-variant", "coreai-pipelined"])
        }
    }
}
