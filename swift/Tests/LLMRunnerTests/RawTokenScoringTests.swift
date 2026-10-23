// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import llm_runner

/// Covers how `llm-runner` decides between generating and scoring from `--raw-tokens`, and each
/// refusal it makes before the model loads.
@Suite("RawTokenScoring")
struct RawTokenScoringTests {
    /// A token file in the `{"tokens": [...]}` format, removed when the test ends.
    private struct TokensFile {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).json")

        var path: String { url.path }

        func write(_ tokens: [Int32]) throws {
            let list = tokens.map(String.init).joined(separator: ", ")
            try Data("{\"tokens\": [\(list)]}".utf8).write(to: url)
        }

        func cleanup() {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func tokensFile(_ tokens: [Int32]) throws -> TokensFile {
        let file = TokensFile()
        try file.write(tokens)
        return file
    }

    private func continuation(
        path: String?,
        context: [Int32]? = [1, 2, 3],
        textContinuation: Bool = false,
        logits: Bool = true
    ) throws -> [Int32]? {
        let contextFile = try context.map(tokensFile)
        defer { contextFile?.cleanup() }
        return try RawTokenScoring.continuation(
            rawContinuationPath: path,
            rawTokensPath: contextFile?.path,
            textContinuationGiven: textContinuation,
            logitsRequested: logits)
    }

    // MARK: Deciding the mode

    @Test("--raw-continuation decides between generating and scoring")
    func rawContinuationDecidesTheMode() throws {
        #expect(try continuation(path: nil) == nil)
        #expect(try continuation(path: nil, logits: false) == nil)

        let file = try tokensFile([7, 8, 9])
        defer { file.cleanup() }
        #expect(try continuation(path: file.path) == [7, 8, 9])
    }

    // MARK: Refusals

    @Test("--raw-continuation without --raw-tokens is refused")
    func refusesWithoutRawTokens() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        #expect {
            try continuation(path: file.path, context: nil)
        } throws: { error in
            guard case .rawContinuationRequiresRawTokens = error as? ContinuationOptionError else { return false }
            return true
        }
    }

    @Test("--raw-continuation without a logits output is refused")
    func refusesWithoutLogitsOutput() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        #expect {
            try continuation(path: file.path, logits: false)
        } throws: { error in
            guard case .rawContinuationRequiresLogitsOutput = error as? ContinuationOptionError else {
                return false
            }
            return true
        }
    }

    @Test("--raw-continuation with the text --continuation is refused")
    func refusesWithTextContinuation() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        #expect {
            try continuation(path: file.path, textContinuation: true)
        } throws: { error in
            guard case .rawContinuationConflictsWithContinuation = error as? ContinuationOptionError else {
                return false
            }
            return true
        }
    }

    @Test("An empty --raw-continuation file is refused")
    func refusesEmptyContinuation() throws {
        let file = try tokensFile([])
        defer { file.cleanup() }
        #expect {
            try continuation(path: file.path)
        } throws: { error in
            guard case .emptyRawContinuation = error as? ContinuationOptionError else { return false }
            return true
        }
    }

    @Test("An empty --raw-tokens context is refused before the model loads")
    func refusesEmptyContext() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        #expect {
            try continuation(path: file.path, context: [])
        } throws: { error in
            guard case .emptyRawContext = error as? ContinuationOptionError else { return false }
            return true
        }
    }

    @Test("A missing or malformed --raw-continuation file is refused")
    func refusesUnreadableContinuation() throws {
        let malformed = TokensFile()
        defer { malformed.cleanup() }
        try Data("{\"ids\": [1, 2]}".utf8).write(to: malformed.url)
        let missing = TokensFile()

        for path in [malformed.path, missing.path] {
            #expect {
                try continuation(path: path)
            } throws: { error in
                guard case .unreadableRawTokens(let reported, _) = error as? ContinuationOptionError else {
                    return false
                }
                return reported == path
            }
        }
    }
}
