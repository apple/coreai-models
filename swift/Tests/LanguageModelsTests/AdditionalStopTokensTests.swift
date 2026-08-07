// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("LanguageConfig.additionalStopTokenIds")
struct AdditionalStopTokensTests {
    /// Vocabulary shared by the tests. `<eos>` must be ID 2 to match
    /// `MockTokenizer.eosTokenId`, so it is expected to be filtered out.
    private static let vocab: [String: Int] = [
        "<eot>": 1,
        "<eos>": 2,
        "<end_of_turn>": 3,
        "<|im_end|>": 4,
        "<|endoftext|>": 5,
    ]

    private static func tokenizer() -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    /// Write `tokenizer_config.json` into a fresh temp directory and return it.
    private static func tokenizerDir(config: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "AdditionalStopTokensTests-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(
            to: dir.appending(path: "tokenizer_config.json"),
            atomically: true, encoding: .utf8
        )
        return dir
    }

    private static func stopIds(config: String) throws -> Set<Int32> {
        let dir = try tokenizerDir(config: config)
        defer { try? FileManager.default.removeItem(at: dir) }
        return Set(
            LanguageConfig.additionalStopTokenIds(from: dir, tokenizer: tokenizer())
        )
    }

    // MARK: - Top-level turn-ending tokens

    @Test("top-level eot_token string is picked up")
    func topLevelEotToken() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>",
                  "eot_token": "<eot>"
                }
                """)
        #expect(ids == [1])
    }

    @Test("top-level end_of_turn / im_end / endoftext keys are picked up")
    func topLevelOtherPatterns() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "end_of_turn": "<end_of_turn>",
                  "im_end": "<|im_end|>",
                  "endoftext": "<|endoftext|>"
                }
                """)
        #expect(ids == [3, 4, 5])
    }

    @Test("top-level token equal to the main EOS is not duplicated")
    func topLevelSkipsMainEos() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>",
                  "eot_token": "<eos>"
                }
                """)
        #expect(ids.isEmpty)
    }

    @Test("top-level token missing from the vocab is ignored")
    func topLevelUnknownToken() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eot_token": "<not_in_vocab>"
                }
                """)
        #expect(ids.isEmpty)
    }

    @Test("non-string top-level value is ignored")
    func topLevelNonStringValue() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eot_token": { "content": "<eot>" }
                }
                """)
        #expect(ids.isEmpty)
    }
}
