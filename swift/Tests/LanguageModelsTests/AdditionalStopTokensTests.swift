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
    /// Vocabulary shared by the parsing tests. `<eos>` must be ID 2 to match
    /// `MockTokenizer.eosTokenId`, so it is expected to be filtered out.
    /// `<|im_end|>` is deliberately absent here so the parsing cases stay focused;
    /// the universal base-vocab `<|im_end|>` fold is covered by `foldsBaseVocabImEnd`.
    private static let vocab: [String: Int] = [
        "<eot>": 1,
        "<eos>": 2,
        "<end_of_turn>": 3,
        "<|endoftext|>": 5,
    ]

    private static func tokenizer() -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    /// Write `tokenizer_config.json` (and optionally `tokenizer.json`) into a
    /// fresh temp directory and return it.
    private static func tokenizerDir(config: String, tokenizerJSON: String? = nil) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "AdditionalStopTokensTests-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(
            to: dir.appending(path: "tokenizer_config.json"),
            atomically: true, encoding: .utf8
        )
        if let tokenizerJSON {
            try tokenizerJSON.write(
                to: dir.appending(path: "tokenizer.json"),
                atomically: true, encoding: .utf8
            )
        }
        return dir
    }

    private static func stopIds(config: String, tokenizerJSON: String? = nil) throws -> Set<Int32> {
        let dir = try tokenizerDir(config: config, tokenizerJSON: tokenizerJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        return LanguageConfig.additionalStopTokenIds(from: dir, tokenizer: tokenizer())
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

    @Test("top-level end_of_turn / endoftext keys are picked up")
    func topLevelOtherPatterns() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "end_of_turn": "<end_of_turn>",
                  "endoftext": "<|endoftext|>"
                }
                """)
        #expect(ids == [3, 5])
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

    // MARK: - tokenizer.json added_tokens (exported bundles)

    @Test("Gemma exported bundle stops on <end_of_turn> via tokenizer.json")
    func gemmaExportedBundleEndOfTurn() throws {
        // save_pretrained drops added_tokens_decoder from tokenizer_config.json
        // and keeps the specials in tokenizer.json, so 106 must be recovered
        // from there.
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>"
                }
                """,
            tokenizerJSON: """
                {
                  "added_tokens": [
                    { "id": 106, "content": "<end_of_turn>", "special": true },
                    { "id": 2, "content": "<eos>", "special": true },
                    { "id": 255999, "content": "<start_of_image>", "special": true }
                  ]
                }
                """)
        #expect(ids == [106])
    }

    @Test("Phi exported bundle stops on <|end|> via tokenizer.json")
    func phiExportedBundleEnd() throws {
        // Phi-4-mini ends assistant turns with <|end|> (ID 200020) while its main
        // EOS is <|endoftext|>, and save_pretrained keeps <|end|> in tokenizer.json.
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>"
                }
                """,
            tokenizerJSON: """
                {
                  "added_tokens": [
                    { "id": 200020, "content": "<|end|>", "special": true }
                  ]
                }
                """)
        #expect(ids == [200020])
    }

    @Test("tokenizer.json non-special turn token is ignored")
    func tokenizerJSONNonSpecialIgnored() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>"
                }
                """,
            tokenizerJSON: """
                {
                  "added_tokens": [
                    { "id": 106, "content": "<end_of_turn>", "special": false }
                  ]
                }
                """)
        #expect(ids.isEmpty)
    }

    @Test("tokenizer.json entry equal to the main EOS is not duplicated")
    func tokenizerJSONSkipsMainEos() throws {
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>"
                }
                """,
            tokenizerJSON: """
                {
                  "added_tokens": [
                    { "id": 2, "content": "<endoftext>", "special": true }
                  ]
                }
                """)
        #expect(ids.isEmpty)
    }

    @Test("same turn-end ID from added_tokens_decoder and tokenizer.json is deduped")
    func dedupAcrossBothSources() throws {
        // <end_of_turn> (3) appears in both added_tokens_decoder and
        // tokenizer.json's added_tokens; <|im_end|> (4) only in the former.
        // Result must still be a set: 3 once, plus 4.
        let ids = try Self.stopIds(
            config: """
                {
                  "eos_token": "<eos>",
                  "added_tokens_decoder": {
                    "3": { "content": "<end_of_turn>", "special": true },
                    "4": { "content": "<|im_end|>", "special": true }
                  }
                }
                """,
            tokenizerJSON: """
                {
                  "added_tokens": [
                    { "id": 3, "content": "<end_of_turn>", "special": true }
                  ]
                }
                """)
        #expect(ids == [3, 4])
    }

    // MARK: - Universal <|im_end|> fold

    @Test("base-vocab <|im_end|> is folded in even when the config lists nothing")
    func foldsBaseVocabImEnd() throws {
        // The config resolves nothing on its own; <|im_end|> (4) reaches the set
        // only via the universal base-vocab fold that keeps the text adapter, VLM
        // adapter, server, and CLI from diverging.
        let dir = try Self.tokenizerDir(config: #"{ "eos_token": "<eos>" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = LanguageConfig.additionalStopTokenIds(
            from: dir, tokenizer: MockTokenizer(vocab: ["<eos>": 2, "<|im_end|>": 4]))
        #expect(ids == [4])
    }
}
