// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - StopTokens.set (runtime terminating set)

/// `StopTokens.set` is the runtime union both adapters check each generated token
/// against: the tokenizer's main EOS plus the IDs resolved at load. These tests
/// prove additional stop tokens are actually unioned in (not stored and unused)
/// and that the main EOS is always present.
@Suite("StopTokens.set")
struct StopTokensSetTests {
    private static let vocab: [String: Int] = [
        "<eos>": 2,
        "<|im_end|>": 4,
    ]

    private static func tokenizer(vocab: [String: Int] = vocab) -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    @Test("Main EOS is always included")
    func includesMainEos() {
        let stopTokens = StopTokens.set(tokenizer: Self.tokenizer(vocab: [:]), additional: [])
        #expect(stopTokens == [2])
    }

    @Test("Additional stop tokens resolved at load are unioned into the stop set")
    func unionsAdditionalStopTokens() {
        // Gemma's <end_of_turn> (106) and Phi's <|end|> (200020) only reach the
        // stop set via the additional IDs resolved at load.
        let stopTokens = StopTokens.set(
            tokenizer: Self.tokenizer(vocab: [:]), additional: [106, 200_020])
        #expect(stopTokens == [2, 106, 200_020])
    }

    @Test("Additional stop tokens combine with the main EOS without duplication")
    func unionsWithMainEos() {
        // 4 is <|im_end|> (folded into the additional IDs at load); 2 is the main
        // EOS and must not be duplicated.
        let stopTokens = StopTokens.set(tokenizer: Self.tokenizer(), additional: [2, 4, 106])
        #expect(stopTokens == [2, 4, 106])
    }

    @Test("Empty additional stop tokens leave only the main EOS")
    func emptyAdditionalStopTokensNoOp() {
        let stopTokens = StopTokens.set(tokenizer: Self.tokenizer(), additional: [])
        #expect(stopTokens == [2])
    }
}

// MARK: - StopTokens.additionalIds (load-time EOS-like IDs)

/// `StopTokens.additionalIds` folds the tokenizer-config turn-end IDs together
/// with a base-vocab `<|im_end|>` and an optional agentic `<|eot|>`. It needs a
/// `LanguageBundle` with an embedded tokenizer directory, built here as a temp
/// bundle mirroring `AdditionalStopTokensTests`.
@Suite("StopTokens.additionalIds")
struct StopTokensAdditionalIdsTests {
    /// `<eos>` is ID 2 to match `MockTokenizer.eosTokenId`.
    private static let vocab: [String: Int] = [
        "<eot>": 1,
        "<eos>": 2,
        "<|im_end|>": 4,
    ]

    private static func tokenizer() -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    /// Build a temp LLM bundle whose embedded `tokenizer/` dir carries the given
    /// tokenizer_config.json (and an empty tokenizer.json so `tokenizerPath` resolves).
    private static func bundle(config: String) throws -> LanguageBundle {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "StopTokensAdditionalIdsTests-\(UUID().uuidString)/model"
        )
        let tokenizerDir = dir.appending(path: "tokenizer")
        try FileManager.default.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "stop-tokens-fixture",
              "assets": { "main": "model.aimodel" },
              "language": {
                "tokenizer": "x/y",
                "vocab_size": 100,
                "max_context_length": 512
              }
            }
            """.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        try config.write(
            to: tokenizerDir.appending(path: "tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: tokenizerDir.appending(path: "tokenizer.json"),
            atomically: true, encoding: .utf8)
        return try LanguageBundle(at: dir)
    }

    @Test("<|im_end|> in the base vocab is folded in")
    func foldsImEnd() throws {
        // Config resolves nothing on its own; <|im_end|> (4) reaches the list only
        // via the base-vocab fold -- the text/VLM unification.
        let bundle = try Self.bundle(config: #"{ "eos_token": "<eos>" }"#)
        let ids = Set(StopTokens.additionalIds(bundle: bundle, tokenizer: Self.tokenizer()))
        #expect(ids == [4])
    }

    @Test("agenticEOT is folded in when provided and in the vocab")
    func foldsAgenticEot() throws {
        let bundle = try Self.bundle(config: #"{ "eos_token": "<eos>" }"#)
        let ids = Set(
            StopTokens.additionalIds(
                bundle: bundle, tokenizer: Self.tokenizer(), agenticEOT: "<eot>"))
        // <eot> (1) from the agentic fold and <|im_end|> (4) from the base-vocab fold.
        #expect(ids == [1, 4])
    }

    @Test("agenticEOT not in the vocab is ignored")
    func ignoresUnknownAgenticEot() throws {
        let bundle = try Self.bundle(config: #"{ "eos_token": "<eos>" }"#)
        let ids = Set(
            StopTokens.additionalIds(
                bundle: bundle, tokenizer: Self.tokenizer(), agenticEOT: "<not_in_vocab>"))
        #expect(ids == [4])
    }

    @Test("an ID from multiple sources is not duplicated")
    func dedupsAcrossSources() throws {
        // Top-level im_end key resolves <|im_end|> (4) via LanguageConfig, and the
        // base-vocab fold would add 4 again -- it must appear once.
        let bundle = try Self.bundle(
            config: #"{ "eos_token": "<eos>", "im_end": "<|im_end|>" }"#)
        let ids = StopTokens.additionalIds(bundle: bundle, tokenizer: Self.tokenizer())
        #expect(ids.filter { $0 == 4 }.count == 1)
        #expect(Set(ids) == [4])
    }
}
