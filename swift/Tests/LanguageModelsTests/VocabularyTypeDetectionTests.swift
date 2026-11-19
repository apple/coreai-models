// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

@Suite("Grammar vocabulary type")
struct VocabularyTypeDetectionTests {
    private static let schema =
        #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}"#

    /// Token ids of the vocabularies below, shared so the tests read the same way.
    private enum Id {
        static let space: Int32 = 256
        static let open: Int32 = 257
        static let quote: Int32 = 258
        static let letter: Int32 = 259
        static let colon: Int32 = 260
        static let eos: Int32 = 261
    }

    /// A SentencePiece byte-fallback vocabulary (Gemma, Llama): all 256 `<0xHH>` byte tokens,
    /// and `▁` standing for a space.
    private static func byteFallbackTokenizer() -> MockTokenizer {
        var vocab = [String: Int]()
        for byte in 0..<256 { vocab[String(format: "<0x%02X>", byte)] = byte }
        for (token, id) in [
            ("▁", Id.space), ("{", Id.open), ("\"", Id.quote), ("a", Id.letter), (":", Id.colon), ("<eos>", Id.eos),
        ] {
            vocab[token] = Int(id)
        }
        return MockTokenizer(vocab: vocab)
    }

    /// A byte-level BPE vocabulary (GPT-2, Qwen): `Ġ` standing for a space.
    private static func byteLevelTokenizer() -> MockTokenizer {
        let tokens = ["Ġ", "{", "\"", "a", ":", "<eos>"]
        var vocab = [String: Int]()
        for (offset, token) in tokens.enumerated() { vocab[token] = Int(Id.space) + offset }
        return MockTokenizer(vocab: vocab)
    }

    /// Whether the grammar allows `token` after the prefix `{"a":`, or nil when it allows no token
    /// at all (a dead end).
    private static func allowsAfterColon(
        _ token: Int32, tokenizer: MockTokenizer, vocabSize: Int, vocabType: VocabularyType? = nil
    ) throws -> Bool? {
        var session = try ConstrainedGenerationSession(
            jsonSchema: schema, tokenizer: tokenizer, vocabSize: vocabSize, vocabType: vocabType,
            stopTokenIds: [Id.eos])
        for prefix in [Id.open, Id.quote, Id.letter, Id.quote, Id.colon] {
            guard session.acceptToken(prefix) else {
                Issue.record("the grammar rejected prefix token \(prefix)")
                return nil
            }
        }
        guard let mask = session.nextTokenBitmask() else { return nil }
        return mask[Int(token) / 32] & (1 << (Int32(token) % 32)) != 0
    }

    @Test("A byte-fallback tokenizer's space token is allowed between JSON tokens")
    func byteFallbackSpaceIsAllowed() throws {
        let allowed = try Self.allowsAfterColon(
            Id.space, tokenizer: Self.byteFallbackTokenizer(), vocabSize: 262)
        #expect(allowed == true, "the grammar must treat '▁' as a space after the colon")
    }

    @Test("A byte-level tokenizer's space token is allowed between JSON tokens")
    func byteLevelSpaceIsAllowed() throws {
        let allowed = try Self.allowsAfterColon(
            Id.space, tokenizer: Self.byteLevelTokenizer(), vocabSize: 262)
        #expect(allowed == true, "the grammar must treat 'Ġ' as a space after the colon")
    }

    private static let byteTokens = (0..<256).map { String(format: "<0x%02X>", $0) }

    @Test("A vocabulary with all 256 byte tokens is byte-fallback")
    func allByteTokensIsByteFallback() {
        let vocab = ["<pad>", "<eos>"] + Self.byteTokens + ["▁the", "▁a", "{"]
        #expect(VocabularyType.detect(vocabulary: vocab) == .byteFallback)
    }

    @Test("Lowercase hex byte tokens count the same as uppercase")
    func lowercaseByteTokens() {
        let vocab = Self.byteTokens.map { $0.lowercased() } + ["▁the"]
        #expect(VocabularyType.detect(vocabulary: vocab) == .byteFallback)
    }

    @Test("A byte-level BPE vocabulary is byte-level")
    func bpeIsByteLevel() {
        let vocab = ["Ġthe", "Ġa", "{", "Ċ", "<|endoftext|>"]
        #expect(VocabularyType.detect(vocabulary: vocab) == .byteLevel)
    }

    @Test("A partial set of byte tokens is not byte-fallback")
    func partialByteTokensAreByteLevel() {
        let vocab = Array(Self.byteTokens.prefix(255)) + ["Ġthe"]
        #expect(VocabularyType.detect(vocabulary: vocab) == .byteLevel)
    }

    @Test("Repeated byte tokens do not add up to a full set")
    func repeatedByteTokensAreByteLevel() {
        let vocab = Array(repeating: "<0x41>", count: 300)
        #expect(VocabularyType.detect(vocabulary: vocab) == .byteLevel)
    }

    @Test("An empty vocabulary is byte-level")
    func emptyVocabulary() {
        #expect(VocabularyType.detect(vocabulary: []) == .byteLevel)
    }

    @Test("An explicit vocabulary type overrides detection")
    func explicitTypeWins() throws {
        // Forced byte-level, the byte-fallback '▁' is not a space: after the colon the compact
        // grammar needs one, so no token is allowed at all.
        let allowed = try Self.allowsAfterColon(
            Id.space, tokenizer: Self.byteFallbackTokenizer(), vocabSize: 262, vocabType: .byteLevel)
        #expect(allowed != true)
    }
}
