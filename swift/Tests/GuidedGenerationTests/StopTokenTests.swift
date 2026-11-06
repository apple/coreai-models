// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("A grammar's stop tokens")
struct StopTokenTests {
    /// Ids into `vocab`: `{"a":1}` and two end tokens, as Qwen3 has both.
    private enum Id {
        static let open: Int32 = 0, close: Int32 = 1, quote: Int32 = 2, colon: Int32 = 3
        static let a: Int32 = 4, one: Int32 = 5, imEnd: Int32 = 6, endOfText: Int32 = 7
    }

    private static let vocab = ["{", "}", "\"", ":", "a", "1", "<|im_end|>", "<|endoftext|>"]
    private static let json = [Id.open, Id.quote, Id.a, Id.quote, Id.colon, Id.one, Id.close]
    private static let schema = """
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}
        """

    private func session(stopTokenIds: [Int32]?) throws -> ConstrainedGenerationSession {
        try ConstrainedGenerationSession(
            jsonSchema: Self.schema, vocabulary: Self.vocab, vocabType: .raw, stopTokenIds: stopTokenIds)
    }

    /// The ids the grammar allows after `tokens`.
    private func allowedAfter(_ tokens: [Int32], stopTokenIds: [Int32]?) throws -> Set<Int32> {
        var grammar = try session(stopTokenIds: stopTokenIds)
        for token in tokens {
            let accepted = grammar.acceptToken(token)
            #expect(accepted)
        }
        return allowedTokenIDs(session: &grammar)
    }

    @Test("The stop tokens given are the ones allowed once the JSON is complete")
    func givenStopTokensAreAllowedAtTheEnd() throws {
        // The model's own end of turn, not the one xgrammar would pick from the vocabulary.
        #expect(try allowedAfter(Self.json, stopTokenIds: [Id.imEnd]) == [Id.imEnd])
        #expect(try allowedAfter(Self.json, stopTokenIds: [Id.imEnd, Id.endOfText]) == [Id.imEnd, Id.endOfText])
    }

    @Test("A stop token is refused before the JSON is complete")
    func stopTokensWaitForTheEnd() throws {
        for count in 0..<Self.json.count {
            let allowed = try allowedAfter(Array(Self.json.prefix(count)), stopTokenIds: [Id.imEnd, Id.endOfText])
            #expect(!allowed.contains(Id.imEnd) && !allowed.contains(Id.endOfText), "after \(count) token(s)")
        }
    }

    @Test("Accepting a stop token ends the grammar")
    func stopTokenTerminates() throws {
        var grammar = try session(stopTokenIds: [Id.imEnd])
        for token in Self.json + [Id.imEnd] {
            let accepted = grammar.acceptToken(token)
            #expect(accepted)
        }
        let terminated = grammar.isTerminated
        #expect(terminated)
    }

    @Test("The stop tokens reach the grammar through TokenizerInfo too")
    func tokenizerInfoCarriesStopTokens() throws {
        let info = TokenizerInfo(vocabulary: Self.vocab, vocabType: .raw, stopTokenIds: [Id.imEnd])
        #expect(info.stopTokenIds == [Id.imEnd])
        var grammar = try ConstrainedGenerationSession(jsonSchema: Self.schema, tokenizerInfo: info)
        for token in Self.json {
            let accepted = grammar.acceptToken(token)
            #expect(accepted)
        }
        let allowed = allowedTokenIDs(session: &grammar)
        #expect(allowed == [Id.imEnd])
    }
}
