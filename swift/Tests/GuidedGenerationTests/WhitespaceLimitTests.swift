// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Whitespace limit")
struct WhitespaceLimitTests {
    private static let space: Int32 = 7
    private static let newline: Int32 = 8
    private static let tab: Int32 = 9
    private static let whitespace: Set<Int32> = [space, newline, tab]

    private func session(limit: Int?) throws -> ConstrainedGenerationSession {
        try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema, vocabulary: sharedToyVocab, vocabType: .raw,
            maxWhitespaceCount: limit)
    }

    /// Feeds `tokens` in order; false at the first one the grammar refuses.
    private func feed(_ s: inout ConstrainedGenerationSession, _ tokens: [Int32]) -> Bool {
        for token in tokens where !s.acceptToken(token) { return false }
        return true
    }

    // MARK: - Configuration

    @Test("The default limit is 32, and a session records the limit it was built with")
    func defaultLimit() throws {
        #expect(ConstrainedGenerationSession.defaultMaxWhitespaceCount == 32)
        let byDefault = try createTestSession().maxWhitespaceCount
        let explicit = try session(limit: 5).maxWhitespaceCount
        let unlimited = try session(limit: nil).maxWhitespaceCount
        #expect(byDefault == 32)
        #expect(explicit == 5)
        #expect(unlimited == nil)
    }

    @Test("A limit that is not positive is rejected")
    func nonPositiveLimitIsRejected() {
        for bad in [0, -1, -100] {
            #expect(throws: ConstrainedGenerationError.self) {
                _ = try ConstrainedGenerationSession(
                    jsonSchema: sharedPersonSchema, vocabulary: sharedToyVocab, vocabType: .raw,
                    maxWhitespaceCount: bad)
            }
        }
    }

    @Test("The limit is carried by every initializer")
    func everyInitializerCarriesTheLimit() throws {
        let info = TokenizerInfo(vocabulary: sharedToyVocab, vocabType: .raw)
        let viaInfo = try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema, tokenizerInfo: info, maxWhitespaceCount: 7)
        let viaInfoDefault = try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema, tokenizerInfo: info)
        #expect(viaInfo.maxWhitespaceCount == 7)
        #expect(viaInfoDefault.maxWhitespaceCount == 32)
    }

    // MARK: - The mask

    @Test("Whitespace tokens are removed from the mask once the run reaches the limit")
    func limitBoundsARun() throws {
        var s = try session(limit: 4)
        let withinLimit = feed(&s, [TestConstants.openBraceToken] + Array(repeating: Self.space, count: 3))
        #expect(withinLimit)
        let beforeLimit = allowedTokenIDs(session: &s)
        #expect(beforeLimit.isSuperset(of: Self.whitespace))
        let atLimit = feed(&s, [Self.space])
        #expect(atLimit)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed.isDisjoint(with: Self.whitespace))
        #expect(allowed == [TestConstants.quoteToken])
    }

    @Test("Mixed whitespace counts toward the same limit")
    func mixedWhitespaceCounts() throws {
        var s = try session(limit: 3)
        let fed = feed(&s, [TestConstants.openBraceToken, Self.newline, Self.space, Self.tab])
        #expect(fed)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed.isDisjoint(with: Self.whitespace))
    }

    @Test("The limit applies to each gap, not to the document")
    func limitIsPerGap() throws {
        var s = try session(limit: 3)
        let three = [Self.space, Self.space, Self.space]
        let firstGap = feed(&s, [TestConstants.openBraceToken] + three)
        #expect(firstGap)
        let capped = allowedTokenIDs(session: &s)
        #expect(capped.isDisjoint(with: Self.whitespace))
        // A key follows; once it ends, whitespace is allowed again.
        let key = feed(&s, [TestConstants.quoteToken, 10, TestConstants.quoteToken])
        #expect(key)
        let secondGap = allowedTokenIDs(session: &s)
        #expect(secondGap.isSuperset(of: Self.whitespace))
    }

    @Test("Whitespace inside a string value is not limited")
    func whitespaceInAStringIsNotCounted() throws {
        var s = try session(limit: 3)
        // {"name": "   ... 40 spaces inside the value
        let opening: [Int32] = [
            TestConstants.openBraceToken, TestConstants.quoteToken, 10, TestConstants.quoteToken, 4,
            TestConstants.quoteToken,
        ]
        let fed = feed(&s, opening + Array(repeating: Self.space, count: 40))
        #expect(fed)
        let allowed = allowedTokenIDs(session: &s)
        // Spaces stay allowed far past the limit. (JSON needs a newline in a string escaped, so the
        // grammar itself refuses a raw one there, with or without a limit.)
        #expect(allowed.contains(Self.space))
    }

    @Test("With no limit, any amount of whitespace stays allowed")
    func unlimitedAcceptsLongRuns() throws {
        var s = try session(limit: nil)
        let fed = feed(&s, [TestConstants.openBraceToken] + Array(repeating: Self.space, count: 200))
        #expect(fed)
        #expect(allowedTokenIDs(session: &s).contains(Self.space))
    }

    @Test("Every token of a pretty-printed document is allowed at its step, with or without a limit")
    func normalLayoutIsUnaffected() throws {
        // {\n  "name": "John",\n  "age": 25\n}
        let tokens: [Int32] = [
            0, 8, 7, 7, 6, 10, 6, 4, 7, 6, 17, 6, 5, 8, 7, 7, 6, 11, 6, 4, 7, 38, 8, 1,
        ]
        for limit: Int? in [ConstrainedGenerationSession.defaultMaxWhitespaceCount, 4, nil] {
            var s = try session(limit: limit)
            for token in tokens {
                let allowed = allowedTokenIDs(session: &s)
                #expect(allowed.contains(token), "limit \(String(describing: limit)) removed token \(token)")
                let accepted = s.acceptToken(token)
                #expect(accepted)
            }
        }
    }

    // MARK: - Rollback and reset

    @Test("Rolling back restores the whitespace run, and so does a reset")
    func rollbackAndReset() throws {
        var s = try session(limit: 3)
        let fed = feed(&s, [TestConstants.openBraceToken, Self.space, Self.space, Self.space])
        #expect(fed)
        let capped = allowedTokenIDs(session: &s)
        #expect(capped.isDisjoint(with: Self.whitespace))
        let rolledBack = s.rollback(2)
        #expect(rolledBack)
        let afterRollback = allowedTokenIDs(session: &s)
        #expect(afterRollback.isSuperset(of: Self.whitespace))
        // One space is in the run again; two more reach the limit.
        let refed = feed(&s, [Self.space, Self.space])
        #expect(refed)
        let cappedAgain = allowedTokenIDs(session: &s)
        #expect(cappedAgain.isDisjoint(with: Self.whitespace))
        s.reset()
        let afterReset = feed(&s, [TestConstants.openBraceToken])
        #expect(afterReset)
        let fresh = allowedTokenIDs(session: &s)
        #expect(fresh.isSuperset(of: Self.whitespace))
    }

    // MARK: - Tokens that start with whitespace, and the three vocabulary encodings

    private static let aSchema =
        #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}"#

    @Test("A token that starts with whitespace is removed with it, and one that ends with it is not")
    func fusedTokens() throws {
        // 0 {  1 }  2 "  3 a  4 :  5 space  6 newline  7 space+quote  8 three newlines  9 quote+newline
        let vocabulary = ["{", "}", "\"", "a", ":", " ", "\n", " \"", "\n\n\n", "\"\n"]
        var s = try ConstrainedGenerationSession(
            jsonSchema: Self.aSchema, vocabulary: vocabulary, vocabType: .raw, maxWhitespaceCount: 2)
        let fed = feed(&s, [0, 5, 6])
        #expect(fed)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed.isDisjoint(with: [5, 6, 7, 8]))
        #expect(allowed.contains(2))
        // A token longer than the room left is removed even before the run starts.
        var fresh = try ConstrainedGenerationSession(
            jsonSchema: Self.aSchema, vocabulary: vocabulary, vocabType: .raw, maxWhitespaceCount: 2)
        let opened = feed(&fresh, [0])
        #expect(opened)
        let atStart = allowedTokenIDs(session: &fresh)
        #expect(!atStart.contains(8))
        #expect(atStart.contains(5))
    }

    @Test("Byte-level markers for space and newline count as whitespace")
    func byteLevelMarkers() throws {
        // 0 {  1 }  2 "  3 a  4 :  5 Ġ (space)  6 Ċ (newline)  7 Ġ"
        let vocabulary = ["{", "}", "\"", "a", ":", "\u{0120}", "\u{010A}", "\u{0120}\""]
        var s = try ConstrainedGenerationSession(
            jsonSchema: Self.aSchema, vocabulary: vocabulary, vocabType: .byteLevel, maxWhitespaceCount: 2)
        let fed = feed(&s, [0, 6, 5])
        #expect(fed)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed.isDisjoint(with: [5, 6, 7]))
        #expect(allowed.contains(2))
    }

    @Test("Byte-fallback markers and byte tokens count as whitespace")
    func byteFallbackMarkers() throws {
        var vocabulary = (0..<256).map { String(format: "<0x%02X>", $0) }
        let base = vocabulary.count
        vocabulary += ["{", "}", "\"", "a", ":", "\u{2581}", "\u{2581}\""]
        let space = Int32(base + 5)
        let spaceQuote = Int32(base + 6)
        let quote = Int32(base + 2)
        var s = try ConstrainedGenerationSession(
            jsonSchema: Self.aSchema, vocabulary: vocabulary, vocabType: .byteFallback, maxWhitespaceCount: 2)
        // { then a newline written as the byte token <0x0A>, then one space
        let fed = feed(&s, [Int32(base), 0x0A, space])
        #expect(fed)
        let allowed = allowedTokenIDs(session: &s)
        #expect(!allowed.contains(space))
        #expect(!allowed.contains(spaceQuote))
        #expect(!allowed.contains(0x0A))
        #expect(allowed.contains(quote))
    }

    @Test("The limit gives way when it would leave no token to continue with")
    func neverLeavesAnEmptyMask() throws {
        // The only token that can close the object starts with a space.
        let vocabulary = ["{", " }", " "]
        let schema = #"{"type": "object", "properties": {}, "additionalProperties": false}"#
        var s = try ConstrainedGenerationSession(
            jsonSchema: schema, vocabulary: vocabulary, vocabType: .raw, maxWhitespaceCount: 1)
        let fed = feed(&s, [0, 2])
        #expect(fed)
        // The run is at the limit, and every remaining token starts with whitespace. (The bitmask
        // has 32 bits for these 3 tokens; the 29 unused ones must not hide that.)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed == [1, 2])
    }

    @Test("The mask filled straight into a caller's buffer honors the limit too")
    func fillBitmaskHonorsTheLimit() throws {
        var s = try session(limit: 2)
        let fed = feed(&s, [TestConstants.openBraceToken, Self.space])
        #expect(fed)
        var words = [Int32](repeating: 0, count: (s.vocabularySize + 31) / 32)
        let before = words.withUnsafeMutableBufferPointer { s.fillBitmask(into: $0.baseAddress!) }
        #expect(before == .constrained)
        func allowed(_ words: [Int32], _ token: Int32) -> Bool {
            words[Int(token) / 32] & (1 << (token % 32)) != 0
        }
        #expect(allowed(words, Self.space))
        let atLimit = feed(&s, [Self.space])
        #expect(atLimit)
        let after = words.withUnsafeMutableBufferPointer { s.fillBitmask(into: $0.baseAddress!) }
        #expect(after == .constrained)
        for token in Self.whitespace { #expect(!allowed(words, token)) }
        #expect(allowed(words, TestConstants.quoteToken))
    }
}
