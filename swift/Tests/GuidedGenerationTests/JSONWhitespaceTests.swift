// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("JSONWhitespace")
struct JSONWhitespaceTests {
    private static let spaceToken: Int32 = 7
    private static let newlineToken: Int32 = 8
    private static let tabToken: Int32 = 9

    private func session(_ whitespace: JSONWhitespace) throws -> ConstrainedGenerationSession {
        try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema,
            vocabulary: sharedToyVocab,
            vocabType: .raw,
            whitespace: whitespace)
    }

    @Test("Default initializer compiles the compact grammar")
    func defaultIsCompact() throws {
        let session = try createTestSession()
        let whitespace = session.whitespace
        #expect(whitespace == .compact)
    }

    @Test("Session records the whitespace policy it was compiled with")
    func sessionRecordsPolicy() throws {
        let flexible = try session(.flexible).whitespace
        let compact = try session(.compact).whitespace
        #expect(flexible == .flexible)
        #expect(compact == .compact)
    }

    @Test("Flexible grammar accepts spaces, newlines, and tabs between tokens")
    func flexibleAllowsAnyWhitespace() throws {
        var s = try session(.flexible)
        let openAccepted = s.acceptToken(TestConstants.openBraceToken)
        #expect(openAccepted)
        let allowed = allowedTokenIDs(session: &s)
        #expect(allowed.contains(Self.spaceToken))
        #expect(allowed.contains(Self.newlineToken))
        #expect(allowed.contains(Self.tabToken))
        #expect(allowed.contains(TestConstants.quoteToken))
    }

    @Test("Compact grammar rejects newlines and tabs; the next key is fully determined")
    func compactRejectsNewlines() throws {
        var s = try session(.compact)
        let openAccepted = s.acceptToken(TestConstants.openBraceToken)
        #expect(openAccepted)
        let allowed = allowedTokenIDs(session: &s)
        #expect(!allowed.contains(Self.newlineToken))
        #expect(!allowed.contains(Self.tabToken))
        #expect(allowed == [TestConstants.quoteToken])
    }

    @Test("Compact grammar still accepts its fixed separators")
    func compactAcceptsFixedSeparators() throws {
        var s = try session(.compact)
        // {"name": "John", "age": 25}
        let tokens: [Int32] = [0, 6, 10, 6, 4, 7, 6, 17, 6, 5, 7, 6, 11, 6, 4, 7, 38, 1]
        for token in tokens {
            let accepted = s.acceptToken(token)
            #expect(accepted, "compact grammar rejected token \(token)")
        }
    }

    @Test("Compact grammar ends the document once the schema is satisfied")
    func compactEndsAfterLastValue() throws {
        var s = try session(.compact)
        let tokens: [Int32] = [0, 6, 10, 6, 4, 7, 6, 17, 6, 5, 7, 6, 11, 6, 4, 7, 38, 1]
        for token in tokens { _ = s.acceptToken(token) }
        // After the closing brace the grammar is complete: whitespace is not accepted.
        let allowed = allowedTokenIDs(session: &s)
        #expect(!allowed.contains(Self.spaceToken))
        #expect(!allowed.contains(Self.newlineToken))
    }
}
