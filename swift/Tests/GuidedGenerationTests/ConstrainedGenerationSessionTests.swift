// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("ConstrainedGenerationSession")
struct ConstrainedGenerationSessionTests {
    // MARK: - Initialization Tests

    @Test func initWithValidSchema() throws {
        let session = try createTestSession()

        let isTerminated = session.isTerminated
        #expect(!isTerminated)
        #expect(session.schema == sharedPersonSchema)
        #expect(session.compiledGrammarMemoryBytes > 0)
    }

    @Test func initWithInvalidSchemaThrows() throws {
        #expect(throws: (any Error).self) {
            _ = try ConstrainedGenerationSession(
                jsonSchema: "not valid json {{{",
                vocabulary: sharedToyVocab,
                vocabType: .raw
            )
        }
    }

    @Test func initWithTokenizerInfo() throws {
        let tokenizerInfo = TokenizerInfo(
            vocabulary: sharedToyVocab,
            vocabType: .raw
        )

        let session = try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema,
            tokenizerInfo: tokenizerInfo
        )

        let isTerminated = session.isTerminated
        #expect(!isTerminated)
    }

    // MARK: - Bitmask Tests

    @Test func firstTokenMustBeOpenBrace() throws {
        var session = try createTestSession()

        let allowed = allowedTokenIDs(session: &session)

        // Token 0 is "{" - must be allowed for JSON object
        #expect(
            allowed.contains(TestConstants.openBraceToken), "'{' (token 0) should be allowed at start of JSON object")
        // Token 1 is "}" - should NOT be allowed at start
        #expect(!allowed.contains(TestConstants.closeBraceToken), "'}' should not be allowed at start")
    }

    @Test func getNextTokenBitmask() throws {
        var session = try createTestSession()

        let bitmask = session.nextTokenBitmask()
        #expect(bitmask != nil)
        #expect(bitmask!.count > 0)

        // Check that "{" (token 0) is allowed via bitmask
        let word0 = bitmask![0]
        #expect((word0 & 1) != 0, "Token 0 ('{') should be set in bitmask")
    }

    @Test func bitmaskNilWhenTerminated() throws {
        var session = try createTestSession()

        // Drive the session to completion with simulated generation
        driveToCompletion(session: &session)

        let isTerminated = session.isTerminated
        #expect(isTerminated)
        #expect(session.nextTokenBitmask() == nil)
        #expect(allowedTokenIDs(session: &session).isEmpty)
    }

    @Test func fillBitmaskIntoPointerMatchesNextTokenBitmask() throws {
        var session = try createTestSession()

        // Get bitmask via the array-returning method
        let arrayBitmask = session.nextTokenBitmask()
        #expect(arrayBitmask != nil)

        // Reset and get bitmask via the pointer method
        session.reset()
        let bitmaskSize = (session.vocabularySize + 31) / 32
        var pointerBitmask = [Int32](repeating: 0, count: bitmaskSize)
        let success = pointerBitmask.withUnsafeMutableBufferPointer { buf in
            session.fillBitmask(into: buf.baseAddress!)
        }
        #expect(success)

        // They should be identical
        #expect(arrayBitmask!.count == pointerBitmask.count)
        for i in 0..<arrayBitmask!.count {
            #expect(
                arrayBitmask![i] == pointerBitmask[i],
                "Mismatch at word \(i): array=\(arrayBitmask![i]) pointer=\(pointerBitmask[i])")
        }
    }

    @Test func fillBitmaskReturnsFalseWhenTerminated() throws {
        var session = try createTestSession()
        driveToCompletion(session: &session)

        let bitmaskSize = (session.vocabularySize + 31) / 32
        var buffer = [Int32](repeating: 0, count: bitmaskSize)
        let success = buffer.withUnsafeMutableBufferPointer { buf in
            session.fillBitmask(into: buf.baseAddress!)
        }
        #expect(!success)
    }

    // MARK: - Token Acceptance Tests

    @Test func acceptToken() throws {
        var session = try createTestSession()

        // Accept "{" (token 0) - should succeed
        let accepted = session.acceptToken(TestConstants.openBraceToken)
        #expect(accepted, "'{' should be accepted at start")
        let isTerminated = session.isTerminated
        #expect(!isTerminated)
    }

    @Test func allowedTokensUpdateAfterAccept() throws {
        var session = try createTestSession()
        let beforeAccept = allowedTokenIDs(session: &session)
        _ = session.acceptToken(TestConstants.openBraceToken)  // accept "{"
        let afterAccept = allowedTokenIDs(session: &session)
        #expect(beforeAccept != afterAccept, "Allowed tokens should change after accepting '{'")
        // After "{", only "\"" (token 6) should be allowed (to start key names)
        #expect(afterAccept.contains(TestConstants.quoteToken), "'\"' should be allowed after '{'")
        #expect(
            !afterAccept.contains(TestConstants.openBraceToken), "'{' should not be allowed again immediately after '{'"
        )
    }

    // MARK: - Reset Tests

    @Test func reset() throws {
        var session = try createTestSession()

        // Accept a token
        session.acceptToken(TestConstants.openBraceToken)  // "{"
        let allowedAfterBrace = allowedTokenIDs(session: &session)

        // Reset
        session.reset()

        // Should be back at initial state
        let allowedAfterReset = allowedTokenIDs(session: &session)
        #expect(allowedAfterReset.contains(TestConstants.openBraceToken), "'{' should be allowed after reset")
        #expect(allowedAfterBrace != allowedAfterReset, "State should differ before/after reset")
    }

    @Test func resetAfterCompletion() throws {
        var session = try createTestSession()
        driveToCompletion(session: &session)
        let isTerminatedBefore = session.isTerminated
        #expect(isTerminatedBefore)
        #expect(session.nextTokenBitmask() == nil)

        session.reset()

        let isTerminatedAfter = session.isTerminated
        #expect(!isTerminatedAfter, "Should not be terminated after reset")
        let allowed = allowedTokenIDs(session: &session)
        #expect(
            allowed.contains(TestConstants.openBraceToken), "'{' should be allowed after reset from completed state")
        #expect(!allowed.isEmpty, "Allowed tokens should not be empty after reset")
    }

    @Test func multipleResets() throws {
        var session = try createTestSession()
        for _ in 0..<3 {
            _ = session.acceptToken(TestConstants.openBraceToken)  // "{"
            session.reset()
            let allowed = allowedTokenIDs(session: &session)
            #expect(allowed.contains(TestConstants.openBraceToken), "'{' should be allowed after each reset")
        }
    }

    // MARK: - Full Generation Tests

    @Test func fullConstrainedGeneration() throws {
        var session = try createTestSession()

        let generatedText = driveToCompletion(session: &session)

        let isTerminated = session.isTerminated
        #expect(isTerminated, "Grammar should terminate after valid JSON")

        // Validate that output is valid JSON
        let jsonData = try #require(generatedText.data(using: .utf8), "Generated text should be valid UTF-8")
        #expect(
            (try? JSONSerialization.jsonObject(with: jsonData)) != nil,
            "Generated text is not valid JSON: \(generatedText)"
        )
    }

    @Test func enumConstraint() throws {
        let enumSchema = """
            {
              "type": "object",
              "properties": {
                "status": {
                  "type": "string",
                  "enum": ["user", "admin", "guest"]
                }
              },
              "required": ["status"],
              "additionalProperties": false
            }
            """

        var session = try createTestSession(schema: enumSchema)

        let generatedText = driveToCompletion(session: &session)

        let jsonData = try #require(generatedText.data(using: .utf8), "Generated text should be valid UTF-8")
        let obj = try #require(
            try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
            "Could not parse output: \(generatedText)"
        )
        let status = try #require(obj["status"] as? String, "Missing 'status' key")

        #expect(
            ["user", "admin", "guest"].contains(status),
            "status '\(status)' should be one of the enum values"
        )
    }

    @Test func integerConstraint() throws {
        let intSchema = """
            {
              "type": "object",
              "properties": {
                "age": {"type": "integer"}
              },
              "required": ["age"],
              "additionalProperties": false
            }
            """

        var session = try createTestSession(schema: intSchema)

        let generatedText = driveToCompletion(session: &session)

        let jsonData = try #require(generatedText.data(using: .utf8), "Generated text should be valid UTF-8")
        #expect(
            (try? JSONSerialization.jsonObject(with: jsonData)) != nil,
            "Output is not valid JSON: \(generatedText)"
        )
    }

    @Test func nestedObjectSchema() throws {
        let nestedSchema = """
            {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "address": {
                  "type": "object",
                  "properties": {
                    "city": {"type": "string"}
                  },
                  "required": ["city"],
                  "additionalProperties": false
                }
              },
              "required": ["name", "address"],
              "additionalProperties": false
            }
            """
        var session = try createTestSession(schema: nestedSchema)
        // Verify the session initialises correctly and produces a valid initial bitmask
        let isTerminatedInit = session.isTerminated
        #expect(!isTerminatedInit)
        #expect(session.compiledGrammarMemoryBytes > 0)
        let initialAllowed = allowedTokenIDs(session: &session)
        #expect(
            initialAllowed.contains(TestConstants.openBraceToken),
            "'{' should be allowed at start of nested object schema")
        // Accept "{" and verify the state advances (grammar accepts the first token)
        let accepted = session.acceptToken(TestConstants.openBraceToken)
        #expect(accepted, "'{' should be accepted by the nested object schema")
        let isTerminatedAfter = session.isTerminated
        #expect(!isTerminatedAfter, "Should not be terminated after single '{' token")
    }

    @Test func arraySchema() throws {
        let arraySchema = """
            {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
            }
            """
        var session = try createTestSession(schema: arraySchema)
        let generatedText = driveToCompletion(session: &session)
        let isTerminated = session.isTerminated
        #expect(isTerminated)
        let data = try #require(generatedText.data(using: .utf8), "Generated text should be valid UTF-8")
        #expect(
            (try? JSONSerialization.jsonObject(with: data)) != nil,
            "Array schema output is not valid JSON: \(generatedText)"
        )
    }

    @Test func allTokensBlockedTermination() throws {
        var session = try createTestSession()
        // Drive to completion -- xgrammar will return all-zeros bitmask when JSON is complete
        driveToCompletion(session: &session)

        // isTerminated must be true (either via matcher.isTerminated or allTokensBlocked)
        let isTerminated = session.isTerminated
        #expect(isTerminated, "Session must be terminated after complete JSON")
        // nextTokenBitmask must return nil when terminated
        #expect(session.nextTokenBitmask() == nil, "nextTokenBitmask must return nil when terminated")
        // applyMask must return false when terminated (logits unchanged)
        var logits = [Float](repeating: 1.0, count: sharedToyVocab.count)
        let maskApplied = session.applyMask(to: &logits)
        #expect(!maskApplied, "applyMask must return false when terminated")
        #expect(logits[0] == 1.0, "Logits must be unchanged when terminated")
    }

    // MARK: - Logit Masking Tests (Float16)

    @Test func applyMaskDisallowsTokens() throws {
        var session = try createTestSession()

        // Create fake logits (all 1.0)
        var logits = [Float16](repeating: Float16(1.0), count: sharedToyVocab.count)
        let applied = session.applyMask(to: &logits)

        #expect(applied, "applyMask should return true when mask is applied")
        #expect(logits.count == sharedToyVocab.count)

        // "{" should still have its original logit
        #expect(logits[Int(TestConstants.openBraceToken)] == Float16(1.0), "'{' logit should be unchanged")

        // "}" should be masked out
        #expect(logits[Int(TestConstants.closeBraceToken)] == Float16(-65504.0), "'}' logit should be masked to -65504")
    }

    // MARK: - Logit Masking Tests (Float)

    @Test func applyMaskFloatDisallowsTokens() throws {
        var session = try createTestSession()

        var logits = [Float](repeating: 1.0, count: sharedToyVocab.count)
        let applied = session.applyMask(to: &logits)

        #expect(applied, "applyMask should return true when mask is applied")
        #expect(logits[Int(TestConstants.openBraceToken)] == 1.0, "'{' logit should be unchanged")
        #expect(logits[Int(TestConstants.closeBraceToken)] == -.infinity, "'}' logit should be -inf")
    }

    @Test func applyMaskReturnsFalseWhenTerminated() throws {
        var session = try createTestSession()

        driveToCompletion(session: &session)
        let isTerminated = session.isTerminated
        #expect(isTerminated)

        var logits = [Float16](repeating: Float16(1.0), count: sharedToyVocab.count)
        let applied = session.applyMask(to: &logits)
        #expect(!applied, "applyMask should return false when terminated")
        // Logits should be unchanged
        #expect(logits[0] == Float16(1.0))
    }

    @Test func applyMaskAndArgmaxPicksAllowedToken() throws {
        var session = try createTestSession()

        // Create logits where "{" (token 0) has highest value among allowed tokens
        var logits = [Float16](repeating: Float16(0.0), count: sharedToyVocab.count)
        logits[Int(TestConstants.openBraceToken)] = Float16(10.0)  // "{" = high
        logits[Int(TestConstants.closeBraceToken)] = Float16(20.0)  // "}" = higher, but should be masked

        session.applyMask(to: &logits)

        // Find argmax
        let sampled = logits.enumerated().max(by: { $0.element < $1.element })?.offset
        #expect(sampled != nil)
        // "}" is not allowed at start, so "{" should win
        #expect(sampled == Int(TestConstants.openBraceToken), "Should sample '{' since '}' is masked")
    }

    @Test func applyMaskPreservesAllowedLogits() throws {
        var session = try createTestSession()
        // Set up logits: all -1.0 except token 0 ("{") at 5.0
        var logits = [Float16](repeating: Float16(-1.0), count: sharedToyVocab.count)
        logits[Int(TestConstants.openBraceToken)] = Float16(5.0)  // "{"
        logits[Int(TestConstants.closeBraceToken)] = Float16(5.0)  // "}" -- should be masked

        session.applyMask(to: &logits)

        #expect(logits[Int(TestConstants.openBraceToken)] == Float16(5.0), "Allowed token logit should be unchanged")
        #expect(
            logits[Int(TestConstants.closeBraceToken)] == -Float16.greatestFiniteMagnitude,
            "Disallowed token should be -greatestFiniteMagnitude")
    }
}
