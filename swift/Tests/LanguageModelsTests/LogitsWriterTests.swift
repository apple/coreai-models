// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

@Suite("LogitsWriter")
struct LogitsWriterTests {
    @Test("saveTopKJSON writes valid JSON structure")
    func saveTopKJSON() throws {
        let tempFile = TempFile()
        defer { tempFile.cleanup() }

        let tokenLogits = [
            TokenLogits(
                tokenId: 42,
                tokenText: "hello",
                topLogits: [
                    TopLogitEntry(tokenId: 42, tokenText: "hello", logit: 5.5),
                    TopLogitEntry(tokenId: 10, tokenText: "world", logit: 3.2),
                ]
            ),
            TokenLogits(
                tokenId: 99,
                tokenText: "test",
                topLogits: [
                    TopLogitEntry(tokenId: 99, tokenText: "test", logit: 4.0)
                ]
            ),
        ]

        try LogitsWriter.saveTopKJSON(tokenLogits: tokenLogits, path: tempFile.path)

        // Verify JSON can be parsed
        let data = try tempFile.readData()
        let decoded = try JSONDecoder().decode(LogitsOutput.self, from: data)

        #expect(decoded.tokens.count == 2)
        #expect(decoded.tokens[0].tokenId == 42)
        #expect(decoded.tokens[0].topLogits.count == 2)
        #expect(decoded.tokens[1].tokenId == 99)
    }

    @Test("saveFullJSON produces valid base64-encoded logits")
    func saveFullJSON() throws {
        let tempFile = TempFile()
        defer { tempFile.cleanup() }

        let logits: [[Float16]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        let mockTokenizer = MockTokenizer()

        try LogitsWriter.saveFullJSON(
            logits: logits,
            generatedTokens: [65, 66],  // 'A', 'B' in UTF-8
            tokenizer: mockTokenizer,
            path: tempFile.path
        )

        let data = try tempFile.readData()
        let decoded = try JSONDecoder().decode(FullLogitsOutput.self, from: data)

        #expect(decoded.tokens.count == 2)

        // Verify base64 decodes to correct size (3 Float16 = 6 bytes)
        let base64Data = Data(base64Encoded: decoded.tokens[0].logitsBase64)
        #expect(base64Data?.count == 6)
    }

    @Test("saveFullJSON rejects empty logits")
    func rejectsEmptyLogits() throws {
        let tempFile = TempFile()
        defer { tempFile.cleanup() }

        let mockTokenizer = MockTokenizer()

        #expect(throws: LogitsWriterError.self) {
            try LogitsWriter.saveFullJSON(
                logits: [] as [[LogitsScalarType]],
                generatedTokens: [],
                tokenizer: mockTokenizer,
                path: tempFile.path
            )
        }
    }

    @Test("saveFullJSON rejects inconsistent vocab sizes")
    func rejectsInconsistentVocabSize() throws {
        let tempFile = TempFile()
        defer { tempFile.cleanup() }

        let logits: [[Float16]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],  // Different size!
        ]
        let mockTokenizer = MockTokenizer()

        #expect(throws: LogitsWriterError.self) {
            try LogitsWriter.saveFullJSON(
                logits: logits,
                generatedTokens: [65, 66],
                tokenizer: mockTokenizer,
                path: tempFile.path
            )
        }
    }

    @Test("saveFullJSON rejects token count mismatch with logits count")
    func rejectsTokenCountMismatch() throws {
        let tempFile = TempFile()
        defer { tempFile.cleanup() }

        let logits: [[Float16]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],  // 3 logits rows
        ]
        let mockTokenizer = MockTokenizer()

        #expect(throws: LogitsWriterError.self) {
            try LogitsWriter.saveFullJSON(
                logits: logits,
                generatedTokens: [65, 66],  // Only 2 tokens - mismatch!
                tokenizer: mockTokenizer,
                path: tempFile.path
            )
        }
    }

    @Test("extractTopK returns correct top entries")
    func extractTopK() {
        let logits: [Float16] = [0.1, 0.9, 0.3, 0.5, 0.2]
        let mockTokenizer = MockTokenizer()

        let top3 = LogitsWriter.extractTopK(from: logits, tokenizer: mockTokenizer, k: 3)

        #expect(top3.count == 3)
        #expect(top3[0].tokenId == 1)  // Index of 0.9
        #expect(abs(top3[0].logit - 0.9) < 0.01)  // Float16 precision tolerance
        #expect(top3[1].tokenId == 3)  // Index of 0.5
        #expect(top3[2].tokenId == 2)  // Index of 0.3
    }

    @Test("extractTopK ranks Float16, Float and Double rows alike, ties toward the lower index")
    func extractTopKElementTypes() {
        let logits: [Float16] = [0.1, 0.5, 0.3, 0.5, 0.2]
        let mockTokenizer = MockTokenizer()

        let fromHalf = LogitsWriter.extractTopK(from: logits, tokenizer: mockTokenizer, k: 3)
        let fromFloat = LogitsWriter.extractTopK(from: logits.map { Float($0) }, tokenizer: mockTokenizer, k: 3)
        let fromDouble = LogitsWriter.extractTopK(from: logits.map { Double($0) }, tokenizer: mockTokenizer, k: 3)

        #expect(fromHalf.map(\.tokenId) == [1, 3, 2])
        #expect(fromFloat.map(\.tokenId) == [1, 3, 2])
        #expect(fromDouble.map(\.tokenId) == [1, 3, 2])
    }
}

@Suite("LogitsLength")
struct LogitsLengthTests {
    @Test("Parse 'full' creates full variant")
    func parseFull() {
        let length = LogitsLength(argument: "full")

        if case .full = length {
            // Success
        } else {
            Issue.record("Expected .full variant")
        }
    }

    @Test("Parse integer within range creates count variant")
    func parseValidInteger() {
        let length = LogitsLength(argument: "10")

        if case .count(let n) = length {
            #expect(n == 10)
        } else {
            Issue.record("Expected .count variant")
        }
    }

    @Test("Parse rejects out-of-range values")
    func rejectsOutOfRange() {
        #expect(LogitsLength(argument: "25") == nil)
        #expect(LogitsLength(argument: "-1") == nil)
    }

    @Test("Helper properties return correct values")
    func helperProperties() {
        #expect(LogitsLength.full.isFull == true)
        #expect(LogitsLength.count(5).isFull == false)
        #expect(LogitsLength.full.topKForConsole == 5)
        #expect(LogitsLength.count(10).topKForConsole == 10)
        #expect(LogitsLength.full.topKForFile == nil)
        #expect(LogitsLength.count(3).topKForFile == 3)
    }
}
