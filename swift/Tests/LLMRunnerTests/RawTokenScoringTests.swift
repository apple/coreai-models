// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation
import Testing

@testable import llm_runner

/// Covers how `llm-runner` decides between generating and scoring from `--raw-tokens`, each
/// refusal it makes before the model loads, and scoring under `--json-schema`.
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

    private func request(
        path: String?,
        context: [Int32]? = [1, 2, 3],
        textContinuation: Bool = false,
        logits: Bool = true,
        conflictingFlags: [String] = []
    ) throws -> RawTokenScoring.Request? {
        let contextFile = try context.map(tokensFile)
        defer { contextFile?.cleanup() }
        return try RawTokenScoring.request(
            rawContinuationPath: path,
            rawTokensPath: contextFile?.path,
            textContinuationGiven: textContinuation,
            logitsRequested: logits,
            conflictingFlags: conflictingFlags)
    }

    /// Expects `body` to throw `expected`.
    private func expectRefusal(
        _ expected: ContinuationOptionError, sourceLocation: SourceLocation = #_sourceLocation,
        _ body: () throws -> Void
    ) {
        #expect(throws: expected, sourceLocation: sourceLocation, performing: body)
    }

    // MARK: Deciding the mode

    @Test("--raw-continuation decides between generating and scoring")
    func rawContinuationDecidesTheMode() throws {
        #expect(try request(path: nil) == nil)
        #expect(try request(path: nil, logits: false) == nil)
        #expect(try request(path: nil, conflictingFlags: ["--image"]) == nil)

        let file = try tokensFile([7, 8, 9])
        defer { file.cleanup() }
        #expect(try request(path: file.path) == .init(context: [1, 2, 3], continuation: [7, 8, 9]))
    }

    // MARK: Refusals before the model loads

    @Test("--raw-continuation without --raw-tokens is refused")
    func refusesWithoutRawTokens() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        expectRefusal(.rawContinuationRequiresRawTokens) { _ = try request(path: file.path, context: nil) }
    }

    @Test("--raw-continuation without a logits output is refused")
    func refusesWithoutLogitsOutput() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        expectRefusal(.rawContinuationRequiresLogitsOutput) { _ = try request(path: file.path, logits: false) }
    }

    @Test("--raw-continuation with the text --continuation is refused")
    func refusesWithTextContinuation() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        expectRefusal(.rawContinuationConflicts(flag: "--continuation")) {
            _ = try request(path: file.path, textContinuation: true)
        }
    }

    @Test(
        "--raw-continuation with a flag that takes its own path is refused",
        arguments: ["--batch-prototype", "--image", "--video"])
    func refusesConflictingFlag(flag: String) throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        expectRefusal(.rawContinuationConflicts(flag: flag)) {
            _ = try request(path: file.path, conflictingFlags: [flag])
        }
        #expect(
            ContinuationOptionError.rawContinuationConflicts(flag: flag).errorDescription
                == "--raw-continuation cannot be used with \(flag)")
    }

    @Test("An empty --raw-continuation file is refused")
    func refusesEmptyContinuation() throws {
        let file = try tokensFile([])
        defer { file.cleanup() }
        expectRefusal(.emptyRawContinuation(path: file.path)) { _ = try request(path: file.path) }
    }

    @Test("An empty --raw-tokens context is refused before the model loads")
    func refusesEmptyContext() throws {
        let file = try tokensFile([7])
        defer { file.cleanup() }
        #expect {
            _ = try request(path: file.path, context: [])
        } throws: { error in
            guard case .emptyRawContext = error as? ContinuationOptionError else { return false }
            return true
        }
    }

    @Test("An empty --raw-tokens file is refused when the run generates")
    func refusesEmptyGenerationContext() throws {
        #expect {
            _ = try request(path: nil, context: [])
        } throws: { error in
            guard case .emptyRawTokens = error as? ContinuationOptionError else { return false }
            return true
        }
        // A file that cannot be read is left to the prompt resolver.
        #expect(
            try RawTokenScoring.request(
                rawContinuationPath: nil, rawTokensPath: "/nonexistent/tokens.json",
                textContinuationGiven: false, logitsRequested: true) == nil)
    }

    // MARK: Migration notice

    @Test("--raw-tokens with a logits flag and no --raw-continuation prints the migration notice")
    func printsGenerationNotice() {
        let notice = RawTokenScoring.generationNotice(
            rawTokensGiven: true, rawContinuationGiven: false, textContinuationGiven: false, logitsRequested: true)
        #expect(notice?.contains("--raw-tokens [t0] --raw-continuation [t1, ..., tn]") == true)

        for (rawTokens, rawContinuation, textContinuation, logits) in [
            (false, false, false, true),
            (true, true, false, true),
            (true, false, true, true),
            (true, false, false, false),
        ] {
            #expect(
                RawTokenScoring.generationNotice(
                    rawTokensGiven: rawTokens, rawContinuationGiven: rawContinuation,
                    textContinuationGiven: textContinuation, logitsRequested: logits) == nil)
        }
    }

    @Test("A missing or malformed --raw-continuation file is refused")
    func refusesUnreadableContinuation() throws {
        let malformed = TokensFile()
        defer { malformed.cleanup() }
        try Data("{\"ids\": [1, 2]}".utf8).write(to: malformed.url)
        let missing = TokensFile()

        for path in [malformed.path, missing.path] {
            // The reason is the decoder's or the file system's wording, so only the path is checked.
            #expect {
                _ = try request(path: path)
            } throws: { error in
                guard case .unreadableRawTokens(let reported, _) = error as? ContinuationOptionError else {
                    return false
                }
                return reported == path
            }
        }
    }

    // MARK: Fitting the model

    @Test("Ids that fit the vocabulary and the context length are accepted")
    func acceptsIdsThatFit() throws {
        // Ids 0 and vocabSize - 1, and exactly maxContextLength ids in total.
        try RawTokenScoring.checkFits(
            .init(context: [0, 1, 2], continuation: [99]), vocabSize: 100, maxContextLength: 4)
    }

    @Test(
        "An id outside the vocabulary is refused, naming the file that holds it",
        arguments: [
            ([100], [1], "--raw-tokens", Int32(100)),
            ([1], [1, -1], "--raw-continuation", Int32(-1)),
            ([1], [250], "--raw-continuation", Int32(250)),
        ] as [([Int32], [Int32], String, Int32)])
    func refusesIdOutsideVocabulary(case: (context: [Int32], continuation: [Int32], flag: String, id: Int32)) {
        expectRefusal(.tokenOutsideVocabulary(flag: `case`.flag, id: `case`.id, vocabSize: 100)) {
            try RawTokenScoring.checkFits(
                .init(context: `case`.context, continuation: `case`.continuation), vocabSize: 100,
                maxContextLength: 64)
        }
    }

    @Test("A context plus continuation longer than the model's context length is refused")
    func refusesMoreIdsThanTheContextHolds() {
        expectRefusal(.exceedsContextLength(context: 3, continuation: 2, maxContextLength: 4)) {
            try RawTokenScoring.checkFits(
                .init(context: [1, 2, 3], continuation: [4, 5]), vocabSize: 100, maxContextLength: 4)
        }
    }

    // MARK: Scoring under --json-schema

    /// Ids into `grammarVocab`.
    private enum Id {
        static let open: Int32 = 0, close: Int32 = 1, quote: Int32 = 2, colon: Int32 = 3
        static let a: Int32 = 4, one: Int32 = 5, b: Int32 = 6, end: Int32 = 7
    }

    private static let grammarVocab = ["{", "}", "\"", ":", "a", "1", "b", "<|im_end|>"]

    private static let schema = """
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}
        """

    /// The grammar with `end` as its stop token, as llm-runner gives it the stops generation uses.
    private func session() throws -> ConstrainedGenerationSession {
        try ConstrainedGenerationSession(
            jsonSchema: Self.schema, vocabulary: Self.grammarVocab, vocabType: .raw, stopTokenIds: [Id.end])
    }

    /// One flat logits row per continuation id, so masking alone decides the log-probabilities.
    private func flatResult(_ continuation: [Int32]) -> ContinuationEvaluationResult {
        ContinuationEvaluationResult(
            contextTokens: [9],
            continuationTokens: continuation,
            logits: Array(repeating: Array(repeating: 0, count: Self.grammarVocab.count), count: continuation.count))
    }

    @Test("A continuation the schema allows is scored over the ids the schema allows")
    func masksAnAllowedContinuation() throws {
        var grammar = try session()
        let continuation = [Id.open, Id.quote, Id.a, Id.quote, Id.colon, Id.one, Id.close]
        let unmasked = flatResult(continuation)
        let masked = try RawTokenScoring.maskedByGrammar(unmasked, session: &grammar)

        #expect(masked.contextTokens == unmasked.contextTokens)
        #expect(masked.continuationTokens == continuation)
        #expect(masked.logits.count == continuation.count)
        // `{` is the only id a JSON object can start with, so it is certain once masked.
        #expect(abs(masked.targetProbabilities()[0] - 1) < 1e-6)
        // `b` is no property of the schema, so it is masked out where `a` is the next id.
        #expect(masked.logits[2][Int(Id.b)] < -1000)
        #expect(masked.logProbability() > unmasked.logProbability())
    }

    @Test("A continuation id the schema does not allow is refused at its position")
    func refusesAnIdTheSchemaDoesNotAllow() throws {
        var grammar = try session()
        expectRefusal(.continuationViolatesSchema(position: 2, id: Id.b)) {
            _ = try RawTokenScoring.maskedByGrammar(flatResult([Id.open, Id.quote, Id.b]), session: &grammar)
        }
    }

    private static let complete = [Id.open, Id.quote, Id.a, Id.quote, Id.colon, Id.one, Id.close]

    @Test("Once the JSON is complete, only a stop token is allowed")
    func onlyAStopTokenFollowsTheJSON() throws {
        var grammar = try session()
        expectRefusal(.continuationViolatesSchema(position: 7, id: Id.open)) {
            _ = try RawTokenScoring.maskedByGrammar(flatResult(Self.complete + [Id.open]), session: &grammar)
        }
    }

    @Test("A continuation that ends in a stop token is scored, as generation would end it")
    func scoresAStopTokenAfterTheJSON() throws {
        var grammar = try session()
        let masked = try RawTokenScoring.maskedByGrammar(flatResult(Self.complete + [Id.end]), session: &grammar)
        // The stop token is the only id allowed there, so it is certain once masked.
        #expect(abs(masked.targetProbabilities()[7] - 1) < 1e-6)
    }

    @Test("A continuation that runs past its stop token is refused")
    func refusesIdsAfterTheStopToken() throws {
        var grammar = try session()
        expectRefusal(.continuationPastSchemaEnd(position: 8, id: Id.open)) {
            _ = try RawTokenScoring.maskedByGrammar(flatResult(Self.complete + [Id.end, Id.open]), session: &grammar)
        }
    }
}
