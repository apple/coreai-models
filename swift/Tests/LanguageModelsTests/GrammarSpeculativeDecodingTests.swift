// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Doubles

/// A target whose greedy continuation is a fixed script: the token after absolute position `p` is
/// `script[p + 1 - promptLength]`, then `stopToken`. `decoy` outscores the scripted token at every
/// position. Counts the verify passes it runs.
private final class ScriptedEngine: RowEngine, @unchecked Sendable {
    static let vocab = 256
    let script: [Int32]
    let promptLength: Int
    let stopToken: Int32
    let decoy: Int32?
    private(set) var verifyPasses = 0

    init(promptLength: Int, script: [Int32], stopToken: Int32, decoy: Int32? = nil) {
        self.promptLength = promptLength
        self.script = script
        self.stopToken = stopToken
        self.decoy = decoy
        super.init(maxContextLength: 4096, block: nil, tile: nil, maxVerifyBlockLength: Int.max)
    }

    override func push(_ token: Int32) -> [Float] {
        let index = processedTokenCount + 1 - promptLength
        var row = [Float](repeating: 0, count: Self.vocab)
        row[Int(index >= 0 && index < script.count ? script[index] : stopToken)] = 10
        if let decoy { row[Int(decoy)] = 20 }
        return row
    }

    override func truncate(to count: Int) {}

    override func forwardProposal(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (logits: [[LogitsScalarType]], features: DrafterFeatures?) {
        verifyPasses += 1
        return try await super.forwardProposal(anchor: anchor, proposal: proposal)
    }
}

/// Always proposes the same token, so a proposal that is accepted in a test came from the grammar.
private final class ConstantDrafter: DrafterEngine, @unchecked Sendable {
    let maxContextLength = 4096
    let token: Int32
    init(token: Int32) { self.token = token }
    func reset(to tokenIndex: Int) async throws {}
    func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using generator: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        DraftProposal(tokens: Array(repeating: token, count: count))
    }
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}
}

private let schema = """
    {"type": "object",
     "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
     "required": ["name", "age"], "additionalProperties": false}
    """
private let stop: Int32 = 2

private func makeConstraint(
    schema: String = schema, forcedRunLimit: Int, whitespaceLimit: Int? = nil
) throws -> GrammarConstraint {
    let session = try ConstrainedGenerationSession(
        jsonSchema: schema, tokenizer: MockTokenizer(), vocabSize: 256, vocabType: .raw,
        stopTokenIds: [stop], maxWhitespaceCount: whitespaceLimit)
    return GrammarConstraint(session, tokenizer: MockTokenizer(), forcedRunLimit: forcedRunLimit)
}

/// Grammar-constrained greedy decoding of `model` one token at a time: the reference a speculative
/// run must reproduce.
private func constrainedGreedy(
    model: HashModel, prompt: [Int32], maxNew: Int, constraint: GrammarConstraint
) -> [Int32] {
    var h: UInt64 = 0
    for token in prompt { h = model.step(h, token) }
    var out: [Int32] = []
    while out.count < maxNew {
        let next = constraint.maskedArgmax(model.row(h).map { LogitsScalarType($0) })
        constraint.accept(next)
        out.append(next)
        if next == stop { break }
        h = model.step(h, next)
    }
    return out
}

// MARK: - Tests

@Suite("Grammar-constrained speculative decoding")
struct GrammarSpeculativeDecodingTests {
    private let script = Array("{\"name\": \"Al\", \"age\": 7}".utf8).map { Int32($0) }

    private func run(
        forcedRunLimit: Int, decoy: Int32? = nil, drafterToken: Int32 = Int32(UInt8(ascii: "z"))
    ) async throws -> (tokens: [Int32], stats: SpeculativeDecoder.Stats, verifyPasses: Int) {
        let prompt: [Int32] = [112]
        let target = ScriptedEngine(
            promptLength: prompt.count, script: script, stopToken: stop, decoy: decoy)
        let decoder = SpeculativeDecoder(
            drafter: ConstantDrafter(token: drafterToken), target: target, numDraftTokens: 3)
        var emitted: [Int32] = []
        let stats = try await decoder.generateGreedy(
            promptTokens: prompt, maxNewTokens: 64, stopTokenIds: [stop],
            constraint: try makeConstraint(forcedRunLimit: forcedRunLimit),
            onToken: { emitted.append($0) })
        return (emitted, stats, target.verifyPasses)
    }

    @Test("Grammar proposals leave the output token for token identical and cut target passes")
    func grammarProposalsAreLosslessAndReducePasses() async throws {
        let expected = script + [stop]
        let without = try await run(forcedRunLimit: 0)
        let with = try await run(forcedRunLimit: GrammarConstraint.defaultForcedRunLimit)

        #expect(without.tokens == expected)
        #expect(with.tokens == expected)
        #expect(without.stats.grammarProposals == 0)
        #expect(with.stats.grammarProposals > 0)
        #expect(with.stats.grammarTokensAccepted == with.stats.grammarTokensProposed)
        #expect(with.verifyPasses < without.verifyPasses)
    }

    @Test("A drafted token the grammar forbids is never emitted")
    func forbiddenDraftsAreRejected() async throws {
        // A control character outscores the scripted token everywhere and is illegal in JSON, inside
        // a string or out of one, so the masked argmax must pick the scripted token, and the constant
        // drafter's `z` (illegal at the start) never lands.
        let result = try await run(forcedRunLimit: 0, decoy: 1)
        #expect(result.tokens == script + [stop])
        #expect(result.stats.draftTokensAccepted == 0)
    }

    @Test(
        "Output equals grammar-constrained greedy decoding of the target, whatever drafts and ties",
        arguments: [
            (levels: 3, k: 1, limit: 0), (levels: 3, k: 4, limit: 7), (levels: 1000, k: 4, limit: 7),
            (levels: 5, k: 7, limit: 3), (levels: 1000, k: 2, limit: 63),
        ])
    func matchesConstrainedGreedy(config: (levels: Int, k: Int, limit: Int)) async throws {
        let prompt: [Int32] = [5, 9]
        for seed in 0..<12 {
            let model = HashModel(vocab: 256, levels: config.levels, salt: UInt64(seed) &* 31)
            let expected = constrainedGreedy(
                model: model, prompt: prompt, maxNew: 48,
                constraint: try makeConstraint(forcedRunLimit: 0))
            let target = HashEngine(model: model, maxVerifyBlockLength: 16)
            let decoder = SpeculativeDecoder(
                drafter: FuzzDrafter(model: model, seed: UInt64(seed) &+ 99), target: target,
                numDraftTokens: config.k)
            var emitted: [Int32] = []
            _ = try await decoder.generateGreedy(
                promptTokens: prompt, maxNewTokens: 48, stopTokenIds: [stop],
                constraint: try makeConstraint(forcedRunLimit: config.limit),
                onToken: { emitted.append($0) })
            #expect(emitted == expected, "seed \(seed)")
        }
    }

    @Test("Stopping at maxNewTokens leaves the grammar exactly at the emitted stream")
    func grammarMatchesEmittedStreamAfterEarlyStop() async throws {
        for limit in [3, 5, 9, 12] {
            let target = ScriptedEngine(promptLength: 1, script: script, stopToken: stop)
            let decoder = SpeculativeDecoder(
                drafter: ConstantDrafter(token: Int32(UInt8(ascii: "z"))), target: target,
                numDraftTokens: 3)
            let constraint = try makeConstraint(forcedRunLimit: GrammarConstraint.defaultForcedRunLimit)
            var emitted: [Int32] = []
            _ = try await decoder.generateGreedy(
                promptTokens: [112], maxNewTokens: limit, stopTokenIds: [stop], constraint: constraint,
                onToken: { emitted.append($0) })
            #expect(emitted == Array(script.prefix(emitted.count)), "limit \(limit)")
            #expect(constraint.acceptedCount == emitted.count, "limit \(limit)")
        }
    }

    @Test("A grammar proposal does not advance the grammar until the decoder commits it")
    func forcedProposalIsSideEffectFree() throws {
        let constraint = try makeConstraint(forcedRunLimit: GrammarConstraint.defaultForcedRunLimit)
        // After `{"` the only legal key is `name`, so a forced span exists.
        constraint.accept(Int32(UInt8(ascii: "{")))
        constraint.accept(Int32(UInt8(ascii: "\"")))
        let first = try constraint.forcedProposal()
        let second = try constraint.forcedProposal()
        #expect(!first.isEmpty)
        #expect(first == second)
        #expect(first.count <= GrammarConstraint.defaultForcedRunLimit)
        #expect(constraint.acceptedCount == 2)
    }

    @Test("A zero run limit disables grammar proposals")
    func zeroLimitDisablesProposals() throws {
        let constraint = try makeConstraint(forcedRunLimit: 0)
        constraint.accept(Int32(UInt8(ascii: "{")))
        #expect(try constraint.forcedProposal().isEmpty)
    }

    @Test("A proposal never exceeds the span the grammar can roll back")
    func proposalIsBoundedByRollbackBudget() throws {
        let key = String(repeating: "k", count: 200)
        let longKey = """
            {"type": "object", "properties": {"\(key)": {"type": "integer"}},
             "required": ["\(key)"], "additionalProperties": false}
            """
        let constraint = try makeConstraint(schema: longKey, forcedRunLimit: 10_000)
        constraint.accept(Int32(UInt8(ascii: "{")))
        constraint.accept(Int32(UInt8(ascii: "\"")))
        let proposal = try constraint.forcedProposal()
        #expect(proposal.count == GrammarConstraint.maxForcedRunLimit)
    }

    @Test("The masked argmax never picks an illegal token, even one with the largest logit")
    func maskedArgmaxFollowsTheMask() throws {
        let z = Int32(UInt8(ascii: "z"))
        // The first token of an object is `{` or whitespace; `z` is illegal.
        for fill in [Float(0), -Float.infinity, Float.nan] {
            let constraint = try makeConstraint(forcedRunLimit: 0)
            var row = [LogitsScalarType](repeating: LogitsScalarType(fill), count: 256)
            row[Int(z)] = 50
            let choice = constraint.maskedArgmax(row)
            #expect(choice != z, "fill \(fill)")
            constraint.accept(choice)
            #expect(constraint.acceptedCount == 1, "the choice must be legal (fill \(fill))")
        }
    }

    @Test("A legal token beats every other when it has the largest logit")
    func maskedArgmaxPicksLegalMaximum() throws {
        let constraint = try makeConstraint(forcedRunLimit: 0)
        var row = [LogitsScalarType](repeating: 0, count: 256)
        row[Int(UInt8(ascii: "z"))] = 50
        row[Int(UInt8(ascii: "{"))] = 10
        #expect(constraint.maskedArgmax(row) == Int32(UInt8(ascii: "{")))
    }
}
