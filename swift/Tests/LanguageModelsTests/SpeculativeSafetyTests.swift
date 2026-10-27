// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// Inputs that would trap or produce NaN throw, or yield valid distributions, instead.
@Suite("Speculative decoding input safety")
struct SpeculativeSafetyTests {
    @Test("Softmax of an all negative-infinity row is a valid distribution")
    func softmaxOfAllNegativeInfinity() {
        let row = [LogitsScalarType](repeating: -.infinity, count: 4)
        let p = SpeculativeMath.temperatureSoftmax(row, temperature: 1)
        #expect(p.allSatisfy { !$0.isNaN })
        #expect(abs(p.reduce(0, +) - 1) < 1e-6)
    }

    @Test("A NaN logit takes no probability mass")
    func softmaxIgnoresNaN() {
        let p = SpeculativeMath.temperatureSoftmax([.nan, 1, 2] as [LogitsScalarType], temperature: 1)
        #expect(p.allSatisfy { !$0.isNaN })
        #expect(p[0] == 0)
        #expect(abs(p.reduce(0, +) - 1) < 1e-6)
    }

    @Test("A near-zero temperature stays finite")
    func tinyTemperature() {
        let p = SpeculativeMath.temperatureSoftmax([0, 1, 2, 3] as [LogitsScalarType], temperature: 1e-5)
        #expect(p.allSatisfy { !$0.isNaN })
        #expect(p[3] > 0.99)
    }

    private func decoderWithoutDraftLogits(
        lookup: Bool
    ) -> SpeculativeDecoder {
        let model = HashModel(vocab: 4, levels: 1, salt: 1)  // constant logits: lookup always matches
        let decoder: SpeculativeDecoder
        if lookup {
            decoder = SpeculativeDecoder(
                draft: HashEngine(model: model),
                target: HashEngine(model: model), numDraftTokens: 3)
            decoder.promptLookup = PromptLookup()
        } else {
            decoder = SpeculativeDecoder(
                drafter: PromptLookupDrafter(),
                target: HashEngine(model: model), numDraftTokens: 3)
        }
        return decoder
    }

    @Test("Sampling with prompt lookup throws instead of trapping")
    func samplingWithPromptLookupThrows() async {
        let decoder = decoderWithoutDraftLogits(lookup: true)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateSampling(
                promptTokens: [0, 0, 0, 0, 0, 0], maxNewTokens: 8, stopTokenIds: [], temperature: 1,
                onToken: { _ in })
        }
    }

    @Test("Sampling with a drafter that supplies no logits throws instead of trapping")
    func samplingWithLogitlessDrafterThrows() async {
        let decoder = decoderWithoutDraftLogits(lookup: false)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateSampling(
                promptTokens: [0, 0, 0, 0, 0, 0], maxNewTokens: 8, stopTokenIds: [], temperature: 1,
                onToken: { _ in })
        }
    }

    @Test("Sampling needs a positive finite temperature")
    func samplingTemperatureIsValidated() async {
        let model = HashModel(vocab: 4, levels: 3, salt: 1)
        let decoder = SpeculativeDecoder(
            draft: HashEngine(model: model),
            target: HashEngine(model: model), numDraftTokens: 2)
        for temperature in [0.0, -1.0, .nan, .infinity] {
            await #expect(throws: InferenceRuntimeError.self) {
                _ = try await decoder.generateSampling(
                    promptTokens: [0, 1], maxNewTokens: 4, stopTokenIds: [], temperature: temperature,
                    onToken: { _ in })
            }
        }
    }

    @Test("A verify that returns fewer rows than slots throws instead of indexing past the end")
    func shortVerifyRowsThrow() async {
        let model = HashModel(vocab: 6, levels: 1000, salt: 3)
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 9),
            target: ShortRowsEngine(inner: HashEngine(model: model)),
            numDraftTokens: 4)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateGreedy(
                promptTokens: [1, 2], maxNewTokens: 10, stopTokenIds: [], onToken: { _ in })
        }
    }
}

extension SpeculativeSafetyTests {
    @Test("Sampling runs to the end of the context, where no draft fits")
    func samplingReachesTheEndOfTheContext() async throws {
        for maxContext in Array(6...40) {
            let model = HashModel(vocab: 40, levels: 1000, salt: UInt64(maxContext))
            let decoder = SpeculativeDecoder(
                draft: HashEngine(model: model, maxContextLength: maxContext),
                target: HashEngine(model: model, maxContextLength: maxContext),
                numDraftTokens: 4)
            var got: [Int32] = []
            _ = try await decoder.generateSampling(
                promptTokens: [1], maxNewTokens: 1000, stopTokenIds: [], temperature: 1,
                onToken: { got.append($0) })
            // The context fills exactly as it does for plain decoding.
            let plain = plainGreedy(
                model: model, prompt: [1], maxNew: 1000, stops: [], capacity: maxContext)
            #expect(got.count == plain.count, "maxContext \(maxContext)")
        }
    }
}

extension SpeculativeSafetyTests {
    @Test("A prompt longer than the context is refused before anything is prefilled")
    func promptLongerThanTheContext() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 3)
        let target = HashEngine(model: model, maxContextLength: 8)
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 1, maxContextLength: 8), target: target,
            numDraftTokens: 3)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateGreedy(
                promptTokens: Array(repeating: 1, count: 9), maxNewTokens: 5, stopTokenIds: [],
                onToken: { _ in })
        }
        #expect(target.processedTokenCount == 0)
        // A prompt that exactly fills the context still yields the one token that fits.
        var got: [Int32] = []
        _ = try await decoder.generateGreedy(
            promptTokens: Array(repeating: 1, count: 8), maxNewTokens: 5, stopTokenIds: [],
            onToken: { got.append($0) })
        #expect(got.count == 1)
    }

    @Test("Sampling never accepts a token the target gives no probability, even if the draft gave none")
    func zeroProbabilityTokensAreRejected() async throws {
        // Token 3 has no mass under the target (row entries of -inf) or under the scripted draft.
        let blocked: Float = -.infinity
        let row: [Float] = [0, 0, 0, blocked]
        let target = TableEngine(table: [row, row, row, row])
        let draft = ScriptedDrafter(token: 3, logits: row)
        let decoder = SpeculativeDecoder(drafter: draft, target: target, numDraftTokens: 1)
        for seed in 0..<40 {
            var got: [Int32] = []
            _ = try await decoder.generateSampling(
                promptTokens: [0], maxNewTokens: 12, stopTokenIds: [], temperature: 1,
                seed: UInt64(seed), onToken: { got.append($0) })
            #expect(!got.contains(3), "seed \(seed): \(got)")
        }
    }

    @Test("A draft token outside the vocabulary throws instead of indexing past the distribution")
    func draftTokenOutsideTheVocabulary() async throws {
        // The engine can be fed token 9 (it has a row for it), but its distributions are 4 wide.
        let row: [Float] = [0, 0, 0, 0]
        let target = TableEngine(table: Array(repeating: row, count: 16))
        let decoder = SpeculativeDecoder(
            drafter: ScriptedDrafter(token: 9, logits: row), target: target, numDraftTokens: 1)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateSampling(
                promptTokens: [0], maxNewTokens: 4, stopTokenIds: [], temperature: 1, seed: 1,
                onToken: { _ in })
        }
    }
}

/// Always proposes one fixed token with a fixed draft distribution.
private final class ScriptedDrafter: DrafterEngine, @unchecked Sendable {
    let token: Int32
    let logits: [LogitsScalarType]
    let maxContextLength = 100_000

    init(token: Int32, logits: [Float]) {
        self.token = token
        self.logits = logits.map { LogitsScalarType($0) }
    }

    func reset(to tokenIndex: Int) async throws {}
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}
    func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        DraftProposal(tokens: [token], logits: [logits])
    }
}

extension SpeculativeSafetyTests {
    @Test("A draft count outside 1...maxDraftTokens is refused at construction")
    func draftCountBounds() async {
        // The closures run in another process, so they build their own values.
        await #expect(processExitsWith: .failure) {
            let model = HashModel(vocab: 10, levels: 100, salt: 1)
            _ = SpeculativeDecoder(
                drafter: PromptLookupDrafter(), target: HashEngine(model: model), numDraftTokens: 0)
        }
        await #expect(processExitsWith: .failure) {
            let model = HashModel(vocab: 10, levels: 100, salt: 1)
            _ = SpeculativeDecoder(
                draft: HashEngine(model: model), target: HashEngine(model: model),
                numDraftTokens: SpeculativeDefaults.maxDraftTokens + 1)
        }
        let model = HashModel(vocab: 10, levels: 100, salt: 1)
        // Both ends are accepted.
        for count in [1, SpeculativeDefaults.maxDraftTokens] {
            let decoder = SpeculativeDecoder(
                drafter: PromptLookupDrafter(), target: HashEngine(model: model), numDraftTokens: count)
            #expect(decoder.numDraftTokens == count)
        }
    }
}

extension SpeculativeSafetyTests {
    @Test("A target with recurrent state is refused before anything is prefilled")
    func recurrentTargetIsRefused() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 8)
        let target = HashEngine(model: model)
        target.reportsRecurrentState = true
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 1), target: target, numDraftTokens: 3)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateGreedy(
                promptTokens: [1, 2, 3], maxNewTokens: 5, stopTokenIds: [], onToken: { _ in })
        }
        #expect(target.processedTokenCount == 0)
    }

    @Test("A target whose rewind silently lands elsewhere stops the run instead of continuing from it")
    func rewindThatLandsElsewhere() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 9)
        let target = HashEngine(model: model)
        target.rewindsOnlyToZero = true  // does not report it: the cursor is the only evidence
        // Random proposals are rejected within a few rounds, which forces a rewind.
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 5), target: target, numDraftTokens: 4)
        await #expect(throws: InferenceRuntimeError.self) {
            for _ in 0..<20 {
                _ = try await decoder.generateGreedy(
                    promptTokens: [1, 2, 3], maxNewTokens: 40, stopTokenIds: [], onToken: { _ in })
            }
        }
    }

    @Test("A target holding the wrong number of tokens is reported before it is verified against")
    func targetOutOfStep() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 10)
        // The target's prefill drops a token, so the cache is one short of the confirmed prefix.
        let target = ShortPrefillEngine(inner: HashEngine(model: model))
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 2), target: target, numDraftTokens: 3)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await decoder.generateGreedy(
                promptTokens: [1, 2, 3, 4], maxNewTokens: 5, stopTokenIds: [], onToken: { _ in })
        }
    }
}

/// Prefills one token fewer than asked, as an engine that lost track of its cache would.
private final class ShortPrefillEngine: SpeculativeEngine, @unchecked Sendable {
    let inner: HashEngine
    init(inner: HashEngine) { self.inner = inner }
    var config: RowEngine.Config { inner.config }
    var processedTokenCount: Int { inner.processedTokenCount }
    func reset(to tokenIndex: Int) async throws { try await inner.reset(to: tokenIndex) }
    func advance(_ tokens: [Int32]) async throws { try await inner.advance(Array(tokens.dropLast())) }
    func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32] {
        try await inner.forwardPerPositionArgmax(tokens)
    }
    func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]] {
        try await inner.forwardWithPerPositionLogits(tokens)
    }
    func generate(
        with input: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> EmptyOutputSequence {
        throw InferenceRuntimeError.invalidState("ShortPrefillEngine implements only the speculative primitives")
    }
}

/// Wraps an engine and drops the first row of every per-position argmax, as an engine would that
/// scores fewer positions than a verify block holds.
private final class ShortRowsEngine: SpeculativeEngine, @unchecked Sendable {
    let inner: HashEngine
    init(inner: HashEngine) { self.inner = inner }
    var config: RowEngine.Config { inner.config }
    var processedTokenCount: Int { inner.processedTokenCount }
    func generate(
        with input: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> EmptyOutputSequence {
        throw InferenceRuntimeError.invalidState("ShortRowsEngine implements only the speculative primitives")
    }
    func reset(to tokenIndex: Int) async throws { try await inner.reset(to: tokenIndex) }
    func advance(_ tokens: [Int32]) async throws { try await inner.advance(tokens) }
    func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32] {
        Array(try await inner.forwardPerPositionArgmax(tokens).dropFirst())
    }
    func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]] {
        try await inner.forwardWithPerPositionLogits(tokens)
    }
    func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?) {
        let (argmax, features) = try await inner.forwardProposalGreedy(anchor: anchor, proposal: proposal)
        return (Array(argmax.dropFirst()), features)
    }
}
