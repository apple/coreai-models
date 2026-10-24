// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// Inputs that used to trap or produce NaN now throw or yield valid distributions.
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

/// Wraps an engine and drops the first row of every per-position argmax, as a verify block wider
/// than the widest graph used to.
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
