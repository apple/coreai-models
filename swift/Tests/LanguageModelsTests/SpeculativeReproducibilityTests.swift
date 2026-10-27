// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Speculative sampling reproducibility")
struct SpeculativeReproducibilityTests {
    private func sample(
        seed: UInt64?, tokens: Int = 80, model: HashModel = HashModel(vocab: 40, levels: 4, salt: 5)
    ) async throws -> [Int32] {
        let decoder = SpeculativeDecoder(
            draft: HashEngine(model: HashModel(vocab: 40, levels: 4, salt: 6)),
            target: HashEngine(model: model), numDraftTokens: 4)
        var got: [Int32] = []
        _ = try await decoder.generateSampling(
            promptTokens: [1, 2, 3], maxNewTokens: tokens, stopTokenIds: [], temperature: 1,
            seed: seed, onToken: { got.append($0) })
        return got
    }

    @Test("The same seed reproduces the same tokens")
    func sameSeed() async throws {
        let first = try await sample(seed: 7)
        let second = try await sample(seed: 7)
        #expect(first == second)
        #expect(first.count == 80)
    }

    @Test("Different seeds give different tokens")
    func differentSeeds() async throws {
        let runs = try await withThrowingTaskGroup(of: [Int32].self) { group in
            for seed in 1...4 { group.addTask { try await self.sample(seed: UInt64(seed)) } }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(Set(runs).count > 1)
    }

    @Test("A caller-supplied generator is the one the run draws from")
    func knownGenerator() async throws {
        let viaSeed = try await sample(seed: 11)

        let decoder = SpeculativeDecoder(
            draft: HashEngine(model: HashModel(vocab: 40, levels: 4, salt: 6)),
            target: HashEngine(model: HashModel(vocab: 40, levels: 4, salt: 5)), numDraftTokens: 4)
        var rng = SeededRandomNumberGenerator(seed: 11)
        var viaGenerator: [Int32] = []
        _ = try await decoder.generateSampling(
            promptTokens: [1, 2, 3], maxNewTokens: 80, stopTokenIds: [], temperature: 1, using: &rng,
            onToken: { viaGenerator.append($0) })
        #expect(viaGenerator == viaSeed)
    }
}
