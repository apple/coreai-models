// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Autoregressive drafter guards")
struct AutoregressiveDrafterGuardTests {
    @Test("A confirmed prefix shorter than the drafter's committed tokens throws instead of trapping")
    func draftAllThrowsWhenPrefixIsBehind() async throws {
        let model = HashModel(vocab: 8, levels: 5, salt: 1)
        let engine = HashEngine(model: model)
        let drafter = AutoregressiveDrafter(engine: engine)
        _ = try await drafter.draftAll(confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy)
        #expect(engine.processedTokenCount >= 3)
        await #expect(throws: (any Error).self) {
            _ = try await drafter.draftAll(confirmedPrefix: [1], count: 2, policy: .greedy)
        }
        await #expect(throws: (any Error).self) {
            _ = try await drafter.draftAll(
                confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy)
        }
    }
}
