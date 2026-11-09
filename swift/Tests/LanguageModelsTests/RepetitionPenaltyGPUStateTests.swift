// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Metal
import Testing

@testable import CoreAILanguageModels

@Suite("RepetitionPenaltyGPUState per generation")
struct RepetitionPenaltyGPUStateTests {
    static func prepare(
        reusing existing: RepetitionPenaltyGPUState?, idle: Bool = true, penalty: Double
    ) throws -> RepetitionPenaltyGPUState {
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try #require(
            try RepetitionPenaltyGPUState.forGeneration(
                reusing: existing, existingIsIdle: idle, device: device, vocabSize: 64, pipelineDepth: 3,
                penalty: penalty, windowSize: nil))
    }

    /// Tokens whose penalty differs from neutral after syncing the slot for `step`.
    static func penalized(_ state: RepetitionPenaltyGPUState, step: Int) -> [Int: Float16] {
        let ptr = state.buffer(forStep: step).contents().assumingMemoryBound(to: Float16.self)
        var result: [Int: Float16] = [:]
        for i in 0..<state.vocabSize where ptr[i] != Float16(1.0) {
            result[i] = ptr[i]
        }
        return result
    }

    @Test("each generation starts with its own penalty and an empty history")
    func eachGenerationStartsFresh() throws {
        let first = try Self.prepare(reusing: nil, penalty: 1.3)
        first.recordToken(5)
        #expect(Self.penalized(first, step: 0) == [5: Float16(1.3)])

        let second = try Self.prepare(reusing: first, penalty: 1.5)
        #expect(second !== first)
        second.recordToken(7)
        #expect(Self.penalized(second, step: 0) == [7: Float16(1.5)])

        let third = try Self.prepare(reusing: second, penalty: 1.5)
        #expect(third === second)
        for step in 0..<3 {
            #expect(Self.penalized(third, step: step).isEmpty)
        }
    }

    @Test("a state that may still be in flight is replaced, not reset")
    func inFlightStateIsReplaced() throws {
        let state = try Self.prepare(reusing: nil, penalty: 1.3)
        state.recordToken(5)
        _ = state.buffer(forStep: 0)

        let next = try Self.prepare(reusing: state, idle: false, penalty: 1.3)
        #expect(next !== state)
        #expect(Self.penalized(next, step: 0).isEmpty)
        #expect(Self.penalized(state, step: 0) == [5: Float16(1.3)])
    }
}
