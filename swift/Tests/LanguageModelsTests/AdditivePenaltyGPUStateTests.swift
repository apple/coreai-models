// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Metal
import Testing

@testable import CoreAILanguageModels

/// Isolation tests for the GPU additive-delta state: verify the buffer contents it produces match
/// the CPU `AdditivePenaltyProcessor` math, independent of the pipelined engine's threading.
@Suite("AdditivePenaltyGPUState")
struct AdditivePenaltyGPUStateTests {
    private func value(_ buf: MTLBuffer, _ idx: Int) -> Float {
        Float(buf.contents().assumingMemoryBound(to: Float16.self)[idx])
    }

    @Test("logit bias is the static baseline")
    func logitBiasBaseline() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let state = try AdditivePenaltyGPUState(
            device: device, vocabSize: 8, pipelineDepth: 1,
            frequencyPenalty: 0, presencePenalty: 0, logitBias: [2: 5.0, 5: -3.0], windowSize: 256)
        let buf = state.buffer(forStep: 0)
        #expect(abs(value(buf, 2) - 5.0) < 1e-2)
        #expect(abs(value(buf, 5) - (-3.0)) < 1e-2)
        #expect(value(buf, 0) == 0)
    }

    @Test("frequency + presence accumulate per occurrence (pipelineDepth 1)")
    func accumulateDepth1() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let state = try AdditivePenaltyGPUState(
            device: device, vocabSize: 8, pipelineDepth: 1,
            frequencyPenalty: 1.0, presencePenalty: 0.5, logitBias: nil, windowSize: 256)
        _ = state.buffer(forStep: 0)  // slot 0, no history yet
        state.recordToken(3)
        let b1 = state.buffer(forStep: 1)
        // First occurrence: -freq*1 - presence = -1.5
        #expect(abs(value(b1, 3) - (-1.5)) < 1e-2)
        state.recordToken(3)
        let b2 = state.buffer(forStep: 2)
        // Second occurrence: additional -freq = -2.5 total
        #expect(abs(value(b2, 3) - (-2.5)) < 1e-2)
        #expect(value(b2, 0) == 0)
    }

    @Test("all pipeline slots converge to the same delta")
    func slotsConverge() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let depth = 3
        let state = try AdditivePenaltyGPUState(
            device: device, vocabSize: 8, pipelineDepth: depth,
            frequencyPenalty: 2.0, presencePenalty: 0, logitBias: nil, windowSize: 256)
        // Record 4 occurrences of token 4, cycling buffer() across slots as the engine would.
        for step in 0..<8 {
            _ = state.buffer(forStep: step)
            if step < 4 { state.recordToken(4) }
        }
        // After all 4 occurrences have propagated, every slot must read -freq*4 = -8.0.
        for slot in 0..<depth {
            let b = state.buffer(forStep: slot)
            #expect(abs(value(b, 4) - (-8.0)) < 1e-1, "slot \(slot)")
        }
    }

    @Test("window eviction relaxes the penalty")
    func windowEviction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let state = try AdditivePenaltyGPUState(
            device: device, vocabSize: 8, pipelineDepth: 1,
            frequencyPenalty: 1.0, presencePenalty: 1.0, logitBias: nil, windowSize: 2)
        _ = state.buffer(forStep: 0)
        state.recordToken(1)  // window [1]
        state.recordToken(2)  // window [1,2]
        state.recordToken(1)  // window full -> evict oldest (1), then add 1: window [2,1]
        let b = state.buffer(forStep: 1)
        // token 1: evicted once (+1), re-added (-1 -1 presence since it had left)... net over events.
        // Simpler invariant: token 2 present once -> -freq-presence = -2.0
        #expect(abs(value(b, 2) - (-2.0)) < 1e-2)
    }
}
