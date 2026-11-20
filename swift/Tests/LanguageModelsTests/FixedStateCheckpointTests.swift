// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

/// The saved copy a model with recurrent state restores when a block sends processed tokens again.
@Suite("Fixed-state checkpoint")
struct FixedStateCheckpointTests {
    private func array(_ values: [Float16], shape: [Int]) -> NDArray {
        var array = NDArray(shape: shape, scalarType: .float16)
        array.mutableRawView().withUnsafeMutableBytes { bytes, _, _ in
            values.withUnsafeBytes { bytes.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        return array
    }

    private func values(_ array: NDArray) -> [Float16] {
        let count = array.shape.reduce(1, *)
        return array.rawView().withUnsafeBytes { bytes, _, _ in
            Array(UnsafeBufferPointer(start: bytes.assumingMemoryBound(to: Float16.self), count: count))
        }
    }

    private func state() -> FixedStaticState {
        FixedStaticState(arrays: [
            "conv_state": array([1, 2, 3, 4, 5, 6], shape: [1, 2, 1, 3]),
            "recurrent_state": array([7, 8, 9, 10], shape: [1, 1, 2, 2]),
        ])
    }

    @Test("A restore puts back what was saved, and can be repeated")
    func restoreRepeats() {
        let fixed = state()
        #expect(fixed.checkpointPosition == nil)
        fixed.saveCheckpoint(at: 16)
        #expect(fixed.checkpointPosition == 16)

        for step in 0..<2 {
            zeroFillNDArray(&fixed.arrays["conv_state"]!)
            fixed.arrays["recurrent_state"] = array([0, 0, 0, Float16(step)], shape: [1, 1, 2, 2])
            fixed.restoreCheckpoint()
            #expect(values(fixed.arrays["conv_state"]!) == [1, 2, 3, 4, 5, 6])
            #expect(values(fixed.arrays["recurrent_state"]!) == [7, 8, 9, 10])
            #expect(fixed.checkpointPosition == 16)
        }
    }

    @Test("A later save replaces the copy; discard and reset forget it")
    func saveReplacesAndResetForgets() {
        let fixed = state()
        fixed.saveCheckpoint(at: 16)
        fixed.arrays["recurrent_state"] = array([1, 1, 1, 1], shape: [1, 1, 2, 2])
        fixed.saveCheckpoint(at: 24)
        fixed.arrays["recurrent_state"] = array([2, 2, 2, 2], shape: [1, 1, 2, 2])
        fixed.restoreCheckpoint()
        #expect(fixed.checkpointPosition == 24)
        #expect(values(fixed.arrays["recurrent_state"]!) == [1, 1, 1, 1])

        fixed.discardCheckpoint()
        #expect(fixed.checkpointPosition == nil)

        fixed.saveCheckpoint(at: 8)
        fixed.reset()
        #expect(fixed.checkpointPosition == nil)
        #expect(values(fixed.arrays["conv_state"]!).allSatisfy { $0 == 0 })
    }
}
