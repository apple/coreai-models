// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Metal

// MARK: - Rotating logits destinations

/// Rotating destinations for `logits`, one per in-flight encode.
///
/// An engine that encodes ahead of the GPU needs one per call in flight: with a call still
/// outstanding the GPU may be writing the step the CPU has already moved past, so writing the next
/// call's logits into the same buffer would corrupt submitted work.
///
/// Each holds one row of `vocab` logits: a decode step's. Calls that produce logits for several
/// positions, such as a prefill on the main graph, write them to the engine's `GrowingLogitsBuffer`
/// instead.
struct RotatingLogitsBuffers {
    private let buffers: [MTLBuffer]

    init(device: MTLDevice, vocabSize: Int, depth: Int) throws {
        self.buffers = try Self.allocate(device: device, vocabSize: vocabSize, depth: max(1, depth))
    }

    private static func allocate(device: MTLDevice, vocabSize: Int, depth: Int) throws -> [MTLBuffer] {
        let byteCount = vocabSize * MemoryLayout<UInt16>.size
        var made: [MTLBuffer] = []
        for _ in 0..<depth {
            guard let buffer = device.makeBuffer(length: max(64, byteCount), options: .storageModeShared)
            else {
                throw InferenceRuntimeError.bufferAllocationFailed(
                    "rotating logits (\(byteCount) bytes)")
            }
            made.append(buffer)
        }
        return made
    }

    /// The destination for a given step. Wraps, so a caller passes its own step counter.
    subscript(slot: Int) -> MTLBuffer { buffers[slot % buffers.count] }
}
