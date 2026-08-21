// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal

/// Manages per-pipeline-depth penalty buffers for GPU repetition penalty.
///
/// Uses a split design to avoid races between GPU reads and CPU writes:
/// - `recordToken()`: updates only the CPU-side ring buffer (no MTLBuffer writes)
/// - `buffer(forStep:)`: writes the full penalty state to a specific buffer
///   slot, called at encode time when the gate guarantees that slot is not in use
///
/// Thread safety: relies on MPSGraph runAsync completions being dispatched in
/// submission order on a single MTLCommandQueue (observed behavior, validated by
/// `MPSGraphCompletionOrderingTests`). The gate further ensures that
/// `buffer(forStep:)` does not overlap with `recordToken` for the same slot.
final class RepetitionPenaltyGPUState: @unchecked Sendable {
    let penaltyBuffers: [MTLBuffer]
    let vocabSize: Int
    let pipelineDepth: Int
    let penalty: Float16
    let windowSize: Int

    private var ring: [Int32]
    private var writeIndex: Int = 0
    private var count: Int = 0
    private var refCounts: [Int32: Int] = [:]
    private var dirtyTokens: [(added: [Int32], evicted: [Int32])]

    init(device: MTLDevice, vocabSize: Int, pipelineDepth: Int, penalty: Double, windowSize: Int?) throws {
        self.vocabSize = vocabSize
        self.pipelineDepth = pipelineDepth
        self.penalty = Float16(penalty)
        self.windowSize = windowSize ?? 256

        let bufferSize = vocabSize * MemoryLayout<Float16>.size
        var buffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw MPSGraphSamplerError.bufferAllocationFailed
            }
            let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<vocabSize {
                ptr[i] = Float16(1.0)
            }
            buffers.append(buffer)
        }
        self.penaltyBuffers = buffers
        self.ring = [Int32](repeating: -1, count: self.windowSize)
        self.dirtyTokens = Array(repeating: (added: [], evicted: []), count: pipelineDepth)
    }

    /// Get the penalty buffer for a given step after syncing pending changes.
    ///
    /// Called at encode time. The gate guarantees this slot's previous GPU read
    /// has completed, so writing to it is safe.
    func buffer(forStep step: Int) -> MTLBuffer {
        let slot = step % pipelineDepth
        let buf = penaltyBuffers[slot]
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)

        let dirty = dirtyTokens[slot]
        for tokenId in dirty.evicted {
            ptr[Int(tokenId)] = Float16(1.0)
        }
        for tokenId in dirty.added {
            ptr[Int(tokenId)] = penalty
        }
        dirtyTokens[slot] = (added: [], evicted: [])

        return buf
    }

    /// Record a newly generated token (CPU-side bookkeeping only).
    ///
    /// Called from the completion callback. Does NOT write to MTLBuffers directly.
    /// Instead, queues changes to be applied per-slot at the next `buffer(forStep:)` call.
    func recordToken(_ token: Int32) {
        guard token >= 0 && Int(token) < vocabSize else { return }

        var evictedToken: Int32 = -1
        if count == windowSize {
            let evictSlot = writeIndex
            let candidate = ring[evictSlot]
            if candidate >= 0 {
                refCounts[candidate, default: 0] -= 1
                if refCounts[candidate, default: 0] <= 0 {
                    refCounts.removeValue(forKey: candidate)
                    evictedToken = candidate
                }
            }
        } else {
            count += 1
        }

        ring[writeIndex] = token
        writeIndex = (writeIndex + 1) % windowSize
        refCounts[token, default: 0] += 1

        for i in 0..<pipelineDepth {
            if evictedToken >= 0 {
                dirtyTokens[i].evicted.append(evictedToken)
            }
            dirtyTokens[i].added.append(token)
        }
    }

    /// Reset all state (called on engine reset).
    func reset() {
        for buf in penaltyBuffers {
            let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<vocabSize {
                ptr[i] = Float16(1.0)
            }
        }
        ring = [Int32](repeating: -1, count: windowSize)
        writeIndex = 0
        count = 0
        refCounts.removeAll(keepingCapacity: true)
        dirtyTokens = Array(repeating: (added: [], evicted: []), count: pipelineDepth)
    }
}
