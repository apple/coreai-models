// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal

/// Manages per-pipeline-depth additive delta buffers for GPU frequency/presence penalties and
/// logit bias.
///
/// Produces an ADDITIVE delta buffer `[1, vocabSize]` (f16) where, for each token id:
///
///     delta[id] = logitBias[id] - frequencyPenalty * count[id] - (count[id] > 0 ? presence : 0)
///
/// The GPU sampler simply adds this delta to the logits. This mirrors the CPU
/// `AdditivePenaltyProcessor` exactly.
///
/// Uses the same split design as ``RepetitionPenaltyGPUState`` to avoid races between GPU reads
/// and CPU writes:
/// - `recordToken()`: updates only the CPU-side ring buffer and queues per-slot delta increments
/// - `buffer(forStep:)`: applies the queued increments to one buffer slot at encode time, when the
///   pipeline gate guarantees that slot is not being read by the GPU
///
/// Unlike the multiplicative repetition penalty (whose neutral value is 1.0), the additive delta's
/// neutral value is 0.0. Each buffer starts at the static `logitBias` baseline and accumulates the
/// negative frequency/presence increments as tokens enter and leave the window.
final class AdditivePenaltyGPUState: @unchecked Sendable {
    let deltaBuffers: [MTLBuffer]
    let vocabSize: Int
    let pipelineDepth: Int
    let frequencyPenalty: Float
    let presencePenalty: Float
    let windowSize: Int

    /// Static per-token bias baseline (applied to every buffer at init and on reset).
    private let logitBias: [Int32: Float]

    private var ring: [Int32]
    private var writeIndex: Int = 0
    private var count: Int = 0
    private var refCounts: [Int32: Int] = [:]
    private var dirtyIncrements: [[(token: Int32, change: Float)]]

    init(
        device: MTLDevice,
        vocabSize: Int,
        pipelineDepth: Int,
        frequencyPenalty: Double,
        presencePenalty: Double,
        logitBias: [Int32: Float]?,
        windowSize: Int?
    ) throws {
        self.vocabSize = vocabSize
        self.pipelineDepth = pipelineDepth
        self.frequencyPenalty = Float(frequencyPenalty)
        self.presencePenalty = Float(presencePenalty)
        self.windowSize = windowSize ?? 256
        self.logitBias = logitBias?.filter { $0.key >= 0 && Int($0.key) < vocabSize } ?? [:]

        let bufferSize = vocabSize * MemoryLayout<Float16>.size
        var buffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw MPSGraphSamplerError.bufferAllocationFailed
            }
            buffers.append(buffer)
        }
        self.deltaBuffers = buffers
        self.ring = [Int32](repeating: -1, count: self.windowSize)
        self.dirtyIncrements = Array(repeating: [], count: pipelineDepth)

        // Initialize every slot to the static bias baseline (0.0 elsewhere).
        for buffer in buffers {
            // Float16 0.0 is all-zero bits, so memset resets to the additive identity fast.
            memset(buffer.contents(), 0, bufferSize)
            let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
            for (tokenId, bias) in self.logitBias {
                ptr[Int(tokenId)] = Float16(bias)
            }
        }
    }

    /// Get the additive delta buffer for a given step after applying pending increments.
    ///
    /// Called at encode time. The gate guarantees this slot's previous GPU read has completed,
    /// so writing to it is safe.
    func buffer(forStep step: Int) -> MTLBuffer {
        let slot = step % pipelineDepth
        let buf = deltaBuffers[slot]
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)

        let applied = dirtyIncrements[slot]
        for (token, change) in applied {
            let idx = Int(token)
            ptr[idx] = Float16(Float(ptr[idx]) + change)
        }
        dirtyIncrements[slot].removeAll(keepingCapacity: true)

        return buf
    }

    /// Record a newly generated token (CPU-side bookkeeping only).
    ///
    /// Called from the completion callback. Does NOT write to MTLBuffers directly. Instead, queues
    /// per-slot delta increments to be applied at the next `buffer(forStep:)` call. Increments are
    /// computed from `refCounts` at event time so the frequency/presence math is exact.
    func recordToken(_ token: Int32) {
        guard token >= 0 && Int(token) < vocabSize else { return }

        // Evict the oldest token if the window is full.
        if count == windowSize {
            let candidate = ring[writeIndex]
            if candidate >= 0 {
                let oldCount = refCounts[candidate, default: 0]
                let newCount = oldCount - 1
                if newCount <= 0 {
                    refCounts.removeValue(forKey: candidate)
                } else {
                    refCounts[candidate] = newCount
                }
                // Removing one occurrence relaxes the frequency penalty; if the token leaves the
                // window entirely, the presence penalty is lifted too.
                var change = frequencyPenalty
                if newCount <= 0 { change += presencePenalty }
                enqueue(token: candidate, change: change)
            }
        } else {
            count += 1
        }

        // Add the new token.
        let oldCount = refCounts[token, default: 0]
        var change = -frequencyPenalty
        if oldCount == 0 { change -= presencePenalty }
        enqueue(token: token, change: change)

        ring[writeIndex] = token
        writeIndex = (writeIndex + 1) % windowSize
        refCounts[token, default: 0] += 1
    }

    private func enqueue(token: Int32, change: Float) {
        guard change != 0 else { return }
        for i in 0..<pipelineDepth {
            dirtyIncrements[i].append((token: token, change: change))
        }
    }

    /// Reset all state to the static bias baseline (called on engine reset).
    func reset() {
        let bufferSize = vocabSize * MemoryLayout<Float16>.size
        for buf in deltaBuffers {
            memset(buf.contents(), 0, bufferSize)
            let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
            for (tokenId, bias) in logitBias {
                ptr[Int(tokenId)] = Float16(bias)
            }
        }
        ring = [Int32](repeating: -1, count: windowSize)
        writeIndex = 0
        count = 0
        refCounts.removeAll(keepingCapacity: true)
        dirtyIncrements = Array(repeating: [], count: pipelineDepth)
    }
}
