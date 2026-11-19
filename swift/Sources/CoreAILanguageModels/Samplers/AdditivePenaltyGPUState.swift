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
/// - `recordToken()`: updates only the CPU-side ring buffer and queues, per slot, the exact new
///   delta of each token whose count changed
/// - `buffer(forStep:)`: writes the queued deltas to one buffer slot at encode time, when the
///   pipeline gate guarantees that slot is not being read by the GPU
///
/// Unlike the multiplicative repetition penalty (whose neutral value is 1.0), the additive delta's
/// neutral value is 0.0. Each buffer starts at the static `logitBias` baseline. A token's delta is
/// recomputed from its count on every change rather than accumulated in place, so the f16 buffer
/// does not drift from the CPU result as counts grow.
final class AdditivePenaltyGPUState: @unchecked Sendable {
    let deltaBuffers: [MTLBuffer]
    let vocabSize: Int
    let pipelineDepth: Int
    let frequencyPenalty: Double
    let presencePenalty: Double
    let windowSize: Int

    /// Static per-token bias baseline (applied to every buffer at init and on reset).
    private let logitBias: [Int32: Float]

    private var ring: [Int32]
    private var writeIndex: Int = 0
    private var count: Int = 0
    private var refCounts: [Int32: Int] = [:]
    private var dirtyDeltas: [[(token: Int32, delta: Double)]]

    init(
        device: MTLDevice,
        vocabSize: Int,
        pipelineDepth: Int,
        frequencyPenalty: Double = 0,
        presencePenalty: Double = 0,
        logitBias: [Int32: Float]? = nil,
        windowSize: Int
    ) throws {
        // No default window: callers pass the same window as the CPU path (the context length
        // when the configuration's window is nil), so both engines penalize the same history.
        precondition(windowSize > 0, "windowSize must be > 0")
        self.vocabSize = vocabSize
        self.pipelineDepth = pipelineDepth
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.windowSize = windowSize
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
        self.dirtyDeltas = Array(repeating: [], count: pipelineDepth)

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

    /// Get the additive delta buffer for a given step after writing pending deltas.
    ///
    /// Called at encode time. The gate guarantees this slot's previous GPU read has completed,
    /// so writing to it is safe.
    func buffer(forStep step: Int) -> MTLBuffer {
        let slot = step % pipelineDepth
        let buf = deltaBuffers[slot]
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)

        // Queue order is event order, so the last write for a token is its current delta.
        for (token, delta) in dirtyDeltas[slot] {
            ptr[Int(token)] = Float16(delta)
        }
        dirtyDeltas[slot].removeAll(keepingCapacity: true)

        return buf
    }

    /// Record a newly generated token (CPU-side bookkeeping only).
    ///
    /// Called from the completion callback. Does NOT write to MTLBuffers directly. Instead, queues
    /// each changed token's new delta, computed from `refCounts` here, for the next
    /// `buffer(forStep:)` call on every slot.
    func recordToken(_ token: Int32) {
        guard token >= 0 && Int(token) < vocabSize else { return }

        // Evict the oldest token if the window is full.
        if count == windowSize {
            let candidate = ring[writeIndex]
            if candidate >= 0 {
                let newCount = refCounts[candidate, default: 0] - 1
                if newCount <= 0 {
                    refCounts.removeValue(forKey: candidate)
                } else {
                    refCounts[candidate] = newCount
                }
                enqueue(candidate)
            }
        } else {
            count += 1
        }

        // Add the new token.
        ring[writeIndex] = token
        writeIndex = (writeIndex + 1) % windowSize
        refCounts[token, default: 0] += 1
        enqueue(token)
    }

    /// Queue `token`'s current delta: its bias, minus the frequency penalty per occurrence and the
    /// presence penalty once while it is in the window.
    private func enqueue(_ token: Int32) {
        var delta = Double(logitBias[token] ?? 0)
        if let count = refCounts[token] {
            delta -= frequencyPenalty * Double(count) + presencePenalty
        }
        for i in 0..<pipelineDepth {
            dirtyDeltas[i].append((token: token, delta: delta))
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
        dirtyDeltas = Array(repeating: [], count: pipelineDepth)
    }
}
