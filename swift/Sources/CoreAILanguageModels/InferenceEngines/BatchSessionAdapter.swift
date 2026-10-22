// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Per-row token bookkeeping and stop checks for one request in a batched decode.
///
/// `BatchScheduler` keeps one adapter per occupied slot and feeds it each sampled token. When
/// `accept(_:)` returns false, the scheduler emits `.finished` and frees the slot for the next queued
/// request while the other rows keep decoding.
struct BatchSessionAdapter {
    let request: BatchRequest

    private(set) var generated: [Int32] = []
    /// False once the row has retired.
    private(set) var live: Bool = true
    /// Set when the row retires.
    private(set) var stopReason: StopReason?

    /// The last `stopSequences.maxLength` tokens, matched against the stop sequences.
    private var recentTokens: [Int32] = []

    /// The per-row step index for the sampler, equal to `generated.count`.
    var step: Int { generated.count }

    init(request: BatchRequest) {
        self.request = request
    }

    /// Appends `token`, then retires the row on an EOS id, a stop-sequence match, or reaching
    /// `maxTokens`, checked in that order. Returns whether the row is still live. Once the row has
    /// retired, it ignores further tokens and returns false.
    @discardableResult
    mutating func accept(_ token: Int32) -> Bool {
        guard live else { return false }
        generated.append(token)

        if request.eosTokenIds.contains(token) {
            retire(.eos)
            return false
        }

        if request.stopSequences.maxLength > 0 {
            recentTokens.append(token)
            if recentTokens.count > request.stopSequences.maxLength {
                recentTokens.removeFirst(recentTokens.count - request.stopSequences.maxLength)
            }
            if let matched = request.stopSequences.matchedSequence(recentTokens: recentTokens) {
                retire(.stopSequence(request.decodeStopText(matched)))
                return false
            }
        }

        if generated.count >= request.maxTokens {
            retire(.maxTokens)
            return false
        }
        return true
    }

    private mutating func retire(_ reason: StopReason) {
        live = false
        stopReason = reason
    }
}
