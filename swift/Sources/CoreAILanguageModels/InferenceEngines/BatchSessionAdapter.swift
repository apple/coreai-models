// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Token history and stop checks for one request in a batched decode.
///
/// Pass each sampled token to `accept(_:)`. Once it returns false, the request is done and
/// `stopReason` holds the reason.
struct BatchSessionAdapter {
    let request: BatchRequest

    private(set) var generated: [Int32] = []
    /// False once the row has retired.
    private(set) var live: Bool = true
    /// The reason the row retired, nil while it is live.
    private(set) var stopReason: StopReason?

    /// The last `stopSequences.maxLength` tokens, matched against the stop sequences.
    private var recentTokens: [Int32] = []

    /// The row's sampler step index: the number of tokens generated so far.
    var step: Int { generated.count }

    init(request: BatchRequest) {
        self.request = request
    }

    /// Appends `token`, then retires the row on an EOS id, a stop-sequence match or `maxTokens`,
    /// checked in that order. Returns true while the row is live. After retirement, ignores tokens
    /// and returns false.
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
