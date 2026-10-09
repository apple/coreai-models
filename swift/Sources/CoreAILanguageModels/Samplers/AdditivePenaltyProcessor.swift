// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// Applies additive logit adjustments to the logits before sampling: count-scaled frequency
/// penalty, once-per-token presence penalty, and a fixed per-token logit bias.
///
/// These are additive and differ from ``RepetitionPenaltyProcessor`` (which is multiplicative and
/// count-agnostic):
/// - **frequency penalty**: `logit -= frequencyPenalty * count` for each token in the history
/// - **presence penalty**: `logit -= presencePenalty` once for any token that appeared at all
/// - **logit bias**: `logit += bias` for each biased token id, regardless of history
///
/// The frequency/presence penalties are computed from the recent token history; the logit bias is
/// a static, request-level vector.
public struct AdditivePenaltyProcessor {
    /// Apply the additive frequency/presence penalties and the logit bias to logits in-place.
    ///
    /// - Parameters:
    ///   - logits: Mutable logits array (vocab-sized). Modified in-place.
    ///   - recentTokenIds: Token IDs from recent generation history, used for the count-scaled
    ///     frequency penalty and the presence penalty.
    ///   - frequencyPenalty: Additive penalty scaled by occurrence count (0 = disabled).
    ///   - presencePenalty: Additive penalty applied once per seen token (0 = disabled).
    ///   - logitBias: Optional per-token-id additive bias applied after the penalties.
    public static func apply<C: Collection<Int32>>(
        to logits: inout [LogitsScalarType],
        recentTokenIds: C,
        frequencyPenalty: Float,
        presencePenalty: Float,
        logitBias: [Int32: Float]?
    ) {
        let vocabSize = logits.count

        if frequencyPenalty != 0.0 || presencePenalty != 0.0 {
            var counts = [Int32: Int](minimumCapacity: min(recentTokenIds.count, 512))
            for tokenId in recentTokenIds {
                guard tokenId >= 0 && Int(tokenId) < vocabSize else { continue }
                counts[tokenId, default: 0] += 1
            }
            for (tokenId, count) in counts {
                let idx = Int(tokenId)
                let logit = Float(logits[idx])
                let delta = frequencyPenalty * Float(count) + presencePenalty
                logits[idx] = LogitsScalarType(logit - delta)
            }
        }

        if let logitBias {
            applyLogitBias(to: &logits, logitBias: logitBias)
        }
    }

    /// Apply only the per-token logit bias in-place (no history required).
    ///
    /// - Parameters:
    ///   - logits: Mutable logits array (vocab-sized). Modified in-place.
    ///   - logitBias: Per-token-id additive bias (`logit += bias`).
    public static func applyLogitBias(
        to logits: inout [LogitsScalarType],
        logitBias: [Int32: Float]
    ) {
        let vocabSize = logits.count
        for (tokenId, bias) in logitBias {
            guard tokenId >= 0 && Int(tokenId) < vocabSize else { continue }
            let idx = Int(tokenId)
            logits[idx] = LogitsScalarType(Float(logits[idx]) + bias)
        }
    }
}
