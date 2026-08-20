// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// Applies repetition penalty to logits based on token generation history.
///
/// For each unique token ID in the recent history:
/// - If logit > 0: divide by penalty factor
/// - If logit < 0: multiply by penalty factor
///
/// This discourages the model from re-emitting recently generated tokens.
public struct RepetitionPenaltyProcessor {
    /// Apply repetition penalty to logits in-place.
    ///
    /// - Parameters:
    ///   - logits: Mutable logits array (vocab-sized). Modified in-place.
    ///   - recentTokenIds: Token IDs from recent generation history.
    ///   - penalty: The penalty factor (> 1.0 penalizes, 1.0 = no-op).
    public static func apply<C: Collection<Int32>>(
        to logits: inout [LogitsScalarType],
        recentTokenIds: C,
        penalty: Float
    ) {
        guard penalty > 1.0 else { return }
        guard !recentTokenIds.isEmpty else { return }

        let vocabSize = logits.count
        var seen = Set<Int32>(minimumCapacity: min(recentTokenIds.count, 512))

        for tokenId in recentTokenIds {
            guard tokenId >= 0 && Int(tokenId) < vocabSize else { continue }
            guard seen.insert(tokenId).inserted else { continue }

            let idx = Int(tokenId)
            let logit = Float(logits[idx])
            if logit > 0 {
                logits[idx] = LogitsScalarType(logit / penalty)
            } else if logit < 0 {
                logits[idx] = LogitsScalarType(logit * penalty)
            }
        }
    }
}
