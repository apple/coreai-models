// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A block of candidate tokens a drafter proposes for the target to verify in a
/// single pass: a linear chain in which candidate `i` follows candidate `i-1`, and
/// candidate `0` follows the anchor (the last confirmed token).
package struct DraftProposal: Sendable {
    /// Candidate tokens, in the order the verifier scores them.
    package var tokens: [Int32]

    /// Per-candidate logits (the distribution each token was drawn from), required
    /// for speculative sampling. `nil` for greedy-only drafters.
    package var logits: [[LogitsScalarType]]?

    /// - Precondition: `logits`, when present, holds one row per token.
    package init(tokens: [Int32], logits: [[LogitsScalarType]]? = nil) {
        precondition(
            logits == nil || logits?.count == tokens.count,
            "logits holds \(logits?.count ?? 0) rows for \(tokens.count) tokens")
        self.tokens = tokens
        self.logits = logits
    }
}

/// Final hidden-state features emitted by a fused target for a feature-driven drafter to
/// consume via ``DrafterEngine/commit(confirmedLength:features:)``.
///
/// Row-major `values[p * hiddenDim + d]`; row `p` is the feature vector for the
/// token at absolute position `startPosition + p`.
package struct DrafterFeatures: Sendable {
    package var values: [Float]
    package var positionCount: Int
    package var hiddenDim: Int
    package var startPosition: Int

    /// - Precondition: `values` holds at least `positionCount * hiddenDim` elements.
    package init(values: [Float], positionCount: Int, hiddenDim: Int, startPosition: Int) {
        precondition(positionCount >= 0, "DrafterFeatures positionCount \(positionCount) is negative")
        precondition(hiddenDim >= 0, "DrafterFeatures hiddenDim \(hiddenDim) is negative")
        precondition(
            values.count >= positionCount * hiddenDim,
            "DrafterFeatures holds \(values.count) values, but \(positionCount) positions of width "
                + "\(hiddenDim) need \(positionCount * hiddenDim)")
        self.values = values
        self.positionCount = positionCount
        self.hiddenDim = hiddenDim
        self.startPosition = startPosition
    }
}

/// How the target's verdict is turned into committed tokens.
///
/// - `greedy`: accept the longest prefix matching the target's
///   argmax; the output is token-for-token identical to greedy target decoding.
/// - `sampling`: the accept/reject scheme of Chen et al. (2023, https://arxiv.org/abs/2302.01318) at the given
///   temperature; the output is distributionally identical to sampling the target.
package enum VerificationPolicy: Sendable {
    case greedy
    case sampling(temperature: Double)

    var isSampling: Bool { if case .sampling = self { return true } else { return false } }

    /// Effective temperature (clamped away from zero for sampling; 1 for greedy).
    var temperature: Double {
        if case .sampling(let t) = self { return Swift.max(SpeculativeDefaults.minSamplingTemperature, t) }
        return 1.0
    }
}
