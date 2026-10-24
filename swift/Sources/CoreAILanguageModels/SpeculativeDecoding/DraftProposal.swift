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

/// Final hidden-state features emitted by a fused target for an attached-head
/// (MTP / EAGLE-style) drafter to consume via ``DrafterEngine/commit(confirmedLength:features:)``.
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
        precondition(
            positionCount >= 0 && hiddenDim >= 0 && values.count >= positionCount * hiddenDim,
            "\(values.count) values for \(positionCount) positions of width \(hiddenDim)")
        self.values = values
        self.positionCount = positionCount
        self.hiddenDim = hiddenDim
        self.startPosition = startPosition
    }

    /// The feature row that seeds a drafter's recurrence once `confirmedLength` tokens are
    /// committed: the row of the last committed token, clamped to the rows present. `nil` when
    /// there are no rows.
    package func seedRow(confirmedLength: Int) -> [Float]? {
        guard hiddenDim > 0, positionCount > 0 else { return nil }
        let row = Swift.max(0, Swift.min(confirmedLength - startPosition - 1, positionCount - 1))
        return Array(values[row * hiddenDim..<(row + 1) * hiddenDim])
    }

    /// The first `count` rows, re-based to `startPosition` — the committed-prefix
    /// features an attached-head drafter injects after acceptance.
    package func prefixRows(_ count: Int) -> DrafterFeatures {
        let n = Swift.max(0, Swift.min(count, positionCount))
        return DrafterFeatures(
            values: Array(values.prefix(n * hiddenDim)),
            positionCount: n,
            hiddenDim: hiddenDim,
            startPosition: startPosition
        )
    }
}

/// How the target's verdict is turned into committed tokens.
///
/// - `greedy`: accept the longest prefix matching the target's
///   argmax; the output is token-for-token identical to greedy target decoding.
/// - `sampling`: the accept/reject scheme of Chen et al. (2023) at the given
///   temperature; the output is distributionally identical to sampling the target.
package enum VerificationPolicy: Sendable {
    case greedy
    case sampling(temperature: Double)

    var isSampling: Bool { if case .sampling = self { return true } else { return false } }

    /// Effective temperature (clamped away from zero for sampling; 1 for greedy).
    var temperature: Double {
        if case .sampling(let t) = self { return Swift.max(1e-5, t) }
        return 1
    }
}
