// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A block of candidate tokens a drafter proposes for the target to verify in a
/// single pass.
///
/// The candidates are laid out as a flat batch that follows the anchor (the last
/// confirmed token). `parents` describes their structure:
/// - `nil` ⇒ a **linear chain**: candidate `i` follows candidate `i-1`, and
///   candidate `0` follows the anchor. This is the classic k-token draft.
/// - non-`nil` ⇒ a **tree**: `parents[i]` is the index (into `tokens`) of the
///   parent of candidate `i`, or `-1` when its parent is the anchor. Verification
///   scores each candidate under an attention mask restricted to its ancestor
///   path, and acceptance walks the tree from the anchor.
///
/// A linear chain is exactly the degenerate depth-`k`, width-1 tree, so both
/// drafting shapes flow through the same verify / accept / rollback machinery.
public struct DraftProposal: Sendable {
    /// Candidate tokens, in the flat order the verifier scores them.
    public var tokens: [Int32]

    /// Tree structure, or `nil` for a linear chain. `parents[i] == -1` means the
    /// anchor is the parent.
    public var parents: [Int]?

    /// Per-candidate logits (the distribution each token was drawn from), required
    /// for speculative sampling. `nil` for greedy-only drafters.
    public var logits: [[LogitsScalarType]]?

    public init(tokens: [Int32], parents: [Int]? = nil, logits: [[LogitsScalarType]]? = nil) {
        self.tokens = tokens
        self.parents = parents
        self.logits = logits
    }

    /// A plain chain with no branching.
    public var isLinear: Bool { parents == nil }

    /// Parent slot of candidate `i` in the verifier's logits array, where slot `0`
    /// is the anchor and slot `j+1` is `tokens[j]`. For a linear chain the parent
    /// of `i` is `i` (i.e. `tokens[i-1]`, slot `i`).
    func parentSlot(of i: Int) -> Int {
        guard let parents else { return i }  // linear: slot i is tokens[i-1] (anchor for i==0)
        return parents[i] == -1 ? 0 : parents[i] + 1
    }
}

/// Final hidden-state features emitted by a fused target for an attached-head
/// (DFlash-style) drafter to consume via ``DrafterEngine/inject(_:)``.
///
/// Row-major `values[p * hiddenDim + d]`; row `p` is the feature vector for the
/// token at absolute position `startPosition + p`.
public struct DrafterFeatures: Sendable {
    public var values: [Float]
    public var positionCount: Int
    public var hiddenDim: Int
    public var startPosition: Int

    public init(values: [Float], positionCount: Int, hiddenDim: Int, startPosition: Int) {
        self.values = values
        self.positionCount = positionCount
        self.hiddenDim = hiddenDim
        self.startPosition = startPosition
    }

    /// The first `count` rows, re-based to `startPosition` — the committed-prefix
    /// features an attached-head drafter injects after acceptance.
    public func prefixRows(_ count: Int) -> DrafterFeatures {
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
/// - `greedy`: accept the longest prefix (or tree path) matching the target's
///   argmax; the output is token-for-token identical to greedy target decoding.
/// - `sampling`: the accept/reject scheme of Chen et al. (2023) at the given
///   temperature; the output is distributionally identical to sampling the target.
public enum VerificationPolicy: Sendable {
    case greedy
    case sampling(temperature: Double)

    var isSampling: Bool { if case .sampling = self { return true } else { return false } }

    /// Effective temperature (clamped away from zero for sampling; 1 for greedy).
    var temperature: Double {
        if case .sampling(let t) = self { return Swift.max(1e-5, t) }
        return 1
    }
}
