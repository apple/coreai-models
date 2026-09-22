// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Produces draft proposals for a speculative decoder.
///
/// This is the seam behind which different drafting mechanisms live, all
/// presenting the same propose / commit surface to the decoder:
/// - a **separate draft model** run autoregressively (``AutoregressiveDrafter``),
/// - an **attached head** (DFlash / MTP-style) that drafts off the target's
///   injected hidden-state features (`usesInjectedFeatures == true`),
/// - a tree drafter that returns a branching ``DraftProposal``.
///
/// The decoder owns verification and acceptance; a drafter only proposes and
/// keeps its own state in step with the confirmed prefix.
public protocol DrafterEngine: Sendable {
    /// Maximum context the drafter can hold; the decoder caps generation by the
    /// smaller of this and the target's.
    var maxContextLength: Int { get }

    /// Roll the drafter's state back to `tokenIndex` committed tokens.
    func reset(to tokenIndex: Int) async throws

    /// Propose up to `count` candidates following `confirmedPrefix` (whose last
    /// token is the anchor). `policy` selects greedy (argmax) vs sampled drafts;
    /// sampled drafts also populate `DraftProposal.logits` (the `q` distribution
    /// the accept/reject scheme needs).
    func draftAll(confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy) async throws
        -> DraftProposal

    /// Bring the drafter into step with the newly committed prefix (of length
    /// `confirmedLength`). `features` carries the target's fused hidden states for
    /// the verified slots when the target emits them; an attached-head drafter
    /// injects the committed rows, a token-model drafter ignores them.
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws
}

/// A drafter backed by a separate small model on any ``SpeculativeEngine``
/// backend, proposing a linear chain autoregressively — the classic draft/target
/// pairing. Keeps its own KV in step with the confirmed prefix so each step only
/// prefills the newly-confirmed tail.
public final class AutoregressiveDrafter: DrafterEngine, @unchecked Sendable {
    private let engine: any SpeculativeEngine
    private var rng = SystemRandomNumberGenerator()

    public var maxContextLength: Int { engine.maxContextLength }

    public init(engine: any SpeculativeEngine) {
        self.engine = engine
    }

    public func reset(to tokenIndex: Int) async throws {
        try await engine.reset(to: tokenIndex)
    }

    public func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {
        // Keep the accepted drafts the engine already holds; the correction (the
        // new anchor) is re-prefilled from `confirmedPrefix` on the next draft.
        try await engine.reset(to: min(confirmedLength, engine.processedTokenCount))
    }

    public func draftAll(confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy) async throws
        -> DraftProposal
    {
        // Prefill any confirmed tail the drafter hasn't seen, leaving the last
        // token to seed the first draft step.
        let pending = Array(confirmedPrefix.suffix(confirmedPrefix.count - engine.processedTokenCount))
        precondition(!pending.isEmpty, "confirmedPrefix must extend past the drafter's committed tokens")
        if pending.count > 1 {
            try await engine.advance(Array(pending.dropLast()))
        }

        var tokens: [Int32] = []
        tokens.reserveCapacity(count)
        var stepLogits: [[LogitsScalarType]]? = policy.isSampling ? [] : nil

        var step = try await engine.forwardWithPerPositionLogits([pending[pending.count - 1]])[0]
        for i in 0..<count {
            let token: Int32
            if policy.isSampling {
                let q = SpeculativeMath.temperatureSoftmax(step, temperature: policy.temperature)
                token = SpeculativeMath.sampleFromProbs(q, using: &rng)
                stepLogits?.append(step)
            } else {
                token = SpeculativeMath.argmax(step)
            }
            tokens.append(token)
            if i < count - 1 {
                step = try await engine.forwardWithPerPositionLogits([token])[0]
            }
        }
        return DraftProposal(tokens: tokens, parents: nil, logits: stepLogits)
    }
}
