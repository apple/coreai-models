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
/// - a drafter that drafts off the target's hidden-state features (for example an attached
///   MTP / EAGLE-style head), which the decoder hands it through
///   ``commit(confirmedLength:features:)``,
/// - a **model-free** drafter such as ``PromptLookupDrafter``.
///
/// The decoder owns verification and acceptance; a drafter only proposes and
/// keeps its own state in step with the confirmed prefix.
package protocol DrafterEngine: Sendable {
    /// Maximum context the drafter can hold; the decoder caps generation by the
    /// smaller of this and the target's.
    var maxContextLength: Int { get }

    /// Roll the drafter's state back to `tokenIndex` committed tokens.
    func reset(to tokenIndex: Int) async throws

    /// Propose up to `count` candidates following `confirmedPrefix` (whose last
    /// token is the anchor). `policy` selects greedy (argmax) vs sampled drafts;
    /// sampled drafts also populate `DraftProposal.logits` (the `q` distribution
    /// the accept/reject scheme needs) and draw from `rng`, the generator the decoder uses for the
    /// whole run, so a seeded run is reproducible.
    func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal

    /// Bring the drafter into step with the newly committed prefix (of length
    /// `confirmedLength`). `features` carries the target's fused hidden states for
    /// the verified slots when the target emits them; a feature-driven drafter
    /// takes the committed rows, a token-model drafter ignores them.
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws
}

/// A drafter backed by a separate small model on any ``SpeculativeEngine``
/// backend, proposing a linear chain autoregressively — the classic draft/target
/// pairing. Keeps its own KV in step with the confirmed prefix so each step only
/// prefills the newly-confirmed tail.
///
/// Single-owner: one task drives it at a time, and its draft engine's cache is one sequence.
package final class AutoregressiveDrafter: DrafterEngine, @unchecked Sendable {
    private let engine: any SpeculativeEngine

    package var maxContextLength: Int { engine.config.maxContextLength }

    package init(engine: any SpeculativeEngine) {
        self.engine = engine
    }

    package func reset(to tokenIndex: Int) async throws {
        try await engine.rewindAndConfirm(to: tokenIndex)
    }

    package func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {
        // Keep the accepted drafts the engine already holds; the correction (the
        // new anchor) is re-prefilled from `confirmedPrefix` on the next draft.
        try await engine.rewindAndConfirm(to: min(confirmedLength, engine.processedTokenCount))
    }

    package func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        // Nothing to propose: leave the draft engine untouched, so a later call with the same prefix
        // still finds its tail unseen.
        guard count > 0 else { return DraftProposal(tokens: []) }
        // Prefill any confirmed tail the drafter hasn't seen, leaving the last
        // token to seed the first draft step.
        let unseen = confirmedPrefix.count - engine.processedTokenCount
        guard unseen > 0 else {
            throw InferenceRuntimeError.invalidState(
                "confirmedPrefix of \(confirmedPrefix.count) tokens must extend past the drafter's "
                    + "\(engine.processedTokenCount) committed tokens")
        }
        let pending = Array(confirmedPrefix.suffix(unseen))
        if pending.count > 1 {
            try await engine.advance(Array(pending.dropLast()))
        }
        let seed = pending[pending.count - 1]

        var tokens: [Int32] = []
        tokens.reserveCapacity(count)

        guard policy.isSampling else {
            // Greedy drafts need only each step's argmax, so the draft engine never copies a
            // `[vocabSize]` row out.
            var token = try await engine.forwardPerPositionArgmax([seed])[0]
            for i in 0..<count {
                tokens.append(token)
                if i < count - 1 { token = try await engine.forwardPerPositionArgmax([token])[0] }
            }
            return DraftProposal(tokens: tokens)
        }

        var stepLogits: [[LogitsScalarType]] = []
        stepLogits.reserveCapacity(count)
        var step = try await engine.forwardWithPerPositionLogits([seed])[0]
        for i in 0..<count {
            let q = SpeculativeMath.temperatureSoftmax(step, temperature: policy.temperature)
            let token = SpeculativeMath.sampleFromProbs(q, using: &rng)
            stepLogits.append(step)
            tokens.append(token)
            if i < count - 1 { step = try await engine.forwardWithPerPositionLogits([token])[0] }
        }
        return DraftProposal(tokens: tokens, logits: stepLogits)
    }
}
