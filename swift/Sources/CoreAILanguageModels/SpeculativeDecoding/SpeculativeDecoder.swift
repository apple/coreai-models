// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Speculative decoding across a **drafter** and a large **target** engine.
///
/// Each iteration:
/// 1. The ``DrafterEngine`` proposes a block of candidate tokens (a linear chain
///    or a tree) following the current prefix.
/// 2. The **target** verifies the whole proposal in a *single* forward pass
///    (``SpeculativeEngine/forwardProposal(anchor:proposal:)``).
/// 3. A ``VerificationPolicy`` accepts a prefix / path of the proposal and picks
///    the next token; both KV caches roll back to the accepted length.
///
/// Correctness is defined by the target:
/// - `.greedy` — output is **token-for-token identical** to greedy target
///   decoding (acceptance is exact argmax match; the correction is always the
///   target's own greedy token).
/// - `.sampling` — output is **distributionally identical** to sampling the
///   target at that temperature (Chen et al. 2023 accept/reject).
///
/// The decoder names no concrete engine type: draft and target may run on
/// different backends, and the drafter may be a separate model
/// (``AutoregressiveDrafter``) or an attached head that drafts off the target's
/// injected hidden-state features. Drafter and target must share a vocabulary.
public final class SpeculativeDecoder: @unchecked Sendable {
    private let drafter: any DrafterEngine
    private let target: any SpeculativeEngine
    private let numDraftTokens: Int

    /// Acceptance / throughput statistics for a speculative run.
    public struct Stats: Sendable {
        /// Tokens actually emitted (accepted drafts + corrections).
        public var generatedTokens: Int = 0
        /// Draft tokens proposed across all iterations.
        public var draftTokensProposed: Int = 0
        /// Draft tokens the target accepted.
        public var draftTokensAccepted: Int = 0
        /// Number of propose/verify iterations (= target forward passes).
        public var iterations: Int = 0
        /// Wall-clock seconds spent in the generation loop only (excludes the
        /// one-time prompt prefill / graph specialization), so it is comparable to
        /// the base runner's "Generation:" metric.
        public var generationSeconds: Double = 0

        /// Fraction of proposed draft tokens the target accepted.
        public var acceptanceRate: Double {
            draftTokensProposed == 0 ? 0 : Double(draftTokensAccepted) / Double(draftTokensProposed)
        }

        /// Mean tokens committed per target forward pass. Values > 1 mean the
        /// target ran fewer forward passes than tokens produced — the speedup.
        public var meanTokensPerTargetPass: Double {
            iterations == 0 ? 0 : Double(generatedTokens) / Double(iterations)
        }
    }

    /// Pair a small draft *model* (run autoregressively) with a target model.
    ///
    /// - Parameters:
    ///   - draft: small proposal engine (any ``SpeculativeEngine`` backend).
    ///   - target: large verification engine; its output defines correctness. May
    ///     run on a different backend than the draft.
    ///   - numDraftTokens: tokens the draft proposes per step (`k`). Default 4.
    public init(draft: any SpeculativeEngine, target: any SpeculativeEngine, numDraftTokens: Int = 4) {
        self.drafter = AutoregressiveDrafter(engine: draft)
        self.target = target
        self.numDraftTokens = max(1, numDraftTokens)
    }

    /// Drive an arbitrary ``DrafterEngine`` (e.g. an attached DFlash / MTP head)
    /// against a target.
    ///
    /// - Parameters:
    ///   - drafter: proposal source.
    ///   - target: verification engine; its output defines correctness.
    ///   - numDraftTokens: candidates the drafter proposes per step (`k`). Default 4.
    public init(drafter: any DrafterEngine, target: any SpeculativeEngine, numDraftTokens: Int = 4) {
        self.drafter = drafter
        self.target = target
        self.numDraftTokens = max(1, numDraftTokens)
    }

    /// Run greedy speculative decoding (lossless; token-for-token identical to the
    /// target alone). See the type overview.
    public func generateGreedy(
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        try await run(
            policy: .greedy, promptTokens: promptTokens, maxNewTokens: maxNewTokens,
            stopTokenIds: stopTokenIds, onToken: onToken)
    }

    /// Run speculative **sampling** at the given temperature (distributionally
    /// identical to sampling the target).
    ///
    /// - Note: temperature-only (no top-k/top-p); those would have to be applied
    ///   consistently to both `p` and `q` to preserve the guarantee.
    public func generateSampling(
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        temperature: Double,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        try await run(
            policy: .sampling(temperature: temperature), promptTokens: promptTokens,
            maxNewTokens: maxNewTokens, stopTokenIds: stopTokenIds, onToken: onToken)
    }

    // MARK: - Core loop

    private func run(
        policy: VerificationPolicy,
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        var stats = Stats()
        guard maxNewTokens > 0, !promptTokens.isEmpty else { return stats }

        var rng = SystemRandomNumberGenerator()
        let maxContext = min(drafter.maxContextLength, target.maxContextLength)

        // `tokens` = prompt + every confirmed generated token.
        var tokens = promptTokens

        // Start both engines clean.
        try await target.reset(to: 0)
        try await drafter.reset(to: 0)

        // Invariant across iterations: the target has processed every token *except
        // the last* (the "anchor"), so verifying [anchor, draft…] yields the
        // target's prediction for the anchor's successor at slot 0. Prefill all but
        // the last prompt token here.
        if promptTokens.count > 1 {
            try await target.advance(Array(promptTokens[0..<promptTokens.count - 1]))
        }

        var generated = 0
        let genClock = ContinuousClock()
        let genStart = genClock.now

        generate: while generated < maxNewTokens {
            // Leave room for the verify batch ([anchor] + k drafts) plus a token.
            if tokens.count + numDraftTokens + 1 > maxContext { break }
            stats.iterations += 1
            let prefixLen = tokens.count
            let anchor = tokens[prefixLen - 1]
            let baseProcessed = target.processedTokenCount  // == prefixLen - 1

            // 1. Drafter proposes.
            let proposal = try await drafter.draftAll(
                confirmedPrefix: tokens, count: numDraftTokens, policy: policy)
            stats.draftTokensProposed += proposal.tokens.count

            // 2-3. Target verifies the whole proposal in one pass, then accept per
            //      policy → the accepted candidate indices + next token. Greedy uses
            //      the argmax-only verify path (no per-slot [vocabSize] copy); only
            //      sampling needs the full distributions.
            let verdict: (path: [Int], next: Int32)
            let features: DrafterFeatures?
            switch policy {
            case .greedy:
                let (targetArgmax, f) = try await target.forwardProposalGreedy(
                    anchor: anchor, proposal: proposal)
                features = f
                verdict = acceptGreedy(proposal: proposal, targetArgmax: targetArgmax)
            case .sampling:
                let (targetLogits, f) = try await target.forwardProposal(
                    anchor: anchor, proposal: proposal)
                features = f
                verdict = acceptSampling(
                    proposal: proposal, targetLogits: targetLogits,
                    temperature: policy.temperature, rng: &rng)
            }
            let acceptedTokens = verdict.path.map { proposal.tokens[$0] }
            stats.draftTokensAccepted += acceptedTokens.count

            // 4. Roll the target KV to [confirmed prefix + accepted path]. A linear
            //    proposal's verify order already matches, so truncation suffices; a
            //    tree's accepted path is a subset, so re-lay it explicitly.
            let confirmedLength = baseProcessed + 1 + acceptedTokens.count
            if proposal.isLinear {
                try await target.reset(to: confirmedLength)
            } else {
                try await target.reset(to: baseProcessed)
                try await target.advance([anchor] + acceptedTokens)
            }

            // 5. Bring the drafter into step (token drafter rolls back; attached-head
            //    drafter injects the committed feature rows).
            try await drafter.commit(confirmedLength: confirmedLength, features: features)

            // 6. Commit accepted drafts + the correction/bonus token.
            for token in acceptedTokens {
                tokens.append(token)
                onToken(token)
                generated += 1
                stats.generatedTokens += 1
                if stopTokenIds.contains(token) { break generate }
                if generated >= maxNewTokens { break generate }
            }
            let correction = verdict.next
            tokens.append(correction)
            onToken(correction)
            generated += 1
            stats.generatedTokens += 1
            if stopTokenIds.contains(correction) { break generate }
        }

        let genElapsed = genClock.now - genStart
        stats.generationSeconds =
            Double(genElapsed.components.seconds)
            + Double(genElapsed.components.attoseconds) / 1e18
        return stats
    }

    // MARK: - Acceptance

    /// Greedy acceptance: walk the proposal from the anchor, following the target's
    /// argmax at each step, until it diverges from the available candidates. Works
    /// for a linear chain and a tree alike (`parentSlot` treats a chain as the
    /// degenerate tree). The correction/bonus is always the target's own argmax, so
    /// the committed stream is target-exact.
    private func acceptGreedy(
        proposal: DraftProposal, targetArgmax: [Int32]
    ) -> (path: [Int], next: Int32) {
        var currentSlot = 0  // slot 0 = anchor
        var path: [Int] = []
        while true {
            let want = targetArgmax[currentSlot]
            var chosen: Int? = nil
            for i in 0..<proposal.tokens.count
            where proposal.parentSlot(of: i) == currentSlot && proposal.tokens[i] == want {
                chosen = i
                break
            }
            guard let child = chosen else { break }
            path.append(child)
            currentSlot = child + 1
        }
        return (path, targetArgmax[currentSlot])
    }

    /// Speculative sampling (Chen et al. 2023). Linear proposals only: accept
    /// `x_i ~ q_i` with probability `min(1, p_i(x_i)/q_i(x_i))`; on rejection
    /// resample the normalized residual `norm(relu(p−q))` and stop; if all accepted,
    /// sample a bonus token from the target's next distribution.
    private func acceptSampling(
        proposal: DraftProposal, targetLogits: [[LogitsScalarType]],
        temperature: Double, rng: inout SystemRandomNumberGenerator
    ) -> (path: [Int], next: Int32) {
        precondition(proposal.isLinear, "Speculative sampling supports linear proposals only.")
        guard let draftLogits = proposal.logits else {
            preconditionFailure("Speculative sampling needs per-candidate draft logits.")
        }

        var accepted = 0
        var next: Int32 = 0
        var haveNext = false
        while accepted < proposal.tokens.count {
            let p = SpeculativeMath.temperatureSoftmax(targetLogits[accepted], temperature: temperature)
            let q = SpeculativeMath.temperatureSoftmax(draftLogits[accepted], temperature: temperature)
            let x = Int(proposal.tokens[accepted])
            let acceptProb = q[x] > 0 ? min(1.0, Double(p[x]) / Double(q[x])) : 1.0
            if Double.random(in: 0..<1, using: &rng) < acceptProb {
                accepted += 1
            } else {
                next = SpeculativeMath.sampleResidual(p: p, q: q, using: &rng)
                haveNext = true
                break
            }
        }
        if !haveNext {
            let p = SpeculativeMath.temperatureSoftmax(targetLogits[accepted], temperature: temperature)
            next = SpeculativeMath.sampleFromProbs(p, using: &rng)
        }
        return (Array(0..<accepted), next)
    }
}
