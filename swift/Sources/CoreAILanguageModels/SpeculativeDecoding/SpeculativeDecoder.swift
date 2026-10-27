// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Speculative decoding across a **drafter** and a large **target** engine.
///
/// Each iteration:
/// 1. The ``DrafterEngine`` proposes a chain of candidate tokens following the
///    current prefix.
/// 2. The **target** verifies the whole proposal in a *single* forward pass
///    (``SpeculativeEngine/forwardProposal(anchor:proposal:)``).
/// 3. A ``VerificationPolicy`` accepts a prefix / path of the proposal and picks
///    the next token; both KV caches roll back to the accepted length.
///
/// Correctness is defined by the target, through a ``VerificationPolicy``:
/// - `.greedy` — output is **token-for-token identical** to greedy target
///   decoding (acceptance is exact argmax match; the correction is always the
///   target's own greedy token).
/// - `.sampling` — output is **distributionally identical** to sampling the
///   target at that temperature (the accept/reject scheme of Chen et al. 2023,
///   https://arxiv.org/abs/2302.01318).
///
/// The decoder names no concrete engine type: draft and target may run on
/// different backends, and the drafter may be a separate model
/// (``AutoregressiveDrafter``), a model-free drafter such as ``PromptLookupDrafter``, or a
/// head that drafts off the target's hidden-state features, which it receives through
/// ``DrafterEngine/commit(confirmedLength:features:)``. Drafter and target must share a vocabulary.
///
/// Single-owner: set its properties before generating, and run one generation at a time; it holds
/// no lock, and the engines it drives each hold one sequence.
package final class SpeculativeDecoder: @unchecked Sendable {
    private let drafter: any DrafterEngine
    private let target: any SpeculativeEngine
    /// Drafts per round (`k`); with ``init(drafter:target:numDraftTokens:alignDraftsToVerifyTile:adaptiveDraftCount:)``'s
    /// adaptive count, the starting value.
    package let numDraftTokens: Int
    private let alignDraftsToVerifyTile: Bool
    private let adaptiveDraftCount: Bool

    /// When `true`, linear proposals are shortened to fit the anchor's single forward
    /// window (``SpeculativeEngine/oneForwardDraftBudget``), so a verify never pays a
    /// second forward to cross a batch boundary. When the window has no room left for a
    /// proposal, the full proposal is used and spans two forwards. Output is unchanged.
    package var alignProposalsToForward: Bool = true

    /// Prompt lookup: when set, a block whose recent tokens reappear earlier in the context
    /// proposes what followed that occurrence instead of drafting. Output is unchanged.
    package var promptLookup: PromptLookup?

    /// Acceptance / throughput statistics for a speculative run.
    package struct Stats: Sendable {
        /// Tokens actually emitted (accepted drafts + corrections).
        package var generatedTokens: Int = 0
        /// Draft tokens proposed across all iterations.
        package var draftTokensProposed: Int = 0
        /// Draft tokens the target accepted.
        package var draftTokensAccepted: Int = 0
        /// Number of propose/verify iterations (= target forward passes).
        package var iterations: Int = 0
        /// Wall-clock seconds spent in the generation loop only (excludes the
        /// one-time prompt prefill / graph specialization), so it is comparable to the time
        /// plain decoding spends generating the same tokens.
        package var generationSeconds: Double = 0
        /// Iterations whose proposal came from prompt lookup, and its proposed/accepted tokens.
        package var lookupProposals: Int = 0
        package var lookupTokensProposed: Int = 0
        package var lookupTokensAccepted: Int = 0

        /// Fraction of proposed draft tokens the target accepted.
        package var acceptanceRate: Double {
            draftTokensProposed == 0 ? 0 : Double(draftTokensAccepted) / Double(draftTokensProposed)
        }

        /// Mean tokens committed per target forward pass. Values > 1 mean the
        /// target ran fewer forward passes than tokens produced — the speedup.
        package var meanTokensPerTargetPass: Double {
            iterations == 0 ? 0 : Double(generatedTokens) / Double(iterations)
        }
    }

    /// Pair a small draft *model* (run autoregressively) with a target model.
    ///
    /// - Parameters:
    ///   - draft: small proposal engine (any ``SpeculativeEngine`` backend).
    ///   - target: large verification engine; its output defines correctness. May
    ///     run on a different backend than the draft.
    ///   - numDraftTokens: tokens the draft proposes per step (`k`), from 1 to
    ///     ``SpeculativeDefaults/maxDraftTokens``. Default ``SpeculativeDefaults/numDraftTokens``.
    ///   - adaptiveDraftCount: see ``init(drafter:target:numDraftTokens:alignDraftsToVerifyTile:adaptiveDraftCount:)``.
    package init(
        draft: any SpeculativeEngine, target: any SpeculativeEngine,
        numDraftTokens: Int = SpeculativeDefaults.numDraftTokens,
        adaptiveDraftCount: Bool = false
    ) {
        self.drafter = AutoregressiveDrafter(engine: draft)
        self.target = target
        precondition(
            (1...SpeculativeDefaults.maxDraftTokens).contains(numDraftTokens),
            "numDraftTokens \(numDraftTokens) must be in 1...\(SpeculativeDefaults.maxDraftTokens)")
        self.numDraftTokens = numDraftTokens
        self.alignDraftsToVerifyTile = false
        self.adaptiveDraftCount = adaptiveDraftCount
    }

    /// Drive an arbitrary ``DrafterEngine`` against a target.
    ///
    /// - Parameters:
    ///   - drafter: proposal source.
    ///   - target: verification engine; its output defines correctness.
    ///   - numDraftTokens: candidates the drafter proposes per step (`k`), from 1 to
    ///     ``SpeculativeDefaults/maxDraftTokens``. Default ``SpeculativeDefaults/numDraftTokens``.
    ///   - alignDraftsToVerifyTile: propose only as many drafts as fit in the target's current
    ///     verify tile (see ``SpeculativeEngine/verifyTileWidth``), so every verify is one forward.
    ///   - adaptiveDraftCount: start at `numDraftTokens`, then raise the draft count by 2 after a
    ///     round whose drafts were all accepted and lower it by 1 after any rejection, within 1 and
    ///     the larger of `numDraftTokens` and the verify tile's last slot.
    package init(
        drafter: any DrafterEngine, target: any SpeculativeEngine,
        numDraftTokens: Int = SpeculativeDefaults.numDraftTokens,
        alignDraftsToVerifyTile: Bool = false, adaptiveDraftCount: Bool = false
    ) {
        self.drafter = drafter
        self.target = target
        precondition(
            (1...SpeculativeDefaults.maxDraftTokens).contains(numDraftTokens),
            "numDraftTokens \(numDraftTokens) must be in 1...\(SpeculativeDefaults.maxDraftTokens)")
        self.numDraftTokens = numDraftTokens
        self.alignDraftsToVerifyTile = alignDraftsToVerifyTile
        self.adaptiveDraftCount = adaptiveDraftCount
    }

    /// Run greedy speculative decoding (lossless; token-for-token identical to the
    /// target alone). See the type overview.
    package func generateGreedy(
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        onPrefillReady: (() async throws -> Void)? = nil,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        // Greedy decoding draws nothing; the run takes a generator all the same.
        var rng = SystemRandomNumberGenerator()
        return try await run(
            policy: .greedy, promptTokens: promptTokens, maxNewTokens: maxNewTokens,
            stopTokenIds: stopTokenIds, onPrefillReady: onPrefillReady, onToken: onToken, rng: &rng)
    }

    /// Run speculative **sampling** at the given temperature (distributionally
    /// identical to sampling the target), drawing from `rng` for the whole run: the drafter's
    /// samples, the accept/reject tests and the correction or bonus token. A known generator
    /// makes the run reproducible.
    ///
    /// - Note: temperature-only (no top-k/top-p); those would have to be applied
    ///   consistently to both `p` and `q` to preserve the guarantee.
    package func generateSampling(
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        temperature: Double,
        using rng: inout some RandomNumberGenerator,
        onPrefillReady: (() async throws -> Void)? = nil,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        try await run(
            policy: .sampling(temperature: temperature), promptTokens: promptTokens,
            maxNewTokens: maxNewTokens, stopTokenIds: stopTokenIds,
            onPrefillReady: onPrefillReady, onToken: onToken, rng: &rng)
    }

    /// Run speculative sampling with a generator made once for the run: a
    /// ``SeededRandomNumberGenerator`` when `seed` is set, so the same seed, prompt and models
    /// reproduce the same tokens, else the system generator.
    package func generateSampling(
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        temperature: Double,
        seed: UInt64? = nil,
        onPrefillReady: (() async throws -> Void)? = nil,
        onToken: (Int32) -> Void
    ) async throws -> Stats {
        if let seed {
            var rng = SeededRandomNumberGenerator(seed: seed)
            return try await generateSampling(
                promptTokens: promptTokens, maxNewTokens: maxNewTokens, stopTokenIds: stopTokenIds,
                temperature: temperature, using: &rng, onPrefillReady: onPrefillReady,
                onToken: onToken)
        }
        var rng = SystemRandomNumberGenerator()
        return try await generateSampling(
            promptTokens: promptTokens, maxNewTokens: maxNewTokens, stopTokenIds: stopTokenIds,
            temperature: temperature, using: &rng, onPrefillReady: onPrefillReady, onToken: onToken)
    }

    // MARK: - Core loop

    private func run(
        policy: VerificationPolicy,
        promptTokens: [Int32],
        maxNewTokens: Int,
        stopTokenIds: Set<Int32>,
        onPrefillReady: (() async throws -> Void)? = nil,
        onToken: (Int32) -> Void,
        rng: inout some RandomNumberGenerator
    ) async throws -> Stats {
        var stats = Stats()
        if case .sampling(let temperature) = policy {
            guard temperature > 0, temperature.isFinite else {
                throw InferenceRuntimeError.invalidArgument(
                    "speculative sampling needs a positive finite temperature (got \(temperature))")
            }
            guard promptLookup == nil else {
                throw InferenceRuntimeError.invalidArgument(
                    "prompt lookup supplies no draft distribution, so it supports greedy decoding only")
            }
        }
        guard maxNewTokens > 0, !promptTokens.isEmpty else { return stats }

        guard !target.hasRecurrentState else {
            throw InferenceRuntimeError.invalidArgument(
                "speculative decoding rewinds the target after a rejection, which an engine with "
                    + "recurrent state cannot do")
        }
        let maxContext = min(drafter.maxContextLength, target.config.maxContextLength)
        guard promptTokens.count <= maxContext else {
            throw InferenceRuntimeError.invalidArgument(
                "the prompt of \(promptTokens.count) tokens exceeds the \(maxContext)-token context")
        }

        // `tokens` = prompt + every confirmed generated token.
        var tokens = promptTokens

        // Start both engines clean.
        try await target.rewindAndConfirm(to: 0)
        try await drafter.reset(to: 0)

        // After the reset (which clears any per-run state the target holds) and before the
        // prompt prefill, let the caller stage per-run target state that the prefill consumes.
        // Only the initial prefill needs this; the verification-rollback resets in the loop
        // re-advance only committed tokens, so no re-staging is required.
        try await onPrefillReady?()

        // Invariant across iterations: the target has processed every token *except
        // the last* (the "anchor"), so verifying [anchor, draft…] yields the
        // target's prediction for the anchor's successor at slot 0. Prefill all but
        // the last prompt token here.
        if promptTokens.count > 1 {
            try await target.advance(Array(promptTokens[0..<promptTokens.count - 1]))
        }

        var generated = 0
        var draftBudget = numDraftTokens
        let maxDraftBudget = max(numDraftTokens, (target.verifyTileWidth ?? 1) - 1)
        let genClock = SuspendingClock()
        let genStart = genClock.now

        generate: while generated < maxNewTokens {
            try Task.checkCancellation()
            let prefixLen = tokens.count
            let anchor = tokens[prefixLen - 1]
            // Every token but the anchor is in the target's cache; the rest of the round counts from
            // here, so a target that holds anything else would verify at the wrong positions.
            let baseProcessed = target.processedTokenCount
            guard baseProcessed == prefixLen - 1 else {
                throw InferenceRuntimeError.invalidState(
                    "the target holds \(baseProcessed) tokens, but \(prefixLen - 1) precede the anchor")
            }
            // The drafter's count this round. A block that crosses the verify tile costs a second
            // target forward -- as much as the whole round -- so a tile-aligned run drafts only what
            // fits, and nothing when only the anchor does.
            // Every proposal, whatever its source, is also cut to the context left after the anchor
            // and to the widest block the target can verify, so a large (adaptive) budget shortens
            // the last rounds instead of ending generation early, and a block never exceeds what the
            // target can score position by position.
            let roomForDrafts = maxContext - prefixLen
            if roomForDrafts < 0 { break }
            let proposalCap = min(roomForDrafts, max(1, target.maxVerifyBlockLength) - 1)
            let k = min(draftCount(anchorPosition: baseProcessed, upTo: draftBudget), proposalCap)
            stats.iterations += 1

            // 1. Propose: prompt lookup when the recent tokens recur in the context, else the
            //    drafter. Both are verified the same way below, and both are sized to the
            //    target's one-forward window when alignment is on.
            let budget = target.oneForwardDraftBudget
            let proposalLimit = min(alignProposalsToForward && budget >= 1 ? budget : Int.max, proposalCap)
            let proposal: DraftProposal
            var lookupTokens: [Int32] = []
            if let promptLookup {
                lookupTokens = promptLookup.propose(
                    context: tokens, count: min(promptLookup.maxTokens, proposalLimit))
            }
            if !lookupTokens.isEmpty {
                proposal = DraftProposal(tokens: lookupTokens)
                stats.lookupProposals += 1
                stats.lookupTokensProposed += lookupTokens.count
            } else {
                let count = min(k, proposalLimit)
                proposal =
                    count > 0
                    ? try await drafter.draftAll(
                        confirmedPrefix: tokens, count: count, policy: policy, using: &rng)
                    : DraftProposal(tokens: [])
            }
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
                verdict = try acceptGreedy(proposal: proposal, targetArgmax: targetArgmax)
            case .sampling:
                let (targetLogits, f) = try await target.forwardProposal(
                    anchor: anchor, proposal: proposal)
                features = f
                verdict = try acceptSampling(
                    proposal: proposal, targetLogits: targetLogits,
                    temperature: policy.temperature, rng: &rng)
            }
            let acceptedTokens = verdict.path.map { proposal.tokens[$0] }
            stats.draftTokensAccepted += acceptedTokens.count
            if !lookupTokens.isEmpty { stats.lookupTokensAccepted += acceptedTokens.count }
            // Only the drafter's own rounds steer its budget: a round with no drafts (anchor on the
            // tile's last slot), or one prompt lookup proposed, says nothing about it.
            if adaptiveDraftCount, lookupTokens.isEmpty, !proposal.tokens.isEmpty {
                draftBudget =
                    acceptedTokens.count == proposal.tokens.count
                    ? min(draftBudget + SpeculativeDefaults.adaptiveGrowth, maxDraftBudget)
                    : max(draftBudget - SpeculativeDefaults.adaptiveShrink, 1)
            }

            // 4. Roll the target KV to [confirmed prefix + accepted drafts]: the verify order
            //    matches the proposal's, so truncation suffices.
            let confirmedLength = baseProcessed + 1 + acceptedTokens.count
            try await target.rewindAndConfirm(to: confirmedLength)

            // 5. Bring the drafter into step (token drafter rolls back; attached-head
            //    drafter injects the committed feature rows).
            try await drafter.commit(confirmedLength: confirmedLength, features: features)

            // 6. Commit accepted drafts + the correction/bonus token.
            for token in acceptedTokens {
                tokens.append(token)
                onToken(token)
                generated += 1
                stats.generatedTokens += 1
                if stopTokenIds.contains(token) || generated >= maxNewTokens { break generate }
            }
            let correction = verdict.next
            tokens.append(correction)
            onToken(correction)
            generated += 1
            stats.generatedTokens += 1
            if stopTokenIds.contains(correction) { break generate }
        }

        stats.generationSeconds = (genClock.now - genStart).inSeconds
        return stats
    }

    /// Verification returns one row per slot of `[anchor] + proposal`; fewer would be indexed past
    /// their end, and a shifted set would silently pair a slot with another position's prediction.
    private static func requireRows(_ count: Int, for proposal: DraftProposal) throws {
        guard count == proposal.tokens.count + 1 else {
            throw InferenceRuntimeError.invalidState(
                "verification returned \(count) rows for \(proposal.tokens.count + 1) slots")
        }
    }

    private func draftCount(anchorPosition: Int, upTo budget: Int) -> Int {
        guard alignDraftsToVerifyTile, let w = target.verifyTileWidth, w > 1 else { return budget }
        return min(budget, w - 1 - anchorPosition % w)
    }

    // MARK: - Acceptance

    /// Greedy acceptance: walk the proposal from the anchor, following the target's
    /// argmax at each step, until it diverges from the proposal. The correction/bonus is
    /// always the target's own argmax, so the committed stream is target-exact.
    private func acceptGreedy(
        proposal: DraftProposal, targetArgmax: [Int32]
    ) throws -> (path: [Int], next: Int32) {
        try Self.requireRows(targetArgmax.count, for: proposal)
        var currentSlot = 0  // slot 0 = anchor
        var path: [Int] = []
        while currentSlot < proposal.tokens.count, proposal.tokens[currentSlot] == targetArgmax[currentSlot] {
            path.append(currentSlot)
            currentSlot += 1
        }
        return (path, targetArgmax[currentSlot])
    }

    /// Speculative sampling (Chen et al. 2023, https://arxiv.org/abs/2302.01318). Linear proposals only: accept
    /// `x_i ~ q_i` with probability `min(1, p_i(x_i)/q_i(x_i))`; on rejection
    /// resample the normalized residual `norm(relu(p−q))` and stop; if all accepted,
    /// sample a bonus token from the target's next distribution.
    private func acceptSampling(
        proposal: DraftProposal, targetLogits: [[LogitsScalarType]],
        temperature: Double, rng: inout some RandomNumberGenerator
    ) throws -> (path: [Int], next: Int32) {
        // An empty proposal (no draft fits, as at the end of the context) needs no draft logits: the
        // round only samples the target's next token.
        let draftLogits = proposal.logits ?? []
        guard draftLogits.count >= proposal.tokens.count else {
            throw InferenceRuntimeError.invalidState(
                "Speculative sampling needs per-candidate draft logits; prompt lookup and drafters "
                    + "that return none support greedy decoding only.")
        }
        try Self.requireRows(targetLogits.count, for: proposal)

        var accepted = 0
        var next: Int32 = 0
        var haveNext = false
        while accepted < proposal.tokens.count {
            let p = SpeculativeMath.temperatureSoftmax(targetLogits[accepted], temperature: temperature)
            let q = SpeculativeMath.temperatureSoftmax(draftLogits[accepted], temperature: temperature)
            let x = Int(proposal.tokens[accepted])
            guard x >= 0, x < p.count, x < q.count else {
                throw InferenceRuntimeError.invalidState(
                    "draft token \(x) is outside the vocabulary (target \(p.count), draft \(q.count))")
            }
            // A token the draft gave no mass cannot have been drawn from it, so the ratio is
            // undefined: accept it only if the target can produce it.
            let acceptProb = q[x] > 0 ? min(1.0, Double(p[x]) / Double(q[x])) : (p[x] > 0 ? 1.0 : 0.0)
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
