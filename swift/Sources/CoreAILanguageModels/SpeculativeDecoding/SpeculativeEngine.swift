// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Backend-agnostic surface a speculative decoder needs from the draft and
/// target engines.
///
/// The decoder drives propose / verify / accept / rollback purely through this
/// protocol and never names a concrete engine type, so:
/// - draft and target may run on *different* backends (e.g. a small draft on the
///   CPU verifying against a target on the GPU or Neural Engine), and
/// - the loop is testable against an in-memory double without any model assets.
///
/// A conforming engine owns a KV cache indexed by committed-token count.
/// `processedTokenCount` is the rollback anchor; `reset(to:)` rewinds it; the
/// `forward…` primitives extend it by the number of tokens passed and return one
/// prediction per input position (the prediction for the token that *follows*
/// that position). `advance(_:)` extends it without materializing logits.
public protocol SpeculativeEngine: Sendable {
    /// Maximum number of tokens (prompt + generated) the engine can hold.
    var maxContextLength: Int { get }

    /// Tokens currently committed to the KV cache — the anchor `reset(to:)` and
    /// the decoder reason about.
    var processedTokenCount: Int { get }

    /// Roll the KV cache back so exactly `tokenIndex` tokens remain committed.
    func reset(to tokenIndex: Int) async throws

    /// Prefill `tokens` into the KV cache, discarding their logits. Advances
    /// `processedTokenCount` by `tokens.count`.
    func advance(_ tokens: [Int32]) async throws

    /// Forward `tokens` in a single pass and return, per input position, the
    /// argmax (greedy) token that follows it. Advances `processedTokenCount`.
    func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32]

    /// Forward `tokens` in a single pass and return, per input position, the full
    /// `[vocabSize]` logits predicting the following token. Advances
    /// `processedTokenCount`.
    func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]]

    /// Verify a draft proposal in a single pass.
    ///
    /// Returns per-slot logits aligned to `[anchor] + proposal.tokens` (slot `0`
    /// predicts the token after the anchor; slot `j+1` predicts the token after
    /// `proposal.tokens[j]`), plus the fused hidden-state `features` for those
    /// slots when the engine emits them (nil otherwise).
    ///
    /// The default implementation handles **linear** proposals by scoring the flat
    /// batch and emits no features. Engines that support tree attention masks or a
    /// fused feature output override this.
    func forwardProposal(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (logits: [[LogitsScalarType]], features: DrafterFeatures?)

    /// Greedy verification of a draft proposal in a single pass.
    ///
    /// Returns, per slot of `[anchor] + proposal.tokens`, the target's argmax
    /// (greedy) successor — everything greedy acceptance needs — plus the fused
    /// hidden-state `features` when the engine emits them. This avoids
    /// materializing a `[vocabSize]` distribution per slot, which for a large vocab
    /// dominates the verify cost and is what previously made speculative decoding
    /// slower than base.
    func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?)
}

extension SpeculativeEngine {
    public func forwardProposal(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (logits: [[LogitsScalarType]], features: DrafterFeatures?) {
        precondition(
            proposal.isLinear,
            "This engine only verifies linear proposals; tree drafting needs a mask-aware verifier.")
        let logits = try await forwardWithPerPositionLogits([anchor] + proposal.tokens)
        return (logits, nil)
    }

    public func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?) {
        precondition(
            proposal.isLinear,
            "This engine only verifies linear proposals; tree drafting needs a mask-aware verifier.")
        let argmax = try await forwardPerPositionArgmax([anchor] + proposal.tokens)
        return (argmax, nil)
    }
}

// The Core AI sequential engine already provides every primitive above; it only
// needs `maxContextLength` surfaced from its configuration to conform.
extension CoreAISequentialEngine: SpeculativeEngine {
    public var maxContextLength: Int { config.maxContextLength }
}

// Other engines (e.g. the static-shape / Neural Engine path) can conform by
// implementing the primitives above; that conformance is added alongside the
// engine changes it requires.
