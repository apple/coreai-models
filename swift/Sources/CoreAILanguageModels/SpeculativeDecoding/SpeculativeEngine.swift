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
/// - draft and target may run on *different* backends (e.g. a small draft on the CPU
///   verifying against a target on an accelerator), and
/// - the loop is testable against an in-memory double without any model assets.
///
/// A conforming engine is an ``InferenceEngine``: it owns a KV cache indexed by committed-token
/// count, `processedTokenCount` is the rollback anchor, and `reset(to:)` rewinds it. The
/// `forward…` primitives extend it by the number of tokens passed and return one prediction per
/// input position (the prediction for the token that *follows* that position). `advance(_:)`
/// extends it without materializing logits.
package protocol SpeculativeEngine: InferenceEngine {
    /// Width of the tile a verify pass runs in, for an engine whose passes start on a multiple of
    /// it. A block that crosses a boundary costs a second forward. nil when there is no such tile.
    var verifyTileWidth: Int? { get }

    /// Most tokens (anchor plus proposal) one verify pass can score, and roll back from, while
    /// returning a prediction for every position. A longer block would score fewer positions than
    /// it holds, so the decoder shortens proposals to fit. `Int.max` when the engine has no limit.
    var maxVerifyBlockLength: Int { get }

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
    /// The default implementation scores the flat batch and emits no features. Engines with a
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
    /// dominates the verify cost.
    func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?)

    /// Most proposal tokens that fit, together with the anchor, in a single forward pass
    /// from the current position. Engines with a fixed batch width return the room left in
    /// the current batch window (a block that crosses it costs a second forward);
    /// engines without that constraint return `Int.max`.
    var oneForwardDraftBudget: Int { get }
}

extension SpeculativeEngine {
    package var verifyTileWidth: Int? { nil }
    package var maxVerifyBlockLength: Int { Int.max }

    package func forwardProposal(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (logits: [[LogitsScalarType]], features: DrafterFeatures?) {
        let logits = try await forwardWithPerPositionLogits([anchor] + proposal.tokens)
        return (logits, nil)
    }

    package func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?) {
        let argmax = try await forwardPerPositionArgmax([anchor] + proposal.tokens)
        return (argmax, nil)
    }

    package var oneForwardDraftBudget: Int { Int.max }
}

extension SpeculativeEngine {
    /// Roll the cache back to `tokenIndex` and confirm the engine is there.
    ///
    /// An engine with recurrent state cannot truncate its cache, so a rewind to part of the
    /// committed tokens full-resets it instead (see ``InferenceEngine/hasRecurrentState``). The
    /// decoder reasons from the cursor, so a cursor that did not land on `tokenIndex` is an error,
    /// not something to continue from.
    package func rewindAndConfirm(to tokenIndex: Int) async throws {
        try await reset(to: tokenIndex)
        guard processedTokenCount == tokenIndex else {
            throw InferenceRuntimeError.invalidState(
                "the engine holds \(processedTokenCount) tokens after a rewind to \(tokenIndex); "
                    + "an engine that cannot rewind part way cannot be speculated on")
        }
    }
}
