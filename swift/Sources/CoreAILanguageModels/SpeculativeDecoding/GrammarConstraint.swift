// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// A grammar threaded through ``SpeculativeDecoder``.
///
/// ``ConstrainedGenerationSession`` is noncopyable (it owns a C++ matcher), so it cannot be passed
/// around by value; this class owns it and offers what constrained acceptance needs: the
/// grammar-masked argmax of a position's logits, advancing the grammar by a committed token,
/// rolling it back, and, when a tokenizer and a run limit are supplied, the tokens of the span the
/// grammar fully determines next, which the decoder verifies as a proposal instead of drafting.
///
/// Single-owner: the session is one sequence's state, driven by one generation at a time.
package final class GrammarConstraint: @unchecked Sendable {
    private var session: ConstrainedGenerationSession
    private let tokenizer: (any Tokenizer)?
    private let forcedRunLimit: Int

    /// Default most tokens a grammar proposal holds: the widest span one verify pass of the
    /// static-shape engine scores together with its anchor.
    package static let defaultForcedRunLimit = 7

    /// Largest grammar proposal that can be verified and rolled back. The session rolls back at most
    /// ``ConstrainedGenerationSession/maxRollbackTokens`` tokens, and a verify block is the anchor
    /// plus the proposal.
    package static let maxForcedRunLimit = ConstrainedGenerationSession.maxRollbackTokens - 1

    /// Tokens the grammar has accepted, net of rollbacks.
    private(set) var acceptedCount = 0

    /// - Parameters:
    ///   - session: the grammar session to enforce.
    ///   - tokenizer: tokenizer used to encode forced spans; required for grammar proposals.
    ///   - forcedRunLimit: most tokens per grammar proposal; `0` disables them. Values above
    ///     ``maxForcedRunLimit`` are clamped to it.
    package init(
        _ session: consuming ConstrainedGenerationSession,
        tokenizer: (any Tokenizer)? = nil,
        forcedRunLimit: Int = 0
    ) {
        self.session = session
        self.tokenizer = tokenizer
        self.forcedRunLimit = min(max(0, forcedRunLimit), Self.maxForcedRunLimit)
    }

    /// The grammar-legal token with the largest logit (the lowest index on a tie, a NaN ranking as
    /// negative infinity). Legality comes from the grammar's token mask, not from the logit values,
    /// so a legal token whose logit is negative infinity still beats every illegal one. When the
    /// grammar has terminated, or its mask allows no token of this row, no mask applies and the
    /// result is the plain argmax.
    func maskedArgmax(_ logits: [LogitsScalarType]) -> Int32 {
        var best = -1
        var bestValue = -Float.infinity
        if let mask = session.nextTokenBitmask() {
            for (word, bits) in mask.enumerated() where bits != 0 {
                let base = word &* 32
                if base >= logits.count { break }
                for bit in 0..<min(32, logits.count - base) where bits & (1 << Int32(bit)) != 0 {
                    let raw = Float(logits[base + bit])
                    let value = raw.isNaN ? -Float.infinity : raw
                    if best < 0 || value > bestValue {
                        best = base + bit
                        bestValue = value
                    }
                }
            }
        }
        if best >= 0 { return Int32(best) }
        var index = 0
        bestValue = -.infinity
        for i in 0..<logits.count where Float(logits[i]) > bestValue {
            bestValue = Float(logits[i])
            index = i
        }
        return Int32(index)
    }

    /// Advance the grammar by a committed token.
    func accept(_ token: Int32) {
        if session.acceptToken(token) { acceptedCount += 1 }
    }

    /// Whether the grammar has reached its terminal state.
    var isTerminated: Bool { session.isTerminated }

    /// Undo the last `count` accepted tokens. Throws when the session cannot roll back that many
    /// (its history is bounded), since the grammar would otherwise be left ahead of the emitted
    /// stream.
    func rollback(_ count: Int) throws {
        guard count > 0 else { return }
        guard session.rollback(count) else {
            throw InferenceRuntimeError.invalidState(
                "grammar rollback of \(count) token(s) failed (limit "
                    + "\(ConstrainedGenerationSession.maxRollbackTokens))")
        }
        acceptedCount -= count
    }

    /// Tokens for the span the grammar fully determines from its current state, without advancing
    /// it; empty when nothing is forced or grammar proposals are disabled.
    ///
    /// The forced string is tokenized without special tokens and trimmed to the tokens whose
    /// cumulative decoded length is strictly shorter than the string, leaving the trailing token to
    /// the target so it can merge with what follows. The tokens are checked against the matcher and
    /// rolled back, so the grammar advances only when the decoder commits them.
    func forcedProposal() throws -> [Int32] {
        guard forcedRunLimit > 0, let tokenizer, !session.isTerminated,
            let forced = session.findJumpForwardString(), !forced.isEmpty
        else { return [] }
        let ids = tokenizer.encode(text: forced, addSpecialTokens: false).map(Int32.init)
        guard !ids.isEmpty else { return [] }

        let forcedByteCount = forced.utf8.count
        var safeCount = 0
        // Only the first `forcedRunLimit` tokens can be proposed, so longer prefixes are never decoded.
        for i in 1...min(ids.count, forcedRunLimit) {
            let decoded = tokenizer.decode(tokens: ids.prefix(i).map { Int($0) })
            guard decoded.utf8.count < forcedByteCount else { break }
            safeCount = i
        }
        let tokens = Array(ids.prefix(safeCount))
        guard !tokens.isEmpty else { return [] }

        var accepted = 0
        for token in tokens {
            guard session.acceptToken(token) else { break }
            accepted += 1
        }
        if accepted > 0 {
            guard session.rollback(accepted) else {
                throw InferenceRuntimeError.invalidState(
                    "grammar rollback of \(accepted) proposed token(s) failed (limit "
                        + "\(ConstrainedGenerationSession.maxRollbackTokens))")
            }
        }
        return accepted == tokens.count ? tokens : []
    }
}
