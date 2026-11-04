// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// How speculative decoding is configured, with static checks of the command-line options that
/// configure it. The checks are kept apart from argument parsing so they can be tested. Each returns
/// a message naming the option and the accepted values, or `nil` when the value is acceptable.
package enum SpeculativeDecodingOptions {
    /// A message when `count` is outside 1 to ``SpeculativeDefaults/maxDraftTokens``, the most any
    /// drafter is asked for. The count is a request, not a promise: the decoder shortens each
    /// proposal to what fits, which is the target's widest verify block
    /// (``SpeculativeEngine/maxVerifyBlockLength``), the room left in the context of the target and of
    /// the drafter, and, with tile alignment on, the rest of the target's current verify tile. A
    /// drafter returns fewer tokens than asked when it has fewer to offer, as prompt lookup does when
    /// nothing recurs. The protocol has no per-drafter maximum yet; a drafter with a fixed block
    /// width would need to report one.
    package static func draftCountProblem(_ count: Int?) -> String? {
        guard let count else { return nil }
        guard count >= 1, count <= SpeculativeDefaults.maxDraftTokens else {
            return "--num-draft-tokens must be between 1 and \(SpeculativeDefaults.maxDraftTokens) (got \(count))"
        }
        return nil
    }

    /// Sampling needs the draft distribution, which only a draft model supplies. Prompt lookup and
    /// the draft head propose tokens without one, so they require greedy decoding, also when a draft
    /// model is given as well.
    package static func samplingProblem(greedy: Bool, promptLookup: Bool, draftHead: Bool = false) -> String? {
        guard !greedy else { return nil }
        if promptLookup {
            return
                "--prompt-lookup does not support sampling (it supplies no draft logits); use --sampling-strategy greedy"
        }
        if draftHead {
            return
                "--speculative does not support sampling (the draft head supplies no draft logits); use --sampling-strategy greedy"
        }
        return nil
    }

    /// `--speculative` drafts with the bundle's draft head, so it excludes a separate `--draft-model`,
    /// and `--drafter-top-k`, which configures the head's scan, needs it and a positive count.
    package static func draftHeadProblem(draftHead: Bool, draftModel: Bool, drafterTopK: Int?) -> String? {
        if draftHead && draftModel { return "--speculative and --draft-model cannot be used together" }
        guard let drafterTopK else { return nil }
        guard draftHead else { return "--drafter-top-k applies to --speculative" }
        return drafterTopK >= 1 ? nil : "--drafter-top-k must be >= 1 (got \(drafterTopK))"
    }

    /// Grammar-masked acceptance is defined for the argmax only, so a schema needs greedy decoding.
    package static func grammarSamplingProblem(greedy: Bool, hasSchema: Bool) -> String? {
        guard hasSchema, !greedy else { return nil }
        return "--json-schema with speculative decoding requires greedy decoding; "
            + "use --sampling-strategy greedy"
    }

    /// A message when `--grammar-proposal-max` is outside 0 to ``GrammarConstraint/maxForcedRunLimit``,
    /// or is given without both speculative decoding and `--json-schema`, which are what it applies to.
    package static func grammarProposalProblem(_ limit: Int?, speculating: Bool, hasSchema: Bool) -> String? {
        guard let limit else { return nil }
        guard speculating, hasSchema else {
            return "--grammar-proposal-max applies to speculative decoding with --json-schema"
        }
        guard (0...GrammarConstraint.maxForcedRunLimit).contains(limit) else {
            return "--grammar-proposal-max must be between 0 and \(GrammarConstraint.maxForcedRunLimit) "
                + "(the grammar session rolls back at most \(ConstrainedGenerationSession.maxRollbackTokens) "
                + "tokens, one of them the anchor); got \(limit)"
        }
        return nil
    }

    /// A message naming the first of `used`, command-line options that have no meaning with
    /// speculative decoding (they need a generation mode it does not provide), or `nil`.
    package static func unsupportedProblem(used: [String]) -> String? {
        used.first.map { "\($0) cannot be used with speculative decoding" }
    }

    /// A warning naming `ignored`, options speculative decoding accepts and does not apply, or `nil`
    /// when there are none.
    package static func ignoredWarning(ignored: [String]) -> String? {
        guard !ignored.isEmpty else { return nil }
        return "Warning: speculative decoding ignores \(ignored.joined(separator: ", ")). "
            + "Generation stops at the tokenizer's end-of-sequence tokens and at single-token "
            + "--stop-tokens, and samples with --temperature alone."
    }

    /// The stops speculative decoding ends a run at, from the stops a plain run uses: the
    /// single-token ones. The others are `ignored`: a verify block can accept the first tokens of a
    /// sequence before the stop is recognizable, so stopping on a sequence would need the decoder to
    /// hold back accepted tokens that might begin one. None of a sequence's tokens joins the stop set,
    /// so a run never stops on part of one.
    package static func stopTokenIds(_ stops: StopSequences) -> (ids: Set<Int32>, ignored: [[Int32]]) {
        let ids = Set(stops.sequences.filter { $0.count == 1 }.map { $0[0] })
        let ignored = stops.sequences.filter { $0.count > 1 }
        return (ids, ignored)
    }

    /// A warning naming the `--stop-tokens` speculative decoding does not stop at, or `nil`.
    /// Each is named once, in the order given.
    package static func multiTokenStopWarning(_ names: [String]) -> String? {
        var multiToken: [String] = []
        for name in names where !multiToken.contains(name) {
            multiToken.append(name)
        }
        guard !multiToken.isEmpty else { return nil }
        return "Warning: speculative decoding stops only at --stop-tokens that encode to a single "
            + "token; ignoring \(multiToken.map({ "'\($0)'" }).joined(separator: ", "))."
    }

    /// Raw tokens carry no text to place around the image, so they cannot be combined with it.
    package static func imageProblem(hasImage: Bool, hasRawTokens: Bool) -> String? {
        hasImage && hasRawTokens ? "--image cannot be used with --raw-tokens" : nil
    }

    /// A prompt that fills the context leaves no position to generate into.
    package static func promptLengthProblem(promptTokens: Int, contextLength: Int) -> String? {
        guard promptTokens >= contextLength else { return nil }
        return "The prompt is \(promptTokens) tokens, which is not shorter than the context length "
            + "(\(contextLength)); shorten the prompt or export the model with a longer context"
    }
}
