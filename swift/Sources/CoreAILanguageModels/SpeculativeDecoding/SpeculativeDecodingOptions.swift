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

    /// Sampling needs the draft distribution, which only a draft model supplies. Prompt lookup
    /// proposes tokens without one, so it requires greedy decoding, also when a draft model is
    /// given as well.
    package static func samplingProblem(greedy: Bool, promptLookup: Bool) -> String? {
        guard !greedy, promptLookup else { return nil }
        return
            "--prompt-lookup does not support sampling (it supplies no draft logits); use --sampling-strategy greedy"
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
            + "Generation stops only at the tokenizer's end-of-sequence tokens and samples with "
            + "--temperature alone."
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
