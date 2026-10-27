// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// How speculative decoding is configured, with static checks of the command-line options that
/// configure it. The checks are kept apart from argument parsing so they can be tested. Each returns
/// a message naming the option and the accepted values, or `nil` when the value is acceptable.
package enum SpeculativeOptions {
    /// A message when `count` is outside 1 to ``SpeculativeDefaults/maxDraftTokens``. The decoder
    /// further shortens each proposal to the target's widest verify block
    /// (``SpeculativeEngine/maxVerifyBlockLength``).
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
