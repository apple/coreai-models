// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A training-free, model-free "cheap self-draft": prompt-lookup / n-gram decoding.
///
/// The draft comes from the model's *own* running output — it finds the most recent
/// earlier occurrence of the last `n` generated tokens and proposes the tokens that
/// followed it. There is no draft model and no forward pass, so drafting is
/// essentially free. Because the target still verifies on the (ANE) engine and the
/// draft costs ~0, *any* accepted draft token reduces the number of expensive target
/// passes — so this beats base whenever acceptance > 0 (unlike full-model
/// self-speculation, whose draft is as expensive as the target).
///
/// Greedy-only (no draft distribution), which is the common prompt-lookup setting.
///
/// Single-owner: one task drives it at a time.
package final class PromptLookupDrafter: DrafterEngine, @unchecked Sendable {
    private let lookup: PromptLookup
    package let maxContextLength: Int

    /// - Parameters:
    ///   - maxNgram: longest suffix to match (prefer longer, more-specific matches).
    ///   - minNgram: shortest suffix to accept a match on.
    package init(maxNgram: Int = 3, minNgram: Int = 1, maxContextLength: Int = 131072) {
        self.lookup = PromptLookup(maxNgram: maxNgram, minNgram: minNgram)
        self.maxContextLength = maxContextLength
    }

    package func reset(to tokenIndex: Int) async throws {}
    package func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}

    package func draftAll(confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy) async throws
        -> DraftProposal
    {
        // No match: no speculation this step (target advances one token as usual).
        DraftProposal(tokens: lookup.propose(context: confirmedPrefix, count: count))
    }
}
