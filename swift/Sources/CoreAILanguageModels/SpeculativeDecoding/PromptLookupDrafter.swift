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
/// essentially free. Because the target still verifies every token and the draft costs ~0,
/// *any* accepted draft token reduces the number of target passes (unlike full-model
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
    ///   - maxContextLength: context the drafter reports; the decoder takes the smaller of this and
    ///     the target's, so any positive value at least the target's is equivalent.
    package init(
        maxNgram: Int = SpeculativeDefaults.lookupMaxNgram,
        minNgram: Int = SpeculativeDefaults.lookupMinNgram,
        maxContextLength: Int = SpeculativeDefaults.maxContextLength
    ) {
        precondition(maxContextLength > 0, "PromptLookupDrafter maxContextLength \(maxContextLength) must be positive")
        // The block size is the `count` of each `draftAll` call, so the lookup's own limit is unused.
        self.lookup = PromptLookup(maxNgram: maxNgram, minNgram: minNgram)
        self.maxContextLength = maxContextLength
    }

    package func reset(to tokenIndex: Int) async throws {}
    package func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}

    package func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        // No match: no speculation this step (target advances one token as usual).
        DraftProposal(tokens: lookup.propose(context: confirmedPrefix, count: count))
    }
}
