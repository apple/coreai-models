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
public final class PromptLookupDrafter: DrafterEngine, @unchecked Sendable {
    private let maxNgram: Int
    private let minNgram: Int
    public let maxContextLength: Int

    /// - Parameters:
    ///   - maxNgram: longest suffix to match (prefer longer, more-specific matches).
    ///   - minNgram: shortest suffix to accept a match on.
    public init(maxNgram: Int = 3, minNgram: Int = 1, maxContextLength: Int = 131072) {
        self.maxNgram = max(1, maxNgram)
        self.minNgram = max(1, minNgram)
        self.maxContextLength = maxContextLength
    }

    public func reset(to tokenIndex: Int) async throws {}
    public func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}

    public func draftAll(confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy) async throws
        -> DraftProposal
    {
        let n = confirmedPrefix.count
        guard n >= 2, count > 0 else { return DraftProposal(tokens: []) }

        // Try the longest suffix first, then shorter, for the most specific match.
        for ng in stride(from: min(maxNgram, n - 1), through: minNgram, by: -1) {
            let suffix = Array(confirmedPrefix[(n - ng)..<n])
            // Search for the most recent earlier occurrence of `suffix` (excluding the
            // final position that produced it).
            var i = n - ng - 1
            while i >= 0 {
                if match(confirmedPrefix, at: i, suffix) {
                    // Propose the tokens that followed this earlier occurrence.
                    let start = i + ng
                    let end = min(start + count, n)
                    if start < end {
                        return DraftProposal(tokens: Array(confirmedPrefix[start..<end]))
                    }
                    break
                }
                i -= 1
            }
        }
        // No match: no speculation this step (target advances one token as usual).
        return DraftProposal(tokens: [])
    }

    private func match(_ tokens: [Int32], at index: Int, _ suffix: [Int32]) -> Bool {
        for j in 0..<suffix.count where tokens[index + j] != suffix[j] { return false }
        return true
    }
}
