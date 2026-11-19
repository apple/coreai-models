// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Prompt-lookup proposals: finds the most recent earlier occurrence of the context's last
/// n tokens and proposes the tokens that followed it. Needs no model work, and is accurate
/// when the output copies spans from the context (rewrites, extraction, document QA).
package struct PromptLookup: Sendable {
    /// Longest and shortest suffix lengths matched; longer matches are tried first.
    package let maxNgram: Int
    package let minNgram: Int
    /// Most tokens proposed per block.
    package let maxTokens: Int

    package init(
        maxNgram: Int = SpeculativeDefaults.lookupMaxNgram,
        minNgram: Int = SpeculativeDefaults.lookupMinNgram,
        maxTokens: Int = SpeculativeDefaults.lookupMaxTokens
    ) {
        precondition(minNgram >= 1, "PromptLookup minNgram \(minNgram) must be at least 1")
        precondition(
            maxNgram >= minNgram, "PromptLookup maxNgram \(maxNgram) must be at least minNgram \(minNgram)")
        precondition(maxTokens >= 1, "PromptLookup maxTokens \(maxTokens) must be at least 1")
        self.minNgram = minNgram
        self.maxNgram = maxNgram
        self.maxTokens = maxTokens
    }

    /// Up to `count` tokens that followed the latest earlier occurrence of the context's
    /// suffix, or an empty array when no suffix of at least `minNgram` tokens recurs.
    package func propose(context: [Int32], count: Int) -> [Int32] {
        guard count > 0 else { return [] }
        let end = context.count
        for n in stride(from: min(maxNgram, end - 1), through: minNgram, by: -1) {
            // Candidate starts i such that context[i..<i+n] == suffix and a token follows.
            let suffix = context[(end - n)...]
            for i in stride(from: end - n - 1, through: 0, by: -1) where context[i..<(i + n)] == suffix {
                let from = i + n
                return Array(context[from..<min(from + count, end)])
            }
        }
        return []
    }
}
