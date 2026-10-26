// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Defaults and limits shared by the speculative decoding types, so each value is stated once.
package enum SpeculativeDefaults {
    /// Draft tokens proposed per step.
    package static let numDraftTokens = 4

    /// Most draft tokens per step. A verify pass scores a block in one forward, and the widest
    /// tile an engine runs is far narrower than this (the Neural Engine's decode tile is 8), so a
    /// larger count only spends draft work on tokens the engine cannot verify together.
    package static let maxDraftTokens = 256

    /// Context length assumed for a model-free drafter, which holds no cache of its own; the
    /// decoder takes the smaller of this and the target's.
    package static let maxContextLength = 131_072

    /// Prompt lookup: longest and shortest suffix matched, and most tokens proposed per block.
    package static let lookupMaxNgram = 4
    package static let lookupMinNgram = 3
    package static let lookupMaxTokens = 7

    /// Adaptive draft count: tokens added after a block whose drafts were all accepted, and
    /// removed after a rejection. A heuristic: growing faster than shrinking probes for headroom
    /// quickly, and an over-long draft costs only draft work. The values are not tuned.
    package static let adaptiveGrowth = 2
    package static let adaptiveShrink = 1

    /// Smallest temperature a sampling policy uses; below it the distribution is the greedy limit.
    package static let minSamplingTemperature = 1e-5
}
