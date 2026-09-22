// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Token-count clamp and next-token selection shared by the sequential engines' iterators.
/// Each engine keeps its own `next()` control flow; only these identical pure-value helpers
/// live here.
enum SequentialIterator {
    /// The generation-length cap for an iterator.
    ///
    /// A forced continuation replays exactly its own tokens; otherwise the requested budget
    /// (`nil` meaning "unbounded") is clamped to the context left after the prompt.
    static func clampMaxTokens(
        requested: Int?,
        forcedCount: Int?,
        inputCount: Int,
        maxContextLength: Int
    ) -> Int {
        if let forcedCount {
            return forcedCount
        }
        return min(requested ?? Int.max, max(0, maxContextLength - inputCount))
    }

    /// Select the next token: the forced-continuation token when replaying, otherwise the
    /// sampler's choice.
    ///
    /// `logits` is taken by value; the sampler mutates a copy-on-write copy so the caller's
    /// buffer (which it may also return to the consumer) is left untouched.
    static func nextToken(
        fromLogits logits: [LogitsScalarType],
        forced: [Int32]?,
        step: Int,
        sampling: SamplingConfiguration,
        tokenHistory: ArraySlice<Int32>
    ) -> Int32 {
        if let forced {
            return forced[step]
        }
        var mutableLogits = logits
        return sampling.fallbackSampler(from: &mutableLogits, tokenHistory: tokenHistory, step: step)
    }
}
