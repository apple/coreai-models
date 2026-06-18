// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Tracks processed token history for implicit prefix caching.
///
/// Used by inference engines to resolve full-context input into only the new tokens
/// that need processing, enabling automatic KV cache reuse across multi-turn conversations
/// without caller intervention.
///
/// ## How it works
///
/// When a caller passes full context (prompt + previous output + new suffix) to
/// `generate(with:)`, the engine calls `resolve(input:)` to find the longest common
/// prefix between the input and the cached history. Only tokens beyond that prefix
/// need to be processed; the KV cache already contains the prefix's representations.
///
/// If the input diverges from history (e.g., "Alpha beta" -> "Alpha romeo"), the engine
/// rewinds its KV cache to the divergence point and reprocesses from there.
struct TokenHistory: Sendable {
    private(set) var tokens: [Int32] = []

    /// Compare input against cached history.
    ///
    /// Uses `memcmp` for fast-path comparison when the prefix matches entirely,
    /// falling back to element-wise scan only on mismatch.
    ///
    /// - Parameter input: The full token sequence from the caller.
    /// - Returns:
    ///   - `commonPrefix`: Number of leading tokens that match (safe to keep in KV cache).
    ///   - `newTokens`: The slice of input beyond the common prefix that needs processing.
    func resolve(input: [Int32]) -> (commonPrefix: Int, newTokens: ArraySlice<Int32>) {
        let limit = min(input.count, tokens.count)
        var commonPrefix = limit
        if limit > 0 {
            input.withUnsafeBufferPointer { inputBuf in
                tokens.withUnsafeBufferPointer { cachedBuf in
                    let bytes = limit * MemoryLayout<Int32>.stride
                    if memcmp(inputBuf.baseAddress!, cachedBuf.baseAddress!, bytes) != 0 {
                        // Slow path: find exact divergence point
                        commonPrefix = 0
                        for i in 0..<limit {
                            if inputBuf[i] != cachedBuf[i] { break }
                            commonPrefix = i + 1
                        }
                    }
                }
            }
        }

        return (commonPrefix, input[commonPrefix...])
    }

    /// Record that tokens were successfully processed.
    mutating func append(contentsOf newTokens: ArraySlice<Int32>) {
        tokens.append(contentsOf: newTokens)
    }

    /// Append a single generated token.
    mutating func append(_ token: Int32) {
        tokens.append(token)
    }

    var count: Int { tokens.count }

    /// Truncate history to a given position (for partial reset).
    ///
    /// If `position` exceeds the current history count, this is a no-op (the history
    /// simply doesn't have entries beyond its count to remove).
    mutating func truncate(to position: Int) {
        precondition(position >= 0)
        guard position < tokens.count else { return }
        tokens.removeSubrange(position...)
    }

    /// Full reset.
    mutating func clear() {
        tokens.removeAll(keepingCapacity: true)
    }
}
