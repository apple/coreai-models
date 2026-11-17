// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Where prefill stops to snapshot a hybrid's recurrent state for the prefix cache.
///
/// A hybrid can only resume at a snapshot, so a prompt that shares a prefix with an earlier one reuses up
/// to the deepest snapshot inside what they share. Snapshots go at a cadence that is a function of depth
/// alone, so two prompts with the same start snapshot at the same places: every chunk at first, so a
/// short shared prefix still has one, then every ``cadenceChunks`` chunks, so a long prompt does not
/// stop for one at every chunk. The prompt's last boundary gets one too, so the same prompt again
/// restores all but its tail.
///
/// ```
/// chunk   1  2  3  4  5  6  7  8  9 10 11 12 ...
///         ●  ●  ●  ●  ·  ·  ·  ●  ·  ·  ·  ●
/// ```
enum PrefixSnapshotPlan {
    /// Chunks from the start that get a snapshot each.
    static let denseChunks = 4

    /// Past ``denseChunks``, chunks between snapshots.
    static let cadenceChunks = 4

    /// Whether a snapshot belongs at the end of chunk `depth`, one-based.
    static func belongs(atDepth depth: Int) -> Bool {
        depth > 0 && (depth <= denseChunks || depth % cadenceChunks == 0)
    }

    /// The chunk boundaries to snapshot at for a prompt of `count` tokens starting at `start`, in order:
    /// every one past `start` that ``belongs(atDepth:)``, up to the last before the token prefill holds
    /// back, and that last one.
    static func boundaries(start: Int, promptTokens count: Int, chunkTokens: Int) -> [Int] {
        guard count > 1 else { return [] }
        let first = start / chunkTokens + 1
        let last = (start + count - 1) / chunkTokens
        guard first <= last else { return [] }
        return (first...last).filter { belongs(atDepth: $0) || $0 == last }.map { $0 * chunkTokens }
    }

    /// `plan`, a prefill's step sizes from position `start`, with every step that crosses one of
    /// `targets` cut there, so a step ends exactly at each.
    static func split(_ plan: [Int], from start: Int, at targets: [Int]) -> [Int] {
        var out: [Int] = []
        var position = start
        for size in plan {
            var cut = position
            for target in targets.sorted() where cut < target && target < position + size {
                out.append(target - cut)
                cut = target
            }
            out.append(position + size - cut)
            position += size
        }
        return out
    }
}
