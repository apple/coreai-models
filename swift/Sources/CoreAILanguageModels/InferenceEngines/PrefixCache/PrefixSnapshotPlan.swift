// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Where prefill stops to snapshot a hybrid's recurrent state for the prefix cache.
enum PrefixSnapshotPlan {
    /// The chunk boundary to snapshot at for a prompt of `count` tokens starting at `start`: the last one
    /// before the token prefill holds back, or nil when that is not past `start`.
    static func boundary(start: Int, promptTokens count: Int, chunkTokens: Int) -> Int? {
        guard count > 1 else { return nil }
        let boundary = (start + count - 1) / chunkTokens * chunkTokens
        return boundary > start ? boundary : nil
    }

    /// `plan`, a prefill's step sizes from position `start`, with the step that crosses `target` cut in
    /// two there, so one ends exactly at it.
    static func split(_ plan: [Int], from start: Int, at target: Int) -> [Int] {
        var out: [Int] = []
        var position = start
        for size in plan {
            if position < target, target < position + size {
                out += [target - position, position + size - target]
            } else {
                out.append(size)
            }
            position += size
        }
        return out
    }
}
