// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Where a recurrent model's prefill stops, and where it snapshots its state for the prefix cache.
///
/// Where prefill splits a prompt changes a recurrent model's arithmetic a little, which can tip a greedy
/// near-tie. So its prefill stops at the same absolute positions however it runs: at every multiple of
/// ``gridTokens`` and of its step size, counted from the start of the sequence, with the cache off, on a
/// miss and after a restore alike. A restore only ever resumes at a grid point, which every run stops
/// at, so the steps after it are the ones a cold run takes, and the reply does not depend on the cache.
///
/// Snapshots go only at grid points, so the cache deciding to take one, or not, never moves a stop.
enum PrefixSnapshotPlan {
    /// Tokens between grid points: where a recurrent model's prefill always stops, and so where it can
    /// snapshot. Steps no longer than this cost little or nothing.
    static let gridTokens = 1024

    /// The positions prefill must stop at between `start` and `end`, both exclusive, in order: every
    /// multiple of ``gridTokens``, of `stepTokens`, and of `ringCapacity` when there is one, since no
    /// step may cross the end of a sliding ring.
    static func stops(after start: Int, before end: Int, stepTokens: Int, ringCapacity: Int? = nil) -> [Int] {
        guard end > start + 1 else { return [] }
        var out = Set<Int>()
        for unit in [gridTokens, stepTokens, ringCapacity ?? 0] where unit > 0 {
            var position = (start / unit + 1) * unit
            while position < end {
                out.insert(position)
                position += unit
            }
        }
        return out.sorted()
    }

    /// A recurrent model's prefill steps over positions `[start, end)`: from one ``stops(after:before:stepTokens:ringCapacity:)``
    /// to the next. Each is a function of absolute position alone, so a run that starts at a grid point
    /// takes the very steps a run from the start takes past it, and none is longer than `stepTokens`.
    static func steps(from start: Int, to end: Int, stepTokens: Int, ringCapacity: Int? = nil) -> [Int] {
        guard end > start else { return [] }
        var out: [Int] = []
        var position = start
        for stop in stops(after: start, before: end, stepTokens: stepTokens, ringCapacity: ringCapacity) + [end] {
            out.append(stop - position)
            position = stop
        }
        return out
    }

    /// The grid points to snapshot at for a prompt of `count` tokens starting at `start`, in order: every
    /// one past `start` that prefill reaches before the token it holds back.
    static func boundaries(start: Int, promptTokens count: Int) -> [Int] {
        guard count > 1 else { return [] }
        let first = start / gridTokens + 1
        let last = (start + count - 1) / gridTokens
        guard first <= last else { return [] }
        return (first...last).map { $0 * gridTokens }
    }
}
