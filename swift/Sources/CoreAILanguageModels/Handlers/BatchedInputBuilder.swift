// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// Model-free input builders for an N-row batched step. Each returns a flat row-major buffer that
/// `decodeBatch` copies into its tensor with `fillNDArray`.
///
/// The uniform builders take one `startPos` for all rows, and each of their rows matches the
/// single-row inputs exactly. The `raggedDecode*` builders take a cursor per row and match the
/// uniform builders when all cursors are equal.
enum BatchedInputBuilder {
    /// Flat `[N, queryLen]` token ids. Every row must have `queryLen` tokens.
    static func tokenIDs(rows: [[Int32]]) -> [Int32] {
        rows.flatMap { $0 }
    }

    /// Flat `[N, keyLen]` position ids with `keyLen = startPos + queryLen`. Each row is `0..<keyLen`,
    /// covering the whole key axis (cached prefix plus new tokens).
    static func positionIDs(rowCount: Int, startPos: Int, queryLen: Int) -> [Int32] {
        let keyLen = startPos + queryLen
        var out = [Int32](repeating: 0, count: rowCount * keyLen)
        for b in 0..<rowCount {
            for c in 0..<keyLen {
                out[b * keyLen + c] = Int32(c)
            }
        }
        return out
    }

    /// Flat `[N, 1, queryLen, keyLen]` additive causal mask with `keyLen = startPos + queryLen`.
    /// Query `i` gets 0 for keys `0...startPos + i` and `sentinel` after that.
    static func additiveCausalMask(
        rowCount: Int, startPos: Int, queryLen: Int,
        sentinel: Float16 = causalMaskSentinel
    ) -> [Float16] {
        let keyLen = startPos + queryLen
        var out = [Float16](repeating: sentinel, count: rowCount * queryLen * keyLen)
        for b in 0..<rowCount {
            for query in 0..<queryLen {
                let attendUpTo = startPos + query  // absolute position of this query
                let base = (b * queryLen + query) * keyLen
                for key in 0...attendUpTo { out[base + key] = 0 }
            }
        }
        return out
    }

    // MARK: - Ragged decode (per-row cursor)

    /// Position ids for a ragged decode step (`queryLen = 1`, one cursor per row). Flat `[N, keyLen]`
    /// with `keyLen = max(cursors) + 1`. Every row writes its new token at slot `max(cursors)`, and
    /// that slot holds the row's own cursor, which the model reads as the query's RoPE position.
    ///
    /// For cursors `[2, 5]`:
    /// ```
    /// row 0: [0, 1, 0, 0, 0, 2]   // slots 0..<2 cached, gap 2..<5 set to 0, slot 5 holds cursor 2
    /// row 1: [0, 1, 2, 3, 4, 5]
    /// ```
    static func raggedDecodePositionIDs(cursors: [Int]) -> [Int32] {
        let maxCursor = cursors.max() ?? 0
        let keyLen = maxCursor + 1
        var out = [Int32](repeating: 0, count: cursors.count * keyLen)
        for (b, cursor) in cursors.enumerated() {
            let base = b * keyLen
            for key in 0..<cursor { out[base + key] = Int32(key) }
            out[base + maxCursor] = Int32(cursor)
        }
        return out
    }

    /// Additive mask for a ragged decode step, flat `[N, 1, 1, keyLen]` with
    /// `keyLen = max(cursors) + 1`. Each row attends its cached slots and the write slot
    /// `max(cursors)`, and masks the gap between them.
    ///
    /// For cursors `[2, 5]` (`s` = `sentinel`):
    /// ```
    /// row 0: [0, 0, s, s, s, 0]
    /// row 1: [0, 0, 0, 0, 0, 0]
    /// ```
    static func raggedDecodeMask(
        cursors: [Int], sentinel: Float16 = causalMaskSentinel
    ) -> [Float16] {
        let maxCursor = cursors.max() ?? 0
        let keyLen = maxCursor + 1
        var out = [Float16](repeating: sentinel, count: cursors.count * keyLen)
        for (b, cursor) in cursors.enumerated() {
            let base = b * keyLen
            for key in 0..<cursor { out[base + key] = 0 }
            out[base + maxCursor] = 0
        }
        return out
    }
}
