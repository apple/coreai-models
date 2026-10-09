// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// Pure, model-free builders for the batched (N-row) dynamic-engine inputs.
///
/// Each returns a flat, row-major buffer that `decodeBatch` copies into its tensors via
/// `fillNDArray` (which iterates logical row-major order, so these buffers map 1:1). Factored out
/// of the engine so the fill math is unit-testable without a compiled model or an `NDArrayDescriptor`
/// (which has no public initializer).
///
/// These reproduce the single-row tensors exactly, so on-device output stays byte-identical.
/// Ragged (per-row-cursor) decode uses the dedicated `raggedDecode*` builders below.
enum BatchedInputBuilder {
    /// Flat `[N*queryLen]` row-major token ids.
    static func tokenIDs(rows: [[Int32]]) -> [Int32] {
        rows.flatMap { $0 }
    }

    /// Flat `[N*keyLen]` row-major position ids, `keyLen = startPos + queryLen` (the fetched
    /// KV-cache prefix length the attn-mask key axis is tied to). Each row numbers its positions
    /// from 0.
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

    /// Flat `[N*queryLen*keyLen]` additive mask — the data of the `[N,1,q,k]` tensor. `0` means
    /// attend, `sentinel` (a large negative that softmaxes to ~0) means masked. Row `b`'s query
    /// `query` attends to keys in `[0, startPos+query]`.
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

    /// Flat `[N*keyLen]` position ids for a **ragged decode step** (`q = 1`, each row at its own
    /// cursor). `keyLen = max(cursors) + 1`; every row's single new token is shared-written at the
    /// longest row's slot `max(cursor)` and attends there. Row `b`'s key positions: real cached slots
    /// `[0, cursor[b])` number themselves; the shared write slot `max(cursor)` carries the new token's
    /// real position `cursor[b]` (the last entry → the query's RoPE position); the masked gap
    /// `[cursor[b], max(cursor))` is pinned to 0 (don't-care). Equal cursors reproduce
    /// `positionIDs(startPos: cursor, queryLen: 1)` exactly.
    static func raggedDecodePositionIDs(cursors: [Int]) -> [Int32] {
        let maxCursor = cursors.max() ?? 0
        let keyLen = maxCursor + 1
        var out = [Int32](repeating: 0, count: cursors.count * keyLen)
        for (b, cursor) in cursors.enumerated() {
            let base = b * keyLen
            for key in 0..<cursor { out[base + key] = Int32(key) }  // real cached slots
            out[base + maxCursor] = Int32(cursor)  // new-token slot carries its real position
        }
        return out
    }

    /// Flat `[N*keyLen]` additive mask for a **ragged decode step** (`q = 1`), the data of the
    /// `[N,1,1,keyLen]` tensor. Row `b`'s single query attends to its real cached keys `[0, cursor[b])`
    /// and its new token at the shared write slot `max(cursor)`; the gap `[cursor[b], max(cursor))`
    /// (other rows' slots) is masked to `sentinel`. Equal cursors reproduce
    /// `additiveCausalMask(startPos: cursor, queryLen: 1)` (all-attend) exactly.
    static func raggedDecodeMask(
        cursors: [Int], sentinel: Float16 = causalMaskSentinel
    ) -> [Float16] {
        let maxCursor = cursors.max() ?? 0
        let keyLen = maxCursor + 1
        var out = [Float16](repeating: sentinel, count: cursors.count * keyLen)
        for (b, cursor) in cursors.enumerated() {
            let base = b * keyLen
            for key in 0..<cursor { out[base + key] = 0 }  // real cached keys
            out[base + maxCursor] = 0  // the new token at the shared write slot
        }
        return out
    }
}
