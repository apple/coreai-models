// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Testing

@testable import CoreAILanguageModels

/// Pure, model-free fills for the batched (N-row) dynamic-engine inputs. The uniform builders must
/// reproduce the hand-built reference tensors exactly (byte-identical); the ragged (per-row-cursor)
/// decode builders generalize that to divergent cursors (per-row cursors).
@Suite("Batched input builder")
struct BatchedInputBuilderTests {
    private let s = causalMaskSentinel

    @Test("token ids are row-major flattened")
    func tokenIDs() {
        #expect(BatchedInputBuilder.tokenIDs(rows: [[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6])
    }

    @Test("uniform position ids count 0..<keyLen per row")
    func uniformPositions() {
        // decode: N=2, startPos=1, q=1 -> keyLen=2, each row [0,1]
        #expect(BatchedInputBuilder.positionIDs(rowCount: 2, startPos: 1, queryLen: 1) == [0, 1, 0, 1])
        // prefill: N=1, startPos=0, q=3 -> [0,1,2]
        #expect(BatchedInputBuilder.positionIDs(rowCount: 1, startPos: 0, queryLen: 3) == [0, 1, 2])
    }

    @Test("uniform causal mask matches the hand-built reference tensor for N=1,2")
    func uniformMask() {
        // decode: startPos=1, q=1, keyLen=2 -> query 0 attends keys 0,1
        #expect(BatchedInputBuilder.additiveCausalMask(rowCount: 1, startPos: 1, queryLen: 1) == [0, 0])
        // prefill: N=1, startPos=0, q=2, keyLen=2 -> [[0, sentinel],[0,0]]
        #expect(BatchedInputBuilder.additiveCausalMask(rowCount: 1, startPos: 0, queryLen: 2) == [0, s, 0, 0])
        // N=2 is the N=1 block repeated per row
        #expect(
            BatchedInputBuilder.additiveCausalMask(rowCount: 2, startPos: 0, queryLen: 2)
                == [0, s, 0, 0, 0, s, 0, 0])
    }

    // MARK: - Ragged decode (per-row cursors)

    @Test("ragged decode positions: real slots number themselves, new-token slot carries the cursor")
    func raggedPositions() {
        // cursors [2, 5] -> maxCursor 5, keyLen 6.
        // row0 (c=2): slots 0,1 -> 0,1 ; gap 2,3,4 -> 0 ; slot5 (new token) -> 2
        // row1 (c=5): slots 0..4 -> 0..4 ; slot5 (new token) -> 5
        #expect(
            BatchedInputBuilder.raggedDecodePositionIDs(cursors: [2, 5])
                == [0, 1, 0, 0, 0, 2, /**/ 0, 1, 2, 3, 4, 5])
    }

    @Test("ragged decode mask: attend real keys + the shared write slot, mask the gap")
    func raggedMask() {
        // cursors [2, 5] -> maxCursor 5, keyLen 6.
        // row0 (c=2): attend keys 0,1 and slot5; gap 2,3,4 masked
        // row1 (c=5): attend keys 0..4 and slot5 (== full causal)
        #expect(
            BatchedInputBuilder.raggedDecodeMask(cursors: [2, 5])
                == [0, 0, s, s, s, 0, /**/ 0, 0, 0, 0, 0, 0])
    }

    @Test("each ragged row equals that row's own B=1 decode (positions + mask)")
    func raggedMatchesPerRowSingleBatch() {
        let cursors = [1, 4, 7]
        let maxCursor = cursors.max()!
        let keyLen = maxCursor + 1
        let pos = BatchedInputBuilder.raggedDecodePositionIDs(cursors: cursors)
        let mask = BatchedInputBuilder.raggedDecodeMask(cursors: cursors)
        for (b, c) in cursors.enumerated() {
            // A B=1 decode at cursor c attends keys 0...c with positions 0...c. In the ragged row the
            // new token sits at slot maxCursor (not c), so compare the *attended* key set + their
            // positions, which must match the standalone decode exactly.
            var attendedPositions: [Int32] = []
            for key in 0..<keyLen where mask[b * keyLen + key] == 0 {
                attendedPositions.append(pos[b * keyLen + key])
            }
            #expect(
                attendedPositions == (0...Int32(c)).map { $0 },
                "row \(b) (cursor \(c)) must attend exactly positions 0...\(c)")
        }
    }

    @Test("equal cursors reproduce the uniform decode tensors byte-for-byte")
    func raggedEqualCursorsMatchUniform() {
        // At equal length the ragged scheme is exactly today's lockstep (strict generalization).
        let c = 3
        let rows = 4
        #expect(
            BatchedInputBuilder.raggedDecodePositionIDs(cursors: Array(repeating: c, count: rows))
                == BatchedInputBuilder.positionIDs(rowCount: rows, startPos: c, queryLen: 1))
        #expect(
            BatchedInputBuilder.raggedDecodeMask(cursors: Array(repeating: c, count: rows))
                == BatchedInputBuilder.additiveCausalMask(rowCount: rows, startPos: c, queryLen: 1))
    }
}
