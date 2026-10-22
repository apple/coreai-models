// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Testing

@testable import CoreAILanguageModels

/// Model-free checks of the batched (N-row) input fills against hand-built tensors. The ragged cases
/// give each row its own decode cursor.
@Suite("Batched input builder")
struct BatchedInputBuilderTests {
    private let s = causalMaskSentinel

    @Test("token ids are row-major flattened")
    func tokenIDs() {
        #expect(BatchedInputBuilder.tokenIDs(rows: [[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6])
    }

    @Test("uniform position ids count 0..<keyLen per row")
    func uniformPositions() {
        // N=2 decode at startPos 1: keyLen is 2, so each row is [0, 1].
        #expect(BatchedInputBuilder.positionIDs(rowCount: 2, startPos: 1, queryLen: 1) == [0, 1, 0, 1])
        // N=1 prefill of 3 tokens from startPos 0.
        #expect(BatchedInputBuilder.positionIDs(rowCount: 1, startPos: 0, queryLen: 3) == [0, 1, 2])
    }

    @Test("uniform causal mask matches the hand-built reference tensor for N=1 and N=2")
    func uniformMask() {
        // Decode at startPos 1 (keyLen 2): the query attends both keys.
        #expect(BatchedInputBuilder.additiveCausalMask(rowCount: 1, startPos: 1, queryLen: 1) == [0, 0])
        // Prefill of 2 tokens from startPos 0: [[0, sentinel], [0, 0]].
        #expect(BatchedInputBuilder.additiveCausalMask(rowCount: 1, startPos: 0, queryLen: 2) == [0, s, 0, 0])
        // With N=2, each row repeats the N=1 block.
        #expect(
            BatchedInputBuilder.additiveCausalMask(rowCount: 2, startPos: 0, queryLen: 2)
                == [0, s, 0, 0, 0, s, 0, 0])
    }

    // MARK: - Ragged decode (per-row cursors)

    @Test("ragged decode positions: real slots keep their index, the write slot gets the cursor")
    func raggedPositions() {
        // Cursors [2, 5] give keyLen 6 with the write slot at 5.
        // Row 0: real slots 0-1, gap 2-4 pinned to 0, write slot holds position 2.
        // Row 1: real slots 0-4, write slot holds position 5.
        #expect(
            BatchedInputBuilder.raggedDecodePositionIDs(cursors: [2, 5])
                == [0, 1, 0, 0, 0, 2] + [0, 1, 2, 3, 4, 5])
    }

    @Test("ragged decode mask: attend real keys and the shared write slot, mask the gap")
    func raggedMask() {
        // Cursors [2, 5]: row 0 masks the gap at slots 2-4, and row 1 gets the full causal mask.
        #expect(
            BatchedInputBuilder.raggedDecodeMask(cursors: [2, 5])
                == [0, 0, s, s, s, 0] + [0, 0, 0, 0, 0, 0])
    }

    @Test("each ragged row matches a single-row decode at its cursor (positions and mask)")
    func raggedMatchesPerRowSingleBatch() {
        let cursors = [1, 4, 7]
        let maxCursor = cursors.max()!
        let keyLen = maxCursor + 1
        let pos = BatchedInputBuilder.raggedDecodePositionIDs(cursors: cursors)
        let mask = BatchedInputBuilder.raggedDecodeMask(cursors: cursors)
        for (b, c) in cursors.enumerated() {
            // The new token sits at slot maxCursor instead of c, so compare the positions of the
            // attended keys, which must be 0...c as in a single-row decode.
            var attendedPositions: [Int32] = []
            for key in 0..<keyLen where mask[b * keyLen + key] == 0 {
                attendedPositions.append(pos[b * keyLen + key])
            }
            #expect(
                attendedPositions == (0...Int32(c)).map { $0 },
                "row \(b) (cursor \(c)) must attend exactly positions 0...\(c)")
        }
    }

    @Test("equal cursors reproduce the uniform decode tensors")
    func raggedEqualCursorsMatchUniform() {
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
