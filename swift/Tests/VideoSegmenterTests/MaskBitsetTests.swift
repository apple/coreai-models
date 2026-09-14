// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Testing

@testable import CoreAIShared

@Suite("MaskBitset")
struct MaskBitsetTests {
    /// Builds a mask from an ASCII picture, so the expectations below read as pictures too.
    private func mask(_ rows: [String]) -> MaskBitset {
        var bitset = MaskBitset(width: rows[0].count, height: rows.count)
        for (y, row) in rows.enumerated() {
            for (x, character) in row.enumerated() where character == "#" {
                bitset[x, y] = true
            }
        }
        return bitset
    }

    @Test("Thresholding is strictly greater, matching HF's `mask > 0`")
    func strictThreshold() {
        // Every SAM 3 call site binarizes with `> 0`. Using `>=` would flip every exactly
        // zero pixel to foreground, which on a padded logit field is most of the frame.
        let values: [Float] = [-1, 0, 0.0001, 1]
        let bitset = MaskBitset(thresholding: values, width: 4, height: 1)
        #expect(bitset[0, 0] == false)
        #expect(bitset[1, 0] == false)
        #expect(bitset[2, 0] == true)
        #expect(bitset[3, 0] == true)
        #expect(bitset.area == 2)
    }

    @Test("IoU of two empty masks is 0, not 1")
    func emptyIoU() {
        // `mask_iou` clamps the union to a minimum of 1 rather than special-casing empty,
        // so two empty masks score 0. Association depends on that: it must treat two
        // occluded tracks as unrelated, not as a perfect match.
        let empty = MaskBitset(width: 8, height: 8)
        #expect(empty.iou(empty) == 0)
    }

    @Test("IoU counts intersection over union")
    func iou() {
        let a = mask([
            "##..",
            "##..",
        ])
        let b = mask([
            ".##.",
            ".##.",
        ])
        // Intersection is the shared column pair (2 pixels), union is 6.
        #expect(a.iou(b) == Float(2) / Float(6))
        #expect(a.iou(a) == 1)
    }

    @Test("IoU survives a mask wider than one 64-bit word")
    func iouAcrossWords() {
        // 100x3 is 300 bits, so the set pixels straddle word boundaries, the case a
        // naive per-word loop gets wrong.
        var a = MaskBitset(width: 100, height: 3)
        var b = MaskBitset(width: 100, height: 3)
        for x in 0..<100 {
            a[x, 1] = true
            if x >= 50 { b[x, 1] = true }
        }
        #expect(a.area == 100)
        #expect(b.area == 50)
        #expect(a.iou(b) == 0.5)
    }

    @Test("boundingBox reports inclusive extremes, like masks_to_boxes")
    func boundingBox() {
        let bitset = mask([
            "....",
            ".##.",
            ".##.",
            "....",
        ])
        // `torchvision.ops.masks_to_boxes` returns [x0, y0, x1, y1] of the extreme set
        // pixels, so a 2x2 block spans 1 unit, not 2.
        #expect(bitset.boundingBox == CGRect(x: 1, y: 1, width: 1, height: 1))
    }

    @Test("boundingBox of one pixel is a zero-sized rect at that pixel")
    func singlePixelBox() {
        var bitset = MaskBitset(width: 10, height: 10)
        bitset[7, 3] = true
        #expect(bitset.boundingBox == CGRect(x: 7, y: 3, width: 0, height: 0))
    }

    @Test("boundingBox of an empty mask is zero")
    func emptyBox() {
        #expect(MaskBitset(width: 10, height: 10).boundingBox == .zero)
    }

    @Test("forEachSetIndex visits row-major indices in order and stops at the logical end")
    func setIndices() {
        // 5x2 is 10 bits inside one 64-bit word, so the trailing 54 bits are padding the
        // walk must not report.
        var bitset = MaskBitset(width: 5, height: 2)
        bitset[0, 0] = true
        bitset[4, 1] = true
        var visited: [Int] = []
        bitset.forEachSetIndex { visited.append($0) }
        #expect(visited == [0, 9])
    }

    @Test("packbits round-trips a mask that is not a multiple of 8 pixels")
    func packedRoundTrip() {
        // `numpy.packbits` is MSB-first within each byte and pads the last one. 5x3 = 15
        // bits, so the second byte carries a partial value.
        let original = mask([
            "#..#.",
            ".#.#.",
            "..#..",
        ])
        var packed = [UInt8](repeating: 0, count: 2)
        original.forEachSetIndex { index in
            packed[index >> 3] |= 0x80 >> UInt8(index & 7)
        }
        let restored = MaskBitset(packedBits: packed, width: 5, height: 3)
        #expect(restored == original)
        #expect(restored.area == original.area)
    }
}
