// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// 8-connected component labelling and the mask cleanup built on it.
///
/// Port of `fill_holes_in_mask_scores` and `_get_connected_components_with_padding`.
enum ConnectedComponents {
    /// Per-pixel component area for the set pixels of `mask`. Unset pixels get 0.
    static func areas(of mask: [Bool], width: Int, height: Int) -> [Int32] {
        let count = width * height
        precondition(mask.count >= count, "ConnectedComponents.areas: mask is too small")
        var parent = [Int32](repeating: -1, count: count)

        func find(_ start: Int32) -> Int32 {
            var node = start
            while parent[Int(node)] != node {
                parent[Int(node)] = parent[Int(parent[Int(node)])]
                node = parent[Int(node)]
            }
            return node
        }
        func union(_ a: Int32, _ b: Int32) {
            let rootA = find(a)
            let rootB = find(b)
            if rootA == rootB { return }
            // Attach the larger index under the smaller so the labelling is deterministic.
            if rootA < rootB { parent[Int(rootB)] = rootA } else { parent[Int(rootA)] = rootB }
        }

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard mask[index] else { continue }
                parent[index] = Int32(index)
                // The already-visited half of the neighbors: left plus the three
                // above. The other four are covered when their own row runs.
                if x > 0, mask[index - 1] { union(Int32(index), Int32(index - 1)) }
                if y > 0 {
                    let above = index - width
                    if mask[above] { union(Int32(index), Int32(above)) }
                    if x > 0, mask[above - 1] { union(Int32(index), Int32(above - 1)) }
                    if x + 1 < width, mask[above + 1] { union(Int32(index), Int32(above + 1)) }
                }
            }
        }

        // `parent` doubles as the result. Safe because `union` always attaches the larger
        // index under the smaller, so no entry is overwritten before it is read.
        var componentArea = [Int32](repeating: 0, count: count)
        for index in 0..<count where parent[index] >= 0 {
            let root = find(Int32(index))
            parent[index] = root
            componentArea[Int(root)] += 1
        }
        for index in 0..<count {
            let root = parent[index]
            parent[index] = root >= 0 ? componentArea[Int(root)] : 0
        }
        return parent
    }

    /// Fill small background holes and remove small foreground specks, in place.
    ///
    /// Port of `fill_holes_in_mask_scores(mask, max_area, fill_holes=True,
    /// remove_sprinkles=True)`. The sentinels `0.1` and `-0.1` are upstream's: the mask stays
    /// a logit field, so a filled hole reads as weakly positive.
    static func fillHoles(_ logits: inout [Float], width: Int, height: Int, maxArea: Int) {
        guard maxArea > 0 else { return }
        let count = width * height
        precondition(logits.count >= count, "fillHoles: logit buffer is too small")

        // Background: components of `logits <= 0` up to `maxArea` become weakly positive.
        // Clamped so a sentinel such as `Int.max` widens the limit rather than trapping.
        let maxArea32 = Int32(clamping: maxArea)
        var background = [Bool](repeating: false, count: count)
        for index in 0..<count { background[index] = logits[index] <= 0 }
        let backgroundAreas = areas(of: background, width: width, height: height)
        for index in 0..<count where background[index] && backgroundAreas[index] <= maxArea32 {
            logits[index] = 0.1
        }

        // Halving the mask's own area is what lets a genuinely tiny object survive.
        var foreground = [Bool](repeating: false, count: count)
        var foregroundArea = 0
        for index in 0..<count where logits[index] > 0 {
            foreground[index] = true
            foregroundArea += 1
        }
        let threshold = min(maxArea32, Int32(foregroundArea / 2))
        guard threshold > 0 else { return }
        let foregroundAreas = areas(of: foreground, width: width, height: height)
        for index in 0..<count where foreground[index] && foregroundAreas[index] <= threshold {
            logits[index] = -0.1
        }
    }
}
