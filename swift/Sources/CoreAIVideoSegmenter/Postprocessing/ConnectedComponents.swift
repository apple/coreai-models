// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// 8-connected component labelling and the mask cleanup built on it.
///
/// Port of `fill_holes_in_mask_scores` and `_get_connected_components_with_padding` in
/// `modeling_sam3_video.py`. Upstream routes the labelling through an optional
/// `kernels-community/cv-utils` CUDA kernel and, when it is not installed, returns fake
/// component areas of `H*W + 1`, which makes both cleanup steps no-ops without saying so.
/// This implementation is always present, so a bundle run here gets the cleanup the model
/// was tuned with. Set ``VideoSegmentationParameters/fillHoleArea`` to 0 to reproduce a
/// Python run that was missing the kernel.
///
/// 8-connectivity matches SAM 2's `get_connected_components`, which documents it
/// explicitly and is what `cc_2d` implements.
enum ConnectedComponents {
    /// Per-pixel component area for the set pixels of `mask`. Unset pixels get 0.
    ///
    /// Union-find with path halving over a single raster pass, then one pass to resolve roots
    /// and a third to broadcast the root's area. Linear in pixels.
    static func areas(of mask: [Bool], width: Int, height: Int) -> [Int32] {
        let count = width * height
        precondition(mask.count >= count, "ConnectedComponents.areas: mask is too small")
        var parent = [Int32](repeating: -1, count: count)

        func find(_ start: Int32) -> Int32 {
            var node = start
            while parent[Int(node)] != node {
                // Path halving: point each node at its grandparent while climbing. Keeps the
                // trees flat without a second pass.
                parent[Int(node)] = parent[Int(parent[Int(node)])]
                node = parent[Int(node)]
            }
            return node
        }
        func union(_ a: Int32, _ b: Int32) {
            let rootA = find(a)
            let rootB = find(b)
            if rootA == rootB { return }
            // Always attach the larger index under the smaller so roots stay stable and
            // the labelling is deterministic.
            if rootA < rootB { parent[Int(rootB)] = rootA } else { parent[Int(rootA)] = rootB }
        }

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard mask[index] else { continue }
                parent[index] = Int32(index)
                // Only the already-visited half of the 8-neighbourhood: west, and the
                // three above. The other four are covered when their own row runs.
                if x > 0, mask[index - 1] { union(Int32(index), Int32(index - 1)) }
                if y > 0 {
                    let above = index - width
                    if mask[above] { union(Int32(index), Int32(above)) }
                    if x > 0, mask[above - 1] { union(Int32(index), Int32(above - 1)) }
                    if x + 1 < width, mask[above + 1] { union(Int32(index), Int32(above + 1)) }
                }
            }
        }

        var componentArea = [Int32](repeating: 0, count: count)
        var roots = [Int32](repeating: -1, count: count)
        for index in 0..<count where parent[index] >= 0 {
            let root = find(Int32(index))
            roots[index] = root
            componentArea[Int(root)] += 1
        }
        var result = [Int32](repeating: 0, count: count)
        for index in 0..<count where roots[index] >= 0 {
            result[index] = componentArea[Int(roots[index])]
        }
        return result
    }

    /// Fill small background holes and remove small foreground specks, in place.
    ///
    /// Port of `fill_holes_in_mask_scores(mask, max_area, fill_holes=True,
    /// remove_sprinkles=True)`. The two sentinel values (`0.1` and `-0.1`) are upstream's:
    /// the mask stays a logit field, so a filled hole becomes weakly positive rather than
    /// saturated.
    ///
    /// The foreground threshold is `min(maxArea, foregroundArea / 2)` and is recomputed after
    /// hole filling, which is what keeps a genuinely tiny object from deleting itself.
    static func fillHoles(_ logits: inout [Float], width: Int, height: Int, maxArea: Int) {
        guard maxArea > 0 else { return }
        let count = width * height
        precondition(logits.count >= count, "fillHoles: logit buffer is too small")

        // Background: components of `logits <= 0` up to `maxArea` become weakly positive.
        var background = [Bool](repeating: false, count: count)
        for index in 0..<count { background[index] = logits[index] <= 0 }
        let backgroundAreas = areas(of: background, width: width, height: height)
        for index in 0..<count where background[index] && backgroundAreas[index] <= Int32(maxArea) {
            logits[index] = 0.1
        }

        // Foreground: components of `logits > 0` up to the smaller of `maxArea` and half
        // the mask's own area become weakly negative.
        var foreground = [Bool](repeating: false, count: count)
        var foregroundArea = 0
        for index in 0..<count where logits[index] > 0 {
            foreground[index] = true
            foregroundArea += 1
        }
        let threshold = Int32(min(maxArea, foregroundArea / 2))
        guard threshold > 0 else { return }
        let foregroundAreas = areas(of: foreground, width: width, height: height)
        for index in 0..<count where foreground[index] && foregroundAreas[index] <= threshold {
            logits[index] = -0.1
        }
    }
}
