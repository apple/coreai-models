// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Resolves objects that claim the same pixels.
///
/// Three related rules, all ported from `Sam3VideoModel`, applied at different points and
/// on different data:
///
/// 1. ``suppressRecentlyOccluded``, before memory encoding, on tracker logits. When two
///    objects overlap heavily, the one that was occluded more recently is assumed to be
///    the drifting copy and is blanked.
/// 2. ``suppressAreaShrinkage``, also before memory encoding. Applies a pixel-level argmax
///    and then drops any object that lost most of its area to it, on the reasoning that
///    such an object was never really there.
/// 3. ``applyObjectWiseNonOverlap``, at output time, on binary masks and per-object scores
///    rather than logits.
///
/// All three are enforced per prompt group: two prompts may legitimately return overlapping
/// masks for the same pixels, and forcing them to compete would make "person" and "pillow"
/// delete each other.
enum OcclusionSuppressor {
    /// Logit written over a suppressed mask. `sigmoid(-10)` is 4.5e-5, so it reads as
    /// background everywhere downstream without being an out-of-range sentinel.
    static let noObjectLogit: Float = -10

    /// Fraction of its own area an object must retain through the pixel-level argmax.
    static let shrinkThreshold: Float = 0.3

    /// Frame index standing in for "never occluded". Must stay negative: the rule below
    /// requires the *other* object to have been occluded at some point, tested as `> -1`.
    static let neverOccluded = -1

    /// Frame index standing in for "removed by hotstart, always loses". Larger than any
    /// real frame index.
    static let alwaysOccluded = 100_000

    // MARK: - 1. Recent-occlusion suppression

    /// Blank objects that overlap an object occluded less recently than they were.
    ///
    /// Port of `_suppress_overlapping_based_on_recent_occlusion`. Also updates each
    /// object's last-occluded frame, which is the state the next frame's comparison reads:
    /// an object counts as occluded this frame if its mask is empty *or* it was suppressed
    /// here.
    ///
    /// - Parameters:
    ///   - logits: Tracker mask logits per object, mutated in place.
    ///   - masks: The same masks binarized, parallel to `logits`.
    ///   - objectIDs: Registry order, parallel to both.
    @VideoSegmentationActor
    static func suppressRecentlyOccluded(
        logits: inout [[Float]],
        masks: [MaskBitset],
        objectIDs: [Int],
        promptIDs: [Int],
        newlyRemovedObjectIDs: Set<Int>,
        frameIndex: Int,
        reverse: Bool,
        session: VideoInferenceSession,
        parameters: VideoSegmentationParameters
    ) {
        guard !objectIDs.isEmpty else { return }

        let lastOccluded = objectIDs.map { id in
            session.lastOccludedByObjectID[id]
                ?? (newlyRemovedObjectIDs.contains(id) ? alwaysOccluded : neverOccluded)
        }

        var suppress = [Bool](repeating: false, count: objectIDs.count)
        for group in Set(promptIDs).sorted() {
            let members = promptIDs.indices.filter { promptIDs[$0] == group }
            guard members.count > 1 else { continue }
            markSuppressed(
                members: members, masks: masks, lastOccluded: lastOccluded,
                threshold: parameters.suppressOverlappingOcclusionThreshold,
                reverse: reverse, into: &suppress)
        }

        var updated = lastOccluded
        for index in objectIDs.indices where masks[index].isEmpty || suppress[index] {
            updated[index] = frameIndex
        }
        for (index, id) in objectIDs.enumerated() {
            session.lastOccludedByObjectID[id] = updated[index]
        }

        for index in objectIDs.indices where suppress[index] {
            for pixel in logits[index].indices { logits[index][pixel] = noObjectLogit }
        }
    }

    /// The pairwise rule itself, over one prompt group.
    ///
    /// For every pair `i < j` overlapping above `threshold`, the one occluded more recently
    /// loses, but only if the winner was itself occluded at some point (`> neverOccluded`).
    /// Two objects that have both always been visible simply coexist.
    private static func markSuppressed(
        members: [Int],
        masks: [MaskBitset],
        lastOccluded: [Int],
        threshold: Float,
        reverse: Bool,
        into suppress: inout [Bool]
    ) {
        // Tracking backwards inverts the comparison: "more recent" is a lower frame index.
        func losesTo(_ a: Int, _ b: Int) -> Bool { reverse ? a < b : a > b }

        for outer in 0..<members.count {
            for inner in (outer + 1)..<members.count {
                let i = members[outer]
                let j = members[inner]
                guard masks[i].iou(masks[j]) >= threshold else { continue }
                if losesTo(lastOccluded[i], lastOccluded[j]), lastOccluded[j] > neverOccluded {
                    suppress[i] = true
                }
                if losesTo(lastOccluded[j], lastOccluded[i]), lastOccluded[i] > neverOccluded {
                    suppress[j] = true
                }
            }
        }
    }

    // MARK: - 2. Area-shrinkage suppression

    /// Drop objects that lose most of their area to the pixel-level argmax.
    ///
    /// Port of `_suppress_object_pw_area_shrinkage`, run per prompt group. Returns the
    /// original masks with whole objects blanked, not the de-overlapped masks; the argmax only
    /// measures how much of each object was contested.
    static func suppressAreaShrinkage(
        logits: inout [[Float]], promptIDs: [Int]
    ) {
        guard logits.count > 1 else { return }
        for group in Set(promptIDs).sorted() {
            let members = promptIDs.indices.filter { promptIDs[$0] == group }
            guard members.count > 1 else { continue }

            let pixelCount = logits[members[0]].count
            // Winner per pixel, by raw logit. Ties go to the lowest index in the group,
            // matching `torch.argmax`.
            var winner = [Int](repeating: members[0], count: pixelCount)
            var best = [Float](repeating: -.greatestFiniteMagnitude, count: pixelCount)
            for member in members {
                let values = logits[member]
                for pixel in 0..<pixelCount where values[pixel] > best[pixel] {
                    best[pixel] = values[pixel]
                    winner[pixel] = member
                }
            }

            var blank: [Int] = []
            for member in members {
                let values = logits[member]
                var areaBefore = 0
                var areaAfter = 0
                for pixel in 0..<pixelCount where values[pixel] > 0 {
                    areaBefore += 1
                    if winner[pixel] == member { areaAfter += 1 }
                }
                let ratio = Float(areaAfter) / Float(max(areaBefore, 1))
                if ratio < shrinkThreshold { blank.append(member) }
            }
            for member in blank {
                for pixel in logits[member].indices {
                    logits[member][pixel] = min(logits[member][pixel], noObjectLogit)
                }
            }
        }
    }

    // MARK: - 3. Output-time non-overlap

    /// Give each contested pixel to the highest-scoring object in its prompt group.
    ///
    /// Port of `Sam3VideoProcessor._apply_object_wise_non_overlapping_constraints` with
    /// `background_value = 0`. The scores here are per-object, not per-pixel, so the
    /// winner of a contested region is the same everywhere the two overlap.
    ///
    /// Upstream compares `pixel_nonoverlap > 0`, so an object whose score is exactly zero
    /// loses every pixel, even uncontested ones. Reproduced by the `bestScore[pixel] > 0` test.
    static func applyObjectWiseNonOverlap(
        masks: inout [MaskBitset], scores: [Float], promptIDs: [Int]
    ) {
        guard masks.count > 1 else { return }
        for group in Set(promptIDs).sorted() {
            let members = promptIDs.indices.filter { promptIDs[$0] == group }
            guard members.count > 1 else { continue }

            let pixelCount = masks[members[0]].width * masks[members[0]].height
            var bestScore = [Float](repeating: 0, count: pixelCount)
            var winner = [Int32](repeating: -1, count: pixelCount)
            // Strict `>` so ties keep the lowest group index, matching `torch.argmax`.
            for member in members {
                let score = scores[member]
                masks[member].forEachSetIndex { pixel in
                    if winner[pixel] == -1 || score > bestScore[pixel] {
                        bestScore[pixel] = score
                        winner[pixel] = Int32(member)
                    }
                }
            }
            for member in members {
                var losses: [Int] = []
                masks[member].forEachSetIndex { pixel in
                    if winner[pixel] != Int32(member) || bestScore[pixel] <= 0 {
                        losses.append(pixel)
                    }
                }
                guard !losses.isEmpty else { continue }
                let width = masks[member].width
                for pixel in losses {
                    masks[member][pixel % width, pixel / width] = false
                }
            }
        }
    }
}
