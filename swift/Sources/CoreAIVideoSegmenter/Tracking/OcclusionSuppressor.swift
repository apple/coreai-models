// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Resolves objects that claim the same pixels.
///
/// Three rules ported from `Sam3VideoModel`. ``suppressRecentlyOccluded`` and
/// ``suppressAreaShrinkage`` run on tracker logits before memory encoding,
/// ``applyObjectWiseNonOverlap`` on binary masks at output time.
///
/// All three apply per prompt group, so overlapping "person" and "pillow" masks coexist.
enum OcclusionSuppressor {
    /// Logit written over a suppressed mask.
    static let noObjectLogit: Float = -10

    /// Fraction of its own area an object must retain through the pixel-level argmax.
    static let shrinkThreshold: Float = 0.3

    /// Frame index standing in for "never occluded".
    static let neverOccluded = -1

    /// Frame index standing in for "removed by hotstart, always loses".
    static let alwaysOccluded = 100_000

    /// Index groups that can contend, in prompt-id order. Groups of one are dropped.
    private static func contendingGroups(_ promptIDs: [Int]) -> [[Int]] {
        Set(promptIDs).sorted()
            .map { group in promptIDs.indices.filter { promptIDs[$0] == group } }
            .filter { $0.count > 1 }
    }

    // MARK: - 1. Recent-occlusion suppression

    /// Blank objects that overlap an object occluded less recently than they were.
    ///
    /// Port of `_suppress_overlapping_based_on_recent_occlusion`. Also records which objects
    /// were occluded this frame, meaning their mask came back empty or they were suppressed
    /// here. The next frame's comparison reads that.
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
        precondition(
            masks.count == objectIDs.count && logits.count == objectIDs.count
                && promptIDs.count == objectIDs.count,
            "suppressRecentlyOccluded: logits, masks, objectIDs and promptIDs must be parallel")

        let lastOccluded = objectIDs.map { id in
            session.lastOccludedByObjectID[id]
                ?? (newlyRemovedObjectIDs.contains(id) ? alwaysOccluded : neverOccluded)
        }

        var suppress = [Bool](repeating: false, count: objectIDs.count)
        for members in contendingGroups(promptIDs) {
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
    /// Of two objects overlapping above `threshold`, the one occluded more recently loses.
    /// The winner must have been occluded at some point too, so two objects that have both
    /// stayed visible coexist.
    private static func markSuppressed(
        members: [Int],
        masks: [MaskBitset],
        lastOccluded: [Int],
        threshold: Float,
        reverse: Bool,
        into suppress: inout [Bool]
    ) {
        // Tracking backwards makes "more recent" a lower frame index.
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
    /// Port of `_suppress_object_pw_area_shrinkage`, run per prompt group. The argmax only
    /// measures how much of each object was contested. What comes back is the original masks
    /// with whole objects blanked.
    static func suppressAreaShrinkage(
        logits: inout [[Float]], promptIDs: [Int]
    ) {
        guard logits.count > 1 else { return }
        precondition(
            promptIDs.count == logits.count,
            "suppressAreaShrinkage: logits and promptIDs must be parallel")
        for members in contendingGroups(promptIDs) {
            let pixelCount = logits[members[0]].count
            precondition(
                members.allSatisfy { logits[$0].count == pixelCount },
                "suppressAreaShrinkage: logits in a prompt group must be the same length")
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
    /// `background_value = 0`.
    ///
    /// Upstream compares `pixel_nonoverlap > 0`, so an object scoring exactly zero loses every
    /// pixel, uncontested ones included. The `bestScore[pixel] > 0` test matches that.
    static func applyObjectWiseNonOverlap(
        masks: inout [MaskBitset], scores: [Float], promptIDs: [Int]
    ) {
        guard masks.count > 1 else { return }
        precondition(
            promptIDs.count == masks.count && scores.count == masks.count,
            "applyObjectWiseNonOverlap: masks, scores and promptIDs must be parallel")
        for members in contendingGroups(promptIDs) {
            let width = masks[members[0]].width
            let height = masks[members[0]].height
            // Both axes, because equal pixel counts at different widths stay in bounds while
            // comparing unrelated locations.
            precondition(
                members.allSatisfy { masks[$0].width == width && masks[$0].height == height },
                "applyObjectWiseNonOverlap: masks in a prompt group must share dimensions")
            let pixelCount = width * height
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
                for pixel in losses {
                    masks[member][pixel % width, pixel / width] = false
                }
            }
        }
    }
}
