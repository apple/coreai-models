// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Matches this frame's detections against the existing masklets.
///
/// Port of `Sam3VideoModel._associate_det_trk`. The two IoU thresholds are easy to conflate:
/// `assocIouThresh` (loose, 0.1) decides whether a detection is covered by some track and so
/// is not new, while `trkAssocIouThresh` (strict, 0.5) decides whether a track was seen this
/// frame. The same pair can be matched by the first test and unmatched by the second.
enum Associator {
    struct Result {
        /// Detection indices that should become new objects.
        var newDetectionIndices: [Int] = []
        /// Non-empty tracks no detection covered this frame.
        var unmatchedTrackIDs: [Int] = []
        /// Detection index to every track it overlaps above the loose threshold.
        var detectionToMatchedTrackIDs: [Int: [Int]] = [:]
        /// Track id to the single high-confidence, high-IoU detection that may recondition
        /// it. At most one per track.
        var trackIDToHighConfidenceDetection: [Int: Int] = [:]
        /// Tracks whose predicted mask came back empty.
        var emptyTrackIDs: [Int] = []
    }

    /// - Parameters:
    ///   - detections: This frame's merged detections.
    ///   - trackMasks: Tracker masks, one per entry of `trackIDs`, in that order.
    ///   - trackIDs: Object ids in registry order.
    ///   - trackPromptIDs: Prompt each tracked object belongs to, parallel to `trackIDs`.
    static func associate(
        detections: MergedDetections,
        trackMasks: [MaskBitset],
        trackIDs: [Int],
        trackPromptIDs: [Int],
        parameters: VideoSegmentationParameters
    ) -> Result {
        var result = Result()

        if trackMasks.isEmpty {
            // Nothing to match against, so every detection is new. This branch skips the
            // `newDetThresh` score gate the general path applies, as upstream does; that is
            // how the first frame seeds tracks from sub-threshold detections.
            result.newDetectionIndices = Array(0..<detections.count)
            return result
        }
        if detections.isEmpty {
            for (index, id) in trackIDs.enumerated() {
                if trackMasks[index].isEmpty {
                    result.emptyTrackIDs.append(id)
                } else {
                    result.unmatchedTrackIDs.append(id)
                }
            }
            return result
        }

        // Full IoU matrix, zeroed across prompt groups so a "person" detection can never
        // claim a "dog" track.
        var ious = [[Float]](
            repeating: [Float](repeating: 0, count: trackMasks.count), count: detections.count)
        for detection in 0..<detections.count {
            for track in 0..<trackMasks.count where detections.promptIDs[detection] == trackPromptIDs[track] {
                ious[detection][track] = detections.masks[detection].iou(trackMasks[track])
            }
        }

        // A track is unmatched when it has area but no detection reaches the strict
        // threshold. Empty tracks are reported separately: they are occluded, not lost.
        for (track, id) in trackIDs.enumerated() {
            if trackMasks[track].isEmpty {
                result.emptyTrackIDs.append(id)
                continue
            }
            let matched = (0..<detections.count).contains {
                ious[$0][track] >= parameters.trkAssocIouThresh
            }
            if !matched { result.unmatchedTrackIDs.append(id) }
        }

        for detection in 0..<detections.count {
            let row = ious[detection]
            let matchedTracks = (0..<trackMasks.count).filter {
                row[$0] >= parameters.assocIouThresh
            }
            result.detectionToMatchedTrackIDs[detection] = matchedTracks.map { trackIDs[$0] }

            let isNew =
                detections.scores[detection] >= parameters.newDetThresh && matchedTracks.isEmpty
            if isNew {
                result.newDetectionIndices.append(detection)
                // A new detection cannot also recondition an existing track.
                continue
            }

            // Reconditioning candidate: confident enough, overlapping enough, and mapped to
            // its single best track. Later detections overwrite earlier ones for the same
            // track, which is upstream's behaviour.
            guard detections.scores[detection] >= parameters.highConfThresh else { continue }
            guard let best = row.indices.max(by: { row[$0] < row[$1] }) else { continue }
            if row[best] >= parameters.highIouThresh {
                result.trackIDToHighConfidenceDetection[trackIDs[best]] = detection
            }
        }
        return result
    }
}
