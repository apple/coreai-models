// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// Turns a frame's low-resolution mask logits into the objects a caller sees.
///
/// Port of `Sam3VideoProcessor.postprocess_outputs`. Four steps, in this order, and the
/// order matters: upsample to video resolution and binarize, drop empty and hidden
/// objects, resolve overlaps within each prompt group, then derive boxes.
struct MaskPostprocessor {
    /// Surviving objects plus, when asked for, the mask logits they came from.
    struct Postprocessed {
        var objects: [TrackedObject] = []
        var lowResolutionMasks: [[Float]] = []
    }

    private let lowResolutionSize: Int
    private let videoWidth: Int
    private let videoHeight: Int
    private let emitLowResolutionMasks: Bool
    private let resampler: BilinearResampler

    init(
        lowResolutionSize: Int, videoWidth: Int, videoHeight: Int,
        emitLowResolutionMasks: Bool = false
    ) {
        self.lowResolutionSize = lowResolutionSize
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.emitLowResolutionMasks = emitLowResolutionMasks
        // Note: no antialiasing. Upstream calls `interpolate(..., mode="bilinear",
        // align_corners=False)` here without the flag, unlike the resizes inside the
        // tracker.
        self.resampler = BilinearResampler(
            sourceWidth: lowResolutionSize, sourceHeight: lowResolutionSize,
            destinationWidth: videoWidth, destinationHeight: videoHeight,
            antialias: false)
    }

    /// Reads the session's prompt map and hidden-object set, so it shares the frame loop's
    /// isolation rather than taking copies of both per frame.
    @VideoSegmentationActor
    func postprocess(_ raw: RawFrameOutput, session: VideoInferenceSession) -> Postprocessed {
        // Sorted ids, so output order is stable frame to frame regardless of how the
        // registry happens to be arranged.
        let candidates = raw.maskLogitsByObjectID.keys.sorted()
        guard !candidates.isEmpty else { return Postprocessed() }

        let hidden = raw.suppressedObjectIDs.union(session.hotstartRemovedObjectIDs)

        var ids: [Int] = []
        var masks: [MaskBitset] = []
        var scores: [Float] = []
        var trackerScores: [Float] = []
        var promptIDs: [Int] = []
        var lowResolution: [[Float]] = []

        for objectID in candidates {
            guard !hidden.contains(objectID) else { continue }
            guard let logits = raw.maskLogitsByObjectID[objectID] else { continue }
            let upsampled = resampler.resample(logits)
            let mask = MaskBitset(
                thresholding: upsampled, width: videoWidth, height: videoHeight)
            // Objects whose mask upsampled to nothing are dropped, not reported empty.
            guard !mask.isEmpty else { continue }

            ids.append(objectID)
            masks.append(mask)
            scores.append(raw.scoreByObjectID[objectID] ?? 0)
            trackerScores.append(raw.trackerScoreByObjectID[objectID] ?? 0)
            promptIDs.append(session.promptIDByObjectID[objectID] ?? 0)
            if emitLowResolutionMasks { lowResolution.append(logits) }
        }
        guard !ids.isEmpty else { return Postprocessed() }

        // Boxes come from the masks before overlap resolution. Upstream computes
        // `masks_to_boxes` and only then applies the non-overlap constraint, so a box can
        // be slightly larger than the mask it labels. Kept that way for parity.
        let boxes = masks.map(\.boundingBox)

        // Overlaps are resolved by tracker score, not detection score, and only within a
        // prompt group: "person" and "pillow" are allowed to claim the same pixels.
        OcclusionSuppressor.applyObjectWiseNonOverlap(
            masks: &masks, scores: trackerScores, promptIDs: promptIDs)

        var objects: [TrackedObject] = []
        objects.reserveCapacity(ids.count)
        for index in ids.indices {
            objects.append(
                TrackedObject(
                    id: ids[index],
                    prompt: session.promptText(promptIDs[index]),
                    mask: masks[index],
                    box: boxes[index],
                    score: scores[index],
                    trackerScore: trackerScores[index]))
        }
        return Postprocessed(objects: objects, lowResolutionMasks: lowResolution)
    }
}
