// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// Turns a frame's low-resolution mask logits into the objects a caller sees.
///
/// Port of `Sam3VideoProcessor.postprocess_outputs`. The step order is load-bearing:
/// boxes come from the masks as they stand before overlap resolution.
struct MaskPostprocessor {
    /// Surviving objects plus the mask logits they came from, when requested.
    struct Postprocessed {
        var objects: [TrackedObject] = []
        var lowResolutionMasks: [[Float]] = []
    }

    private let lowResolutionSize: Int
    private let videoWidth: Int
    private let videoHeight: Int
    private let emitLowResolutionMasks: Bool
    private let resampler: BilinearResampler

    /// Reused across every object on every frame: the upsampled mask is megabytes at video
    /// resolution and lives only until `MaskBitset` has thresholded it.
    private final class Buffers {
        var upsampled: [Float]
        var scratch: [Float]
        init(pixels: Int, scratchCount: Int) {
            upsampled = [Float](repeating: 0, count: pixels)
            scratch = [Float](repeating: 0, count: scratchCount)
        }
    }
    private let buffers: Buffers

    init(
        lowResolutionSize: Int, videoWidth: Int, videoHeight: Int,
        emitLowResolutionMasks: Bool = false
    ) {
        self.lowResolutionSize = lowResolutionSize
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.emitLowResolutionMasks = emitLowResolutionMasks
        // Upstream calls `interpolate(..., mode="bilinear", align_corners=False)` here with
        // antialiasing off, unlike the resizes inside the tracker.
        self.resampler = BilinearResampler(
            sourceWidth: lowResolutionSize, sourceHeight: lowResolutionSize,
            destinationWidth: videoWidth, destinationHeight: videoHeight,
            antialias: false)
        self.buffers = Buffers(
            pixels: videoWidth * videoHeight, scratchCount: resampler.scratchCount)
    }

    /// Reads the session's prompt map and hidden-object set directly, sharing the frame
    /// loop's isolation so neither needs copying per frame.
    @VideoSegmentationActor
    func postprocess(_ raw: RawFrameOutput, session: VideoInferenceSession) -> Postprocessed {
        // Sorted ids keep output order stable frame to frame, whatever order the registry
        // holds.
        let candidates = raw.maskLogitsByObjectID.keys.sorted()
        guard !candidates.isEmpty else { return Postprocessed() }

        var ids: [Int] = []
        var masks: [MaskBitset] = []
        var scores: [Float] = []
        var trackerScores: [Float] = []
        var promptIDs: [Int] = []
        var lowResolution: [[Float]] = []

        for objectID in candidates {
            guard !raw.suppressedObjectIDs.contains(objectID),
                !session.hotstartRemovedObjectIDs.contains(objectID)
            else { continue }
            guard let logits = raw.maskLogitsByObjectID[objectID] else { continue }
            resampler.resample(logits, into: &buffers.upsampled, scratch: &buffers.scratch)
            let mask = MaskBitset(
                thresholding: buffers.upsampled, width: videoWidth, height: videoHeight)
            // An object whose mask upsampled to nothing is dropped from the frame.
            guard !mask.isEmpty else { continue }

            ids.append(objectID)
            masks.append(mask)
            scores.append(raw.scoreByObjectID[objectID] ?? 0)
            trackerScores.append(raw.trackerScoreByObjectID[objectID] ?? 0)
            promptIDs.append(session.promptIDByObjectID[objectID] ?? 0)
            if emitLowResolutionMasks { lowResolution.append(logits) }
        }
        guard !ids.isEmpty else { return Postprocessed() }

        // Upstream boxes the masks before resolving overlap, so a box can be slightly
        // larger than the mask it labels.
        let boxes = masks.map(\.boundingBox)

        // Overlaps are resolved by tracker score, matching upstream.
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
