// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// One tracked object on one frame.
public struct TrackedObject: Sendable {
    /// Stable id for the masklet; survives across frames until the track ends.
    public let id: Int
    /// Text prompt that discovered this object.
    public let prompt: String
    /// Binary mask at the source video's resolution.
    public let mask: MaskBitset
    /// Mask-derived bounding box in pixels, top-left origin. Derived the way
    /// `torchvision.ops.masks_to_boxes` does, so the extremes are inclusive.
    public let box: CGRect
    /// Detection score in [0, 1] for the frame the object was found on.
    public let score: Float
    /// Tracker confidence on this frame, or 0 before the tracker has seen the object.
    public let trackerScore: Float
}

/// The result for one frame of video.
public struct VideoSegmentationFrame: Sendable {
    /// Index into the decoded frame sequence, 0-based.
    public let frameIndex: Int
    /// Surviving objects, ordered by id.
    public let objects: [TrackedObject]
    /// The source frame, so callers can composite without decoding twice.
    public let image: CGImage

    /// Per-object mask logits at the model's low resolution, ordered like `objects`.
    ///
    /// Empty unless ``VideoSegmentationParameters/emitLowResolutionMasks`` is set. See
    /// that flag for why it exists.
    public let lowResolutionMasks: [[Float]]

    /// Wall clock spent processing this frame.
    ///
    /// Measured around the frame's own work, not between emissions: hotstart holds the first
    /// `hotstartDelay` results back, so timing the gap between emissions would report that
    /// buffering rather than the model.
    public let processingTime: Duration
}

// MARK: - Internal frame-loop types

/// Detections for one frame, merged across every prompt.
///
/// Port of what `_merge_detections_from_prompts` returns, minus `bbox`: nothing downstream
/// reads the detector's boxes, and the boxes the caller finally sees come from
/// `masks_to_boxes` on the upsampled mask.
struct MergedDetections {
    /// Low-resolution mask logits, one flat `size * size` buffer per detection.
    var maskLogits: [[Float]] = []
    /// The same masks binarized at 0, for every IoU in the frame loop.
    var masks: [MaskBitset] = []
    var scores: [Float] = []
    /// Prompt that produced each detection, parallel to the arrays above.
    var promptIDs: [Int] = []

    var count: Int { scores.count }
    var isEmpty: Bool { scores.isEmpty }
}

/// What the planning phase decided, consumed by the execution phase and the output
/// builder. Port of `tracker_update_plan`.
struct TrackerUpdatePlan {
    var newDetectionIndices: [Int] = []
    var newObjectIDs: [Int] = []
    var unmatchedTrackIDs: [Int] = []
    var detectionToMatchedTrackIDs: [Int: [Int]] = [:]
    var newlyRemovedObjectIDs: Set<Int> = []
    var trackIDToHighConfidenceDetection: [Int: Int] = [:]
    var reconditionedObjectIDs: Set<Int> = []
}

/// The per-frame model output before postprocessing. Port of `Sam3VideoSegmentationOutput`.
struct RawFrameOutput {
    let frameIndex: Int
    /// Object id to low-resolution mask logits.
    var maskLogitsByObjectID: [Int: [Float]]
    var scoreByObjectID: [Int: Float]
    var trackerScoreByObjectID: [Int: Float]
    var suppressedObjectIDs: Set<Int>
}
