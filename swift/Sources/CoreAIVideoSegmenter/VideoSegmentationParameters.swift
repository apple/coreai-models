// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// Runtime knobs for video segmentation.
///
/// The tracking fields mirror `transformers.Sam3VideoConfig` field for field, defaults
/// included. Every threshold below governs host-side logic that HF reads off the model
/// config at load time. A bundle can override them through a `tracking` block in its
/// `metadata.json`.
public struct VideoSegmentationParameters: Sendable {
    // MARK: - Detection

    /// Probability threshold for keeping a detection. `score_threshold_detection`.
    public var scoreThresholdDetection: Float = 0.5

    /// Mask-IoU threshold for detection NMS. `det_nms_thresh`. Zero disables NMS.
    ///
    /// HF runs NMS through an optional `kernels-community/cv-utils` kernel and keeps every
    /// above-threshold detection without it. Set this to 0 to compare against such a run.
    public var detNmsThresh: Float = 0.1

    /// Probability threshold for promoting a detection to a new tracked object.
    /// `new_det_thresh`.
    public var newDetThresh: Float = 0.7

    // MARK: - Association

    /// Loose IoU threshold: above this a detection counts as matching a track.
    /// `assoc_iou_thresh`.
    public var assocIouThresh: Float = 0.1

    /// Stricter IoU threshold deciding whether a track is "unmatched" this frame.
    /// `trk_assoc_iou_thresh`.
    public var trkAssocIouThresh: Float = 0.5

    /// Confidence a detection needs before it may recondition a track. `high_conf_thresh`.
    public var highConfThresh: Float = 0.8

    /// IoU a detection needs before it may recondition a track. `high_iou_thresh`.
    public var highIouThresh: Float = 0.8

    // MARK: - Reconditioning

    /// Recondition every Nth frame, or 0 to disable. `recondition_every_nth_frame`.
    public var reconditionEveryNthFrame: Int = 16

    /// True strengthens memory with the tracked mask, treating the detector as validation.
    /// False replaces it with the detection mask, treating the detector as correction.
    /// `recondition_on_trk_masks`.
    ///
    /// Defaults to the checkpoint's value: `Sam3VideoConfig` declares `True` while
    /// `facebook/sam3`'s `config.json` sets `False`, the one field where the two disagree.
    /// Bundles with a `tracking` block carry the exported value.
    public var reconditionOnTrkMasks: Bool = false

    // MARK: - Hotstart

    /// Frames of output held back while the removal heuristics decide. `hotstart_delay`.
    /// Zero disables both the delay and the removal rules that depend on it.
    public var hotstartDelay: Int = 15

    /// Unmatched frames before a track is removed during hotstart. `hotstart_unmatch_thresh`.
    public var hotstartUnmatchThresh: Int = 8

    /// Overlapping frames before a duplicate track is removed. `hotstart_dup_thresh`.
    public var hotstartDupThresh: Int = 8

    /// Restrict unmatched-suppression to the hotstart window.
    /// `suppress_unmatched_only_within_hotstart`.
    public var suppressUnmatchedOnlyWithinHotstart: Bool = true

    /// Keep-alive counter a new track starts with. `init_trk_keep_alive`.
    public var initTrkKeepAlive: Int = 30

    /// Ceiling for the keep-alive counter. `max_trk_keep_alive`.
    public var maxTrkKeepAlive: Int = 30

    /// Floor for the keep-alive counter. `min_trk_keep_alive`.
    public var minTrkKeepAlive: Int = -1

    /// Decrement keep-alive for tracks the tracker returned empty. Upstream default is off.
    /// `decrease_trk_keep_alive_for_empty_masklets`.
    public var decreaseTrkKeepAliveForEmptyMasklets: Bool = false

    // MARK: - Suppression

    /// IoU above which two overlapping objects contend, resolved by which was occluded
    /// more recently. Zero disables. `suppress_overlapping_based_on_recent_occlusion_threshold`.
    public var suppressOverlappingOcclusionThreshold: Float = 0.7

    /// Maximum simultaneous tracked objects. `max_num_objects`.
    public var maxNumObjects: Int = 10000

    // MARK: - Tracker memory geometry

    // These three come from `Sam3TrackerVideoConfig`. The exported asset pins their sum
    // as `spatial_slots` but leaves the split open, so each is carried separately.

    /// Total mask-memory frames: the current one plus `num_maskmem - 1` recent.
    /// `num_maskmem`.
    public var numMaskmem: Int = 7

    /// Conditioning frames the memory bank may draw on. `max_cond_frame_num`.
    public var maxCondFrameNum: Int = 4

    /// Look-back window for object pointers. `max_object_pointers_in_encoder`.
    public var maxObjectPointers: Int = 16

    // MARK: - Mask cleanup

    /// Maximum area of a connected component that gets filled (background) or removed
    /// (foreground). `fill_hole_area`. Zero disables both.
    ///
    /// Like ``detNmsThresh``, HF no-ops this when `kernels-community/cv-utils` is absent.
    public var fillHoleArea: Int = 16

    // MARK: - Preprocessing

    /// Per-channel means applied after scaling pixels to [0, 1]. SAM 3 uses
    /// `IMAGENET_STANDARD_MEAN`.
    public var normalizationMeans: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)

    /// Per-channel standard deviations. SAM 3 uses `IMAGENET_STANDARD_STD`.
    public var normalizationStds: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)

    // MARK: - Rendering

    /// Opacity of a mask fill over the source frame, in [0, 1].
    public var maskOpacity: Float = 0.5

    /// Draw the mask-derived bounding box for each object.
    public var drawBoxes: Bool = true

    /// Draw a `#id prompt score` caption above each box.
    public var drawLabels: Bool = true

    /// Box and outline stroke width in pixels.
    public var strokeWidth: CGFloat = 3

    // MARK: - Diagnostics

    /// Also report each object's mask logits at the model's own resolution, before the
    /// upsample to video size.
    ///
    /// Off by default, at 82,944 floats per object per frame. On, it separates a tracker
    /// disagreement from an upsampling one.
    public var emitLowResolutionMasks: Bool = false

    public init() {}

    public static let `default` = VideoSegmentationParameters()

    /// Whether the hotstart removal rules apply at all.
    var hotstartEnabled: Bool { hotstartDelay > 0 }
}
