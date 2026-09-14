// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A `kind: video_segmenter` model bundle: the asset, the tokenizer, and the slot
/// geometry the host has to pack memory to.
///
/// The `runtime` block is required. The host builds fixed-capacity memory-bank tensors
/// whose shapes must match what the graph was traced with, and deriving those from a
/// hardcoded default would break silently the moment somebody re-exports with
/// `--spatial-slots`.
///
/// The `tracking` block is optional and carries the `Sam3VideoConfig` thresholds. A bundle
/// without it gets ``VideoSegmentationParameters``'s defaults, which are the upstream config's
/// own defaults.
public struct VideoSegmenterBundle: Sendable {
    public let bundle: ModelBundle
    public let modelURL: URL
    public let tokenizerFolder: URL
    public let geometry: Geometry

    /// Slot geometry from the bundle's `runtime` block.
    public struct Geometry: Sendable, Decodable, Equatable {
        /// Square input resolution the vision encoder was traced at.
        public let imageSize: Int
        /// Spatial memory slots per object in `tracker_step`.
        public let spatialSlots: Int
        /// Object-pointer slots per object in `tracker_step`.
        public let ptrSlots: Int
        /// Token count `text_encode` was traced at.
        public let maxTextSeqLen: Int

        enum CodingKeys: String, CodingKey {
            case imageSize = "image_size"
            case spatialSlots = "spatial_slots"
            case ptrSlots = "ptr_slots"
            case maxTextSeqLen = "max_text_seq_len"
        }
    }

    public init(from path: String) throws {
        try self.init(bundle: ModelBundle(from: path))
    }

    public init(bundle: ModelBundle) throws {
        guard bundle.kind == .videoSegmenter else {
            throw ModelBundle.BundleError.kindMismatch(expected: .videoSegmenter, got: bundle.kind)
        }
        self.bundle = bundle
        self.modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        self.tokenizerFolder = bundle.bundlePath.appending(path: "tokenizer")

        guard let runtime = try? JSONDecoder().decode(RuntimeEnvelope.self, from: bundle.raw),
            let geometry = runtime.runtime
        else {
            throw VideoSegmentationError.invalidConfiguration(
                "\(bundle.bundlePath.lastPathComponent)/metadata.json has no 'runtime' block. "
                    + "A video_segmenter bundle must declare image_size, spatial_slots, "
                    + "ptr_slots, and max_text_seq_len.")
        }
        guard geometry.imageSize > 0, geometry.spatialSlots > 0, geometry.ptrSlots > 0,
            geometry.maxTextSeqLen > 0
        else {
            throw VideoSegmentationError.invalidConfiguration(
                "metadata.json 'runtime' values must all be positive; got \(geometry).")
        }
        self.geometry = geometry
    }

    /// Parameters with any `tracking` overrides from metadata.json applied on top of
    /// `base`. Absent keys keep their value from `base`.
    public func parameters(overriding base: VideoSegmentationParameters = .default)
        -> VideoSegmentationParameters
    {
        guard let envelope = try? JSONDecoder().decode(TrackingEnvelope.self, from: bundle.raw),
            let tracking = envelope.tracking
        else { return base }

        var parameters = base
        tracking.scoreThresholdDetection.map { parameters.scoreThresholdDetection = $0 }
        tracking.detNmsThresh.map { parameters.detNmsThresh = $0 }
        tracking.newDetThresh.map { parameters.newDetThresh = $0 }
        tracking.assocIouThresh.map { parameters.assocIouThresh = $0 }
        tracking.trkAssocIouThresh.map { parameters.trkAssocIouThresh = $0 }
        tracking.highConfThresh.map { parameters.highConfThresh = $0 }
        tracking.highIouThresh.map { parameters.highIouThresh = $0 }
        tracking.reconditionEveryNthFrame.map { parameters.reconditionEveryNthFrame = $0 }
        tracking.reconditionOnTrkMasks.map { parameters.reconditionOnTrkMasks = $0 }
        tracking.hotstartDelay.map { parameters.hotstartDelay = $0 }
        tracking.hotstartUnmatchThresh.map { parameters.hotstartUnmatchThresh = $0 }
        tracking.hotstartDupThresh.map { parameters.hotstartDupThresh = $0 }
        tracking.suppressUnmatchedOnlyWithinHotstart.map {
            parameters.suppressUnmatchedOnlyWithinHotstart = $0
        }
        tracking.initTrkKeepAlive.map { parameters.initTrkKeepAlive = $0 }
        tracking.maxTrkKeepAlive.map { parameters.maxTrkKeepAlive = $0 }
        tracking.minTrkKeepAlive.map { parameters.minTrkKeepAlive = $0 }
        tracking.decreaseTrkKeepAliveForEmptyMasklets.map {
            parameters.decreaseTrkKeepAliveForEmptyMasklets = $0
        }
        tracking.suppressOverlappingOcclusionThreshold.map {
            parameters.suppressOverlappingOcclusionThreshold = $0
        }
        tracking.maxNumObjects.map { parameters.maxNumObjects = $0 }
        tracking.fillHoleArea.map { parameters.fillHoleArea = $0 }
        tracking.numMaskmem.map { parameters.numMaskmem = $0 }
        tracking.maxCondFrameNum.map { parameters.maxCondFrameNum = $0 }
        tracking.maxObjectPointers.map { parameters.maxObjectPointers = $0 }
        return parameters
    }

    // MARK: - Codable shapes

    private struct RuntimeEnvelope: Decodable {
        let runtime: Geometry?
    }

    private struct TrackingEnvelope: Decodable {
        let tracking: Tracking?
    }

    /// Every field optional so a partial `tracking` block is legal and an unfamiliar key
    /// from a newer exporter is ignored rather than fatal.
    private struct Tracking: Decodable {
        let scoreThresholdDetection: Float?
        let detNmsThresh: Float?
        let newDetThresh: Float?
        let assocIouThresh: Float?
        let trkAssocIouThresh: Float?
        let highConfThresh: Float?
        let highIouThresh: Float?
        let reconditionEveryNthFrame: Int?
        let reconditionOnTrkMasks: Bool?
        let hotstartDelay: Int?
        let hotstartUnmatchThresh: Int?
        let hotstartDupThresh: Int?
        let suppressUnmatchedOnlyWithinHotstart: Bool?
        let initTrkKeepAlive: Int?
        let maxTrkKeepAlive: Int?
        let minTrkKeepAlive: Int?
        let decreaseTrkKeepAliveForEmptyMasklets: Bool?
        let suppressOverlappingOcclusionThreshold: Float?
        let maxNumObjects: Int?
        let fillHoleArea: Int?
        let numMaskmem: Int?
        let maxCondFrameNum: Int?
        let maxObjectPointers: Int?

        // Snake case throughout, matching the HF config field names the exporter copies.
        enum CodingKeys: String, CodingKey {
            case scoreThresholdDetection = "score_threshold_detection"
            case detNmsThresh = "det_nms_thresh"
            case newDetThresh = "new_det_thresh"
            case assocIouThresh = "assoc_iou_thresh"
            case trkAssocIouThresh = "trk_assoc_iou_thresh"
            case highConfThresh = "high_conf_thresh"
            case highIouThresh = "high_iou_thresh"
            case reconditionEveryNthFrame = "recondition_every_nth_frame"
            case reconditionOnTrkMasks = "recondition_on_trk_masks"
            case hotstartDelay = "hotstart_delay"
            case hotstartUnmatchThresh = "hotstart_unmatch_thresh"
            case hotstartDupThresh = "hotstart_dup_thresh"
            case suppressUnmatchedOnlyWithinHotstart = "suppress_unmatched_only_within_hotstart"
            case initTrkKeepAlive = "init_trk_keep_alive"
            case maxTrkKeepAlive = "max_trk_keep_alive"
            case minTrkKeepAlive = "min_trk_keep_alive"
            case decreaseTrkKeepAliveForEmptyMasklets =
                "decrease_trk_keep_alive_for_empty_masklets"
            case suppressOverlappingOcclusionThreshold =
                "suppress_overlapping_based_on_recent_occlusion_threshold"
            case maxNumObjects = "max_num_objects"
            case fillHoleArea = "fill_hole_area"
            case numMaskmem = "num_maskmem"
            case maxCondFrameNum = "max_cond_frame_num"
            case maxObjectPointers = "max_object_pointers_in_encoder"
        }
    }
}
