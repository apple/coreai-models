// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Half-precision payload held on the host between frames.
///
/// `limit` caps how much is read. The packer copies `values` wholesale into a fixed-size
/// slot, so a longer array would spill into the next one.
struct MemoryPayload: Sendable {
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    let values: [Float16]

    init(reading array: NDArray, limit: Int = .max) {
        let count = min(array.shape.reduce(1, *), limit)
        self.values = readNDArray(array, as: Float16.self, count: count)
    }
    #else
    let values: [Float]

    init(reading array: NDArray, limit: Int = .max) {
        self.values = Array(flattenAsFloat(array).prefix(limit))
    }
    #endif
}

/// One object's stored result for one frame.
///
/// Smaller than HF's equivalent dict, which retains `high_res_masks` (4 MB a frame) for the
/// whole video. Dropping those is what makes a long video fit.
struct StoredFrameOutput {
    /// Low-resolution mask logits. Read back only when a frame is revisited, so pruning
    /// drops these early.
    var predictedMasks: [Float]?
    /// `(1, 1, hidden)` pointer, fed to `tracker_step` as one of the pointer slots.
    var objectPointer: NDArray
    var objectScoreLogit: Float
    /// Set by the frame's memory encoding pass.
    var memoryFeatures: MemoryPayload?
    var memoryPositionEncoding: MemoryPayload?
}

/// Per-object output history, split the way `output_dict_per_obj` is.
///
/// The split drives behaviour. `_select_closest_cond_frames` picks from the conditioning
/// side while `_gather_memory_frame_outputs` walks a recent window of the other.
/// Reconditioning moves an entry across mid-frame.
struct ObjectOutputHistory {
    /// Frames where the object was seeded or reconditioned. Insertion-ordered, because
    /// `_get_object_pointers` iterates them in that order.
    var conditioning: [Int: StoredFrameOutput] = [:]
    var conditioningOrder: [Int] = []
    var nonConditioning: [Int: StoredFrameOutput] = [:]
    /// Frames the object has been propagated through, and in which direction.
    var framesTracked: [Int: Bool] = [:]
    /// Pending mask prompts, keyed by frame.
    var maskInputs: [Int: MaskPrompt] = [:]

    mutating func store(_ output: StoredFrameOutput, at frame: Int, conditioning isConditioning: Bool) {
        if isConditioning {
            if conditioning[frame] == nil { conditioningOrder.append(frame) }
            conditioning[frame] = output
            // A frame belongs to exactly one bucket. A stale copy left behind would show up
            // in `_gather_memory_frame_outputs` as a recent memory.
            nonConditioning[frame] = nil
        } else {
            nonConditioning[frame] = output
        }
    }

    /// Move an existing non-conditioning entry into the conditioning bucket, which is what
    /// `_tracker_update_memories` does for a reconditioned object.
    mutating func promoteToConditioning(frame: Int) {
        guard let existing = nonConditioning.removeValue(forKey: frame) else { return }
        if conditioning[frame] == nil { conditioningOrder.append(frame) }
        conditioning[frame] = existing
    }

    /// Attach a frame's encoded memory to whichever bucket holds it. Reconditioning may have
    /// moved the object a moment ago, so the bucket is resolved here.
    mutating func attachMemory(
        features: MemoryPayload, positionEncoding: MemoryPayload, at frame: Int
    ) {
        if conditioning[frame] != nil {
            conditioning[frame]?.memoryFeatures = features
            conditioning[frame]?.memoryPositionEncoding = positionEncoding
        } else {
            nonConditioning[frame]?.memoryFeatures = features
            nonConditioning[frame]?.memoryPositionEncoding = positionEncoding
        }
    }

    /// Drop history that can no longer be read.
    ///
    /// Upstream retains everything, which a long video outgrows. The window covers what the
    /// two readers can still reach. Conditioning entries are kept whole. Pointers iterate all
    /// of them and there are few.
    mutating func prune(before frame: Int, memoryWindow: Int, pointerWindow: Int) {
        let memoryFloor = frame - memoryWindow
        let pointerFloor = frame - pointerWindow
        for key in nonConditioning.keys {
            if key < pointerFloor {
                nonConditioning[key] = nil
            } else if key < memoryFloor {
                nonConditioning[key]?.memoryFeatures = nil
                nonConditioning[key]?.memoryPositionEncoding = nil
                nonConditioning[key]?.predictedMasks = nil
            }
        }
        for key in conditioningOrder where key < memoryFloor {
            // Keep the pointer and the memory, since a conditioning frame can be selected
            // from any distance. Only the low-res mask is unreachable once the frame is past.
            conditioning[key]?.predictedMasks = nil
        }
    }
}

/// A mask prompt queued for an object on a frame, as a binary low-resolution mask.
struct MaskPrompt {
    let mask: MaskBitset
}

/// Host-side state for one video.
///
/// Port of `Sam3VideoInferenceSession` plus the per-object tracker state HF keeps inside
/// `Sam3TrackerVideoInferenceSession`. Plain data only. The logic that reads it lives in
/// `Tracking/`.
@VideoSegmentationActor
final class VideoInferenceSession {
    let videoWidth: Int
    let videoHeight: Int

    // MARK: - Prompts

    /// Prompt text by id, in registration order.
    private(set) var prompts: [String] = []
    /// Encoded text features, filled on first use and reused for the whole video.
    var promptEncodings: [Int: PromptEncoding] = [:]
    /// Token ids and attention mask per prompt.
    var promptTokens: [Int: (ids: [Int32], attentionMask: [Int32])] = [:]
    /// Which prompt discovered each object.
    var promptIDByObjectID: [Int: Int] = [:]

    // MARK: - Objects

    var registry = ObjectRegistry()
    /// Parallel to `registry.ids`.
    var histories: [ObjectOutputHistory] = []
    /// Object ids seeded on this frame and not yet consumed by the tracker loop.
    var objectsWithNewInputs: [Int] = []

    // MARK: - Tracking metadata

    var scoreByObjectID: [Int: Float] = [:]
    var trackerScoreByFrame: [Int: [Int: Float]] = [:]
    var lastOccludedByObjectID: [Int: Int] = [:]
    var maxObjectID: Int = -1

    // MARK: - Hotstart metadata

    var firstFrameByObjectID: [Int: Int] = [:]
    var unmatchedFramesByObjectID: [Int: [Int]] = [:]
    var overlapFramesByPair: [OverlapPair: [Int]] = [:]
    var keepAliveByObjectID: [Int: Int] = [:]
    var removedObjectIDs: Set<Int> = []
    var suppressedObjectIDsByFrame: [Int: Set<Int>] = [:]
    /// Objects hotstart removed at any point. The postprocessor hides them retroactively.
    var hotstartRemovedObjectIDs: Set<Int> = []

    // MARK: - Frame cache

    /// Tracker features for the frame being processed, matching HF's
    /// `max_vision_features_cache_size` default of 1.
    var cachedFrameIndex: Int?
    var cachedFeatures: TrackerFeatures?

    struct OverlapPair: Hashable {
        let firstAppearing: Int
        let duplicate: Int
    }

    init(videoWidth: Int, videoHeight: Int) {
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
    }

    // MARK: - Prompts

    /// Register `text`, returning its id. Duplicate text reuses the existing id, matching
    /// `Sam3VideoInferenceSession.add_prompt`.
    func addPrompt(_ text: String) -> Int {
        if let existing = prompts.firstIndex(of: text) { return existing }
        prompts.append(text)
        return prompts.count - 1
    }

    func promptText(_ id: Int) -> String { prompts[id] }

    var promptIDs: [Int] { Array(prompts.indices) }

    // MARK: - Objects

    var objectIDs: [Int] { registry.ids }
    var objectCount: Int { registry.count }

    /// Index of `id`, creating its history if this is the first sighting.
    @discardableResult
    func index(ofObject id: Int) -> Int {
        let (index, isNew) = registry.index(of: id)
        if isNew { histories.append(ObjectOutputHistory()) }
        return index
    }

    /// Remove an object, compacting the parallel storage and dropping its per-object metadata.
    ///
    /// Port of `Sam3VideoInferenceSession.remove_object`. Upstream resets the whole session
    /// when the last object goes. Clearing per-object state has the same effect and keeps
    /// the prompts.
    ///
    /// `removedObjectIDs`, `hotstartRemovedObjectIDs` and `overlapFramesByPair` survive as
    /// tombstones, since the hotstart rules and the postprocessor still read them for ids
    /// that are gone.
    func removeObject(_ id: Int) {
        guard let survivors = registry.remove(id) else { return }
        histories = survivors.map { histories[$0] }
        promptIDByObjectID[id] = nil
        lastOccludedByObjectID[id] = nil
        scoreByObjectID[id] = nil
        firstFrameByObjectID[id] = nil
        unmatchedFramesByObjectID[id] = nil
        keepAliveByObjectID[id] = nil
        objectsWithNewInputs.removeAll { $0 == id }
    }

    /// Set `frame`'s tracker features, replacing whatever the previous frame left.
    func cache(_ features: TrackerFeatures, forFrame frame: Int) {
        cachedFrameIndex = frame
        cachedFeatures = features
    }

    func features(forFrame frame: Int) throws -> TrackerFeatures {
        guard cachedFrameIndex == frame, let cachedFeatures else {
            throw VideoSegmentationError.invalidConfiguration(
                "No cached tracker features for frame \(frame); the frame loop ran out of order.")
        }
        return cachedFeatures
    }

    /// Drop history no reader can reach any more, for every object.
    func prune(currentFrame: Int, memoryWindow: Int, pointerWindow: Int) {
        for index in histories.indices {
            histories[index].prune(
                before: currentFrame, memoryWindow: memoryWindow, pointerWindow: pointerWindow)
        }
    }
}
