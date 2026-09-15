// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Half-precision payload held on the host between frames.
///
/// Encoded memory leaves `memory_encode` as fp16 and goes straight back into `tracker_step`
/// as fp16, so it is never interpreted here. Keeping it as `Float16` rather than widening
/// halves the memory bank's footprint.
///
/// `limit` caps how much is read, for callers writing into a fixed-size slot: the packer
/// copies `values` wholesale, so a longer-than-expected array would spill into the next slot.
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
/// Deliberately smaller than HF's equivalent dict. Upstream keeps `high_res_masks` (1008²
/// floats, 4 MB) forever and never reads it back; the memory encoder consumes the in-flight
/// value on the frame it is produced. Dropping it is what makes a long video fit.
struct StoredFrameOutput {
    /// Low-resolution mask logits. Only re-read when a frame is revisited and the object
    /// already has a conditioning output there, so it is pruned aggressively.
    var predictedMasks: [Float]?
    /// `(1, 1, hidden)` pointer, fed to `tracker_step` as one of the pointer slots.
    var objectPointer: NDArray
    var objectScoreLogit: Float
    /// Set after the frame's memory encoding pass; absent until then.
    var memoryFeatures: MemoryPayload?
    var memoryPositionEncoding: MemoryPayload?
}

/// Per-object output history, split the way `output_dict_per_obj` is.
///
/// The conditioning / non-conditioning split is not bookkeeping: `_select_closest_cond_frames`
/// picks from the first, `_gather_memory_frame_outputs` walks a fixed recent window of the
/// second, and reconditioning promotes an entry from one to the other mid-frame.
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
            // A frame is in exactly one bucket. Promoting without this leaves a stale
            // copy that `_gather_memory_frame_outputs` could pick up as a recent memory.
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

    /// Attach a frame's encoded memory to whichever bucket holds it. A reconditioned object
    /// was promoted to conditioning a moment ago, so neither bucket can be assumed.
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
    /// Upstream never prunes, which does not survive a long video. The window is derived
    /// from what the two readers can still reach: `_gather_memory_frame_outputs` looks back
    /// `numMaskmem - 1` frames, `_get_object_pointers` looks back `maxObjectPointers - 1`.
    /// Conditioning entries are kept whole — pointers iterate all of them, and there are few.
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
            // Keep the pointer and the memory (a conditioning frame can be selected from
            // any distance), but the stored low-res mask is only read when that exact frame
            // is revisited, which forward propagation never does.
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
/// `Sam3TrackerVideoInferenceSession`. Everything here is plain data; the logic that reads
/// it lives in `Tracking/`.
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
    /// Objects hotstart removed at any point; the postprocessor hides them retroactively.
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

    /// Remove an object and every trace of it, compacting the parallel storage.
    ///
    /// Port of `Sam3VideoInferenceSession.remove_object`. Upstream resets the whole session
    /// when the last object goes; clearing here is equivalent and keeps prompt state.
    func removeObject(_ id: Int) {
        guard let survivors = registry.remove(id) else { return }
        histories = survivors.map { histories[$0] }
        promptIDByObjectID[id] = nil
        lastOccludedByObjectID[id] = nil
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
