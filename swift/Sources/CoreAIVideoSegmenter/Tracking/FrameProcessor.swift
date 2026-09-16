// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// Runs one frame end to end: encode, detect, propagate, plan, execute, emit.
///
/// Port of `Sam3VideoModel._det_track_one_frame`, `run_tracker_update_planning_phase`,
/// `run_tracker_update_execution_phase`, `build_outputs`, and `forward`.
///
/// The phase split is load-bearing. Planning resolves every heuristic and encodes memory
/// from the de-overlapped masks. Only then does execution mutate the object set. That keeps
/// a track pending removal out of the memory bank, and puts a track pending creation into it
/// for this frame.
@VideoSegmentationActor
final class FrameProcessor {
    private let engine: VideoSegmentationEngine
    private let shapes: VideoSegmentationEngine.Shapes
    private let parameters: VideoSegmentationParameters
    private let tracker: TrackerLoop
    private let preprocessor: FramePreprocessor

    /// Wall-clock spent in each entrypoint, for the CLI's timing summary.
    private(set) var timings: [String: Double] = [:]

    init(
        engine: VideoSegmentationEngine,
        shapes: VideoSegmentationEngine.Shapes,
        parameters: VideoSegmentationParameters,
        tracker: TrackerLoop
    ) {
        self.engine = engine
        self.shapes = shapes
        self.parameters = parameters
        self.tracker = tracker
        self.preprocessor = FramePreprocessor(
            targetSize: shapes.imageSize,
            mean: parameters.normalizationMeans,
            standardDeviation: parameters.normalizationStds)
    }

    /// Process `image` as frame `frameIndex`, returning the raw per-object masks.
    func process(
        session: VideoInferenceSession,
        image: CGImage,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool
    ) async throws -> RawFrameOutput {
        // One ViT pass, shared by the detector and the tracker as upstream does.
        let pixels = try preprocessor.preprocess(image)
        let backbone = try await timed(VideoSegmentationEngine.Function.imageEncode) {
            try await engine.imageEncode(pixelValues: pixels)
        }

        let detections = try await runDetection(session: session, backbone: backbone)

        let features = try await timed(VideoSegmentationEngine.Function.trackerEncode) {
            try await engine.trackerEncode(lastHiddenState: backbone)
        }
        session.cache(features, forFrame: frameIndex)

        // 2. Propagate existing tracks. Memory encoding is deferred to the planning phase,
        // which first has to resolve non-overlap.
        var trackerLogits: [[Float]] = []
        var trackerScoreLogits: [Float] = []
        if !session.objectIDs.isEmpty {
            let propagation = try await timed(VideoSegmentationEngine.Function.trackerStep) {
                try await tracker.propagate(
                    session: session, frameIndex: frameIndex, totalFrames: totalFrames,
                    reverse: reverse, runMemoryEncoder: false)
            }
            trackerLogits = propagation.maskLogits
            trackerScoreLogits = propagation.objectScoreLogits
            for index in trackerLogits.indices {
                fillHoles(&trackerLogits[index])
            }
        }

        // Plan: run every heuristic and encode this frame's memory.
        let planning = try await plan(
            session: session, frameIndex: frameIndex, reverse: reverse,
            detections: detections, trackerLogits: &trackerLogits,
            trackerScoreLogits: trackerScoreLogits)

        // Execute: seed new objects, drop removed ones.
        try await execute(
            session: session, frameIndex: frameIndex, totalFrames: totalFrames,
            reverse: reverse, detections: detections, plan: planning.plan)

        let output = buildOutputs(
            session: session, frameIndex: frameIndex, detections: detections,
            trackerLogits: trackerLogits, trackerScoreLogits: trackerScoreLogits,
            plan: planning.plan, newScores: planning.newScores)

        // Shed history no reader can reach. Upstream retains it for the whole video.
        session.prune(
            currentFrame: frameIndex,
            memoryWindow: max(parameters.numMaskmem, 1),
            pointerWindow: max(parameters.maxObjectPointers, parameters.numMaskmem))
        return output
    }

    /// What the planning phase produced.
    private struct Planning {
        var plan: TrackerUpdatePlan
        /// Scores assigned to objects created or removed on this frame.
        var newScores: [Int: Float]
    }

    private func fillHoles(_ logits: inout [Float]) {
        ConnectedComponents.fillHoles(
            &logits, width: shapes.lowResMaskSize, height: shapes.lowResMaskSize,
            maxArea: parameters.fillHoleArea)
    }

    // MARK: - Detection

    private func runDetection(
        session: VideoInferenceSession, backbone: NDArray
    ) async throws -> MergedDetections {
        guard !session.promptIDs.isEmpty else { throw VideoSegmentationError.noPrompts }

        var perPrompt: [MergedDetections] = []
        for promptID in session.promptIDs {
            let encoding: PromptEncoding
            if let cached = session.promptEncodings[promptID] {
                encoding = cached
            } else {
                guard let tokens = session.promptTokens[promptID] else {
                    throw VideoSegmentationError.invalidConfiguration(
                        "Prompt \(promptID) was registered but never tokenized.")
                }
                encoding = try await timed(VideoSegmentationEngine.Function.textEncode) {
                    try await engine.textEncode(
                        inputIDs: tokens.ids, attentionMask: tokens.attentionMask)
                }
                session.promptEncodings[promptID] = encoding
            }

            let outputs = try await timed(VideoSegmentationEngine.Function.detect) {
                try await engine.detect(lastHiddenState: backbone, prompt: encoding)
            }
            perPrompt.append(
                DetectionDecoder.decode(
                    outputs, promptID: promptID, maskSize: shapes.lowResMaskSize,
                    parameters: parameters))
        }
        return DetectionDecoder.merge(perPrompt)
    }

    // MARK: - Planning phase

    /// Port of `run_tracker_update_planning_phase`.
    ///
    /// Mutates `trackerLogits`. Occlusion suppression and the area-shrinkage rule both blank
    /// objects in place. Those blanked masks are what get encoded into memory and reported as
    /// output.
    private func plan(
        session: VideoInferenceSession,
        frameIndex: Int,
        reverse: Bool,
        detections: MergedDetections,
        trackerLogits: inout [[Float]],
        trackerScoreLogits: [Float]
    ) async throws -> Planning {
        var plan = TrackerUpdatePlan()
        let objectIDsSnapshot = session.objectIDs

        let trackMasks = trackerLogits.map {
            MaskBitset(thresholding: $0, width: shapes.lowResMaskSize, height: shapes.lowResMaskSize)
        }
        let trackPromptIDs = objectIDsSnapshot.map { session.promptIDByObjectID[$0] ?? 0 }

        let association = Associator.associate(
            detections: detections, trackMasks: trackMasks, trackIDs: objectIDsSnapshot,
            trackPromptIDs: trackPromptIDs, parameters: parameters)

        plan.unmatchedTrackIDs = association.unmatchedTrackIDs
        plan.detectionToMatchedTrackIDs = association.detectionToMatchedTrackIDs
        plan.trackIDToHighConfidenceDetection = association.trackIDToHighConfidenceDetection

        // Object-count ceiling: keep the highest-scoring new detections.
        var newIndices = association.newDetectionIndices
        let existing = objectIDsSnapshot.count
        if existing + newIndices.count > parameters.maxNumObjects {
            let keepCount = max(0, parameters.maxNumObjects - existing)
            CLILogger.log(
                "Frame \(frameIndex): hit max_num_objects (\(parameters.maxNumObjects)); dropping "
                    + "\(newIndices.count - keepCount) of \(newIndices.count) new detections.")
            newIndices =
                newIndices
                .sorted { detections.scores[$0] > detections.scores[$1] }
                .prefix(keepCount)
                .sorted()
        }
        plan.newDetectionIndices = newIndices

        // Ids are assigned by position, so the sort above fixes their order.
        let firstNewID = session.maxObjectID + 1
        plan.newObjectIDs = (0..<newIndices.count).map { firstNewID + $0 }
        for (objectID, detectionIndex) in zip(plan.newObjectIDs, newIndices) {
            session.promptIDByObjectID[objectID] = detections.promptIDs[detectionIndex]
        }

        plan.newlyRemovedObjectIDs = HotstartHeuristics.process(
            session: session, frameIndex: frameIndex, reverse: reverse,
            detectionToMatchedTrackIDs: association.detectionToMatchedTrackIDs,
            newObjectIDs: plan.newObjectIDs,
            emptyTrackIDs: association.emptyTrackIDs,
            unmatchedTrackIDs: association.unmatchedTrackIDs,
            parameters: parameters)

        // Reconditioning, on a fixed cadence and only when there is something to
        // recondition against.
        var reconditionedMasks: [Int: ReconditionSource] = [:]
        let shouldRecondition =
            parameters.reconditionEveryNthFrame > 0
            && frameIndex % parameters.reconditionEveryNthFrame == 0
            && !association.trackIDToHighConfidenceDetection.isEmpty
        if shouldRecondition {
            (reconditionedMasks, plan.reconditionedObjectIDs) = prepareReconditionMasks(
                session: session, detections: detections,
                trackerScoreLogits: trackerScoreLogits,
                candidates: association.trackIDToHighConfidenceDetection)
        }

        // Memory encoding for this frame, from the de-overlapped masks.
        if !objectIDsSnapshot.isEmpty {
            if parameters.suppressOverlappingOcclusionThreshold > 0 {
                OcclusionSuppressor.suppressRecentlyOccluded(
                    logits: &trackerLogits, masks: trackMasks, objectIDs: objectIDsSnapshot,
                    promptIDs: trackPromptIDs,
                    newlyRemovedObjectIDs: plan.newlyRemovedObjectIDs,
                    frameIndex: frameIndex, reverse: reverse, session: session,
                    parameters: parameters)
            }
            try await updateMemories(
                session: session, frameIndex: frameIndex, trackerLogits: trackerLogits,
                reconditionedMasks: reconditionedMasks, promptIDs: trackPromptIDs)
        }

        // New objects inherit their detection score. Removed ones are pushed far negative
        // and kept in the map, which is what keeps upstream's output assembly uniform.
        var newScores: [Int: Float] = [:]
        for (objectID, detectionIndex) in zip(plan.newObjectIDs, newIndices) {
            let score = detections.scores[detectionIndex]
            session.scoreByObjectID[objectID] = score
            newScores[objectID] = score
        }
        if let maximum = plan.newObjectIDs.max() {
            session.maxObjectID = max(session.maxObjectID, maximum)
        }
        for objectID in plan.newlyRemovedObjectIDs {
            session.scoreByObjectID[objectID] = -1e4
            newScores[objectID] = -1e4
            session.lastOccludedByObjectID[objectID] = nil
        }
        return Planning(plan: plan, newScores: newScores)
    }

    /// Where a reconditioned object's memory mask comes from.
    ///
    /// `trackerMask` resolves at encode time rather than capture time. Occlusion suppression
    /// mutates that tensor in place in between, so a copy taken beforehand would reinstate a
    /// mask upstream blanks.
    private enum ReconditionSource {
        case trackerMask
        case detectionMask([Float])
    }

    /// Port of `_prepare_recondition_masks`.
    ///
    /// The flag reads backwards from its name: `reconditionOnTrkMasks == true` reinforces
    /// memory with the tracker's mask, false overwrites it with the detection's.
    private func prepareReconditionMasks(
        session: VideoInferenceSession,
        detections: MergedDetections,
        trackerScoreLogits: [Float],
        candidates: [Int: Int]
    ) -> ([Int: ReconditionSource], Set<Int>) {
        var masks: [Int: ReconditionSource] = [:]
        var reconditioned: Set<Int> = []
        for (trackID, detectionIndex) in candidates.sorted(by: { $0.key < $1.key }) {
            guard let objectIndex = session.registry.existingIndex(of: trackID) else { continue }
            // Upstream compares the raw `tracker_obj_scores_global` logit against a
            // probability-shaped threshold, which admits any object with a positive score.
            guard objectIndex < trackerScoreLogits.count,
                trackerScoreLogits[objectIndex] > parameters.highConfThresh
            else { continue }

            if parameters.reconditionOnTrkMasks {
                masks[objectIndex] = .trackerMask
            } else {
                // Upstream stores `det_mask >= 0.5` as a bool tensor here, which becomes
                // 0.0/1.0 when the memory encoder casts it to float.
                masks[objectIndex] = .detectionMask(
                    detections.maskLogits[detectionIndex].map { $0 >= 0.5 ? 1 : 0 })
            }
            reconditioned.insert(trackID)
        }
        return (masks, reconditioned)
    }

    /// Port of `_tracker_update_memories`.
    private func updateMemories(
        session: VideoInferenceSession,
        frameIndex: Int,
        trackerLogits: [[Float]],
        reconditionedMasks: [Int: ReconditionSource],
        promptIDs: [Int]
    ) async throws {
        var masks = trackerLogits
        for (objectIndex, source) in reconditionedMasks {
            // `.trackerMask` resolves against the post-suppression tracker logits. See
            // `ReconditionSource`.
            if case .detectionMask(let mask) = source { masks[objectIndex] = mask }
            // A reconditioned object's frame becomes a conditioning frame, which changes
            // both what the memory bank may select and which pointers are eligible.
            session.histories[objectIndex].promoteToConditioning(frame: frameIndex)
        }
        OcclusionSuppressor.suppressAreaShrinkage(logits: &masks, promptIDs: promptIDs)
        try await timed(VideoSegmentationEngine.Function.memoryEncode) {
            try await tracker.encodeFinalMemories(
                session: session, frameIndex: frameIndex, maskLogits: masks)
        }
    }

    // MARK: - Execution phase

    /// Port of `run_tracker_update_execution_phase`.
    private func execute(
        session: VideoInferenceSession,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool,
        detections: MergedDetections,
        plan: TrackerUpdatePlan
    ) async throws {
        if !plan.newDetectionIndices.isEmpty {
            try await timed(VideoSegmentationEngine.Function.trackerMaskInit) {
                try await tracker.addNewObjects(
                    session: session, frameIndex: frameIndex, totalFrames: totalFrames,
                    newObjectIDs: plan.newObjectIDs,
                    newObjectMaskLogits: plan.newDetectionIndices.map { detections.maskLogits[$0] },
                    reverse: reverse)
            }
        }
        // Sorted so removal order is deterministic, and with it the index renumbering.
        // Upstream iterates a set and reaches the same end state in arbitrary order.
        for objectID in plan.newlyRemovedObjectIDs.sorted() {
            session.removeObject(objectID)
        }
    }

    // MARK: - Output assembly

    /// Port of `build_outputs` plus the metadata bookkeeping at the end of `forward`.
    ///
    /// Both zips below run the post-execution object list against pre-execution arrays and
    /// truncate at the shorter, matching upstream's indexing.
    private func buildOutputs(
        session: VideoInferenceSession,
        frameIndex: Int,
        detections: MergedDetections,
        trackerLogits: [[Float]],
        trackerScoreLogits: [Float],
        plan: TrackerUpdatePlan,
        newScores: [Int: Float]
    ) -> RawFrameOutput {
        var maskByObjectID: [Int: [Float]] = [:]
        let currentObjectIDs = session.objectIDs

        for (objectID, mask) in zip(currentObjectIDs, trackerLogits) {
            maskByObjectID[objectID] = mask
        }

        // New objects show their detection mask, ahead of the tracker's own output for the
        // frame that seeded them.
        for (objectID, detectionIndex) in zip(plan.newObjectIDs, plan.newDetectionIndices) {
            var mask = detections.maskLogits[detectionIndex]
            fillHoles(&mask)
            maskByObjectID[objectID] = mask
        }

        // Reconditioned objects are overridden by the detection that reconditioned them,
        // whichever mode `reconditionOnTrkMasks` selected for memory.
        for objectID in plan.reconditionedObjectIDs.sorted() {
            guard let detectionIndex = plan.trackIDToHighConfidenceDetection[objectID] else {
                continue
            }
            maskByObjectID[objectID] = detections.maskLogits[detectionIndex]
        }

        // Tracker scores for the frame, as probabilities. New objects first, then the
        // tracker's own scores, matching upstream's update order.
        var trackerScores = session.trackerScoreByFrame[frameIndex] ?? [:]
        for (objectID, score) in newScores { trackerScores[objectID] = score }
        for (objectID, logit) in zip(currentObjectIDs, trackerScoreLogits) {
            trackerScores[objectID] = DetectionDecoder.sigmoid(logit)
        }
        session.trackerScoreByFrame[frameIndex] = trackerScores

        // Hotstart hides removed objects retroactively. That only works while the output is
        // delayed long enough for the decision to precede the display.
        if parameters.hotstartEnabled {
            session.hotstartRemovedObjectIDs.formUnion(session.removedObjectIDs)
        }

        // Keyed off the mask map: as upstream does, a removed object keeps its entry for
        // this frame and the postprocessor hides it.
        return RawFrameOutput(
            frameIndex: frameIndex,
            maskLogitsByObjectID: maskByObjectID,
            scoreByObjectID: session.scoreByObjectID,
            trackerScoreByObjectID: trackerScores,
            suppressedObjectIDs: session.suppressedObjectIDsByFrame[frameIndex] ?? [])
    }

    // MARK: - Timing

    @discardableResult
    private func timed<T>(_ name: String, _ body: () async throws -> T) async rethrows -> T {
        let started = ContinuousClock.now
        let result = try await body()
        timings[name, default: 0] += (ContinuousClock.now - started).inSeconds
        return result
    }
}
