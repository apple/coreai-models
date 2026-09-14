// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Runs the tracker over every object on one frame, and encodes the resulting memories.
///
/// Port of `Sam3TrackerVideoModel.forward`, `_run_single_frame_inference`,
/// `_use_mask_as_output`, and `_batch_encode_memories`.
///
/// Point prompts are not implemented: SAM 3's video path only ever seeds a track from a
/// detection mask, in `_tracker_add_new_objects`.
@VideoSegmentationActor
final class TrackerLoop {
    private let engine: VideoSegmentationEngine
    private let shapes: VideoSegmentationEngine.Shapes
    private let parameters: VideoSegmentationParameters
    private let packer: MemoryBankPacker

    /// Resamplers are built once: their weight tables depend only on the sizes, and the
    /// same four resizes repeat for every object on every frame.
    private let lowResToImage: BilinearResampler
    private let imageToLowRes: BilinearResampler
    private let highResToMemory: BilinearResampler
    private let lowResToMemory: BilinearResampler

    /// Scale and bias `_use_mask_as_output` applies to turn a binary mask into logits.
    private static let maskOutScale: Float = 20
    private static let maskOutBias: Float = -10

    init(
        engine: VideoSegmentationEngine,
        shapes: VideoSegmentationEngine.Shapes,
        parameters: VideoSegmentationParameters,
        packer: MemoryBankPacker
    ) {
        self.engine = engine
        self.shapes = shapes
        self.parameters = parameters
        self.packer = packer
        self.lowResToImage = BilinearResampler(
            sourceWidth: shapes.lowResMaskSize, sourceHeight: shapes.lowResMaskSize,
            destinationWidth: shapes.highResMaskSize, destinationHeight: shapes.highResMaskSize,
            antialias: true)
        self.imageToLowRes = BilinearResampler(
            sourceWidth: shapes.highResMaskSize, sourceHeight: shapes.highResMaskSize,
            destinationWidth: shapes.lowResMaskSize, destinationHeight: shapes.lowResMaskSize,
            antialias: true)
        self.highResToMemory = BilinearResampler(
            sourceWidth: shapes.highResMaskSize, sourceHeight: shapes.highResMaskSize,
            destinationWidth: shapes.memoryMaskSize, destinationHeight: shapes.memoryMaskSize,
            antialias: true)
        self.lowResToMemory = BilinearResampler(
            sourceWidth: shapes.lowResMaskSize, sourceHeight: shapes.lowResMaskSize,
            destinationWidth: shapes.memoryMaskSize, destinationHeight: shapes.memoryMaskSize,
            antialias: true)
    }

    struct Propagation {
        /// Low-resolution mask logits per object, in registry order.
        var maskLogits: [[Float]] = []
        /// Object score logits per object, in registry order.
        var objectScoreLogits: [Float] = []
    }

    /// One object's fresh result, before it is stored.
    private struct SingleFrameResult {
        var predictedMasks: [Float]
        var highResolutionMasks: [Float]
        var objectPointer: NDArray
        var objectScoreLogit: Float
    }

    /// The weight-free part of `_use_mask_as_output`, which has no object pointer of its own;
    /// that comes from `tracker_mask_init`.
    private struct MaskAsOutput {
        var predictedMasks: [Float]
        var highResolutionMasks: [Float]
        var objectScoreLogit: Float
    }

    /// Propagate every registered object through `frameIndex`.
    ///
    /// - Parameter runMemoryEncoder: Encode this frame's memory as part of the pass. False
    ///   on the plain propagation call, where memory is deferred until the planning phase
    ///   has resolved non-overlap; true when a new object was just seeded.
    func propagate(
        session: VideoInferenceSession,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool,
        runMemoryEncoder: Bool
    ) async throws -> Propagation {
        // Upstream mutates its `reverse` argument inside the object loop and the change
        // leaks into every later object in the same call. Reproduced deliberately: a
        // freshly seeded object forces forward tracking, and that then applies to the
        // objects after it.
        var reverse = reverse

        var propagation = Propagation(
            maskLogits: Array(repeating: [], count: session.objectCount),
            objectScoreLogits: Array(repeating: 0, count: session.objectCount))

        var memoryIndices: [Int] = []
        var memoryMasks: [[Float]] = []
        var memoryScores: [Float] = []
        var memoryFromMask: [Bool] = []

        for objectIndex in 0..<session.objectCount {
            let objectID = session.registry.id(at: objectIndex)
            let hasNewInputs = session.objectsWithNewInputs.contains(objectID)
            let hasConditioningOutput =
                session.histories[objectIndex].conditioning[frameIndex] != nil

            if !hasNewInputs, hasConditioningOutput,
                let stored = session.histories[objectIndex].conditioning[frameIndex]
            {
                // Already computed on this frame as a conditioning output, so reuse it
                // rather than re-running the tracker.
                guard let masks = stored.predictedMasks else {
                    throw VideoSegmentationError.invalidConfiguration(
                        "Object \(objectID) has a conditioning output on frame \(frameIndex) with "
                            + "no stored mask. Pruning ran on the frame still being processed.")
                }
                propagation.maskLogits[objectIndex] = masks
                propagation.objectScoreLogits[objectIndex] = stored.objectScoreLogit
                continue
            }

            var isInitialConditioningFrame = false
            var maskPrompt: MaskPrompt?
            if hasNewInputs {
                isInitialConditioningFrame =
                    session.histories[objectIndex].framesTracked[frameIndex] == nil
                if isInitialConditioningFrame { reverse = false }
                maskPrompt = session.histories[objectIndex].maskInputs[frameIndex]
                if maskPrompt != nil {
                    session.objectsWithNewInputs.removeAll { $0 == objectID }
                }
            }

            let result = try await runSingleFrame(
                session: session, frameIndex: frameIndex, objectIndex: objectIndex,
                totalFrames: totalFrames, maskPrompt: maskPrompt, reverse: reverse)

            session.histories[objectIndex].store(
                StoredFrameOutput(
                    predictedMasks: result.predictedMasks,
                    objectPointer: result.objectPointer,
                    objectScoreLogit: result.objectScoreLogit),
                at: frameIndex,
                conditioning: isInitialConditioningFrame)

            propagation.maskLogits[objectIndex] = result.predictedMasks
            propagation.objectScoreLogits[objectIndex] = result.objectScoreLogit

            if runMemoryEncoder, parameters.numMaskmem > 0 {
                memoryIndices.append(objectIndex)
                memoryMasks.append(result.highResolutionMasks)
                memoryScores.append(result.objectScoreLogit)
                memoryFromMask.append(maskPrompt != nil)
            }

            if !isInitialConditioningFrame {
                session.histories[objectIndex].framesTracked[frameIndex] = reverse
            }
        }

        try await encodeMemories(
            session: session, frameIndex: frameIndex, objectIndices: memoryIndices,
            highResolutionMasks: memoryMasks, scoreLogits: memoryScores,
            fromMask: memoryFromMask, alreadyAtMemoryResolution: false)

        return propagation
    }

    /// Encode memory for every object from the frame's final, de-overlapped masks.
    ///
    /// Port of `_tracker_update_memories`'s second half. Distinct from the pass inside
    /// ``propagate`` in three ways: the masks are low-resolution rather than
    /// image-resolution, the score is derived from mask area rather than from the decoder,
    /// and binarization is always off.
    func encodeFinalMemories(
        session: VideoInferenceSession,
        frameIndex: Int,
        maskLogits: [[Float]]
    ) async throws {
        guard !maskLogits.isEmpty else { return }
        // Mask area stands in for an object score, exactly as upstream:
        // `torch.where((high_res_masks > 0).any(...), 10.0, -10.0)`.
        let scores = maskLogits.map { logits in
            logits.contains { $0 > 0 } ? -Self.maskOutBias : Self.maskOutBias
        }
        try await encodeMemories(
            session: session, frameIndex: frameIndex,
            objectIndices: Array(maskLogits.indices),
            highResolutionMasks: maskLogits, scoreLogits: scores,
            fromMask: [Bool](repeating: false, count: maskLogits.count),
            alreadyAtMemoryResolution: false, sourceIsLowResolution: true)
    }

    /// Seed tracks from detection masks, then re-run the tracker with memory encoding on.
    ///
    /// Port of `_tracker_add_new_objects`. Two things here are upstream's and expensive: the
    /// binarization is `>= 0.5` on raw mask logits rather than on probabilities, and the re-run
    /// covers every object, not just the new ones, which also overwrites the memory the
    /// planning phase just wrote, now conditioned on this frame.
    func addNewObjects(
        session: VideoInferenceSession,
        frameIndex: Int,
        totalFrames: Int,
        newObjectIDs: [Int],
        newObjectMaskLogits: [[Float]],
        reverse: Bool
    ) async throws {
        for (objectID, logits) in zip(newObjectIDs, newObjectMaskLogits) {
            let objectIndex = session.index(ofObject: objectID)
            // `>= 0.5` on raw mask logits. `nextDown` turns the bitset's strict `>` into
            // exactly that, with no float slop.
            let mask = MaskBitset(
                thresholding: logits, width: shapes.lowResMaskSize,
                height: shapes.lowResMaskSize, above: Float(0.5).nextDown)
            session.histories[objectIndex].maskInputs[frameIndex] = MaskPrompt(mask: mask)
        }
        session.objectsWithNewInputs = newObjectIDs

        _ = try await propagate(
            session: session, frameIndex: frameIndex, totalFrames: totalFrames,
            reverse: reverse, runMemoryEncoder: true)
    }

    // MARK: - One object, one frame

    private var lowResPixels: Int { shapes.lowResMaskSize * shapes.lowResMaskSize }
    private var highResPixels: Int { shapes.highResMaskSize * shapes.highResMaskSize }

    private func runSingleFrame(
        session: VideoInferenceSession,
        frameIndex: Int,
        objectIndex: Int,
        totalFrames: Int,
        maskPrompt: MaskPrompt?,
        reverse: Bool
    ) async throws -> SingleFrameResult {
        let features = try session.features(forFrame: frameIndex)

        if let maskPrompt {
            // Seeding path: only the object pointer needs the network, and everything else
            // in `_use_mask_as_output` is weight-free arithmetic on the prompt mask.
            let maskFloats = maskPrompt.mask.toBytes().map { Float($0) }
            let pointer = try await engine.trackerMaskInit(
                features: features, maskInput: maskFloats)
            let derived = maskAsOutput(maskPrompt.mask, maskFloats: maskFloats)
            // Consumed; drop it so a long video doesn't accumulate one prompt per seeding.
            session.histories[objectIndex].maskInputs[frameIndex] = nil
            return SingleFrameResult(
                predictedMasks: derived.predictedMasks,
                highResolutionMasks: derived.highResolutionMasks,
                objectPointer: pointer,
                objectScoreLogit: derived.objectScoreLogit)
        }

        let memory = try packer.pack(
            history: session.histories[objectIndex], objectIndex: objectIndex,
            frameIndex: frameIndex, totalFrames: totalFrames, reverse: reverse)
        let outputs = try await engine.trackerStep(features: features, memory: memory)
        return SingleFrameResult(
            predictedMasks: floatElements(outputs.predictedMasks, in: 0..<lowResPixels),
            highResolutionMasks: floatElements(outputs.highResolutionMasks, in: 0..<highResPixels),
            objectPointer: outputs.objectPointer,
            objectScoreLogit: flattenAsFloat(outputs.objectScoreLogits).first ?? Self.maskOutBias)
    }

    /// The weight-free half of `_use_mask_as_output`.
    ///
    /// The low-resolution output is computed for completeness but is effectively write-only
    /// during forward propagation: `build_outputs` shows a new object its detection mask, and
    /// the stored value is only re-read if the same frame is revisited. The image-resolution
    /// mask and the score are what the memory encoder consumes.
    private func maskAsOutput(_ mask: MaskBitset, maskFloats: [Float]) -> MaskAsOutput {
        let upscaled = lowResToImage.resample(maskFloats)
        var highResolution = [Float](repeating: 0, count: upscaled.count)
        for index in upscaled.indices {
            highResolution[index] = upscaled[index] * Self.maskOutScale + Self.maskOutBias
        }
        let lowResolution = imageToLowRes.resample(highResolution)
        // `is_obj_appearing` tests the *prompt* mask, not the resampled one.
        let appearing = !mask.isEmpty
        return MaskAsOutput(
            predictedMasks: lowResolution,
            highResolutionMasks: highResolution,
            objectScoreLogit: appearing ? -Self.maskOutBias : Self.maskOutBias)
    }

    // MARK: - Memory encoding

    /// Encode one memory per object and attach it to that object's stored frame output.
    ///
    /// `binarize` is `any(fromMask)` across the batch, not per object: upstream computes
    /// `is_mask_from_pts` once for the whole batch, so a single newly seeded object turns
    /// binarization on for every object encoded on that frame. That coupling changes the
    /// numbers, so it is preserved.
    private func encodeMemories(
        session: VideoInferenceSession,
        frameIndex: Int,
        objectIndices: [Int],
        highResolutionMasks: [[Float]],
        scoreLogits: [Float],
        fromMask: [Bool],
        alreadyAtMemoryResolution: Bool,
        sourceIsLowResolution: Bool = false
    ) async throws {
        guard !objectIndices.isEmpty else { return }
        let features = try session.features(forFrame: frameIndex)
        let binarize = fromMask.contains(true)
        let resampler = sourceIsLowResolution ? lowResToMemory : highResToMemory

        for (position, objectIndex) in objectIndices.enumerated() {
            // Upstream resizes inside `_encode_new_memory`, which is handed two different
            // resolutions depending on the caller. A traced graph takes one, so the host
            // normalizes to `memoryMaskSize` first. Both callers resize *up*, where
            // antialiasing is a no-op, so this stays a single plain bilinear pass.
            let mask =
                alreadyAtMemoryResolution
                ? highResolutionMasks[position]
                : resampler.resample(highResolutionMasks[position])

            let encoded = try await engine.memoryEncode(
                visionFeatureLevel2: features.level2,
                maskLogits: mask,
                objectScoreLogit: scoreLogits[position],
                binarize: binarize)

            // Attach to whichever bucket holds this frame. A reconditioned object was
            // promoted to conditioning a moment ago, so this cannot assume either.
            let memoryFeatures = MemoryPayload(reading: encoded.features)
            let positionEncoding = MemoryPayload(reading: encoded.positionEncoding)
            if session.histories[objectIndex].conditioning[frameIndex] != nil {
                session.histories[objectIndex].conditioning[frameIndex]?.memoryFeatures =
                    memoryFeatures
                session.histories[objectIndex].conditioning[frameIndex]?.memoryPositionEncoding =
                    positionEncoding
            } else if session.histories[objectIndex].nonConditioning[frameIndex] != nil {
                session.histories[objectIndex].nonConditioning[frameIndex]?.memoryFeatures =
                    memoryFeatures
                session.histories[objectIndex].nonConditioning[frameIndex]?
                    .memoryPositionEncoding = positionEncoding
            }
        }
    }
}
