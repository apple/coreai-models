// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import Foundation

/// Runs the tracker over every object on one frame, and encodes the resulting memories.
///
/// Port of `Sam3TrackerVideoModel.forward`, `_run_single_frame_inference`,
/// `_use_mask_as_output`, and `_batch_encode_memories`.
///
/// Mask prompts only: SAM 3's video path seeds a track from a detection mask, in
/// `_tracker_add_new_objects`.
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

    /// Scratch shared by all four resamplers: only one is ever in use at a time.
    private var scratch: [Float]

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
        self.scratch = [Float](
            repeating: 0,
            count: max(
                lowResToImage.scratchCount, imageToLowRes.scratchCount,
                highResToMemory.scratchCount, lowResToMemory.scratchCount))
    }

    /// Resample through the shared scratch, so the only per-object allocation left is the
    /// destination the caller goes on to keep.
    private func resample(
        _ resampler: BilinearResampler, _ source: [Float], into destination: inout [Float]
    ) {
        resampler.resample(source, into: &destination, scratch: &scratch)
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

    /// The weight-free part of `_use_mask_as_output`. The object pointer comes separately,
    /// from `tracker_mask_init`.
    private struct MaskAsOutput {
        var predictedMasks: [Float]
        var highResolutionMasks: [Float]
        var objectScoreLogit: Float
    }

    /// Propagate every registered object through `frameIndex`.
    ///
    /// - Parameter runMemoryEncoder: Encode this frame's memory as part of the pass. False
    ///   on the plain propagation call, which defers memory until the planning phase has
    ///   resolved non-overlap. True when a new object was just seeded.
    func propagate(
        session: VideoInferenceSession,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool,
        runMemoryEncoder: Bool
    ) async throws -> Propagation {
        // Upstream mutates its `reverse` argument inside the object loop. The change carries
        // to every later object in the same call, so a freshly seeded object forces forward
        // tracking for the objects after it. Matched here.
        var reverse = reverse

        var propagation = Propagation(
            maskLogits: Array(repeating: [], count: session.objectCount),
            objectScoreLogits: Array(repeating: 0, count: session.objectCount))

        var memoryInputs: [MemoryEncodingInput] = []

        for objectIndex in 0..<session.objectCount {
            let objectID = session.registry.id(at: objectIndex)
            let hasNewInputs = session.objectsWithNewInputs.contains(objectID)
            let hasConditioningOutput =
                session.histories[objectIndex].conditioning[frameIndex] != nil

            if !hasNewInputs, hasConditioningOutput,
                let stored = session.histories[objectIndex].conditioning[frameIndex]
            {
                // Already computed on this frame as a conditioning output, so reuse it.
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
                memoryInputs.append(
                    MemoryEncodingInput(
                        objectIndex: objectIndex,
                        mask: result.highResolutionMasks,
                        scoreLogit: result.objectScoreLogit,
                        fromMask: maskPrompt != nil))
            }

            if !isInitialConditioningFrame {
                session.histories[objectIndex].framesTracked[frameIndex] = reverse
            }
        }

        try await encodeMemories(
            session: session, frameIndex: frameIndex, inputs: memoryInputs)

        return propagation
    }

    /// Encode memory for every object from the frame's final, de-overlapped masks.
    ///
    /// Port of `_tracker_update_memories`'s second half. Unlike the pass inside ``propagate``
    /// the masks are low-resolution, the score comes from mask area, and binarization is
    /// off.
    func encodeFinalMemories(
        session: VideoInferenceSession,
        frameIndex: Int,
        maskLogits: [[Float]]
    ) async throws {
        guard !maskLogits.isEmpty else { return }
        // Mask area stands in for an object score, exactly as upstream:
        // `torch.where((high_res_masks > 0).any(...), 10.0, -10.0)`.
        let inputs = maskLogits.enumerated().map { objectIndex, logits in
            MemoryEncodingInput(
                objectIndex: objectIndex,
                mask: logits,
                scoreLogit: logits.contains { $0 > 0 } ? -Self.maskOutBias : Self.maskOutBias,
                fromMask: false)
        }
        try await encodeMemories(
            session: session, frameIndex: frameIndex, inputs: inputs,
            sourceIsLowResolution: true)
    }

    /// Seed tracks from detection masks, then re-run the tracker with memory encoding on.
    ///
    /// Port of `_tracker_add_new_objects`. Upstream's re-run covers every object, which also
    /// overwrites the memory the planning phase just wrote. This is the expensive step.
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
            // `>= 0.5` on raw mask logits, expressed through the bitset's strict `>` by
            // stepping the threshold one ULP down.
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
            // Seeding path: only the object pointer needs the network. The rest of
            // `_use_mask_as_output` is weight-free arithmetic on the prompt mask.
            let maskFloats = maskPrompt.mask.toBytes().map { Float($0) }
            let pointer = try await engine.trackerMaskInit(
                features: features, maskInput: maskFloats)
            let derived = maskAsOutput(maskPrompt.mask, maskFloats: maskFloats)
            // Consumed, so drop it. A long video accumulates one prompt per seeding.
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
    private func maskAsOutput(_ mask: MaskBitset, maskFloats: [Float]) -> MaskAsOutput {
        var highResolution = [Float](repeating: 0, count: highResPixels)
        resample(lowResToImage, maskFloats, into: &highResolution)
        var scale = Self.maskOutScale
        var bias = Self.maskOutBias
        highResolution.withUnsafeMutableBufferPointer {
            if let base = $0.baseAddress {
                vDSP_vsmsa(base, 1, &scale, &bias, base, 1, vDSP_Length($0.count))
            }
        }
        var lowResolution = [Float](repeating: 0, count: lowResPixels)
        resample(imageToLowRes, highResolution, into: &lowResolution)
        // `is_obj_appearing` tests the prompt mask, ahead of the resample.
        let appearing = !mask.isEmpty
        return MaskAsOutput(
            predictedMasks: lowResolution,
            highResolutionMasks: highResolution,
            objectScoreLogit: appearing ? -Self.maskOutBias : Self.maskOutBias)
    }

    // MARK: - Memory encoding

    /// One object's contribution to a frame's memory-encoding pass.
    private struct MemoryEncodingInput {
        let objectIndex: Int
        let mask: [Float]
        let scoreLogit: Float
        let fromMask: Bool
    }

    /// Encode one memory per object and attach it to that object's stored frame output.
    ///
    /// `binarize` is `any(fromMask)` across the batch: upstream computes `is_mask_from_pts`
    /// once for the whole batch. That coupling moves the numbers, so it is matched here.
    private func encodeMemories(
        session: VideoInferenceSession,
        frameIndex: Int,
        inputs: [MemoryEncodingInput],
        sourceIsLowResolution: Bool = false
    ) async throws {
        guard !inputs.isEmpty else { return }
        let features = try session.features(forFrame: frameIndex)
        let binarize = inputs.contains { $0.fromMask }
        let resampler = sourceIsLowResolution ? lowResToMemory : highResToMemory

        for input in inputs {
            // Upstream resizes inside `_encode_new_memory`, at whichever of two resolutions
            // the caller supplies. A traced graph takes one, so the host normalizes first.
            var mask = [Float](
                repeating: 0, count: shapes.memoryMaskSize * shapes.memoryMaskSize)
            resample(resampler, input.mask, into: &mask)

            let encoded = try await engine.memoryEncode(
                visionFeatureLevel2: features.level2,
                maskLogits: mask,
                objectScoreLogit: input.scoreLogit,
                binarize: binarize)

            session.histories[input.objectIndex].attachMemory(
                features: MemoryPayload(reading: encoded.features),
                positionEncoding: MemoryPayload(reading: encoded.positionEncoding),
                at: frameIndex)
        }
    }
}
