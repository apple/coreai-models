// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// The seven Core AI entrypoints of a SAM 3 video asset, behind typed calls.
///
/// Everything above this type is host logic: the inference session, the memory ring
/// buffer, and the tracking heuristics. This is the only place that touches `AIModel`.
///
/// Two deliberate choices in the signatures below:
///
/// * Tensors that go straight from one graph to the next stay as `NDArray` handles rather
///   than being flattened to `[Float]`. `vision_feat_0` alone is 2.6 M elements and is fed
///   to `tracker_step` once per tracked object per frame, so a host round-trip would be
///   repeated for every object.
/// * `tracker_encode` also emits `vision_pos_0` and `vision_pos_1`, which no other
///   entrypoint consumes. They are left unread.
public actor VideoSegmentationEngine: ResourceManaging {
    /// Entrypoint names, in the order the exporter declares them.
    public enum Function {
        public static let imageEncode = "image_encode"
        public static let textEncode = "text_encode"
        public static let detect = "detect"
        public static let trackerEncode = "tracker_encode"
        public static let trackerStep = "tracker_step"
        public static let memoryEncode = "memory_encode"
        public static let trackerMaskInit = "tracker_mask_init"

        static let all = [
            imageEncode, textEncode, detect, trackerEncode, trackerStep, memoryEncode,
            trackerMaskInit,
        ]
    }

    private let modelURL: URL
    private var loaded: Loaded?

    private struct Loaded {
        let model: AIModel
        let functions: [String: InferenceFunction]
        let descriptors: [String: InferenceFunctionDescriptor]
    }

    /// Shapes read off the asset at load time, so the host packs to what was traced
    /// rather than to what it assumes.
    public struct Shapes: Sendable, Equatable {
        /// Square input resolution of `image_encode`.
        public let imageSize: Int
        /// Text token count `text_encode` was traced at.
        public let textSequenceLength: Int
        /// Detector query count.
        public let queryCount: Int
        /// Side of the detector's and tracker's low-resolution masks.
        public let lowResMaskSize: Int
        /// Side of the `memory_encode` mask input.
        public let memoryMaskSize: Int
        /// Side of `tracker_step`'s high-resolution mask output.
        public let highResMaskSize: Int
        /// Flattened spatial extent of the deepest feature level (`H * W`).
        public let memoryTokenCount: Int
        /// Channel width of a spatial memory slot.
        public let memoryDim: Int
        /// Tracker hidden width, and the width of an object pointer.
        public let hiddenDim: Int
        /// Spatial memory slots per object.
        public let spatialSlots: Int
        /// Object-pointer slots per object.
        public let ptrSlots: Int
    }

    public private(set) var shapes: Shapes?

    public init(modelURL: URL) {
        self.modelURL = modelURL
    }

    /// Load and specialize the asset, then resolve its shapes.
    ///
    /// A static multi-function asset commits workspace for every declared function on the
    /// first `loadFunction`, so all seven are resolved here rather than on demand; deferring
    /// them would cost the same memory and add a first-use stall.
    public func loadResources() async throws {
        guard loaded == nil else { return }
        let prepared = try await PreparedModel.prepare(at: modelURL)
        guard prepared.structure == .videoSegmenter else {
            throw VideoSegmentationError.invalidConfiguration(
                "\(modelURL.lastPathComponent) classified as \(prepared.structure), not a video "
                    + "segmenter. Its functions are: \(prepared.model.functionNames.sorted()).")
        }

        var functions: [String: InferenceFunction] = [:]
        var descriptors: [String: InferenceFunctionDescriptor] = [:]
        for name in Function.all {
            guard let descriptor = prepared.model.functionDescriptor(for: name),
                let function = try prepared.model.loadFunction(named: name)
            else {
                throw VideoSegmentationError.missingFunction(
                    name: name, available: prepared.model.functionNames)
            }
            functions[name] = function
            descriptors[name] = descriptor
        }
        self.loaded = Loaded(
            model: prepared.model, functions: functions, descriptors: descriptors)
        self.shapes = try Self.resolveShapes(descriptors)
    }

    public func unloadResources() async {
        loaded = nil
        shapes = nil
    }

    /// Run every entrypoint once on zeros so the first real frame isn't paying for
    /// kernel compilation.
    public func warmup() async throws {
        let state = try require()
        for name in Function.all {
            var inputs: [String: NDArray] = [:]
            for input in state.descriptors[name]!.inputNames {
                inputs[input] = NDArray(descriptor: try arrayDescriptor(name, input: input))
            }
            _ = try await invoke(name, inputs)
        }
    }

    // MARK: - Entrypoints

    /// ViT backbone. `pixelValues` is planar CHW at `shapes.imageSize`, already normalized.
    public func imageEncode(pixelValues: [Float]) async throws -> NDArray {
        var array = NDArray(descriptor: try arrayDescriptor(Function.imageEncode, input: "pixel_values"))
        fillFloatNDArray(&array, with: pixelValues)
        let outputs = try await invoke(Function.imageEncode, ["pixel_values": array])
        return try output(outputs, "last_hidden_state", from: Function.imageEncode)
    }

    /// CLIP text tower. Run once per distinct prompt for the whole video.
    ///
    /// Returns the attention mask alongside the features because `detect` needs the same
    /// mask on every frame, so there is no point rebuilding it per frame per prompt.
    public func textEncode(
        inputIDs: [Int32], attentionMask: [Int32]
    ) async throws -> PromptEncoding {
        var ids = NDArray(descriptor: try arrayDescriptor(Function.textEncode, input: "input_ids"))
        fillNDArray(&ids, as: Int32.self, with: inputIDs)
        var mask = NDArray(
            descriptor: try arrayDescriptor(Function.textEncode, input: "attention_mask"))
        fillNDArray(&mask, as: Int32.self, with: attentionMask)

        let outputs = try await invoke(
            Function.textEncode, ["input_ids": ids, "attention_mask": mask])
        return PromptEncoding(
            textFeatures: try output(outputs, "text_features", from: Function.textEncode),
            attentionMask: mask)
    }

    /// FPN + DETR + mask decoder, for one prompt.
    public func detect(
        lastHiddenState: NDArray, prompt: PromptEncoding
    ) async throws -> DetectOutputs {
        let outputs = try await invoke(
            Function.detect,
            [
                "last_hidden_state": lastHiddenState,
                "text_features": prompt.textFeatures,
                "attention_mask": prompt.attentionMask,
            ])
        return DetectOutputs(
            predictedMasks: try output(outputs, "pred_masks", from: Function.detect),
            predictedBoxes: try output(outputs, "pred_boxes", from: Function.detect),
            predictedLogits: try output(outputs, "pred_logits", from: Function.detect),
            presenceLogits: try output(outputs, "presence_logits", from: Function.detect))
    }

    /// Tracker FPN neck plus the two pre-projected decoder levels.
    public func trackerEncode(lastHiddenState: NDArray) async throws -> TrackerFeatures {
        let outputs = try await invoke(
            Function.trackerEncode, ["last_hidden_state": lastHiddenState])
        return TrackerFeatures(
            level0: try output(outputs, "vision_feat_0", from: Function.trackerEncode),
            level1: try output(outputs, "vision_feat_1", from: Function.trackerEncode),
            level2: try output(outputs, "vision_feat_2", from: Function.trackerEncode),
            positionLevel2: try output(outputs, "vision_pos_2", from: Function.trackerEncode))
    }

    /// Memory attention plus the SAM mask decoder, for one object on one frame.
    ///
    /// Internal because `PackedMemory` is: the memory bank's layout is a consequence of
    /// how this asset was traced, not something a caller should be assembling.
    func trackerStep(
        features: TrackerFeatures, memory: PackedMemory
    ) async throws -> TrackerStepOutputs {
        let outputs = try await invoke(
            Function.trackerStep,
            [
                "vision_feat_0": features.level0,
                "vision_feat_1": features.level1,
                "vision_feat_2": features.level2,
                "vision_pos_2": features.positionLevel2,
                "spatial_memory": memory.spatialMemory,
                "spatial_memory_pos": memory.spatialMemoryPosition,
                "spatial_tpos_idx": memory.spatialTemporalIndex,
                "spatial_valid": memory.spatialValid,
                "object_pointers": memory.objectPointers,
                "ptr_tpos": memory.pointerTemporalPosition,
                "ptr_valid": memory.pointerValid,
            ])
        return TrackerStepOutputs(
            predictedMasks: try output(outputs, "pred_masks", from: Function.trackerStep),
            highResolutionMasks: try output(outputs, "high_res_masks", from: Function.trackerStep),
            objectPointer: try output(outputs, "object_pointer", from: Function.trackerStep),
            objectScoreLogits: try output(
                outputs, "object_score_logits", from: Function.trackerStep))
    }

    /// Encode one predicted mask into a spatial memory slot.
    ///
    /// `maskLogits` must already be at `shapes.memoryMaskSize`. Upstream `_encode_new_memory`
    /// resizes whatever it is handed, and it is handed two different resolutions depending
    /// on the caller; a traced graph accepts one, so normalizing is the host's job.
    ///
    /// `binarize` reproduces `is_mask_from_pts`, which upstream computes as `any(...)` over
    /// the batch, so one newly seeded object turns it on for every object encoded on that
    /// frame.
    public func memoryEncode(
        visionFeatureLevel2: NDArray,
        maskLogits: [Float],
        objectScoreLogit: Float,
        binarize: Bool
    ) async throws -> EncodedMemory {
        var mask = NDArray(descriptor: try arrayDescriptor(Function.memoryEncode, input: "mask_logits"))
        fillFloatNDArray(&mask, with: maskLogits)
        var score = NDArray(
            descriptor: try arrayDescriptor(Function.memoryEncode, input: "object_score_logits"))
        fillFloatNDArray(&score, with: [objectScoreLogit])
        var flag = NDArray(
            descriptor: try arrayDescriptor(Function.memoryEncode, input: "binarize_mask"))
        fillFloatNDArray(&flag, with: [binarize ? 1 : 0])

        let outputs = try await invoke(
            Function.memoryEncode,
            [
                "vision_feat_2": visionFeatureLevel2,
                "mask_logits": mask,
                "object_score_logits": score,
                "binarize_mask": flag,
            ])
        return EncodedMemory(
            features: try output(outputs, "maskmem_features", from: Function.memoryEncode),
            positionEncoding: try output(outputs, "maskmem_pos_enc", from: Function.memoryEncode))
    }

    /// Seed a track from a detection mask, producing only its object pointer.
    ///
    /// The rest of `_use_mask_as_output` is weight-free and stays on the host; see
    /// `TrackerLoop.maskAsOutput`.
    public func trackerMaskInit(
        features: TrackerFeatures, maskInput: [Float]
    ) async throws -> NDArray {
        var mask = NDArray(
            descriptor: try arrayDescriptor(Function.trackerMaskInit, input: "mask_input"))
        fillFloatNDArray(&mask, with: maskInput)
        let outputs = try await invoke(
            Function.trackerMaskInit,
            [
                "vision_feat_0": features.level0,
                "vision_feat_1": features.level1,
                "vision_feat_2": features.level2,
                "mask_input": mask,
            ])
        return try output(outputs, "object_pointer", from: Function.trackerMaskInit)
    }

    /// A zero-filled input array for `function`'s `input`, for the memory packer to fill.
    public func makeInput(for function: String, named input: String) throws -> NDArray {
        NDArray(descriptor: try arrayDescriptor(function, input: input))
    }

    // MARK: - Invocation

    private func invoke(_ name: String, _ inputs: [String: NDArray]) async throws
        -> [String: NDArray]
    {
        let state = try require()
        try validate(name, inputs, against: state.descriptors[name]!)
        var raw = try await state.functions[name]!.run(inputs: inputs)
        var outputs: [String: NDArray] = [:]
        for output in state.descriptors[name]!.outputNames {
            if let array = raw.remove(output)?.ndArray {
                outputs[output] = array
            }
        }
        return outputs
    }

    /// Reject shape mismatches before they reach the runtime.
    ///
    /// A static Core AI function handed a wrongly shaped input does not raise: it kills the
    /// process with SIGKILL and no traceback. The descriptors are right here, so the check is
    /// cheap insurance against a failure that is otherwise very hard to debug.
    private func validate(
        _ name: String, _ inputs: [String: NDArray], against descriptor: InferenceFunctionDescriptor
    ) throws {
        for input in descriptor.inputNames {
            guard let array = inputs[input] else {
                throw VideoSegmentationError.invalidConfiguration(
                    "\(name): missing input '\(input)'.")
            }
            guard case .ndArray(let expected) = descriptor.inputDescriptor(of: input) else {
                throw VideoSegmentationError.invalidConfiguration(
                    "\(name): input '\(input)' is not an array.")
            }
            if array.shape != expected.shape {
                throw VideoSegmentationError.shapeMismatch(
                    function: name, input: input, expected: expected.shape, actual: array.shape)
            }
        }
        for input in inputs.keys where !descriptor.inputNames.contains(input) {
            throw VideoSegmentationError.invalidConfiguration(
                "\(name): unexpected input '\(input)'. Expected \(descriptor.inputNames.sorted()).")
        }
    }

    private func require() throws -> Loaded {
        guard let loaded else {
            throw VideoSegmentationError.invalidConfiguration(
                "Engine resources are not loaded; call loadResources() first.")
        }
        return loaded
    }

    private func output(
        _ outputs: [String: NDArray], _ name: String, from function: String
    ) throws -> NDArray {
        guard let array = outputs[name] else {
            throw VideoSegmentationError.missingOutput(function: function, name: name)
        }
        return array
    }

    private func arrayDescriptor(_ function: String, input: String) throws -> NDArrayDescriptor {
        let state = try require()
        guard let functionDescriptor = state.descriptors[function],
            case .ndArray(let descriptor) = functionDescriptor.inputDescriptor(of: input)
        else {
            throw VideoSegmentationError.invalidConfiguration(
                "\(function) has no array input named '\(input)'.")
        }
        return descriptor
    }

    // MARK: - Shape resolution

    /// Read every geometric constant the host needs off the traced descriptors.
    ///
    /// Nothing here is hardcoded from the 1008 export: a 336 "lite" variant would report its
    /// own grid and the packer would follow. What is checked is internal agreement, because a
    /// mismatch between two entrypoints is the failure mode that SIGKILLs.
    private static func resolveShapes(
        _ descriptors: [String: InferenceFunctionDescriptor]
    ) throws -> Shapes {
        func shape(_ function: String, input: String) throws -> [Int] {
            guard let descriptor = descriptors[function],
                case .ndArray(let array) = descriptor.inputDescriptor(of: input)
            else {
                throw VideoSegmentationError.unsupportedGeometry(
                    "\(function) has no array input '\(input)'.")
            }
            return array.shape
        }
        func outputShape(_ function: String, _ name: String) throws -> [Int] {
            guard let descriptor = descriptors[function],
                case .ndArray(let array) = descriptor.outputDescriptor(of: name)
            else {
                throw VideoSegmentationError.unsupportedGeometry(
                    "\(function) has no array output '\(name)'.")
            }
            return array.shape
        }

        let pixelValues = try shape(Function.imageEncode, input: "pixel_values")
        guard pixelValues.count == 4, pixelValues[2] == pixelValues[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "image_encode expects a square [1, 3, S, S] input; got \(pixelValues).")
        }
        let inputIDs = try shape(Function.textEncode, input: "input_ids")
        let predictedLogits = try outputShape(Function.detect, "pred_logits")
        let predictedMasks = try outputShape(Function.detect, "pred_masks")
        let spatialMemory = try shape(Function.trackerStep, input: "spatial_memory")
        let objectPointers = try shape(Function.trackerStep, input: "object_pointers")
        let highResolution = try outputShape(Function.trackerStep, "high_res_masks")
        let memoryMask = try shape(Function.memoryEncode, input: "mask_logits")

        guard spatialMemory.count == 4 else {
            throw VideoSegmentationError.unsupportedGeometry(
                "tracker_step expects [slots, HW, 1, mem_dim] spatial memory; got \(spatialMemory).")
        }
        guard memoryMask.count == 4, memoryMask[2] == memoryMask[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "memory_encode expects a square mask input; got \(memoryMask).")
        }
        guard predictedMasks.count == 4, predictedMasks[2] == predictedMasks[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "detect expects square masks; got \(predictedMasks).")
        }

        let shapes = Shapes(
            imageSize: pixelValues[2],
            textSequenceLength: inputIDs[1],
            queryCount: predictedLogits[1],
            lowResMaskSize: predictedMasks[2],
            memoryMaskSize: memoryMask[2],
            highResMaskSize: highResolution[2],
            memoryTokenCount: spatialMemory[1],
            memoryDim: spatialMemory[3],
            hiddenDim: objectPointers[2],
            spatialSlots: spatialMemory[0],
            ptrSlots: objectPointers[0])

        // Cross-entrypoint agreement. The tracker's low-res mask must line up with the
        // detector's for association to compare them, and `memory_encode`'s score input has
        // a trailing singleton that is easy to get wrong.
        let trackerLowRes = try outputShape(Function.trackerStep, "pred_masks")
        guard trackerLowRes[2] == shapes.lowResMaskSize else {
            throw VideoSegmentationError.unsupportedGeometry(
                "tracker_step emits \(trackerLowRes[2])px masks but detect emits "
                    + "\(shapes.lowResMaskSize)px; association compares the two.")
        }
        let maskInit = try shape(Function.trackerMaskInit, input: "mask_input")
        guard maskInit[2] == shapes.lowResMaskSize else {
            throw VideoSegmentationError.unsupportedGeometry(
                "tracker_mask_init takes \(maskInit[2])px masks but detections are "
                    + "\(shapes.lowResMaskSize)px.")
        }
        let scoreInput = try shape(Function.memoryEncode, input: "object_score_logits")
        let scoreOutput = try outputShape(Function.trackerStep, "object_score_logits")
        guard scoreInput == scoreOutput else {
            throw VideoSegmentationError.unsupportedGeometry(
                "memory_encode takes object_score_logits\(scoreInput) but tracker_step emits "
                    + "\(scoreOutput).")
        }
        let encodedMemory = try outputShape(Function.memoryEncode, "maskmem_features")
        guard encodedMemory[0] == shapes.memoryTokenCount, encodedMemory[2] == shapes.memoryDim
        else {
            throw VideoSegmentationError.unsupportedGeometry(
                "memory_encode emits \(encodedMemory) but a tracker_step slot is "
                    + "[\(shapes.memoryTokenCount), 1, \(shapes.memoryDim)].")
        }
        return shapes
    }
}

// MARK: - Output bundles

/// A prompt's encoded text plus the attention mask `detect` needs alongside it.
public struct PromptEncoding: Sendable {
    public let textFeatures: NDArray
    public let attentionMask: NDArray
}

/// `detect` outputs, kept as device arrays.
///
/// `pred_masks` is `[1, queries, 288, 288]`, 66 MB as `Float` at 200 queries. Callers read
/// the logits first and pull only the surviving mask slices.
public struct DetectOutputs: Sendable {
    public let predictedMasks: NDArray
    public let predictedBoxes: NDArray
    public let predictedLogits: NDArray
    public let presenceLogits: NDArray
}

/// The four `tracker_encode` outputs anything downstream actually reads.
public struct TrackerFeatures: Sendable {
    public let level0: NDArray
    public let level1: NDArray
    public let level2: NDArray
    public let positionLevel2: NDArray
}

public struct TrackerStepOutputs: Sendable {
    public let predictedMasks: NDArray
    public let highResolutionMasks: NDArray
    public let objectPointer: NDArray
    public let objectScoreLogits: NDArray
}

public struct EncodedMemory: Sendable {
    public let features: NDArray
    public let positionEncoding: NDArray
}
