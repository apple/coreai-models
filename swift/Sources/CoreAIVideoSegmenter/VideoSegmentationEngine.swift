// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// The seven Core AI entrypoints of a SAM 3 video asset, behind typed calls.
///
/// The single point of contact with `AIModel`. Everything above it is host logic: the
/// inference session, the memory ring buffer, and the tracking heuristics.
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
    /// In-flight load. Concurrent callers suspended at `prepare` share one asset instead of
    /// each preparing their own.
    private var loadTask: Task<Void, Error>?

    private struct Loaded {
        let model: AIModel
        let functions: [String: InferenceFunction]
        let descriptors: [String: InferenceFunctionDescriptor]
    }

    /// Shapes read off the asset at load time, so the host packs to what was traced.
    public struct Shapes: Sendable, Equatable {
        /// Square input resolution of `image_encode`.
        public let imageSize: Int
        /// Text token count `text_encode` was traced at.
        public let textSequenceLength: Int
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
    /// first `loadFunction`, so loading all seven here costs no extra memory.
    public func loadResources() async throws {
        guard loaded == nil else { return }
        if let loadTask { return try await loadTask.value }
        let task = Task { try await performLoad() }
        loadTask = task
        do {
            try await task.value
        } catch {
            loadTask = nil
            throw error
        }
        loadTask = nil
    }

    private func performLoad() async throws {
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
        // A concurrent `unloadResources` cancels this task, and committing afterwards would
        // silently undo it.
        try Task.checkCancellation()
        self.loaded = Loaded(
            model: prepared.model, functions: functions, descriptors: descriptors)
        self.shapes = try Self.resolveShapes(descriptors)
    }

    public func unloadResources() async {
        loadTask?.cancel()
        loadTask = nil
        loaded = nil
        shapes = nil
    }

    /// Run every entrypoint once on zeros, moving kernel compilation off the first frame.
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
        let outputs = try await invoke(
            Function.imageEncode,
            ["pixel_values": try input(Function.imageEncode, "pixel_values", pixelValues)])
        return try outputs("last_hidden_state")
    }

    /// CLIP text tower. Run once per distinct prompt for the whole video.
    ///
    /// Returns the attention mask alongside the features, since `detect` needs the same mask
    /// on every frame.
    public func textEncode(
        inputIDs: [Int32], attentionMask: [Int32]
    ) async throws -> PromptEncoding {
        let mask = try input(Function.textEncode, "attention_mask", attentionMask)
        let outputs = try await invoke(
            Function.textEncode,
            [
                "input_ids": try input(Function.textEncode, "input_ids", inputIDs),
                "attention_mask": mask,
            ])
        return PromptEncoding(textFeatures: try outputs("text_features"), attentionMask: mask)
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
            predictedMasks: try outputs("pred_masks"),
            predictedBoxes: try outputs("pred_boxes"),
            predictedLogits: try outputs("pred_logits"),
            presenceLogits: try outputs("presence_logits"))
    }

    /// Tracker FPN neck plus the two pre-projected decoder levels.
    public func trackerEncode(lastHiddenState: NDArray) async throws -> TrackerFeatures {
        let outputs = try await invoke(
            Function.trackerEncode, ["last_hidden_state": lastHiddenState])
        return TrackerFeatures(
            level0: try outputs("vision_feat_0"),
            level1: try outputs("vision_feat_1"),
            level2: try outputs("vision_feat_2"),
            positionLevel2: try outputs("vision_pos_2"))
    }

    /// Memory attention plus the SAM mask decoder, for one object on one frame.
    ///
    /// Internal because `PackedMemory` is. ``MemoryBankPacker`` owns that layout.
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
                "spatial_valid": memory.spatialSlotOccupancy,
                "object_pointers": memory.objectPointers,
                "ptr_tpos": memory.pointerTemporalPosition,
                "ptr_valid": memory.pointerSlotOccupancy,
            ])
        return TrackerStepOutputs(
            predictedMasks: try outputs("pred_masks"),
            highResolutionMasks: try outputs("high_res_masks"),
            objectPointer: try outputs("object_pointer"),
            objectScoreLogits: try outputs("object_score_logits"))
    }

    /// Encode one predicted mask into a spatial memory slot.
    ///
    /// `maskLogits` must already be at `shapes.memoryMaskSize`, since a traced graph accepts
    /// one resolution where upstream `_encode_new_memory` resizes whatever it is handed.
    ///
    /// `binarize` reproduces `is_mask_from_pts`, which upstream computes as `any(...)` over
    /// the batch, so one newly seeded object turns it on for every object on that frame.
    public func memoryEncode(
        visionFeatureLevel2: NDArray,
        maskLogits: [Float],
        objectScoreLogit: Float,
        binarize: Bool
    ) async throws -> EncodedMemory {
        let function = Function.memoryEncode
        let outputs = try await invoke(
            function,
            [
                "vision_feat_2": visionFeatureLevel2,
                "mask_logits": try input(function, "mask_logits", maskLogits),
                "object_score_logits": try input(
                    function, "object_score_logits", [objectScoreLogit]),
                "binarize_mask": try input(function, "binarize_mask", [binarize ? 1 : 0] as [Float]),
            ])
        return EncodedMemory(
            features: try outputs("maskmem_features"),
            positionEncoding: try outputs("maskmem_pos_enc"))
    }

    /// Seed a track from a detection mask, producing only its object pointer.
    ///
    /// The rest of `_use_mask_as_output` is weight-free and stays on the host, in
    /// `TrackerLoop.maskAsOutput`.
    public func trackerMaskInit(
        features: TrackerFeatures, maskInput: [Float]
    ) async throws -> NDArray {
        let function = Function.trackerMaskInit
        let outputs = try await invoke(
            function,
            [
                "vision_feat_0": features.level0,
                "vision_feat_1": features.level1,
                "vision_feat_2": features.level2,
                "mask_input": try input(function, "mask_input", maskInput),
            ])
        return try outputs("object_pointer")
    }

    /// A zero-filled input array for `function`'s `input`, for the memory packer to fill.
    public func makeInput(for function: String, named input: String) throws -> NDArray {
        NDArray(descriptor: try arrayDescriptor(function, input: input))
    }

    // MARK: - Invocation

    /// One call's outputs, carrying the function name so a lookup can report it.
    private struct FunctionOutputs {
        let function: String
        let arrays: [String: NDArray]

        func callAsFunction(_ name: String) throws -> NDArray {
            guard let array = arrays[name] else {
                throw VideoSegmentationError.missingOutput(function: function, name: name)
            }
            return array
        }
    }

    private func invoke(
        _ name: String, _ inputs: [String: NDArray]
    ) async throws -> FunctionOutputs {
        let state = try require()
        let descriptor = state.descriptors[name]!
        try validate(name, inputs, against: descriptor)
        var raw = try await state.functions[name]!.run(inputs: inputs)
        var arrays: [String: NDArray] = [:]
        for output in descriptor.outputNames {
            if let array = raw.remove(output)?.ndArray {
                arrays[output] = array
            }
        }
        return FunctionOutputs(function: name, arrays: arrays)
    }

    /// A zero-filled array shaped for `function`'s `input`, then filled with `values`.
    private func input(_ function: String, _ name: String, _ values: [Float]) throws -> NDArray {
        var array = NDArray(descriptor: try arrayDescriptor(function, input: name))
        fillFloatNDArray(&array, with: values)
        return array
    }

    private func input(_ function: String, _ name: String, _ values: [Int32]) throws -> NDArray {
        var array = NDArray(descriptor: try arrayDescriptor(function, input: name))
        fillNDArray(&array, as: Int32.self, with: values)
        return array
    }

    /// Reject shape mismatches before they reach the runtime.
    ///
    /// A static Core AI function handed a wrongly shaped input SIGKILLs the process with no
    /// traceback, so the check has to happen here.
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
    /// Everything comes from the descriptors, so a 336 "lite" variant reports its own grid
    /// and the packer follows. What this checks is agreement between entrypoints. That
    /// mismatch SIGKILLs at run time.
    private static func resolveShapes(
        _ descriptors: [String: InferenceFunctionDescriptor]
    ) throws -> Shapes {
        func checked(_ shape: [Int], _ rank: Int?, _ what: String) throws -> [Int] {
            if let rank, shape.count != rank {
                throw VideoSegmentationError.unsupportedGeometry(
                    "\(what) has rank \(shape.count) \(shape); this runtime expects rank \(rank).")
            }
            return shape
        }
        func shape(_ function: String, input: String, rank: Int? = nil) throws -> [Int] {
            guard let descriptor = descriptors[function],
                case .ndArray(let array) = descriptor.inputDescriptor(of: input)
            else {
                throw VideoSegmentationError.unsupportedGeometry(
                    "\(function) has no array input '\(input)'.")
            }
            return try checked(array.shape, rank, "\(function) input '\(input)'")
        }
        func outputShape(_ function: String, _ name: String, rank: Int? = nil) throws -> [Int] {
            guard let descriptor = descriptors[function],
                case .ndArray(let array) = descriptor.outputDescriptor(of: name)
            else {
                throw VideoSegmentationError.unsupportedGeometry(
                    "\(function) has no array output '\(name)'.")
            }
            return try checked(array.shape, rank, "\(function) output '\(name)'")
        }

        let pixelValues = try shape(Function.imageEncode, input: "pixel_values", rank: 4)
        guard pixelValues[2] == pixelValues[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "image_encode expects a square [1, 3, S, S] input; got \(pixelValues).")
        }
        let inputIDs = try shape(Function.textEncode, input: "input_ids", rank: 2)
        let predictedMasks = try outputShape(Function.detect, "pred_masks", rank: 4)
        let spatialMemory = try shape(Function.trackerStep, input: "spatial_memory", rank: 4)
        let objectPointers = try shape(Function.trackerStep, input: "object_pointers", rank: 3)
        let highResolution = try outputShape(Function.trackerStep, "high_res_masks", rank: 4)
        let memoryMask = try shape(Function.memoryEncode, input: "mask_logits", rank: 4)

        guard memoryMask[2] == memoryMask[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "memory_encode expects a square mask input; got \(memoryMask).")
        }
        guard predictedMasks[2] == predictedMasks[3] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "detect expects square masks; got \(predictedMasks).")
        }

        let shapes = Shapes(
            imageSize: pixelValues[2],
            textSequenceLength: inputIDs[1],
            lowResMaskSize: predictedMasks[2],
            memoryMaskSize: memoryMask[2],
            highResMaskSize: highResolution[2],
            memoryTokenCount: spatialMemory[1],
            memoryDim: spatialMemory[3],
            hiddenDim: objectPointers[2],
            spatialSlots: spatialMemory[0],
            ptrSlots: objectPointers[0])

        // Cross-entrypoint agreement. The tracker's low-res mask must line up with the
        // detector's for association to compare them.
        let trackerLowRes = try outputShape(Function.trackerStep, "pred_masks", rank: 4)
        guard trackerLowRes[2] == shapes.lowResMaskSize else {
            throw VideoSegmentationError.unsupportedGeometry(
                "tracker_step emits \(trackerLowRes[2])px masks but detect emits "
                    + "\(shapes.lowResMaskSize)px; association compares the two.")
        }
        let maskInit = try shape(Function.trackerMaskInit, input: "mask_input", rank: 4)
        guard maskInit[2] == shapes.lowResMaskSize else {
            throw VideoSegmentationError.unsupportedGeometry(
                "tracker_mask_init takes \(maskInit[2])px masks but detections are "
                    + "\(shapes.lowResMaskSize)px.")
        }
        // `DetectionDecoder` thresholds the flattened logits and then slices `pred_masks` by
        // the surviving indices, so one logit per mask is what keeps that slice in range.
        let predictedLogits = try outputShape(Function.detect, "pred_logits")
        guard predictedLogits.reduce(1, *) == predictedMasks[1] else {
            throw VideoSegmentationError.unsupportedGeometry(
                "detect emits pred_logits\(predictedLogits) but \(predictedMasks[1]) masks; "
                    + "the decoder indexes one into the other.")
        }
        let scoreInput = try shape(Function.memoryEncode, input: "object_score_logits")
        let scoreOutput = try outputShape(Function.trackerStep, "object_score_logits")
        guard scoreInput == scoreOutput else {
            throw VideoSegmentationError.unsupportedGeometry(
                "memory_encode takes object_score_logits\(scoreInput) but tracker_step emits "
                    + "\(scoreOutput).")
        }
        let encodedMemory = try outputShape(Function.memoryEncode, "maskmem_features", rank: 3)
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
/// `pred_masks` is 66 MB as `Float` at 200 queries, so callers read the logits first and
/// pull only the surviving mask slices.
public struct DetectOutputs: Sendable {
    public let predictedMasks: NDArray
    public let predictedBoxes: NDArray
    public let predictedLogits: NDArray
    public let presenceLogits: NDArray
}

/// The `tracker_encode` outputs downstream reads. The graph also emits `vision_pos_0` and
/// `vision_pos_1`, which no entrypoint consumes.
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
