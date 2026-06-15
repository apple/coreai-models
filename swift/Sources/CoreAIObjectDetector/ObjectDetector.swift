// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

// MARK: - ObjectDetector

/// Core AI-backed object detector.
public struct ObjectDetector {
    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    private let imageInputName: String
    private let logitsOutputName: String
    private let boxesOutputName: String

    /// Loads the `.aimodel` at `path` and initializes a detector.
    public init(resourcesAt path: String) async throws {
        let modelURL = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            modelURL.pathExtension == "aimodel"
        else {
            throw DetectionRuntimeError.modelNotFound(modelURL.path)
        }

        let model = try await AIModel(contentsOf: modelURL)

        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Cannot find 'main' function in model"
            )
        }

        // Discover input names
        guard let imageInputName = Self.findImageInputName(in: descriptor.inputNames) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Cannot find image input in model. Inputs: \(descriptor.inputNames)"
            )
        }

        // Discover output names
        guard let logitsOutputName = Self.findLogitsOutputName(in: descriptor.outputNames) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Cannot find logits output in model. Outputs: \(descriptor.outputNames)"
            )
        }
        guard let boxesOutputName = Self.findBoxesOutputName(in: descriptor.outputNames) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Cannot find boxes output in model. Outputs: \(descriptor.outputNames)"
            )
        }

        guard case .ndArray = descriptor.outputDescriptor(of: logitsOutputName) else {
            throw DetectionRuntimeError.outputMissing(logitsOutputName)
        }
        guard case .ndArray = descriptor.outputDescriptor(of: boxesOutputName) else {
            throw DetectionRuntimeError.outputMissing(boxesOutputName)
        }

        guard let fn = try model.loadFunction(named: "main") else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Cannot load 'main' function from model"
            )
        }

        self.function = fn
        self.functionDescriptor = descriptor
        self.imageInputName = imageInputName
        self.logitsOutputName = logitsOutputName
        self.boxesOutputName = boxesOutputName
    }

    // MARK: - Inference

    /// Warm up the backend (e.g. trigger Metal kernel compilation) with a dummy
    /// pass at the same `(B, H, W)` that subsequent `detect()` calls will use.
    /// For static-shape models the arguments are ignored — `planBatch` falls
    /// back to the descriptor's fixed dims.
    public func warmup(imageCount: Int = 1, parameters: DetectionParameters = .default) async throws {
        guard case .ndArray(let imageDescriptor) = functionDescriptor.inputDescriptor(of: imageInputName) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(imageInputName)'"
            )
        }
        let expectedShape = imageDescriptor.shape
        guard expectedShape.count == 4 else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Expected 4-dimensional input shape, got \(expectedShape.count)"
            )
        }
        let plan = try Self.planBatch(
            expectedShape: expectedShape,
            imageCount: imageCount,
            parameters: parameters
        )
        let resolved = imageDescriptor.resolvingDynamicDimensions(
            [plan.batch, 3, plan.height, plan.width])
        _ = try await function.run(inputs: [imageInputName: NDArray(descriptor: resolved)])
    }

    /// Detect objects in `image` using `.default` parameters.
    public func detect(image: CGImage) async throws -> [DetectedObject] {
        try await detect(image: image, parameters: .default)
    }

    /// Detect objects in `image` — convenience wrapper over the batched API.
    public func detect(image: CGImage, parameters: DetectionParameters) async throws -> [DetectedObject] {
        let results = try await detect(images: [image], parameters: parameters)
        return results.first ?? []
    }

    /// Detect objects in each of `images` using `.default` parameters.
    public func detect(images: [CGImage]) async throws -> [[DetectedObject]] {
        try await detect(images: images, parameters: .default)
    }

    /// Detect objects across `images` in a single batched forward pass.
    ///
    /// Pipeline:
    /// 1. Resolve a batch plan `(B, H, W)` from the model descriptor and
    ///    parameters. Batch is always `images.count`. Dynamic spatial dims
    ///    are filled from `parameters.inputHeight` / `inputWidth` (which
    ///    have struct-level defaults).
    /// 2. Allocate the `[B, 3, H, W]` input NDArray and preprocess each
    ///    image directly into its batch slot, then run a single forward pass.
    /// 3. Slice each batch slot from the outputs and decode independently,
    ///    returning `images.count` detection lists in input order.
    public func detect(images: [CGImage], parameters: DetectionParameters) async throws
        -> [[DetectedObject]]
    {
        guard !images.isEmpty else {
            throw DetectionRuntimeError.invalidConfiguration("detect requires at least one image")
        }
        guard case .ndArray(let imageDescriptor) = functionDescriptor.inputDescriptor(of: imageInputName) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(imageInputName)'"
            )
        }
        let expectedShape = imageDescriptor.shape
        guard expectedShape.count == 4 else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Expected 4-dimensional input shape, got \(expectedShape.count)"
            )
        }

        let plan = try Self.planBatch(
            expectedShape: expectedShape,
            imageCount: images.count,
            parameters: parameters
        )

        // 1. Allocate the batched input NDArray and write each image's
        //    preprocessed CHW pixels directly into its batch slot.
        let resolvedDescriptor = imageDescriptor.resolvingDynamicDimensions(
            [plan.batch, 3, plan.height, plan.width])
        let imageArray = try buildInputNDArray(
            images: images, plan: plan, descriptor: resolvedDescriptor, parameters: parameters)

        var outputs = try await function.run(inputs: [imageInputName: imageArray])
        guard let logitsArray = outputs.remove(logitsOutputName)?.ndArray,
            let boxesArray = outputs.remove(boxesOutputName)?.ndArray
        else {
            throw DetectionRuntimeError.invalidConfiguration(
                "Missing one or more outputs after run."
            )
        }

        // 3. Decode each input image's batch slot.
        return Self.decodePerImage(
            logitsArray: logitsArray,
            boxesArray: boxesArray,
            images: images,
            parameters: parameters
        )
    }

    // MARK: - Preprocessing

    /// Preprocess each image and write its `[3, H, W]` Float pixels directly
    /// into the corresponding batch slot of a freshly allocated `[B, 3, H, W]`
    /// NDArray. Avoids materializing a per-image `[Float]` array-of-arrays
    /// or a flattened `B*3*H*W` intermediate.
    private func buildInputNDArray(
        images: [CGImage],
        plan: BatchPlan,
        descriptor: NDArrayDescriptor,
        parameters: DetectionParameters
    ) throws -> NDArray {
        let preprocessor = ImagePreprocessor(
            targetSize: CGSize(width: plan.width, height: plan.height),
            mean: parameters.normalizationMeans,
            std: parameters.normalizationStds,
            rescaleFactor: 1.0
        )
        let slotCount = 3 * plan.height * plan.width
        var imageArray = NDArray(descriptor: descriptor)

        if descriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            var view = imageArray.mutableView(as: Float16.self)
            for (b, image) in images.enumerated() {
                let chw = try preprocessor.preprocessCHW(cgImage: image)
                view.withUnsafeMutablePointer { ptr, _, _ in
                    let slot = ptr.advanced(by: b * slotCount)
                    for i in 0..<slotCount { slot[i] = Float16(chw[i]) }
                }
            }
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            var view = imageArray.mutableView(as: Float.self)
            for (b, image) in images.enumerated() {
                let chw = try preprocessor.preprocessCHW(cgImage: image)
                view.withUnsafeMutablePointer { ptr, _, _ in
                    let slot = ptr.advanced(by: b * slotCount)
                    chw.withUnsafeBufferPointer { src in
                        slot.update(from: src.baseAddress!, count: slotCount)
                    }
                }
            }
        }
        return imageArray
    }

    // MARK: - Output decoding

    private static func decodePerImage(
        logitsArray: NDArray,
        boxesArray: NDArray,
        images: [CGImage],
        parameters: DetectionParameters
    ) -> [[DetectedObject]] {
        let logitsShape = logitsArray.shape  // [B, Q, C]
        let boxesShape = boxesArray.shape  // [B, Q, 4]
        let logitsAll = flattenAsFloat(logitsArray)
        let boxesAll = flattenAsFloat(boxesArray)
        let perBatchLog = logitsShape.dropFirst().reduce(1, *)
        let perBatchBox = boxesShape.dropFirst().reduce(1, *)
        let singleBatchLogitsShape = [1] + logitsShape.dropFirst()

        return images.enumerated().map { i, image in
            let raw = DetectionOutput(
                logits: Array(logitsAll[i * perBatchLog..<(i + 1) * perBatchLog]),
                logitsShape: singleBatchLogitsShape,
                predictedBoxes: Array(boxesAll[i * perBatchBox..<(i + 1) * perBatchBox])
            )
            return DetectionPostprocessor.decode(
                output: raw,
                inputSize: CGSize(width: image.width, height: image.height),
                parameters: parameters
            )
        }
    }

    // MARK: - Batch planning

    struct BatchPlan: Equatable {
        let batch: Int
        let height: Int
        let width: Int
    }

    /// Resolve the concrete `(B, H, W)` to bind the model with, given the
    /// model's expected shape (which may contain `-1` for dynamic dims), the
    /// number of input images, and the user's parameter overrides.
    ///
    /// Resolution rules:
    /// - **Batch**: always `imageCount`. A static-batch model must match.
    /// - **Spatial dims**: a dynamic `-1` dim is filled from
    ///   `parameters.inputHeight` / `inputWidth`. A static dim is taken
    ///   from the model descriptor (the parameters' values are ignored for
    ///   that axis).
    static func planBatch(
        expectedShape: [Int],
        imageCount: Int,
        parameters: DetectionParameters
    ) throws -> BatchPlan {
        guard imageCount >= 1 else {
            throw DetectionRuntimeError.invalidConfiguration("planBatch requires imageCount >= 1")
        }

        // Verify image count matches a static batch dim.
        let batchExpected = expectedShape[0]
        if batchExpected >= 0 && batchExpected != imageCount {
            throw DetectionRuntimeError.invalidConfiguration(
                "Model expects fixed batch=\(batchExpected) but caller supplied \(imageCount) image(s)"
            )
        }

        let heightExpected = expectedShape[2]
        let widthExpected = expectedShape[3]
        let height = heightExpected < 0 ? parameters.inputHeight : heightExpected
        let width = widthExpected < 0 ? parameters.inputWidth : widthExpected

        return BatchPlan(batch: imageCount, height: height, width: width)
    }

    // MARK: - Name Discovery

    static func findImageInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("pixel") || l.contains("image")
        }
    }

    static func findLogitsOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("logit") }
    }

    static func findBoxesOutputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("box")
        }
    }
}

// MARK: - Errors

/// Runtime errors thrown by the detection pipeline.
public enum DetectionRuntimeError: Error, LocalizedError, Sendable {
    case modelLoadFailed(String)
    case outputMissing(String)
    case invalidConfiguration(String)
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .outputMissing(let name):
            return "Expected output tensor missing: \(name)"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        case .modelNotFound(let path):
            return "No .aimodel directory at: \(path)"
        }
    }
}
