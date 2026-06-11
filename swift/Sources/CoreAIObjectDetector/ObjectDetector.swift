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

    /// Warm up the backend (e.g. trigger Metal kernel compilation) with a dummy pass.
    public func warmup() async throws {
        guard case .ndArray(let imageDescriptor) = functionDescriptor.inputDescriptor(of: imageInputName) else {
            throw DetectionRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(imageInputName)'"
            )
        }
        let defaults = DetectionParameters()
        let warmupShape = zip(imageDescriptor.shape, [1, 3, defaults.inputHeight, defaults.inputWidth])
            .map { actual, fallback in actual >= 0 ? actual : fallback }
        let resolved = imageDescriptor.resolvingDynamicDimensions(warmupShape)
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
    /// 2. Preprocess each image sequentially into a `[3, H, W]` Float buffer.
    /// 3. Concatenate the per-image buffers into the `[B, 3, H, W]` input
    ///    NDArray and run a single forward pass.
    /// 4. Slice each batch slot from the outputs and decode independently,
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
            imageSizes: images.map { CGSize(width: $0.width, height: $0.height) },
            parameters: parameters
        )

        // 1. Preprocess each input image (sequential).
        let perImagePixels = try preprocessImages(images, plan: plan, parameters: parameters)

        // 2. Build batched NDArray and run inference once.
        let resolvedDescriptor = imageDescriptor.resolvingDynamicDimensions(
            [plan.batch, 3, plan.height, plan.width])
        let imageArray = try buildInputNDArray(descriptor: resolvedDescriptor, perImagePixels: perImagePixels)

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

    /// Sequentially preprocess each image to a `[3 * H * W]` Float buffer at
    /// the plan's target spatial dimensions.
    private func preprocessImages(
        _ images: [CGImage], plan: BatchPlan, parameters: DetectionParameters
    ) throws -> [[Float]] {
        let preprocessor = ImagePreprocessor(
            targetSize: CGSize(width: plan.width, height: plan.height),
            mean: parameters.normalizationMeans,
            std: parameters.normalizationStds,
            rescaleFactor: 1.0
        )
        return try images.map { try preprocessor.preprocessCHW(cgImage: $0) }
    }

    /// Build the input NDArray for a `[B, 3, H, W]` resolved descriptor by
    /// concatenating per-image CHW buffers in batch order. Each per-image
    /// entry is `3*H*W` floats; the buffers are written contiguously to match
    /// row-major batch-leading layout.
    private func buildInputNDArray(
        descriptor: NDArrayDescriptor, perImagePixels: [[Float]]
    ) throws -> NDArray {
        var imageArray = NDArray(descriptor: descriptor)
        let flat = Array(perImagePixels.joined())
        if descriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            fillNDArray(&imageArray, as: Float16.self, with: flat.map(Float16.init))
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            fillNDArray(&imageArray, as: Float.self, with: flat)
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
    /// list of input image sizes, and the user's parameter overrides.
    ///
    /// Resolution rules:
    /// - **Batch**: always `images.count`. A static-batch model must match.
    /// - **Spatial dims**: a dynamic `-1` dim is filled from
    ///   `parameters.inputHeight` / `inputWidth`. A static dim is taken
    ///   from the model descriptor (the parameters' values are ignored for
    ///   that axis).
    static func planBatch(
        expectedShape: [Int],
        imageSizes: [CGSize],
        parameters: DetectionParameters
    ) throws -> BatchPlan {
        guard !imageSizes.isEmpty else {
            throw DetectionRuntimeError.invalidConfiguration("planBatch requires at least one image")
        }

        // Resolve batch from image count; verify it matches a static batch dim.
        let targetBatch = imageSizes.count
        let batchExpected = expectedShape[0]
        if batchExpected >= 0 && batchExpected != targetBatch {
            throw DetectionRuntimeError.invalidConfiguration(
                "Model expects fixed batch=\(batchExpected) but caller supplied \(targetBatch) image(s)"
            )
        }

        let heightExpected = expectedShape[2]
        let widthExpected = expectedShape[3]
        let height = heightExpected < 0 ? parameters.inputHeight : heightExpected
        let width = widthExpected < 0 ? parameters.inputWidth : widthExpected

        return BatchPlan(batch: targetBatch, height: height, width: width)
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
