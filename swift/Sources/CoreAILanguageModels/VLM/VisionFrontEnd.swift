// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreImage
import Foundation

/// The vision side of a vision-language model: image preprocessing, the vision encoder, and the
/// projector into the decoder's embedding space.
///
/// Owns the vision asset's functions (separate `encode_image` + `project`, or a single fused
/// `main`) and turns images or video frames into `InputEmbeddings`. The encoder's pixel input is
/// `[1, 3, H, W]`, or `[1, num_patches, patch_dim]` for a pre-patchified export, whose patch rows
/// the host builds (see `PixelInput`).
final class VisionFrontEnd {
    let visionConfig: VisionConfig

    // MARK: - Vision Model Handles

    private let visionFunction: InferenceFunction
    private let visionFunctionDescriptor: InferenceFunctionDescriptor
    private let projectFunction: InferenceFunction
    private let projectFunctionDescriptor: InferenceFunctionDescriptor
    private let visionProjectorFused: Bool
    private let pixelInput: PixelInput

    // MARK: - Image Preprocessor

    private let imagePreprocessor: ImagePreprocessor

    init(visionModel: PreparedModel, visionConfig: VisionConfig) throws {
        self.visionConfig = visionConfig

        // --- Vision pipeline ---
        // Vision model may have separate "encode_image"+"project" functions (internal export)
        // or a single fused "main" function (public export). Support both.

        let hasSeparateVision = visionModel.model.functionDescriptor(for: "encode_image") != nil
        self.visionProjectorFused = !hasSeparateVision
        if hasSeparateVision {
            guard let visionDesc = visionModel.model.functionDescriptor(for: "encode_image") else {
                throw InferenceRuntimeError.functionNotFound("encode_image")
            }
            self.visionFunctionDescriptor = visionDesc

            guard let visionFn = try visionModel.model.loadFunction(named: "encode_image") else {
                throw InferenceRuntimeError.functionNotFound("encode_image")
            }
            self.visionFunction = visionFn

            guard let projectDesc = visionModel.model.functionDescriptor(for: "project") else {
                throw InferenceRuntimeError.functionNotFound("project")
            }
            self.projectFunctionDescriptor = projectDesc

            guard let projectFn = try visionModel.model.loadFunction(named: "project") else {
                throw InferenceRuntimeError.functionNotFound("project")
            }
            self.projectFunction = projectFn
        } else {
            // Fused vision+projector as single "main" function
            guard let visionDesc = visionModel.model.functionDescriptor(for: "main") else {
                throw InferenceRuntimeError.functionNotFound(
                    "Vision model needs 'encode_image' or 'main' function")
            }
            self.visionFunctionDescriptor = visionDesc
            self.projectFunctionDescriptor = visionDesc

            guard let visionFn = try visionModel.model.loadFunction(named: "main") else {
                throw InferenceRuntimeError.functionNotFound("main (vision)")
            }
            self.visionFunction = visionFn
            self.projectFunction = visionFn
        }

        // Build image preprocessor from vision config normalization fields.
        let vc = visionConfig
        self.imagePreprocessor = ImagePreprocessor(
            targetSize: CGSize(width: vc.imageSize, height: vc.imageSize),
            mean: (CGFloat(vc.imageMean[0]), CGFloat(vc.imageMean[1]), CGFloat(vc.imageMean[2])),
            std: (CGFloat(vc.imageStd[0]), CGFloat(vc.imageStd[1]), CGFloat(vc.imageStd[2])),
            rescaleFactor: CGFloat(vc.rescaleFactor)
        )

        let pixelInputName = visionFunctionDescriptor.inputNames[0]
        guard case .ndArray(let pixelDesc) = visionFunctionDescriptor.inputDescriptor(of: pixelInputName)
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Cannot get descriptor for vision input '\(pixelInputName)'")
        }
        self.pixelInput = try PixelInput(declaredShape: pixelDesc.shape, visionConfig: visionConfig)
        CLILogger.log(
            "VLM vision input '\(pixelInputName)' \(pixelDesc.shape): "
                + (pixelInput.patchify == nil ? "CHW pixels" : "host patchify"))
    }

    // MARK: - Image Encoding

    /// Encode an image into embeddings suitable for injection into the LLM.
    ///
    /// Pipeline:
    /// 1. Resize to `visionConfig.imageSize`, normalize channels
    /// 2. Rearrange into patch rows when the encoder takes pre-patchified input
    /// 3. Run vision encoder (`encode_image`) to get patch features
    /// 4. Run projector (`project`) to map features to LLM hidden dimension
    /// 5. Return as `InputEmbeddings` with placeholder token positions
    func encodeImage(cgImage: CGImage) async throws -> InputEmbeddings {
        let encodeSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLM EncodeImage",
            details: "cgImage"
        )
        defer {
            InstrumentsProfiler.endCustomInterval(name: "CoreAIVLM EncodeImage", signpostID: encodeSignpost)
        }

        // Step 1: Preprocess image to CHW Float32
        let chwPixels = try preprocessImage(cgImage: cgImage)

        // Steps 2-4: Lay the pixels out as the encoder's input, run encode_image, then the
        // projector (skipped if fused with encoder)
        let projectedEmbeddings = try await visionForward(pixels: visionInput(fromCHW: chwPixels))

        // The image token positions will be determined during generate() when we know the
        // full token sequence. For now, record the expected token count.
        // Use a placeholder range; the actual positions are resolved at generate() time
        // by scanning for imageTokenId in the token sequence.
        let tokenCount = visionConfig.imageTokenCount
        let placeholderRange = 0..<tokenCount

        CLILogger.log("VLM encodeImage complete: \(tokenCount) embedding tokens")

        return try InputEmbeddings(
            embeddings: projectedEmbeddings,
            embeddingPositions: placeholderRange
        )
    }

    // MARK: - Video Encoding

    /// Encode video frames into concatenated embeddings.
    ///
    /// Each frame is processed independently through the vision encoder + projector.
    /// Embeddings are concatenated along the sequence dimension to produce a single
    /// `InputEmbeddings` with shape `[1, N * tokensPerFrame, hidden_dim]`.
    ///
    /// Frames are encoded sequentially because the GPU vision encoder cannot run
    /// multiple inference calls concurrently on the same model function.
    func encodeVideo(_ video: VideoInput) async throws -> InputEmbeddings {
        var frameEmbeddings: [InputEmbeddings] = []

        for try await frame in video.frames {
            let chwPixels = try preprocessImage(cgImage: frame.image)
            let projected = try await visionForward(pixels: visionInput(fromCHW: chwPixels))
            frameEmbeddings.append(
                try InputEmbeddings(embeddings: projected, embeddingPositions: 0..<projected.shape[1]))
            CLILogger.log("  - encoded video frame \(frameEmbeddings.count)", level: 2)
        }

        guard !frameEmbeddings.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("encodeVideo: no frames in video input")
        }

        let concatenated = try InputEmbeddings.concatenating(frameEmbeddings)
        CLILogger.log("VLM encodeVideo complete: \(frameEmbeddings.count) frames, \(concatenated.tokenCount) tokens")
        return concatenated
    }

    // MARK: - Encoder Stages

    /// Preprocess an image into the normalized CHW pixels `visionInput(fromCHW:)` consumes.
    func preprocessImage(cgImage: CGImage) throws -> [Float] {
        try imagePreprocessor.preprocessCHW(cgImage: cgImage, strategy: visionConfig.imageStrategy)
    }

    /// Turn CHW pixels into the buffer the encoder takes: the patch rows of a pre-patchified
    /// export, or the CHW pixels themselves.
    func visionInput(fromCHW pixels: [Float]) throws -> [Float] {
        try pixelInput.buffer(fromCHW: pixels)
    }

    /// Run the vision encoder, and the projector when it is separate, on a buffer from
    /// `visionInput(fromCHW:)`.
    func visionForward(pixels: [Float]) async throws -> NDArray {
        let encoderOutput = try await runVisionEncoder(pixels: pixels)
        return visionProjectorFused ? encoderOutput : try await runProjector(encoderOutput: encoderOutput)
    }

    /// Run the vision encoder on the encoder-ready pixel buffer.
    ///
    /// - Parameter pixels: Float32 array matching the encoder's pixel input: CHW `[3, H, W]`, or
    ///   patch rows from `visionInput(fromCHW:)`
    /// - Returns: NDArray of encoder hidden states
    private func runVisionEncoder(pixels: [Float]) async throws -> NDArray {
        let pixelInputName = visionFunctionDescriptor.inputNames[0]
        let featuresOutputName = visionFunctionDescriptor.outputNames[0]

        // Resolve pixel_values input descriptor
        guard case .ndArray(let pixelDesc) = visionFunctionDescriptor.inputDescriptor(of: pixelInputName)
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Cannot get descriptor for vision input '\(pixelInputName)'")
        }

        try pixelInput.validate(pixels, name: pixelInputName)
        let resolvedPixelDesc = pixelDesc.resolvingDynamicDimensions(pixelInput.shape)
        var pixelArray = NDArray(descriptor: resolvedPixelDesc)
        fillNDArray(&pixelArray, as: Float.self, with: pixels)

        // Resolve output descriptor
        guard case .ndArray(let featuresDesc) = visionFunctionDescriptor.outputDescriptor(of: featuresOutputName)
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Cannot get descriptor for vision output '\(featuresOutputName)'")
        }
        // Output shape is static (determined by model); resolve any dynamic dims
        let resolvedFeaturesDesc = featuresDesc.resolvingDynamicDimensions(
            featuresDesc.shape.map { $0 < 0 ? 1 : $0 })
        var featuresArray = NDArray(descriptor: resolvedFeaturesDesc)

        // Execute encode_image
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&featuresArray, for: featuresOutputName)

        _ = try await visionFunction.run(
            inputs: [pixelInputName: pixelArray],
            outputViews: consume outputViews
        )

        CLILogger.log("  - encode_image complete, output shape: \(featuresArray.shape)")
        return featuresArray
    }

    /// Run the projector to map vision features to LLM hidden dimension.
    ///
    /// - Parameter encoderOutput: NDArray from vision encoder
    /// - Returns: NDArray of projected embeddings, shape `[1, image_token_count, hidden_dim]`
    private func runProjector(encoderOutput: NDArray) async throws -> NDArray {
        let projInputName = projectFunctionDescriptor.inputNames[0]
        let projOutputName = projectFunctionDescriptor.outputNames[0]

        // Resolve output descriptor
        guard case .ndArray(let projOutDesc) = projectFunctionDescriptor.outputDescriptor(of: projOutputName)
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Cannot get descriptor for project output '\(projOutputName)'")
        }
        let resolvedProjOutDesc = projOutDesc.resolvingDynamicDimensions(
            projOutDesc.shape.map { $0 < 0 ? visionConfig.imageTokenCount : $0 })
        var projectedArray = NDArray(descriptor: resolvedProjOutDesc)

        // Execute project
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&projectedArray, for: projOutputName)

        _ = try await projectFunction.run(
            inputs: [projInputName: encoderOutput],
            outputViews: consume outputViews
        )

        CLILogger.log("  - project complete, output shape: \(projectedArray.shape)")
        return projectedArray
    }
}

// MARK: - Pixel Input

extension VisionFrontEnd {
    /// The encoder's pixel input, read from its declared shape: the shape it is bound at, and the
    /// rearrangement the host applies to CHW pixels to fill it.
    ///
    /// A rank-4 input is `[1, 3, H, W]` and takes the normalized CHW pixels unchanged. A rank-3
    /// input is `[1, num_patches, patch_dim]`, the form ANE-mappable exports declare because the
    /// in-graph patchify exceeds the Neural Engine's rank limit; the host builds its patch rows
    /// with `VisionPatchify`. Dynamic dimensions resolve to the `[1, 3, H, W]` image shape, and a
    /// fully static input is bound at its declared shape.
    struct PixelInput {
        /// The shape the input is bound at.
        let shape: [Int]
        /// The host-side patchify of a rank-3 input; nil when the input takes CHW pixels.
        let patchify: VisionPatchify?

        init(declaredShape: [Int], visionConfig: VisionConfig) throws {
            let imageSize = visionConfig.imageSize
            self.shape =
                declaredShape.contains(where: { $0 < 0 }) ? [1, 3, imageSize, imageSize] : declaredShape
            self.patchify =
                declaredShape.count == 3
                ? try VisionPatchify(
                    visionConfig: visionConfig, numPatches: declaredShape[1], patchDim: declaredShape[2])
                : nil
        }

        /// The number of floats a buffer bound at `shape` holds.
        var elementCount: Int { shape.reduce(1, *) }

        /// The buffer this input takes for normalized CHW pixels `[3, imageSize, imageSize]`.
        func buffer(fromCHW pixels: [Float]) throws -> [Float] {
            guard let patchify else { return pixels }
            return try patchify(chw: pixels)
        }

        /// Throws unless `pixels` fills the input exactly.
        func validate(_ pixels: [Float], name: String) throws {
            guard pixels.count == elementCount else {
                throw InferenceRuntimeError.invalidArgument(
                    "Vision input '\(name)' takes \(elementCount) floats \(shape), got \(pixels.count)")
            }
        }
    }
}
