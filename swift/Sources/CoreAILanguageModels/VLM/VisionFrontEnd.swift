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
/// `main`) and turns images or video frames into `InputEmbeddings`.
final class VisionFrontEnd {
    let visionConfig: VisionConfig

    // MARK: - Vision Model Handles

    private let visionFunction: InferenceFunction
    private let visionFunctionDescriptor: InferenceFunctionDescriptor
    private let projectFunction: InferenceFunction
    private let projectFunctionDescriptor: InferenceFunctionDescriptor
    private let visionProjectorFused: Bool

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
    }

    // MARK: - Image Encoding

    /// Encode an image into embeddings suitable for injection into the LLM.
    ///
    /// Pipeline:
    /// 1. Resize to `visionConfig.imageSize`, normalize channels
    /// 2. Run vision encoder (`encode_image`) to get patch features
    /// 3. Run projector (`project`) to map features to LLM hidden dimension
    /// 4. Return as `InputEmbeddings` with placeholder token positions
    func encodeImage(cgImage: CGImage) async throws -> InputEmbeddings {
        let encodeSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLM EncodeImage",
            details: "cgImage"
        )

        // Step 1: Preprocess image to CHW Float32
        let chwPixels = try imagePreprocessor.preprocessCHW(
            cgImage: cgImage, strategy: visionConfig.imageStrategy)

        // Step 2: Run encode_image
        let encoderOutput = try await runVisionEncoder(pixels: chwPixels)

        // Step 3: Run projector (skip if fused with encoder)
        let projectedEmbeddings =
            visionProjectorFused ? encoderOutput : try await runProjector(encoderOutput: encoderOutput)

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIVLM EncodeImage",
            signpostID: encodeSignpost
        )

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
        let tokensPerFrame = visionConfig.tokensPerFrame ?? visionConfig.imageTokenCount

        var frameEmbeddings: [NDArray] = []
        var frameIndex = 0

        for try await frame in video.frames {
            let chwPixels = try imagePreprocessor.preprocessCHW(
                cgImage: frame.image, strategy: visionConfig.imageStrategy)
            let encoderOutput = try await runVisionEncoder(pixels: chwPixels)
            let projected =
                visionProjectorFused ? encoderOutput : try await runProjector(encoderOutput: encoderOutput)
            frameEmbeddings.append(projected)
            frameIndex += 1
            CLILogger.log("  - encoded video frame \(frameIndex)", level: 2)
        }

        guard !frameEmbeddings.isEmpty else {
            throw InferenceRuntimeError.invalidArgument("encodeVideo: no frames in video input")
        }

        if frameEmbeddings.count == 1 {
            CLILogger.log("VLM encodeVideo complete: 1 frame, \(tokensPerFrame) tokens")
            return try InputEmbeddings(
                embeddings: frameEmbeddings[0],
                embeddingPositions: 0..<tokensPerFrame
            )
        }

        let totalTokens = frameEmbeddings.count * tokensPerFrame
        let hiddenDim = frameEmbeddings[0].shape[2]
        let scalarType = frameEmbeddings[0].scalarType

        var concatenated = NDArray(shape: [1, totalTokens, hiddenDim], scalarType: scalarType)
        let elementsPerFrame = tokensPerFrame * hiddenDim

        switch scalarType {
        case .float16, .bfloat16:
            var destView = concatenated.mutableView(as: Float16.self)
            destView.withUnsafeMutablePointer { destPtr, _, _ in
                for (i, embedding) in frameEmbeddings.enumerated() {
                    embedding.view(as: Float16.self).withUnsafePointer { srcPtr, _, _ in
                        (destPtr + i * elementsPerFrame).update(from: srcPtr, count: elementsPerFrame)
                    }
                }
            }
        case .float32:
            var destView = concatenated.mutableView(as: Float.self)
            destView.withUnsafeMutablePointer { destPtr, _, _ in
                for (i, embedding) in frameEmbeddings.enumerated() {
                    embedding.view(as: Float.self).withUnsafePointer { srcPtr, _, _ in
                        (destPtr + i * elementsPerFrame).update(from: srcPtr, count: elementsPerFrame)
                    }
                }
            }
        default:
            throw InferenceRuntimeError.invalidInputType(
                "encodeVideo: unsupported embedding scalar type \(scalarType)")
        }

        CLILogger.log("VLM encodeVideo complete: \(frameEmbeddings.count) frames, \(totalTokens) tokens")
        return try InputEmbeddings(
            embeddings: concatenated,
            embeddingPositions: 0..<totalTokens
        )
    }

    /// Run the vision encoder on preprocessed pixel values.
    ///
    /// - Parameter pixels: Float32 array in CHW layout, shape `[3, H, W]`
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

        // Shape: [1, 3, imageSize, imageSize]
        let imageSize = visionConfig.imageSize
        let resolvedPixelDesc = pixelDesc.resolvingDynamicDimensions([1, 3, imageSize, imageSize])
        var pixelArray = NDArray(descriptor: resolvedPixelDesc)

        // Fill with CHW pixel data
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
