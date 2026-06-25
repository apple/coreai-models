// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// TODO: Refactor to re-use common components with CoreAISequentialEngine
// TODO: Add pipelined engine variant for higher throughput

import CoreAI
import CoreAIShared
import CoreImage
import Foundation
import Synchronization

// MARK: - VLM Model Config

/// Configuration for a Vision-Language Model engine.
///
/// Extends the base `ModelConfig` with vision-specific parameters: image size,
/// patch geometry, placeholder token ID, and per-image embedding token count.
public struct VLMModelConfig: InferenceConfiguration, Codable, Sendable {
    public let base: ModelConfig
    public let visionConfig: VisionConfig

    public var maxContextLength: Int { base.maxContextLength }
    public var vocabSize: Int { base.vocabSize }
    public var function: String { base.function }
    public var name: String { base.name }

    /// Prefill chunk size from base config.
    public var prefillChunkSize: Int { base.prefillChunkSize }
    public var chunkThreshold: Int { base.chunkThreshold }

    public init(base: ModelConfig, visionConfig: VisionConfig) {
        self.base = base
        self.visionConfig = visionConfig
    }
}

// MARK: - Core AI Sequential VLM Engine

/// Sequential inference engine for Vision-Language Models using Core AI APIs.
///
/// ## Model Contract
///
/// Manages three model functions (potentially from separate `.aimodel` bundles):
///
/// 1. **Vision encoder** (`encode_image`):
///    - Input: `pixel_values` (Float32, shape `[1, 3, H, W]`)
///    - Output: encoder hidden states (Float32, shape `[1, num_patches, vision_hidden_dim]`)
///
/// 2. **Vision projector** (`project`):
///    - Input: encoder hidden states
///    - Output: projected embeddings (Float16/BFloat16, shape `[1, image_token_count, hidden_dim]`)
///
/// 3. **Embedding lookup** (`embed_tokens`):
///    - Input: `input_ids` (Int32, shape `[1, seq_len]`)
///    - Output: token embeddings (Float16/BFloat16, shape `[1, seq_len, hidden_dim]`)
///
/// 4. **LLM decoder** (`main`):
///    - Inputs: `in_embeddings` (Float16/BFloat16), `position_ids` (Int32)
///    - States: `keyCache`, `valueCache` (persistent KV cache)
///    - Output: `logits` (Float16, shape `[1, seq_len, vocab_size]`)
///
/// ## Inference Flow
///
/// 1. `encodeImage(at:)` — preprocess image, run vision encoder + projector, return `EmbeddedInput`
/// 2. `generate(with: EmbeddedInput, tokens:, ...)` — embed tokens, scatter-merge with vision
///    embeddings at placeholder positions, run LLM prefill, then standard autoregressive decode
///
/// KV cache is managed identically to `CoreAISequentialEngine`: starts small and grows
/// dynamically with 2x expansion.
public final class CoreAISequentialVLMEngine: MultimodalInferenceEngine, @unchecked Sendable {
    public typealias ConfigType = VLMModelConfig
    public typealias OutputSequence = GenerationSequence

    public var supportsLogits: Bool { true }
    public var vocabSize: Int { config.vocabSize }
    public let config: VLMModelConfig

    // MARK: - Vision Model Handles

    private let visionFunction: InferenceFunction
    private let visionFunctionDescriptor: InferenceFunctionDescriptor
    private let projectFunction: InferenceFunction
    private let projectFunctionDescriptor: InferenceFunctionDescriptor
    private let visionProjectorFused: Bool

    // MARK: - Embed Model Handle

    private let embedFunction: InferenceFunction
    private let embedFunctionDescriptor: InferenceFunctionDescriptor

    // MARK: - LLM Model Handle

    private let llmFunction: InferenceFunction
    private let llmFunctionDescriptor: InferenceFunctionDescriptor

    // LLM I/O names from descriptor
    private let embeddingsInputName: String
    private let positionIdsName: String
    private let keyCacheName: String
    private let valueCacheName: String
    private let logitsName: String

    // LLM descriptors for dynamic shape resolution
    private let embeddingsInputDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor

    // MARK: - Persistent State

    private var keyCache: NDArray
    private var valueCache: NDArray
    private var logitsArray: NDArray
    private var cachedLogitsBatchSize: Int
    private var currentKVCapacity: Int
    private let keyCacheDescriptor: NDArrayDescriptor
    private let valueCacheDescriptor: NDArrayDescriptor

    // Track processed tokens for incremental inference
    public private(set) var processedTokenCount: Int = 0

    // MARK: - Image Preprocessor

    private let imagePreprocessor: ImagePreprocessor

    // MARK: - Generation Token

    private let _activeToken = Mutex<GenerationToken?>(nil)
    public var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    func clearTokenIfActive(_ token: GenerationToken) {
        _activeToken.withLock { if $0 === token { $0 = nil } }
    }

    // MARK: - Init

    /// Initialize the VLM engine with separate model assets for vision, embed, and LLM.
    ///
    /// - Parameters:
    ///   - config: VLM model configuration (includes vision config)
    ///   - visionModel: Prepared model containing `encode_image` and `project` functions
    ///   - embedModel: Prepared model containing `embed_tokens` function
    ///   - llmModel: Prepared model containing `main` function (embedding-input decoder)
    ///   - options: Engine options including KV cache strategy
    public init(
        config: VLMModelConfig,
        visionModel: PreparedModel,
        embedModel: PreparedModel,
        llmModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        self.config = config

        let modelLoadSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLMModelLoading",
            details: "Loading VLM \(config.name)"
        )

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

        // --- Embed pipeline ---

        // embed_tokens may be named "main" or "embed_tokens" depending on the asset
        let embedFunctionName = embedModel.model.functionDescriptor(for: "embed_tokens") != nil
            ? "embed_tokens" : "main"

        guard let embedDesc = embedModel.model.functionDescriptor(for: embedFunctionName) else {
            throw InferenceRuntimeError.functionNotFound("embed_tokens (tried 'embed_tokens' and 'main')")
        }
        self.embedFunctionDescriptor = embedDesc

        guard let embedFn = try embedModel.model.loadFunction(named: embedFunctionName) else {
            throw InferenceRuntimeError.functionNotFound(embedFunctionName)
        }
        self.embedFunction = embedFn

        // --- LLM pipeline ---

        guard let llmDesc = llmModel.model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.functionNotFound(config.function)
        }
        self.llmFunctionDescriptor = llmDesc

        // Validate LLM architecture: expects inputs for embeddings + position_ids, states for KV cache
        guard llmDesc.inputNames.count == 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "VLM LLM function expected 2 inputs (in_embeddings, position_ids), "
                    + "got \(llmDesc.inputNames.count): \(llmDesc.inputNames)")
        }
        guard llmDesc.stateNames.count == 2 else {
            throw InferenceRuntimeError.invalidOutputType(
                "VLM LLM function expected 2 states (KV cache), "
                    + "got \(llmDesc.stateNames.count): \(llmDesc.stateNames)")
        }
        guard llmDesc.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "VLM LLM function expected at least 1 output (logits), "
                    + "got \(llmDesc.outputNames.count): \(llmDesc.outputNames)")
        }

        // Extract I/O names
        self.embeddingsInputName = llmDesc.inputNames[0]
        self.positionIdsName = llmDesc.inputNames[1]
        self.keyCacheName = llmDesc.stateNames[0]
        self.valueCacheName = llmDesc.stateNames[1]
        self.logitsName = llmDesc.outputNames[0]

        // Extract and validate descriptors
        guard case .ndArray(let embedsDesc) = llmDesc.inputDescriptor(of: embeddingsInputName) else {
            throw InferenceRuntimeError.invalidInputType(
                "Cannot get descriptor for '\(embeddingsInputName)'")
        }
        self.embeddingsInputDescriptor = embedsDesc

        guard case .ndArray(let posIdsDesc) = llmDesc.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType(
                "Cannot get descriptor for '\(positionIdsName)'")
        }
        self.positionIdsDescriptor = posIdsDesc

        guard case .ndArray(let logitsDesc) = llmDesc.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.invalidOutputType(
                "Cannot get descriptor for '\(logitsName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }
        self.logitsDescriptor = logitsDesc

        // Extract KV cache state descriptors
        guard case .ndArray(let keyCacheDesc) = llmDesc.stateDescriptor(of: keyCacheName),
            case .ndArray(let valueCacheDesc) = llmDesc.stateDescriptor(of: valueCacheName)
        else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get KV cache state descriptors")
        }
        self.keyCacheDescriptor = keyCacheDesc
        self.valueCacheDescriptor = valueCacheDesc

        // Allocate KV cache
        let isDynamic = keyCacheDesc.shape.contains(where: { $0 < 0 })
        let initialCapacity: Int
        if options.kvCacheStrategy == .fixedSize || !isDynamic {
            initialCapacity = config.maxContextLength
        } else {
            initialCapacity = min(256, config.maxContextLength)
        }
        self.currentKVCapacity = initialCapacity

        let resolvedKeyDesc = keyCacheDesc.resolvingDynamicDimensions(
            keyCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
        let resolvedValueDesc = valueCacheDesc.resolvingDynamicDimensions(
            valueCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
        self.keyCache = NDArray(descriptor: resolvedKeyDesc)
        self.valueCache = NDArray(descriptor: resolvedValueDesc)

        CLILogger.log(
            "VLM KV cache: dynamic=\(isDynamic), initial=\(initialCapacity), key=\(keyCacheDesc.shape) -> \(resolvedKeyDesc.shape)"
        )

        // Allocate initial logits (1 token)
        let initLogitsDesc = logitsDesc.resolvingDynamicDimensions([1, 1, config.vocabSize])
        self.logitsArray = NDArray(descriptor: initLogitsDesc)
        self.cachedLogitsBatchSize = 1

        // Load LLM inference function
        guard let llmFn = try llmModel.model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot load function '\(config.function)'")
        }
        self.llmFunction = llmFn

        // Select image preprocessor based on vision config
        // Default to gemma3/SigLIP preset; can be extended for other encoders
        self.imagePreprocessor = ImagePreprocessor(
            targetSize: CGSize(
                width: config.visionConfig.imageSize,
                height: config.visionConfig.imageSize
            ),
            mean: (0.485, 0.456, 0.406),
            std: (0.229, 0.224, 0.225),
            rescaleFactor: 1.0
        )

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIVLMModelLoading",
            signpostID: modelLoadSignpost
        )

        CLILogger.log(
            "CoreAI VLM engine initialized — vision: encode_image+project, "
                + "embed: \(embedFunctionName), llm: \(config.function)"
        )
    }

    // MARK: - Image Encoding (MultimodalInferenceEngine)

    /// Encode an image into embeddings suitable for injection into the LLM.
    ///
    /// Pipeline:
    /// 1. Load image, resize to `visionConfig.imageSize`, normalize channels
    /// 2. Run vision encoder (`encode_image`) to get patch features
    /// 3. Run projector (`project`) to map features to LLM hidden dimension
    /// 4. Return as `EmbeddedInput` with placeholder token positions
    ///
    /// - Parameter url: URL to the image file (JPEG, PNG, HEIC, etc.)
    /// - Returns: `EmbeddedInput` containing projected embeddings and token positions
    public func encodeImage(at url: URL) async throws -> EmbeddedInput {
        let encodeSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLM EncodeImage",
            details: url.lastPathComponent
        )

        // Step 1: Preprocess image to CHW Float32
        guard let ciImage = CIImage(contentsOf: url) else {
            throw ImagePreprocessorError.loadFailed(url)
        }
        guard let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) else {
            throw ImagePreprocessorError.renderFailed
        }
        let chwPixels = try imagePreprocessor.preprocessCHW(cgImage: cgImage)

        // Step 2: Run encode_image
        let encoderOutput = try await runVisionEncoder(pixels: chwPixels)

        // Step 3: Run projector (skip if fused with encoder)
        let projectedEmbeddings = visionProjectorFused ? encoderOutput : try await runProjector(encoderOutput: encoderOutput)

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIVLM EncodeImage",
            signpostID: encodeSignpost
        )

        // The image token positions will be determined during generate() when we know the
        // full token sequence. For now, record the expected token count.
        // Use a placeholder range; the actual positions are resolved at generate() time
        // by scanning for imageTokenId in the token sequence.
        let tokenCount = config.visionConfig.imageTokenCount
        let placeholderRange = 0..<tokenCount

        CLILogger.log("VLM encodeImage complete: \(tokenCount) embedding tokens")

        return EmbeddedInput(
            embeddings: projectedEmbeddings,
            imageTokenPositions: placeholderRange
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
        let imageSize = config.visionConfig.imageSize
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
            projOutDesc.shape.map { $0 < 0 ? config.visionConfig.imageTokenCount : $0 })
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

    // MARK: - Embed Tokens

    /// Run embed_tokens to convert token IDs to embeddings.
    ///
    /// - Parameter tokens: Token IDs to embed
    /// - Returns: NDArray of embeddings, shape `[1, seq_len, hidden_dim]`
    private func embedTokens(_ tokens: ArraySlice<Int32>) async throws -> NDArray {
        let batchSize = tokens.count
        let embedInputName = embedFunctionDescriptor.inputNames[0]
        let embedOutputName = embedFunctionDescriptor.outputNames[0]

        // Resolve input descriptor for this batch size
        guard case .ndArray(let inputDesc) = embedFunctionDescriptor.inputDescriptor(of: embedInputName)
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Cannot get descriptor for embed input '\(embedInputName)'")
        }
        let resolvedInputDesc = inputDesc.resolvingDynamicDimensions([1, batchSize])
        var inputArray = NDArray(descriptor: resolvedInputDesc)
        fillNDArray(&inputArray, as: Int32.self, with: tokens)

        // Resolve output descriptor
        guard case .ndArray(let outputDesc) = embedFunctionDescriptor.outputDescriptor(of: embedOutputName)
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Cannot get descriptor for embed output '\(embedOutputName)'")
        }
        // Output shape: [1, batchSize, hidden_dim]
        let resolvedOutputDesc = outputDesc.resolvingDynamicDimensions(
            outputDesc.shape.enumerated().map { idx, dim in
                if dim < 0 && idx == 1 { return batchSize }
                return dim < 0 ? 1 : dim
            })
        var outputArray = NDArray(descriptor: resolvedOutputDesc)

        // Execute embed_tokens
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&outputArray, for: embedOutputName)

        _ = try await embedFunction.run(
            inputs: [embedInputName: inputArray],
            outputViews: consume outputViews
        )

        return outputArray
    }

    // MARK: - Scatter Merge

    /// Merge text embeddings with vision embeddings by replacing image placeholder positions.
    ///
    /// Finds positions where `tokens[i] == visionConfig.imageTokenId` and copies the
    /// corresponding vision embedding vectors into those positions. All other positions
    /// retain their text embeddings.
    ///
    /// - Parameters:
    ///   - textEmbeddings: Text token embeddings NDArray, shape `[1, seq_len, hidden_dim]`
    ///   - imageEmbeddings: Vision embeddings NDArray, shape `[1, image_token_count, hidden_dim]`
    ///   - tokens: Token IDs for locating image placeholder positions
    /// - Returns: Merged embeddings NDArray, shape `[1, seq_len, hidden_dim]`
    private func scatterMerge(
        textEmbeddings: NDArray,
        imageEmbeddings: NDArray,
        tokens: [Int32]
    ) throws -> NDArray {
        let imageTokenId = config.visionConfig.imageTokenId
        guard let hiddenDim = textEmbeddings.shape.last, hiddenDim > 0 else {
            throw InferenceRuntimeError.invalidState(
                "scatterMerge: text embeddings have invalid shape \(textEmbeddings.shape)")
        }

        // Find image placeholder positions
        var imagePositions: [Int] = []
        for (i, token) in tokens.enumerated() {
            if token == imageTokenId {
                imagePositions.append(i)
            }
        }

        CLILogger.log("  - scatter merge: \(imagePositions.count) image positions, hidden_dim=\(hiddenDim)")

        var merged = textEmbeddings
        guard !imagePositions.isEmpty else { return merged }

        let imageTokenCount = config.visionConfig.imageTokenCount
        guard imagePositions.count == imageTokenCount else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: found \(imagePositions.count) image placeholder tokens, "
                + "expected \(imageTokenCount) from config. Check prompt template.")
        }

        let seqLen = textEmbeddings.shape.count >= 2 ? textEmbeddings.shape[1] : 0
        let imgSeqLen = imageEmbeddings.shape.count >= 2 ? imageEmbeddings.shape[1] : 0
        guard imgSeqLen >= imageTokenCount else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: image embeddings have \(imgSeqLen) tokens, need \(imageTokenCount)")
        }

        // Validate all positions are within bounds
        guard let maxPos = imagePositions.last, maxPos < seqLen else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: image position \(imagePositions.last ?? -1) exceeds sequence length \(seqLen)")
        }

        imageEmbeddings.view(as: Float16.self).withUnsafePointer { imgPtr, _, _ in
            var mutableView = merged.mutableView(as: Float16.self)
            mutableView.withUnsafeMutablePointer { mergedPtr, _, _ in
                for (i, pos) in imagePositions.enumerated() {
                    let srcOffset = i * hiddenDim
                    let dstOffset = pos * hiddenDim
                    (mergedPtr + dstOffset).update(
                        from: imgPtr + srcOffset,
                        count: hiddenDim
                    )
                }
            }
        }

        return merged
    }

    // MARK: - Token Batch Processing (Decode Path)

    /// Process a batch of embeddings through the LLM decoder.
    ///
    /// Unlike the text-only sequential engine which takes token IDs directly, the VLM
    /// decoder always takes pre-computed embeddings. Each decode step:
    /// 1. Caller provides embeddings (from embed_tokens or scatter-merge)
    /// 2. This method runs the LLM forward pass with embeddings + position_ids + KV cache
    ///
    /// - Parameters:
    ///   - embeddings: Pre-computed embeddings NDArray, shape `[1, batch_size, hidden_dim]`
    ///   - batchSize: Number of tokens in this batch
    /// - Returns: Logits for all tokens in the batch
    private func processEmbeddingBatch(
        embeddings: NDArray,
        batchSize: Int
    ) async throws -> [LogitsScalarType] {
        guard batchSize > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot process empty embedding batch")
        }

        try ensureKVCapacity(forContextLength: processedTokenCount + batchSize)

        let batchSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLM Batch",
            details: "\(batchSize) tokens at position \(processedTokenCount)"
        )

        // Build position_ids: [0, 1, ..., processedTokenCount + batchSize - 1]
        let totalPositions = processedTokenCount + batchSize
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions([1, totalPositions])
        var positionIds = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIds, as: Int32.self, count: totalPositions) { Int32($0) }

        // Reallocate logits if batch size changed
        if cachedLogitsBatchSize != batchSize {
            let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions(
                [1, batchSize, config.vocabSize])
            logitsArray = NDArray(descriptor: resolvedLogitsDesc)
            cachedLogitsBatchSize = batchSize
        }

        // Build states (KV cache -- persistent, inout)
        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyCacheName)
        states.insert(&valueCache, for: valueCacheName)

        // Build output backings (logits -- written in-place)
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logitsArray, for: logitsName)

        // Execute LLM forward pass
        _ = try await llmFunction.run(
            inputs: [embeddingsInputName: embeddings, positionIdsName: positionIds],
            states: consume states,
            outputViews: consume outputViews
        )

        // Read logits from NDArray
        let totalLogits = batchSize * config.vocabSize
        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: totalLogits)

        processedTokenCount += batchSize

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIVLM Batch",
            signpostID: batchSignpost
        )

        return logitBuffer
    }

    /// Process a batch of token IDs: embed then run through LLM.
    ///
    /// Convenience method for decode steps where we start from token IDs.
    /// Runs embed_tokens then processEmbeddingBatch.
    private func processTokenBatch(_ tokens: ArraySlice<Int32>) async throws -> [LogitsScalarType] {
        let batchSize = tokens.count
        guard batchSize > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot process empty token batch")
        }

        // Step 1: embed tokens
        let embeddings = try await embedTokens(tokens)

        // Step 2: run LLM with embeddings
        return try await processEmbeddingBatch(embeddings: embeddings, batchSize: batchSize)
    }

    // MARK: - VLM Prefill

    /// Run VLM prefill: embed all tokens, scatter-merge image embeddings, decode.
    ///
    /// - Parameters:
    ///   - embeddedInput: Pre-computed image embeddings from `encodeImage(at:)`
    ///   - tokens: Full token sequence including image placeholder tokens
    /// - Returns: Logits for the last token (shape: [vocabSize])
    private func vlmPrefill(
        embeddedInput: EmbeddedInput,
        tokens: [Int32]
    ) async throws -> [LogitsScalarType] {
        let prefillSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIVLM Prefill",
            details: "\(tokens.count) tokens"
        )

        CLILogger.log("VLM prefill: \(tokens.count) tokens")

        // Step 1: Embed all tokens
        let textEmbeddings = try await embedTokens(tokens[...])

        // Step 2: Scatter merge -- replace image placeholder positions with vision embeddings
        let merged = try scatterMerge(
            textEmbeddings: textEmbeddings,
            imageEmbeddings: embeddedInput.embeddings,
            tokens: tokens
        )

        CLILogger.log("  - scatter merge complete")

        // Step 3: Run LLM with merged embeddings
        let logitBuffer = try await processEmbeddingBatch(
            embeddings: merged,
            batchSize: tokens.count
        )

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIVLM Prefill",
            signpostID: prefillSignpost
        )

        return lastTokenLogits(from: logitBuffer, vocabSize: config.vocabSize)
    }

    // MARK: - Prefill Strategy

    private func selectPrefillStrategy(newTokenCount: Int) -> PrefillStrategy {
        if newTokenCount > config.chunkThreshold {
            return .chunked(chunkSize: config.prefillChunkSize)
        }
        return .wholeBatch
    }

    // MARK: - Chunked Prefill (text-only decode path)

    private func processChunkedPrompt(
        tokens: ArraySlice<Int32>,
        chunkSize: Int
    ) async throws -> [LogitsScalarType] {
        let totalChunks = (tokens.count + chunkSize - 1) / chunkSize

        var lastLogits: [LogitsScalarType] = []
        var remainingTokens = tokens
        var chunkIndex = 0

        while !remainingTokens.isEmpty {
            let currentChunkSize = min(chunkSize, remainingTokens.count)
            let chunkEnd = remainingTokens.startIndex + currentChunkSize
            let chunk = remainingTokens[remainingTokens.startIndex..<chunkEnd]

            CLILogger.log(
                "VLM decode chunk \(chunkIndex + 1)/\(totalChunks): \(chunk.count) tokens at position \(processedTokenCount)"
            )

            lastLogits = try await processTokenBatch(chunk)
            remainingTokens = remainingTokens[chunkEnd...]
            chunkIndex += 1
        }

        return lastTokenLogits(from: lastLogits, vocabSize: config.vocabSize)
    }

    // MARK: - Generate (text-only, InferenceEngine protocol)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        let token = GenerationToken()
        _activeToken.withLock { $0 = token }
        return GenerationSequence(
            engine: self,
            input: input,
            embeddedInput: nil,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    // MARK: - Generate (multimodal, MultimodalInferenceEngine protocol)

    /// Generate tokens from a token sequence with embedded image regions.
    ///
    /// The first forward pass performs VLM prefill: embeds all tokens, scatter-merges the
    /// image embeddings at placeholder positions, then runs the LLM. Subsequent steps use
    /// standard token-by-token decode (embed_tokens -> main).
    public func generate(
        with input: EmbeddedInput,
        tokens: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        let token = GenerationToken()
        _activeToken.withLock { $0 = token }
        return GenerationSequence(
            engine: self,
            input: tokens,
            embeddedInput: input,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    // MARK: - Lifecycle

    public func reset() async throws {
        try await reset(to: 0)
    }

    public func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(processedTokenCount)]")
        if tokenIndex == 0 {
            _activeToken.withLock {
                $0?.cancel()
                $0 = nil
            }
            let resetSpan = InstrumentsProfiler.beginReset(engine: "CoreAIVLM")
            processedTokenCount = 0
            zeroFill(&keyCache)
            zeroFill(&valueCache)
            resetSpan.end()
        } else {
            processedTokenCount = tokenIndex
        }
    }

    public func cancel() async throws {
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
    }

    public func cleanup() {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAIVLM")
        CLILogger.log("CoreAI VLM engine cleanup complete")
        cleanupSpan.end()
    }

    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        // Warmup the decode path with a single dummy token through embed + LLM
        let dummyTokens: ArraySlice<Int32> = [Int32(1)][...]
        _ = try await processTokenBatch(dummyTokens)
        // Reset state after warmup
        processedTokenCount = 0
        zeroFill(&keyCache)
        zeroFill(&valueCache)
    }

    // MARK: - KV Cache (dynamic growth)

    private func ensureKVCapacity(forContextLength needed: Int) throws {
        guard needed > currentKVCapacity else { return }
        guard needed <= config.maxContextLength else {
            throw InferenceRuntimeError.invalidState(
                "Context length \(needed) exceeds maximum \(config.maxContextLength)")
        }

        var newCapacity = currentKVCapacity
        while newCapacity < needed { newCapacity *= 2 }
        newCapacity = min(newCapacity, config.maxContextLength)

        let resolvedKeyDesc = keyCacheDescriptor.resolvingDynamicDimensions(
            keyCacheDescriptor.shape.map { $0 < 0 ? newCapacity : $0 })
        let resolvedValueDesc = valueCacheDescriptor.resolvingDynamicDimensions(
            valueCacheDescriptor.shape.map { $0 < 0 ? newCapacity : $0 })

        var newKeyCache = NDArray(descriptor: resolvedKeyDesc)
        var newValueCache = NDArray(descriptor: resolvedValueDesc)
        _ = newKeyCache.mutableRawView()
        _ = newValueCache.mutableRawView()

        try Self.copyCache(from: keyCache, to: &newKeyCache)
        try Self.copyCache(from: valueCache, to: &newValueCache)

        CLILogger.log("VLM KV cache grew: \(currentKVCapacity) -> \(newCapacity)")
        keyCache = newKeyCache
        valueCache = newValueCache
        currentKVCapacity = newCapacity
    }

    private static func copyCache(from source: NDArray, to destination: inout NDArray) throws {
        let srcShape = source.shape
        let dstShape = destination.shape
        guard let headDim = srcShape.last else {
            throw InferenceRuntimeError.invalidState("KV cache has empty shape -- cannot copy")
        }
        let seqDim = KVCacheFactory.detectSequenceDim(shape: srcShape)

        let numBlocks = srcShape[..<seqDim].reduce(1, *)
        let oldSeqLen = srcShape[seqDim]
        let copySize = oldSeqLen * headDim

        let srcBlockStride = srcShape[seqDim...].reduce(1, *)
        let dstBlockStride = dstShape[seqDim...].reduce(1, *)

        source.view(as: LogitsScalarType.self).withUnsafePointer { srcPtr, _, _ in
            var dstView = destination.mutableView(as: LogitsScalarType.self)
            dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                for block in 0..<numBlocks {
                    let srcOff = block * srcBlockStride
                    let dstOff = block * dstBlockStride
                    dstPtr.advanced(by: dstOff).update(
                        from: srcPtr.advanced(by: srcOff), count: copySize)
                }
            }
        }
    }

    // MARK: - Helpers

    private func zeroFill(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        var view = array.mutableView(as: LogitsScalarType.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count {
                ptr[i] = 0
            }
        }
    }
}

// MARK: - Generation Sequence

extension CoreAISequentialVLMEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let engine: CoreAISequentialVLMEngine
        let input: [CoreAISequentialVLMEngine.TokenId]
        let embeddedInput: EmbeddedInput?
        let samplingConfiguration: SamplingConfiguration
        let inferenceOptions: InferenceOptions
        let generationToken: GenerationToken

        let stopReasonStore = StopReasonStore()

        public var stopReason: StopReason? { stopReasonStore.stopReason }

        public func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                engine: engine,
                input: input,
                embeddedInput: embeddedInput,
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore,
                generationToken: generationToken
            )
        }
    }
}

// MARK: - Generation Iterator

extension CoreAISequentialVLMEngine.GenerationSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: CoreAISequentialVLMEngine
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [CoreAISequentialVLMEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore
        private let generationToken: GenerationToken

        private var inputTokens: [CoreAISequentialVLMEngine.TokenId]
        private var embeddedInput: EmbeddedInput?
        private var step: Int = 0
        private var finished: Bool = false
        private var prefillDone: Bool = false

        init(
            engine: CoreAISequentialVLMEngine,
            input: [CoreAISequentialVLMEngine.TokenId],
            embeddedInput: EmbeddedInput?,
            samplingConfiguration: SamplingConfiguration,
            inferenceOptions: InferenceOptions,
            stopReasonStore: StopReasonStore,
            generationToken: GenerationToken
        ) {
            self.engine = engine
            self.samplingConfiguration = samplingConfiguration
            self.returnsLogits = inferenceOptions.includeLogits
            self.forcedContinuation = inferenceOptions.forcedContinuation
            self.stopReasonStore = stopReasonStore
            self.generationToken = generationToken
            self.inputTokens = input
            self.embeddedInput = embeddedInput
            if let forced = inferenceOptions.forcedContinuation {
                self.maxTokens = forced.count
            } else {
                self.maxTokens = Swift.min(
                    inferenceOptions.maxTokens ?? Int.max,
                    Swift.max(0, engine.config.maxContextLength - input.count)
                )
            }
        }

        deinit {
            engine.clearTokenIfActive(generationToken)
        }

        public func next() async throws -> InferenceOutput? {
            if finished { return nil }

            if generationToken.isCancelled {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                return nil
            }

            guard step < maxTokens else {
                stopReasonStore.setIfUnset(.maxTokens)
                finishAndRelease()
                return nil
            }

            do {
                try Task.checkCancellation()

                let logitBuffer: [LogitsScalarType]

                if !prefillDone {
                    // First iteration: run prefill
                    if let embedded = embeddedInput {
                        // VLM path: scatter-merge image embeddings and prefill
                        logitBuffer = try await engine.vlmPrefill(
                            embeddedInput: embedded,
                            tokens: inputTokens
                        )
                        embeddedInput = nil
                    } else {
                        // Text-only path: standard prefill
                        let newTokens = inputTokens[engine.processedTokenCount...]
                        let strategy = engine.selectPrefillStrategy(newTokenCount: newTokens.count)

                        switch strategy {
                        case .chunked(let chunkSize):
                            logitBuffer = try await engine.processChunkedPrompt(
                                tokens: newTokens, chunkSize: chunkSize)
                        case .wholeBatch:
                            let allLogits = try await engine.processTokenBatch(newTokens)
                            logitBuffer = lastTokenLogits(
                                from: allLogits, vocabSize: engine.config.vocabSize)
                        case .oneAtATime:
                            var lastLogits: [LogitsScalarType] = []
                            for j in newTokens.indices {
                                lastLogits = try await engine.processTokenBatch(newTokens[j...j])
                            }
                            logitBuffer = lastLogits
                        }
                    }
                    prefillDone = true
                } else {
                    // Subsequent iterations: single-token decode
                    guard engine.processedTokenCount < inputTokens.count else {
                        throw InferenceRuntimeError.invalidState("No new tokens to process")
                    }
                    let newTokens = inputTokens[engine.processedTokenCount...]
                    logitBuffer = try await engine.processTokenBatch(newTokens)
                }

                // Check cancellation after inference step
                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    return nil
                }

                // Sample next token
                let nextToken: Int32
                if let forced = forcedContinuation {
                    nextToken = forced[step]
                } else {
                    var mutableLogits = logitBuffer
                    nextToken = samplingConfiguration.fallbackSampler(from: &mutableLogits)
                }

                inputTokens.append(nextToken)
                step += 1

                return InferenceOutput(
                    tokenId: nextToken,
                    logits: returnsLogits ? logitBuffer : nil
                )
            } catch is CancellationError {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                finishAndRelease()
                throw error
            }
        }

        private func finishAndRelease() {
            guard !finished else { return }
            finished = true
            engine.clearTokenIfActive(generationToken)
        }
    }
}
