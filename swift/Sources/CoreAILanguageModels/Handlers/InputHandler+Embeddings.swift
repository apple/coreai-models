// Embeddings input handler for decoders that take token embeddings instead of token IDs.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Input handler for embeddings-input decoders (vision-language models): token embeddings
/// (Float16) and `position_ids` (Int32).
///
/// Runs `embed_tokens` on each step's tokens. On the multimodal prefill step the caller also
/// passes the projected image embeddings, which are scatter-merged into the positions holding
/// the image placeholder token. Decode steps and text-only prompts embed tokens only.
public struct EmbeddingsInputHandler: SyncInputHandler {
    /// Maps a token batch to its embeddings, shape `[1, tokens.count, hidden_dim]`.
    public typealias Embed = (ArraySlice<Int32>) async throws -> NDArray

    public let inputNames: [String]

    private let embeddingsInputName: String
    private let positionIdsName: String
    private let positionIdsDescriptor: NDArrayDescriptor
    private let imageTokenId: Int32
    private let embed: Embed

    public init(
        embeddingsInputName: String,
        positionIdsName: String,
        positionIdsDescriptor: NDArrayDescriptor,
        imageTokenId: Int32,
        embed: @escaping Embed
    ) {
        self.embeddingsInputName = embeddingsInputName
        self.positionIdsName = positionIdsName
        self.positionIdsDescriptor = positionIdsDescriptor
        self.imageTokenId = imageTokenId
        self.embed = embed
        self.inputNames = [embeddingsInputName, positionIdsName]
    }

    /// Embeds the step's tokens.
    public mutating func prepare(_ context: InputContext) async throws -> [String: NDArray] {
        try await prepare(context, imageEmbeddings: nil)
    }

    /// Embeds the step's tokens and, when `imageEmbeddings` is given, scatter-merges them into
    /// the image placeholder positions of `context.tokens`.
    public mutating func prepare(
        _ context: InputContext,
        imageEmbeddings: NDArray?
    ) async throws -> [String: NDArray] {
        let tokens = context.tokens
        let batchSize = tokens.count
        precondition(batchSize > 0, "EmbeddingsInputHandler: empty token batch")

        var embeddings = try await embed(tokens)
        if let imageEmbeddings {
            embeddings = try Self.scatterMerge(
                textEmbeddings: embeddings,
                imageEmbeddings: imageEmbeddings,
                tokens: tokens,
                imageTokenId: imageTokenId
            )
            CLILogger.log("  - scatter merge complete")
        }

        // Build position_ids: [0, 1, ..., processedTokenCount + batchSize - 1]
        let totalPositions = context.processedTokenCount + batchSize
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions([1, totalPositions])
        var positionIds = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIds, as: Int32.self, count: totalPositions) { Int32($0) }

        return [
            embeddingsInputName: embeddings,
            positionIdsName: positionIds,
        ]
    }

    // MARK: - Scatter Merge

    /// Merge text embeddings with vision embeddings by replacing image placeholder positions.
    ///
    /// Finds positions where `tokens[i] == imageTokenId` and copies the corresponding vision
    /// embedding vectors into those positions. All other positions retain their text embeddings.
    ///
    /// - Parameters:
    ///   - textEmbeddings: Text token embeddings NDArray, shape `[1, seq_len, hidden_dim]`
    ///   - imageEmbeddings: Vision embeddings NDArray, shape `[1, image_token_count, hidden_dim]`
    ///   - tokens: Token IDs for locating image placeholder positions
    ///   - imageTokenId: The image placeholder token
    /// - Returns: Merged embeddings NDArray, shape `[1, seq_len, hidden_dim]`
    static func scatterMerge(
        textEmbeddings: NDArray,
        imageEmbeddings: NDArray,
        tokens: ArraySlice<Int32>,
        imageTokenId: Int32
    ) throws -> NDArray {
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

        let expectedCount = imageEmbeddings.shape[1]
        guard imagePositions.count == expectedCount else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: found \(imagePositions.count) image placeholder tokens, "
                    + "expected \(expectedCount) from embeddings. Check prompt template.")
        }

        let seqLen = textEmbeddings.shape[1]
        let imgSeqLen = imageEmbeddings.shape[1]
        guard imgSeqLen >= expectedCount else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: image embeddings have \(imgSeqLen) tokens, need \(expectedCount)")
        }

        // Validate all positions are within bounds
        guard let maxPos = imagePositions.last, maxPos < seqLen else {
            throw InferenceRuntimeError.invalidArgument(
                "scatterMerge: image position \(imagePositions.last ?? -1) exceeds sequence length \(seqLen)")
        }

        // Copy image embeddings into placeholder positions.
        guard imageEmbeddings.scalarType == .float16 else {
            throw InferenceRuntimeError.invalidInputType(
                "scatterMerge only supports float16 embeddings; got \(imageEmbeddings.scalarType)")
        }
        imageEmbeddings.view(as: Float16.self).withUnsafePointer { imgPtr, _, _ in
            let mutableView = merged.mutableView(as: Float16.self)
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
}

extension EmbeddingsInputHandler {
    /// Creates a handler whose `embed` runs the model's `embed_tokens` function.
    public init(
        embeddingsInputName: String,
        positionIdsName: String,
        positionIdsDescriptor: NDArrayDescriptor,
        imageTokenId: Int32,
        embedFunction: InferenceFunction,
        embedFunctionDescriptor: InferenceFunctionDescriptor
    ) {
        self.init(
            embeddingsInputName: embeddingsInputName,
            positionIdsName: positionIdsName,
            positionIdsDescriptor: positionIdsDescriptor,
            imageTokenId: imageTokenId,
            embed: { tokens in
                try await Self.runEmbedTokens(
                    tokens, function: embedFunction, descriptor: embedFunctionDescriptor)
            }
        )
    }

    /// Run embed_tokens to convert token IDs to embeddings.
    ///
    /// - Returns: NDArray of embeddings, shape `[1, seq_len, hidden_dim]`
    static func runEmbedTokens(
        _ tokens: ArraySlice<Int32>,
        function embedFunction: InferenceFunction,
        descriptor embedFunctionDescriptor: InferenceFunctionDescriptor
    ) async throws -> NDArray {
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
}
