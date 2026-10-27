// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreImage
import Foundation

/// Multimodal (image input) overlay for the static-shape ladder engine.
///
/// The Gemma4 ladder gathers token embeddings *outside* the compiled graph
/// (`runGather` → `transformer_input`), so vision support is a clean overlay: encode
/// the image with the separate `vision.aimodel`, scatter the soft tokens into the
/// gathered embeddings at placeholder positions, and everything else (decoder graph,
/// PLE, masks, KV cache) is unchanged — image tokens ride the ordinary causal prefill
/// path. Unlike `CoreAISequentialVLMEngine` (separate `embed_tokens`+`project`, a
/// growing single cache), this reuses the ladder's dual-cache/PLE engine and only
/// adds the encode + scatter steps.
///
/// `vision.aimodel` is itself a ladder of canvas sizes
/// (`VisionConfig.imageSizes`), not one fixed size: `encodeImage` picks the
/// smallest size that fits the input image without upscaling it, letterboxes
/// (aspect-preserving resize + zero-pad, block-aligned — see `letterbox`) into that
/// canvas, builds the `attn_mask` excluding the padding patches from
/// attention, and drops the padding-only pooled cells from the output before
/// wrapping it as `InputEmbeddings` — so the number of soft tokens per image varies
/// with its content, not just the chosen canvas size.
extension StaticShapeEngine: MultimodalInferenceEngine {
    /// Overwrite image-placeholder rows of this batch's gathered embeddings with the
    /// encoded image soft tokens. Called from `buildInputs`, after the token-embedding
    /// gather and before the result is bound to the decoder's `transformer_input`.
    func scatterImageEmbeddings(
        into gathered: inout NDArray, alignedStep: Int, batchSize: Int
    ) throws {
        guard let image = pendingImageEmbeddings, !imageTokenPositions.isEmpty else { return }
        try Self.scatterImageSoftTokens(
            into: &gathered,
            image: image,
            positions: imageTokenPositions,
            alignedStep: alignedStep,
            batchSize: batchSize)
    }

    /// Core scatter math, factored out of ``scatterImageEmbeddings(into:alignedStep:batchSize:)``
    /// so it can be unit-tested against synthetic `NDArray`s without a real model asset.
    ///
    /// `positions[i]` is the absolute prompt position of the `i`-th image soft token;
    /// only positions inside `[alignedStep, alignedStep + batchSize)` are written.
    static func scatterImageSoftTokens(
        into gathered: inout NDArray, image: NDArray, positions: [Int], alignedStep: Int, batchSize: Int
    ) throws {
        guard let hiddenDim = gathered.shape.last, hiddenDim > 0 else { return }

        // The caller (`generate(with: InputEmbeddings, ...)`) already checked
        // `positions.count == image.shape[1]` before setting this state, so every
        // position's rank in `positions` is a valid row of `image` — but that check
        // says nothing about the row *width*: a misconfigured vision.aimodel (wrong
        // hidden size) would otherwise read/write past the end of one array's rows
        // without either array's element count disagreeing.
        guard image.shape.last == hiddenDim else {
            throw InferenceRuntimeError.invalidState(
                "scatterImageEmbeddings: image embedding hidden dim \(image.shape.last ?? -1) "
                    + "does not match the decoder's transformer_input hidden dim \(hiddenDim)")
        }

        let batchEnd = alignedStep + batchSize  // exclusive
        // The decoder graph multiplies the *entire* transformer_input by
        // sqrt(hidden_size) unconditionally (Gemma4Extend.forward), but the vision
        // projector's output is not pre-scaled — HF's own reference scatters image
        // features in *before* that scaling is applied to the merged sequence. Predivide
        // here so that after the graph's uniform embed_scale multiply, the image rows
        // land back at the projector's original (unscaled) magnitude.
        let invEmbedScale = LogitsScalarType(1.0 / Double(hiddenDim).squareRoot())
        var scattered = 0
        image.view(as: LogitsScalarType.self).withUnsafePointer { imgPtr, _, _ in
            var dstView = gathered.mutableView(as: LogitsScalarType.self)
            dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                for (imgIndex, pos) in positions.enumerated()
                where pos >= alignedStep && pos < batchEnd {
                    let src = imgIndex * hiddenDim
                    let dst = (pos - alignedStep) * hiddenDim
                    for d in 0..<hiddenDim {
                        dstPtr[dst + d] = imgPtr[src + d] * invEmbedScale
                    }
                    scattered += 1
                }
            }
        }
        if scattered > 0 {
            CLILogger.log(
                "Scattered \(scattered) image soft tokens into batch [\(alignedStep),\(batchEnd)) "
                    + "(hiddenDim=\(hiddenDim), /embed_scale=\(invEmbedScale))")
        }
    }

    // MARK: - MultimodalInferenceEngine

    // CIContext is thread-safe and expensive to create; shared across every
    // encodeImage(at:) call rather than allocated fresh each time.
    private static let ciContext = CIContext()

    public func encodeImage(at url: URL) async throws -> InputEmbeddings {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw ImagePreprocessorError.loadFailed(url)
        }
        guard let cgImage = Self.ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw ImagePreprocessorError.renderFailed
        }
        return try await encodeImage(cgImage: cgImage)
    }

    public func encodeImage(cgImage: CGImage) async throws -> InputEmbeddings {
        guard let visionModel, let vc = visionConfig else {
            throw InferenceRuntimeError.invalidState(
                "encodeImage called on an engine with no vision model/config")
        }
        guard let imageSizes = vc.imageSizes, !imageSizes.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "encodeImage requires VisionConfig.imageSizes (a Gemma4-style size ladder); "
                    + "none present in this bundle's metadata.json")
        }
        let patchSize = vc.patchSize
        let blockPx = vc.blockPixels

        let canvasSize = Self.chooseImageSize(for: cgImage, imageSizes: imageSizes)
        let functionName = VisionConfig.encoderFunctionName(imageSize: canvasSize)
        let pooledGrid = vc.pooledGrid(imageSize: canvasSize)
        let softTokenCount = vc.softTokenCount(imageSize: canvasSize)
        let letterboxed = try Self.letterbox(cgImage: cgImage, bucketSize: canvasSize, blockPx: blockPx)
        let grid = canvasSize / patchSize
        let numPatches = grid * grid
        let realPatchWidth = letterboxed.realWidthPx / patchSize
        let realPatchHeight = letterboxed.realHeightPx / patchSize
        let realPooledWidth = letterboxed.realWidthPx / blockPx
        let realPooledHeight = letterboxed.realHeightPx / blockPx

        guard let fn = try visionModel.loadFunction(named: functionName) else {
            throw InferenceRuntimeError.invalidState("Cannot load vision function '\(functionName)'")
        }
        let desc = fn.descriptor
        guard let outName = desc.outputNames.first else {
            throw InferenceRuntimeError.invalidState(
                "Vision function '\(functionName)' has no output names")
        }

        guard case .ndArray(let pixelDesc) = desc.inputDescriptor(of: "pixel_values") else {
            throw InferenceRuntimeError.invalidInputType("No ndArray descriptor for 'pixel_values'")
        }
        let resolvedPixel = pixelDesc.resolvingDynamicDimensions([1, 3, canvasSize, canvasSize])
        var pixelArray = NDArray(descriptor: resolvedPixel)
        fillNDArray(&pixelArray, as: Float16.self, with: letterboxed.chw.map { Float16($0) })

        guard case .ndArray(let maskDesc) = desc.inputDescriptor(of: "attn_mask") else {
            throw InferenceRuntimeError.invalidInputType("No ndArray descriptor for 'attn_mask'")
        }
        let resolvedMask = maskDesc.resolvingDynamicDimensions([1, numPatches, 1, numPatches])
        var maskArray = NDArray(descriptor: resolvedMask)
        let maskValues = Self.buildAttentionMask(
            numPatches: numPatches, grid: grid,
            realPatchWidth: realPatchWidth, realPatchHeight: realPatchHeight)
        fillNDArray(&maskArray, as: Float16.self, with: maskValues)

        guard case .ndArray(let outDesc) = desc.outputDescriptor(of: outName) else {
            throw InferenceRuntimeError.invalidOutputType("No ndArray descriptor for '\(outName)'")
        }
        let resolvedOut = outDesc.resolvingDynamicDimensions(
            outDesc.shape.map { $0 < 0 ? softTokenCount : $0 })
        var outArray = NDArray(descriptor: resolvedOut)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&outArray, for: outName)
        _ = try await fn.run(
            inputs: ["pixel_values": pixelArray, "attn_mask": maskArray],
            outputViews: consume outputViews)

        let realCount = realPooledWidth * realPooledHeight
        let compacted: NDArray
        if realCount >= softTokenCount {
            // Fully real (image filled the whole bucket) -- nothing to drop.
            compacted = outArray
        } else {
            compacted = try Self.compactRealSoftTokens(
                from: outArray, pooledGrid: pooledGrid,
                realPooledWidth: realPooledWidth, realPooledHeight: realPooledHeight)
        }

        CLILogger.log(
            "Vision encode complete: canvas=\(canvasSize) real="
                + "\(letterboxed.realWidthPx)x\(letterboxed.realHeightPx)px "
                + "\(realCount)/\(softTokenCount) soft tokens")
        return try InputEmbeddings(embeddings: compacted, embeddingPositions: 0..<realCount)
    }

    // MARK: - Size-ladder helpers

    /// Smallest canvas size that fits `cgImage`'s longer side without upscaling it; falls back
    /// to the largest size (downscale) if the image exceeds every size. `imageSizes` must be
    /// sorted ascending (checked when `VisionConfig` is decoded).
    static func chooseImageSize(for cgImage: CGImage, imageSizes: [Int]) -> Int {
        let longSide = max(cgImage.width, cgImage.height)
        return imageSizes.first(where: { $0 >= longSide }) ?? imageSizes[imageSizes.count - 1]
    }

    struct LetterboxedImage {
        let chw: [Float]
        let realWidthPx: Int
        let realHeightPx: Int
    }

    /// Aspect-preserving letterbox of `cgImage` into a `bucketSize`x`bucketSize`
    /// canvas. The image is scaled by `min(1, bucketSize / longSide)` (never
    /// upscaled) and drawn at the top-left of the row-major pixel buffer `_patchify`
    /// (python side) consumes; the rest of the canvas stays zero. `realWidthPx` /
    /// `realHeightPx` are the scaled extent rounded up to whole `blockPx`
    /// (`patchSize * poolingKernelSize`) units, so the padding boundary is always
    /// pool-block-aligned (see `Gemma4VisionEncoderForiOS`'s docstring, python side,
    /// for why that's required for correctness). Pixels inside the real extent but
    /// beyond the scaled image (at most `blockPx - 1` per axis) are zero and attended to.
    ///
    /// CoreGraphics places the origin of a `CGContext` at the bottom-left, while row 0
    /// of the context's backing buffer is the top row, so the draw rect's `y` is
    /// `bucketSize - drawHeight`.
    static func letterbox(cgImage: CGImage, bucketSize: Int, blockPx: Int) throws -> LetterboxedImage {
        guard blockPx > 0, bucketSize > 0, bucketSize % blockPx == 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "encodeImage: bucket size \(bucketSize) must be a positive multiple of the block size "
                    + "\(blockPx) (patch size * pooling kernel)")
        }
        let srcW = CGFloat(cgImage.width)
        let srcH = CGFloat(cgImage.height)
        guard srcW > 0, srcH > 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "encodeImage: source image has zero width or height")
        }
        let longSide = max(srcW, srcH)
        let scale = min(1.0, CGFloat(bucketSize) / longSide)
        let drawW = min(bucketSize, max(1, Int((srcW * scale).rounded())))
        let drawH = min(bucketSize, max(1, Int((srcH * scale).rounded())))

        func roundUpToBlock(_ v: Int) -> Int {
            let blocks = max(1, (v + blockPx - 1) / blockPx)
            return min(blocks * blockPx, bucketSize)
        }
        let targetW = roundUpToBlock(drawW)
        let targetH = roundUpToBlock(drawH)

        // Rest of the canvas is already zero from the fresh context; mean=0/std=1/
        // rescale=1 (this model's convention -- see export/gemma4_vision.py) keeps a
        // true zero after normalization.
        let preprocessor = ImagePreprocessor(
            targetSize: CGSize(width: bucketSize, height: bucketSize),
            mean: (0, 0, 0), std: (1, 1, 1), rescaleFactor: 1.0)
        let canvasImage = try preprocessor.renderToContext(
            cgImage: cgImage, canvasWidth: bucketSize, canvasHeight: bucketSize,
            drawRect: CGRect(x: 0, y: bucketSize - drawH, width: drawW, height: drawH))
        let chw = try preprocessor.preprocessCHW(cgImage: canvasImage)

        return LetterboxedImage(chw: chw, realWidthPx: targetW, realHeightPx: targetH)
    }

    /// `(1, numPatches, 1, numPatches)` mask, row-major: key axis is dim 1, query
    /// axis is dim 3 (matches `primitives/ios/sdpa.py`'s `SDPA.forward` convention --
    /// `causal_mask: (1, max_seq_len, 1, seq_len)`, i.e. key then query). A patch at
    /// `(row, col)` outside the real `[0, realPatchHeight) x [0, realPatchWidth)`
    /// rectangle is masked as a key for every query.
    static func buildAttentionMask(
        numPatches: Int, grid: Int, realPatchWidth: Int, realPatchHeight: Int
    ) -> [Float16] {
        var mask = [Float16](repeating: 0, count: numPatches * numPatches)
        for row in 0..<grid {
            let rowIsReal = row < realPatchHeight
            for col in 0..<grid {
                let isReal = rowIsReal && col < realPatchWidth
                if isReal { continue }
                let key = row * grid + col
                let base = key * numPatches
                for q in 0..<numPatches {
                    mask[base + q] = causalMaskSentinel
                }
            }
        }
        return mask
    }

    /// Drop the padding-only pooled cells from `outArray` (shape `(1, pooledGrid *
    /// pooledGrid, hidden)`), keeping only the top-left `realPooledWidth x
    /// realPooledHeight` real rectangle (row-major over the full pooled grid) --
    /// which, per `letterbox`, is always exactly the real content (padding is
    /// always pool-block-aligned, so a pooled cell is never a mix of the two).
    static func compactRealSoftTokens(
        from outArray: NDArray, pooledGrid: Int, realPooledWidth: Int, realPooledHeight: Int
    ) throws -> NDArray {
        let hiddenDim = outArray.shape.last ?? 0
        let outputRows = outArray.shape.count >= 2 ? outArray.shape[outArray.shape.count - 2] : 0
        let (cells, overflow) = pooledGrid.multipliedReportingOverflow(by: pooledGrid)
        guard pooledGrid > 0, !overflow, cells == outputRows, realPooledWidth >= 0, realPooledHeight >= 0,
            realPooledWidth <= pooledGrid, realPooledHeight <= pooledGrid
        else {
            throw InferenceRuntimeError.invalidState(
                "vision encoder returned \(outputRows) soft tokens, but the bucket's pooled grid is "
                    + "\(pooledGrid)x\(pooledGrid) with a real area of \(realPooledWidth)x\(realPooledHeight); "
                    + "the bundle's image_sizes do not match its vision program")
        }
        // The copy below views both arrays as `LogitsScalarType`, which traps on any other type.
        guard outArray.scalarType == (LogitsScalarType.self == Float16.self ? .float16 : .float32) else {
            throw InferenceRuntimeError.invalidState(
                "vision encoder output is \(outArray.scalarType), but soft-token compaction expects "
                    + "\(LogitsScalarType.self)")
        }
        let realCount = realPooledWidth * realPooledHeight
        var compacted = NDArray(shape: [1, realCount, hiddenDim], scalarType: outArray.scalarType)
        guard hiddenDim > 0, realCount > 0 else { return compacted }
        outArray.view(as: LogitsScalarType.self).withUnsafePointer { srcPtr, _, _ in
            var dstView = compacted.mutableView(as: LogitsScalarType.self)
            dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                var dstIndex = 0
                for r in 0..<realPooledHeight {
                    for c in 0..<realPooledWidth {
                        let srcOffset = (r * pooledGrid + c) * hiddenDim
                        let dstOffset = dstIndex * hiddenDim
                        for d in 0..<hiddenDim {
                            dstPtr[dstOffset + d] = srcPtr[srcOffset + d]
                        }
                        dstIndex += 1
                    }
                }
            }
        }
        return compacted
    }

    public func encodeVideo(_ video: VideoInput) async throws -> InputEmbeddings {
        throw InferenceRuntimeError.invalidState("Gemma4 vision does not support video input")
    }

    public func generate(
        with input: InputEmbeddings,
        tokens: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        let imageTokenId = visionConfig?.imageTokenId ?? -1
        let positions = tokens.enumerated().compactMap { $0.element == imageTokenId ? $0.offset : nil }

        guard positions.count == input.tokenCount else {
            throw InferenceRuntimeError.invalidArgument(
                "generate(with: InputEmbeddings): \(positions.count) image placeholder(s) in prompt "
                    + "but \(input.tokenCount) encoded soft tokens — counts must match")
        }

        pendingImageEmbeddings = input.embeddings
        imageTokenPositions = positions
        CLILogger.log(
            "Multimodal generate: imageTokenId=\(imageTokenId), \(positions.count) placeholder "
                + "positions in \(tokens.count) tokens, \(input.tokenCount) soft tokens"
                + (positions.isEmpty ? "" : " [first=\(positions.first!), last=\(positions.last!)]"))
        return try startGeneration(
            with: tokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions)
    }

    /// Stage image embeddings + placeholder positions for the NEXT prefill without
    /// going through ``generate(with:tokens:samplingConfiguration:inferenceOptions:)`` —
    /// for callers that drive the engine with their own prefill rather than `generate`.
    /// Call it AFTER `reset(to:)` (which clears pending image state) and BEFORE the prompt
    /// prefill. `tokens` must contain exactly `input.tokenCount` occurrences of
    /// `visionConfig.imageTokenId`.
    public func setPendingImage(_ input: InputEmbeddings, tokens: [TokenId]) throws {
        let imageTokenId = visionConfig?.imageTokenId ?? -1
        let positions = tokens.enumerated().compactMap { $0.element == imageTokenId ? $0.offset : nil }
        guard positions.count == input.tokenCount else {
            throw InferenceRuntimeError.invalidArgument(
                "setPendingImage: \(positions.count) image placeholder(s) in prompt but "
                    + "\(input.tokenCount) encoded soft tokens — counts must match")
        }
        pendingImageEmbeddings = input.embeddings
        imageTokenPositions = positions
    }
}
