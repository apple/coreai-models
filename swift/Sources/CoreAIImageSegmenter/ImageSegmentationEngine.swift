// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

// MARK: - CoreAISegmentationEngine

/// Core AI-backed segmentation engine.
///
/// Supports two asset shapes, autodetected at init time:
///   * Single-function — one ``main`` graph that consumes the image (and a text or point
///     prompt) and emits all detection outputs in one call. Produced by the baseline
///     SAM3 export and EfficientSAM.
///   * Multi-function — three graphs (``image_encode``, ``text_encode``, ``detect``) wired
///     together at runtime. Produced by the SAM3 lite export. The engine pipes the encoder
///     outputs into the detector and returns the same `SegmentationOutput` shape as the
///     single-function path.
public struct CoreAISegmentationEngine {
    private let backend: Backend

    // MARK: - Capabilities

    public var supportsTextQuery: Bool {
        switch backend {
        case .single(let s): return s.textInputName != nil
        case .multi: return true
        }
    }

    public var supportsPointQuery: Bool {
        switch backend {
        case .single(let s): return s.pointsInputName != nil && s.pointLabelsInputName != nil
        case .multi: return false
        }
    }

    // MARK: - Init

    public init(parameters: SegmentationParameters, modelURL: URL) async throws {
        let preparedAsset = try await PreparedModel.prepare(at: modelURL)
        let model = preparedAsset.model

        // `PreparedModel` already classified the asset (and used that classification to pick
        // the compute-unit specialization at load time). Reuse it as the single source of
        // truth for multi- vs single-function dispatch rather than re-probing here.
        if preparedAsset.structure == .multiFunctionSegmenter {
            // Structure detection guarantees all three entrypoints exist; fetch the
            // descriptors the contexts need to validate and wire their I/O.
            guard let imageEncodeDescriptor = model.functionDescriptor(for: GraphNames.imageEncode),
                let textEncodeDescriptor = model.functionDescriptor(for: GraphNames.textEncode),
                let detectDescriptor = model.functionDescriptor(for: GraphNames.detect)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Model classified as multi-function segmenter but is missing one of "
                        + "{'image_encode','text_encode','detect'}. Available functions: \(model.functionNames)."
                )
            }
            self.backend = .multi(
                try await MultiFunctionContext(
                    model: model,
                    imageEncodeDescriptor: imageEncodeDescriptor,
                    textEncodeDescriptor: textEncodeDescriptor,
                    detectDescriptor: detectDescriptor
                )
            )
            return
        }

        guard let mainDescriptor = model.functionDescriptor(for: GraphNames.main) else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Model has no 'main' function and no {'image_encode','text_encode','detect'} bundle. "
                    + "Available functions: \(model.functionNames)."
            )
        }

        self.backend = .single(
            try await SingleFunctionContext(model: model, descriptor: mainDescriptor)
        )
    }

    // MARK: - Warmup

    public func warmup() async throws {
        switch backend {
        case .single(let s):
            if s.textInputName != nil {
                try await warmupSingleFunctionTextModel(state: s)
            } else if s.pointsInputName != nil, s.pointLabelsInputName != nil {
                try await warmupSingleFunctionPointModel(state: s)
            }
        case .multi(let m):
            try await warmupMultiFunctionTextModel(state: m)
        }
    }

    // MARK: - Public segment(image:textQuery:)

    public func segment(image: CGImage, textQuery: TextQuery, parameters: SegmentationParameters) async throws
        -> SegmentationOutput
    {
        switch backend {
        case .single(let s):
            return try await runSingleFunctionTextSegment(
                state: s, image: image, textQuery: textQuery, parameters: parameters
            )
        case .multi(let m):
            return try await runMultiFunctionTextSegment(
                state: m, image: image, textQuery: textQuery, parameters: parameters
            )
        }
    }

    // MARK: - Public segment(image:pointQuery:)

    public func segment(image: CGImage, pointQuery: PointQuery, parameters: SegmentationParameters) async throws
        -> SegmentationOutput
    {
        switch backend {
        case .single(let s):
            return try await runSingleFunctionPointSegment(
                state: s, image: image, pointQuery: pointQuery, parameters: parameters
            )
        case .multi:
            throw SegmentationRuntimeError.unsupportedEngine(
                "Multi-function segmentation assets do not accept point queries — "
                    + "use segment(image:textQuery:parameters:) instead."
            )
        }
    }

    // MARK: - Backend storage

    private enum Backend {
        case single(SingleFunctionContext)
        case multi(MultiFunctionContext)
    }

    /// Backing state for a single-`main`-function asset (baseline SAM3, EfficientSAM).
    fileprivate struct SingleFunctionContext {
        let function: InferenceFunction
        let descriptor: InferenceFunctionDescriptor

        let imageInputName: String

        // Text-based inputs (SAM3-style baseline).
        let textInputName: String?
        let embeddingsInputName: String?

        // Point-based inputs (EfficientSAM).
        let pointsInputName: String?
        let pointLabelsInputName: String?

        // Required for every model.
        let masksOutputName: String

        // Text-model outputs. Validated non-nil at init when textInputName != nil.
        let boxesOutputName: String?
        let logitsOutputName: String?
        let presenceLogitsOutputName: String?
        let semanticSegOutputName: String?

        // Point-model output. Validated non-nil at init when both point inputs present.
        let iouScoresOutputName: String?

        init(model: AIModel, descriptor: InferenceFunctionDescriptor) async throws {
            guard let imageInputName = findImageInputName(in: descriptor.inputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find image input in model. Inputs: \(descriptor.inputNames)"
                )
            }

            let textInputName = findTextInputName(in: descriptor.inputNames)
            let embeddingsInputName = findEmbeddingsInputName(in: descriptor.inputNames)
            let pointsInputName = findPointsInputName(in: descriptor.inputNames)
            let pointLabelsInputName = findPointLabelsInputName(in: descriptor.inputNames)

            guard let masksOutputName = findMasksOutputName(in: descriptor.outputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find masks output in model. Outputs: \(descriptor.outputNames)"
                )
            }
            guard case .ndArray = descriptor.outputDescriptor(of: masksOutputName) else {
                throw SegmentationRuntimeError.outputMissing(masksOutputName)
            }

            let boxesOutputName = findBoxesOutputName(in: descriptor.outputNames)
            let logitsOutputName = findLogitsOutputName(in: descriptor.outputNames)
            let presenceLogitsOutputName = findPresenceOutputName(in: descriptor.outputNames)
            let semanticSegOutputName = findSemanticOutputName(in: descriptor.outputNames)
            let iouScoresOutputName = findIouScoresOutputName(in: descriptor.outputNames)

            // Validate that the expected outputs are present for each model type.
            if textInputName != nil {
                if boxesOutputName == nil {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Text model missing boxes output. Outputs: \(descriptor.outputNames)"
                    )
                }
                if logitsOutputName == nil {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Text model missing logits output. Outputs: \(descriptor.outputNames)"
                    )
                }
                if presenceLogitsOutputName == nil {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Text model missing presence logits output. Outputs: \(descriptor.outputNames)"
                    )
                }
                if semanticSegOutputName == nil {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Text model missing semantic segmentation output. Outputs: \(descriptor.outputNames)"
                    )
                }
            } else if pointsInputName != nil && pointLabelsInputName != nil {
                if iouScoresOutputName == nil {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Point model missing iou_scores output. Outputs: \(descriptor.outputNames)"
                    )
                }
            } else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Model has neither text nor point inputs. Inputs: \(descriptor.inputNames)"
                )
            }

            guard let fn = try model.loadFunction(named: GraphNames.main) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot load 'main' function from model"
                )
            }

            self.function = fn
            self.descriptor = descriptor
            self.imageInputName = imageInputName
            self.textInputName = textInputName
            self.embeddingsInputName = embeddingsInputName
            self.pointsInputName = pointsInputName
            self.pointLabelsInputName = pointLabelsInputName
            self.masksOutputName = masksOutputName
            self.boxesOutputName = boxesOutputName
            self.logitsOutputName = logitsOutputName
            self.presenceLogitsOutputName = presenceLogitsOutputName
            self.semanticSegOutputName = semanticSegOutputName
            self.iouScoresOutputName = iouScoresOutputName
        }
    }

    /// Backing state for the SAM3 lite export (`image_encode` → `text_encode` → `detect`).
    fileprivate struct MultiFunctionContext {
        let imageEncode: InferenceFunction
        let imageEncodeDescriptor: InferenceFunctionDescriptor
        let textEncode: InferenceFunction
        let textEncodeDescriptor: InferenceFunctionDescriptor
        let detect: InferenceFunction
        let detectDescriptor: InferenceFunctionDescriptor

        // image_encode i/o.
        let imageInputName: String
        let backboneFeaturesOutputName: String

        // text_encode i/o.
        let textInputName: String
        let textFeaturesOutputName: String

        // detect i/o. The two intermediate inputs are matched by name against the encoder
        // outputs so re-export naming changes can be absorbed without code edits.
        let backboneFeaturesInputName: String
        let textFeaturesInputName: String
        let masksOutputName: String
        let boxesOutputName: String
        let logitsOutputName: String
        let presenceLogitsOutputName: String
        let semanticSegOutputName: String

        init(
            model: AIModel,
            imageEncodeDescriptor: InferenceFunctionDescriptor,
            textEncodeDescriptor: InferenceFunctionDescriptor,
            detectDescriptor: InferenceFunctionDescriptor
        ) async throws {
            // image_encode: needs an image input + backbone-features output.
            guard let imageInputName = findImageInputName(in: imageEncodeDescriptor.inputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find image input in 'image_encode'. Inputs: \(imageEncodeDescriptor.inputNames)"
                )
            }
            guard
                let backboneFeaturesOutputName = findBackboneFeaturesName(
                    in: imageEncodeDescriptor.outputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find backbone-features output in 'image_encode'. "
                        + "Outputs: \(imageEncodeDescriptor.outputNames)"
                )
            }

            // text_encode: needs a token-id input + text-features output.
            guard let textInputName = findTextInputName(in: textEncodeDescriptor.inputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find text input in 'text_encode'. Inputs: \(textEncodeDescriptor.inputNames)"
                )
            }
            guard let textFeaturesOutputName = findTextFeaturesName(in: textEncodeDescriptor.outputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find text-features output in 'text_encode'. "
                        + "Outputs: \(textEncodeDescriptor.outputNames)"
                )
            }

            // detect: needs both encoder outputs as inputs + the five detection outputs.
            guard
                let backboneFeaturesInputName = findBackboneFeaturesName(
                    in: detectDescriptor.inputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find backbone-features input in 'detect'. Inputs: \(detectDescriptor.inputNames)"
                )
            }
            guard let textFeaturesInputName = findTextFeaturesName(in: detectDescriptor.inputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find text-features input in 'detect'. Inputs: \(detectDescriptor.inputNames)"
                )
            }
            guard let masksOutputName = findMasksOutputName(in: detectDescriptor.outputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find masks output in 'detect'. Outputs: \(detectDescriptor.outputNames)"
                )
            }
            guard case .ndArray = detectDescriptor.outputDescriptor(of: masksOutputName) else {
                throw SegmentationRuntimeError.outputMissing(masksOutputName)
            }
            guard let boxesOutputName = findBoxesOutputName(in: detectDescriptor.outputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find boxes output in 'detect'. Outputs: \(detectDescriptor.outputNames)"
                )
            }
            guard let logitsOutputName = findLogitsOutputName(in: detectDescriptor.outputNames) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find logits output in 'detect'. Outputs: \(detectDescriptor.outputNames)"
                )
            }
            guard let presenceLogitsOutputName = findPresenceOutputName(in: detectDescriptor.outputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find presence-logits output in 'detect'. Outputs: \(detectDescriptor.outputNames)"
                )
            }
            guard let semanticSegOutputName = findSemanticOutputName(in: detectDescriptor.outputNames)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot find semantic segmentation output in 'detect'. Outputs: \(detectDescriptor.outputNames)"
                )
            }

            guard let imageEncode = try model.loadFunction(named: GraphNames.imageEncode) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot load 'image_encode' function from model"
                )
            }
            guard let textEncode = try model.loadFunction(named: GraphNames.textEncode) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot load 'text_encode' function from model"
                )
            }
            guard let detect = try model.loadFunction(named: GraphNames.detect) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Cannot load 'detect' function from model"
                )
            }

            self.imageEncode = imageEncode
            self.imageEncodeDescriptor = imageEncodeDescriptor
            self.textEncode = textEncode
            self.textEncodeDescriptor = textEncodeDescriptor
            self.detect = detect
            self.detectDescriptor = detectDescriptor

            self.imageInputName = imageInputName
            self.backboneFeaturesOutputName = backboneFeaturesOutputName

            self.textInputName = textInputName
            self.textFeaturesOutputName = textFeaturesOutputName

            self.backboneFeaturesInputName = backboneFeaturesInputName
            self.textFeaturesInputName = textFeaturesInputName
            self.masksOutputName = masksOutputName
            self.boxesOutputName = boxesOutputName
            self.logitsOutputName = logitsOutputName
            self.presenceLogitsOutputName = presenceLogitsOutputName
            self.semanticSegOutputName = semanticSegOutputName
        }
    }

    // MARK: - Single-function warmup helpers

    private func warmupSingleFunctionTextModel(state: SingleFunctionContext) async throws {
        guard let textInputName = state.textInputName else { return }
        guard
            case .ndArray(let imageDescriptor) = state.descriptor.inputDescriptor(of: state.imageInputName),
            case .ndArray(let textDescriptor) = state.descriptor.inputDescriptor(of: textInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image or text input"
            )
        }
        let imageArray = NDArray(descriptor: imageDescriptor)
        var textArray = NDArray(descriptor: textDescriptor)
        fillNDArray(&textArray, as: Int32.self, count: textDescriptor.shape.reduce(1, *)) { _ in
            CLIPTokenizer.eotTokenId
        }
        try await runSingleFunctionTextInference(
            state: state,
            inputs: [state.imageInputName: imageArray, textInputName: textArray]
        )
    }

    private func warmupSingleFunctionPointModel(state: SingleFunctionContext) async throws {
        guard let pointsInputName = state.pointsInputName,
            let pointLabelsInputName = state.pointLabelsInputName
        else { return }
        guard
            case .ndArray(let imageDescriptor) = state.descriptor.inputDescriptor(of: state.imageInputName),
            case .ndArray(let pointsDescriptor) = state.descriptor.inputDescriptor(of: pointsInputName),
            case .ndArray(let labelsDescriptor) = state.descriptor.inputDescriptor(of: pointLabelsInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image or point inputs"
            )
        }
        try await runSingleFunctionPointInference(
            state: state,
            inputs: [
                state.imageInputName: NDArray(descriptor: imageDescriptor),
                pointsInputName: NDArray(descriptor: pointsDescriptor),
                pointLabelsInputName: NDArray(descriptor: labelsDescriptor),
            ],
            pointQuery: PointQuery(),
            imageSize: .zero
        )
    }

    // MARK: - Single-function text path (preserves existing behavior)

    private func runSingleFunctionTextSegment(
        state: SingleFunctionContext,
        image: CGImage,
        textQuery: TextQuery,
        parameters: SegmentationParameters
    ) async throws -> SegmentationOutput {
        guard let textInputName = state.textInputName else {
            throw SegmentationRuntimeError.unsupportedEngine(
                "This model has no text input — use segment(image:pointQuery:parameters:) instead."
            )
        }

        guard case .ndArray(let imageDescriptor) = state.descriptor.inputDescriptor(of: state.imageInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(state.imageInputName)'"
            )
        }
        let imageArray = try Self.buildImageNDArray(
            from: image, descriptor: imageDescriptor, parameters: parameters
        )

        var inputs: [String: NDArray] = [state.imageInputName: imageArray]

        switch textQuery {
        case .prompt:
            throw SegmentationRuntimeError.invalidConfiguration(
                "TextQuery.prompt must be resolved to .tokens by ImageSegmenter before reaching the engine."
            )
        case .tokens(let textTokensBatch):
            guard case .ndArray(let textDescriptor) = state.descriptor.inputDescriptor(of: textInputName) else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "No array descriptor for text input '\(textInputName)'"
                )
            }
            inputs[textInputName] = Self.buildTextTokensNDArray(
                tokensBatch: textTokensBatch, descriptor: textDescriptor
            )

        case .embeddings(let embeddingsBatch):
            guard let embInputName = state.embeddingsInputName else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "TextQuery.embeddings provided but no embeddings input found in model. "
                        + "Inputs: \(state.descriptor.inputNames)"
                )
            }
            guard case .ndArray(let embeddingsDescriptor) = state.descriptor.inputDescriptor(of: embInputName)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "No array descriptor for embeddings input '\(embInputName)'"
                )
            }
            let batchSize = embeddingsDescriptor.shape[0]
            let sequenceLength = embeddingsDescriptor.shape[1]
            let hiddenSize = embeddingsDescriptor.shape[2]
            var embeddingsArray = NDArray(descriptor: embeddingsDescriptor)
            fillNDArray(&embeddingsArray, as: Float.self, count: batchSize * sequenceLength * hiddenSize) { idx in
                Self.embeddingValue(
                    at: idx, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: embeddingsBatch)
            }
            inputs[embInputName] = embeddingsArray
        }

        return try await runSingleFunctionTextInference(state: state, inputs: inputs)
    }

    // MARK: - Single-function point path (preserves existing behavior)

    private func runSingleFunctionPointSegment(
        state: SingleFunctionContext,
        image: CGImage,
        pointQuery: PointQuery,
        parameters: SegmentationParameters
    ) async throws -> SegmentationOutput {
        guard let pointsInputName = state.pointsInputName,
            let pointLabelsInputName = state.pointLabelsInputName
        else {
            throw SegmentationRuntimeError.unsupportedEngine(
                "This model has no point inputs — use segment(image:textQuery:parameters:) instead."
            )
        }

        let (imageArray, modelSize) = try preprocessImageForPoints(state: state, image: image)

        let (pointsDescriptor, labelsDescriptor, batchSize, queryCount, pointsPerQuery) =
            try pointInputShapes(
                state: state,
                pointsInputName: pointsInputName,
                pointLabelsInputName: pointLabelsInputName
            )

        let imageHeight = Float(image.height)
        let imageWidth = Float(image.width)

        let resolvedQueries = try Self.resolveQueries(
            pointQuery, queryCount: queryCount, pointsPerQuery: pointsPerQuery,
            imageWidth: imageWidth, imageHeight: imageHeight
        )
        let resolvedQuery = PointQuery(queries: resolvedQueries)

        let (pointsArray, labelsArray) = buildPointTensors(
            queries: resolvedQueries,
            pointsDescriptor: pointsDescriptor,
            labelsDescriptor: labelsDescriptor,
            batchSize: batchSize,
            queryCount: queryCount,
            pointsPerQuery: pointsPerQuery,
            scaleX: modelSize.width / imageWidth,
            scaleY: modelSize.height / imageHeight
        )

        return try await runSingleFunctionPointInference(
            state: state,
            inputs: [
                state.imageInputName: imageArray, pointsInputName: pointsArray,
                pointLabelsInputName: labelsArray,
            ],
            pointQuery: resolvedQuery,
            imageSize: CGSize(width: CGFloat(imageWidth), height: CGFloat(imageHeight))
        )
    }

    /// Resize + dtype-convert `image` into the model's image-input NDArray.
    /// EfficientSAM bakes `(x - mean) / std` into the graph, so we feed raw `[0, 1]` pixels
    /// (`rescaleFactor=1/255`, identity mean/std).
    /// Returns the filled NDArray and the model's spatial size in pixels (`width × height`).
    private func preprocessImageForPoints(state: SingleFunctionContext, image: CGImage) throws
        -> (NDArray, (width: Float, height: Float))
    {
        guard case .ndArray(let imageDescriptor) = state.descriptor.inputDescriptor(of: state.imageInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(state.imageInputName)'"
            )
        }
        let modelWidth = imageDescriptor.shape[3]
        let modelHeight = imageDescriptor.shape[2]
        let preprocessor = ImagePreprocessor(
            targetSize: CGSize(width: modelWidth, height: modelHeight),
            mean: (0, 0, 0),
            std: (1, 1, 1),
            rescaleFactor: 1.0
        )
        let floatPixels = try preprocessor.preprocessCHW(cgImage: image)
        var imageArray = NDArray(descriptor: imageDescriptor)
        if imageDescriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            fillNDArray(&imageArray, as: Float16.self, with: floatPixels.map(Float16.init))
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            fillNDArray(&imageArray, as: Float.self, with: floatPixels)
        }
        return (imageArray, (Float(modelWidth), Float(modelHeight)))
    }

    /// Look up and validate the point-prompt tensor descriptors.
    /// Returns the two descriptors plus `[B, Q, P]` derived from the points input shape.
    /// Throws if either descriptor is missing, the ranks differ from `[B,Q,P,2]`/`[B,Q,P]`,
    /// or the shapes disagree.
    private func pointInputShapes(
        state: SingleFunctionContext,
        pointsInputName: String,
        pointLabelsInputName: String
    ) throws -> (
        pointsDescriptor: NDArrayDescriptor, labelsDescriptor: NDArrayDescriptor,
        batchSize: Int, queryCount: Int, pointsPerQuery: Int
    ) {
        guard case .ndArray(let pointsDescriptor) = state.descriptor.inputDescriptor(of: pointsInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for points input '\(pointsInputName)'"
            )
        }
        guard case .ndArray(let labelsDescriptor) = state.descriptor.inputDescriptor(of: pointLabelsInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for point labels input '\(pointLabelsInputName)'"
            )
        }
        // batched_points shape: [B, Q, P, 2]; batched_point_labels: [B, Q, P]
        guard pointsDescriptor.shape.count == 4, labelsDescriptor.shape.count == 3 else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Unexpected point input ranks: points=\(pointsDescriptor.shape), labels=\(labelsDescriptor.shape)"
            )
        }
        let batchSize = pointsDescriptor.shape[0]
        let queryCount = pointsDescriptor.shape[1]
        let pointsPerQuery = pointsDescriptor.shape[2]
        guard
            batchSize == labelsDescriptor.shape[0],
            queryCount == labelsDescriptor.shape[1],
            pointsPerQuery == labelsDescriptor.shape[2]
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Points/labels shape mismatch: points=\(pointsDescriptor.shape) labels=\(labelsDescriptor.shape)"
            )
        }
        return (pointsDescriptor, labelsDescriptor, batchSize, queryCount, pointsPerQuery)
    }

    /// Build the `points` `[B, Q, P, 2]` and `labels` `[B, Q, P]` NDArrays from `queries`.
    ///
    /// Sentinel `-1` marks unused slots: the EfficientSAM prompt encoder routes them to its
    /// `invalid_points` embedding so they contribute nothing to the mask. The user's queries
    /// fill batch slot 0 and replicate identically across any additional batches.
    private func buildPointTensors(
        queries: [[PointQuery.Point]],
        pointsDescriptor: NDArrayDescriptor,
        labelsDescriptor: NDArrayDescriptor,
        batchSize: Int,
        queryCount: Int,
        pointsPerQuery: Int,
        scaleX: Float,
        scaleY: Float
    ) -> (points: NDArray, labels: NDArray) {
        let totalElements = batchSize * queryCount * pointsPerQuery
        var pointFloats = [Float](repeating: -1.0, count: totalElements * 2)
        var labelFloats = [Float](repeating: -1.0, count: totalElements)
        for batchIndex in 0..<batchSize {
            for (queryIndex, query) in queries.enumerated() {
                for (pointIndex, point) in query.enumerated() {
                    let queryPointIndex =
                        (batchIndex * queryCount + queryIndex) * pointsPerQuery + pointIndex
                    pointFloats[queryPointIndex * 2 + 0] = point.x * scaleX
                    pointFloats[queryPointIndex * 2 + 1] = point.y * scaleY
                    labelFloats[queryPointIndex] = Float(point.label.rawValue)
                }
            }
        }

        var pointsArray = NDArray(descriptor: pointsDescriptor)
        if pointsDescriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            fillNDArray(&pointsArray, as: Float16.self, with: pointFloats.map(Float16.init))
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            fillNDArray(&pointsArray, as: Float.self, with: pointFloats)
        }

        var labelsArray = NDArray(descriptor: labelsDescriptor)
        if labelsDescriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            fillNDArray(&labelsArray, as: Float16.self, with: labelFloats.map(Float16.init))
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            fillNDArray(&labelsArray, as: Float.self, with: labelFloats)
        }
        return (pointsArray, labelsArray)
    }

    // MARK: - Single-function inference helpers

    @discardableResult
    private func runSingleFunctionTextInference(
        state: SingleFunctionContext,
        inputs: [String: NDArray]
    ) async throws -> SegmentationOutput {
        guard let boxesOutputName = state.boxesOutputName,
            let logitsOutputName = state.logitsOutputName,
            let presenceLogitsOutputName = state.presenceLogitsOutputName,
            let semanticSegOutputName = state.semanticSegOutputName
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Text inference invoked on a non-text model."
            )
        }
        var outputs = try await state.function.run(inputs: inputs)
        guard let masks = outputs.remove(state.masksOutputName)?.ndArray,
            let boxes = outputs.remove(boxesOutputName)?.ndArray,
            let logits = outputs.remove(logitsOutputName)?.ndArray,
            let presence = outputs.remove(presenceLogitsOutputName)?.ndArray,
            let semantic = outputs.remove(semanticSegOutputName)?.ndArray
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Missing one or more outputs after run."
            )
        }

        return SegmentationOutput(
            predictedMasks: flattenAsFloat(masks),
            masksShape: masks.shape,
            predictedBoxes: flattenAsFloat(boxes),
            predictedLogits: flattenAsFloat(logits),
            presenceLogits: flattenAsFloat(presence),
            semanticSegment: flattenAsFloat(semantic),
            semanticSegmentShape: semantic.shape
        )
    }

    @discardableResult
    private func runSingleFunctionPointInference(
        state: SingleFunctionContext,
        inputs: [String: NDArray],
        pointQuery: PointQuery,
        imageSize: CGSize
    ) async throws -> SegmentationOutput {
        guard let iouScoresOutputName = state.iouScoresOutputName else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Point inference invoked on a non-point model."
            )
        }
        var outputs = try await state.function.run(inputs: inputs)
        guard let masksOutput = outputs.remove(state.masksOutputName)?.ndArray,
            let iouScoresOutput = outputs.remove(iouScoresOutputName)?.ndArray
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Missing one or more outputs after run."
            )
        }

        // EfficientSAM emits [B, Q, K, H, W] (K=3 candidates per query) and [B, Q, K] scores.
        // Pick the highest-scoring candidate per query so output is [B, Q, H, W] / [B, Q].
        let (bestMasks, bestShape, bestScores) = try bestOfKMasks(
            masks: masksOutput, scores: iouScoresOutput
        )

        // Drop sentinel-padded query slots so the postprocessor never surfaces phantom
        // segments from EfficientSAM's `invalid_points` embedding. `pointQuery` here is
        // the resolved query — its count equals the user's queries (or the segment-
        // everything grid, which fully fills Q anyway).
        let (predictedMasks, masksShape, predictedScores) = Self.sliceUserQueries(
            flatMasks: bestMasks, flatScores: bestScores,
            shape: bestShape, userQueryCount: pointQuery.queries.count
        )
        let predictedBoxes = Self.extractBoxesFromPointQuery(pointQuery, imageSize: imageSize)
        return SegmentationOutput(
            predictedMasks: predictedMasks,
            masksShape: masksShape,
            predictedBoxes: predictedBoxes,
            predictedLogits: [],
            predictedScores: predictedScores,
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )
    }

    // MARK: - Multi-function text path

    private func warmupMultiFunctionTextModel(state: MultiFunctionContext) async throws {
        guard
            case .ndArray(let imageDescriptor) = state.imageEncodeDescriptor.inputDescriptor(
                of: state.imageInputName),
            case .ndArray(let textDescriptor) = state.textEncodeDescriptor.inputDescriptor(of: state.textInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image_encode/text_encode inputs"
            )
        }
        let imageArray = NDArray(descriptor: imageDescriptor)
        var textArray = NDArray(descriptor: textDescriptor)
        fillNDArray(&textArray, as: Int32.self, count: textDescriptor.shape.reduce(1, *)) { _ in
            CLIPTokenizer.eotTokenId
        }

        try await runMultiFunctionInference(
            state: state,
            imageArray: imageArray,
            textArray: textArray
        )
    }

    private func runMultiFunctionTextSegment(
        state: MultiFunctionContext,
        image: CGImage,
        textQuery: TextQuery,
        parameters: SegmentationParameters
    ) async throws -> SegmentationOutput {
        guard
            case .ndArray(let imageDescriptor) = state.imageEncodeDescriptor.inputDescriptor(
                of: state.imageInputName)
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "No array descriptor for image input '\(state.imageInputName)'"
            )
        }
        let imageArray = try Self.buildImageNDArray(
            from: image, descriptor: imageDescriptor, parameters: parameters
        )

        let textArray: NDArray
        switch textQuery {
        case .prompt:
            throw SegmentationRuntimeError.invalidConfiguration(
                "TextQuery.prompt must be resolved to .tokens by ImageSegmenter before reaching the engine."
            )
        case .tokens(let textTokensBatch):
            guard
                case .ndArray(let textDescriptor) = state.textEncodeDescriptor.inputDescriptor(
                    of: state.textInputName)
            else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "No array descriptor for text input '\(state.textInputName)'"
                )
            }
            textArray = Self.buildTextTokensNDArray(
                tokensBatch: textTokensBatch, descriptor: textDescriptor
            )
        case .embeddings:
            throw SegmentationRuntimeError.unsupportedEngine(
                "Multi-function segmentation assets accept token IDs only — "
                    + "the text_encode graph already projects them internally."
            )
        }

        return try await runMultiFunctionInference(
            state: state,
            imageArray: imageArray,
            textArray: textArray
        )
    }

    /// Run image_encode → text_encode → detect, threading encoder outputs into `detect`.
    /// Outputs are pulled out of each `function.run` return dict — never pre-allocated.
    @discardableResult
    private func runMultiFunctionInference(
        state: MultiFunctionContext,
        imageArray: NDArray,
        textArray: NDArray
    ) async throws -> SegmentationOutput {
        var imageOutputs = try await state.imageEncode.run(
            inputs: [state.imageInputName: imageArray]
        )
        guard let backboneFeatures = imageOutputs.remove(state.backboneFeaturesOutputName)?.ndArray
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Missing '\(state.backboneFeaturesOutputName)' output from image_encode."
            )
        }

        var textOutputs = try await state.textEncode.run(inputs: [state.textInputName: textArray])
        guard let textFeatures = textOutputs.remove(state.textFeaturesOutputName)?.ndArray else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Missing '\(state.textFeaturesOutputName)' output from text_encode."
            )
        }

        var detectOutputs = try await state.detect.run(
            inputs: [
                state.backboneFeaturesInputName: backboneFeatures,
                state.textFeaturesInputName: textFeatures,
            ]
        )
        guard let masks = detectOutputs.remove(state.masksOutputName)?.ndArray,
            let boxes = detectOutputs.remove(state.boxesOutputName)?.ndArray,
            let logits = detectOutputs.remove(state.logitsOutputName)?.ndArray,
            let presence = detectOutputs.remove(state.presenceLogitsOutputName)?.ndArray,
            let semantic = detectOutputs.remove(state.semanticSegOutputName)?.ndArray
        else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Missing one or more outputs after detect.run."
            )
        }

        return SegmentationOutput(
            predictedMasks: flattenAsFloat(masks),
            masksShape: masks.shape,
            predictedBoxes: flattenAsFloat(boxes),
            predictedLogits: flattenAsFloat(logits),
            presenceLogits: flattenAsFloat(presence),
            semanticSegment: flattenAsFloat(semantic),
            semanticSegmentShape: semantic.shape
        )
    }

    // MARK: - Shared NDArray builders

    /// Resize, normalize, and dtype-convert `image` into the model's image-input NDArray.
    /// Used by both single-function and multi-function text paths — the preprocessing config
    /// (mean/std, target size) is identical between them.
    static func buildImageNDArray(
        from image: CGImage,
        descriptor: NDArrayDescriptor,
        parameters: SegmentationParameters
    ) throws -> NDArray {
        let expectedShape = descriptor.shape
        guard expectedShape.count == 4 else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Expected 4-dimensional input shape, got \(expectedShape.count)"
            )
        }
        let height = expectedShape[2]
        let width = expectedShape[3]
        let floatPixels = try ImagePreprocessor(
            targetSize: CGSize(width: width, height: height),
            mean: parameters.normalizationMeans,
            std: parameters.normalizationStds,
            rescaleFactor: 1.0
        ).preprocessCHW(cgImage: image)

        var array = NDArray(descriptor: descriptor)
        if descriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            fillNDArray(&array, as: Float16.self, with: floatPixels.map(Float16.init))
            #else
            fatalError("Float16 is not supported on this platform")
            #endif
        } else {
            fillNDArray(&array, as: Float.self, with: floatPixels)
        }
        return array
    }

    /// Pack `tokensBatch` into an `[batch, sequenceLength]` int32 NDArray, padding with
    /// `CLIPTokenizer.eotTokenId` when a row is shorter than `sequenceLength`.
    static func buildTextTokensNDArray(
        tokensBatch: [[Int32]],
        descriptor: NDArrayDescriptor
    ) -> NDArray {
        let batchSize = descriptor.shape[0]
        let sequenceLength = descriptor.shape[1]
        var array = NDArray(descriptor: descriptor)
        fillNDArray(&array, as: Int32.self, count: batchSize * sequenceLength) { idx in
            tokenValue(
                at: idx, sequenceLength: sequenceLength, batch: tokensBatch,
                eotTokenId: CLIPTokenizer.eotTokenId)
        }
        return array
    }

    // MARK: - Best-of-K helpers (point path; pure-data, easily unit testable)

    /// For [B, Q, K, H, W] masks + [B, Q, K] scores, pick the highest-scoring K per (B, Q).
    /// Returns flat `[B, Q, H, W]` masks, the new shape, and `[B, Q]` scores.
    /// Throws `invalidConfiguration` if the masks tensor is not 5D.
    private func bestOfKMasks(
        masks: NDArray, scores: NDArray
    ) throws -> (masks: [Float], shape: [Int], scores: [Float]) {
        let masksShape = masks.shape
        let allMasks = flattenAsFloat(masks)
        let allScores = flattenAsFloat(scores)
        guard masksShape.count == 5 else {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Point inference expected [B, Q, K, H, W] masks; got rank \(masksShape.count) shape \(masksShape)."
            )
        }
        return Self.reduceBestOfK(flatMasks: allMasks, flatScores: allScores, shape: masksShape)
    }

    /// Pure-data reduction for `bestOfKMasks` — accepts already-flattened `[B,Q,K,H,W]` masks
    /// and `[B,Q,K]` scores, picks the top-scoring K per (B, Q), and returns flat `[B,Q,H,W]`
    /// masks plus `[B,Q]` scores. Exists for unit testing without touching `NDArray`.
    static func reduceBestOfK(
        flatMasks: [Float], flatScores: [Float], shape: [Int]
    ) -> (masks: [Float], shape: [Int], scores: [Float]) {
        precondition(shape.count == 5, "reduceBestOfK expects [B, Q, K, H, W] shape")
        let batchSize = shape[0]
        let queryCount = shape[1]
        let candidateCount = shape[2]
        let height = shape[3]
        let width = shape[4]
        let pixelCount = height * width
        var outMasks = [Float]()
        outMasks.reserveCapacity(batchSize * queryCount * pixelCount)
        var outScores = [Float]()
        outScores.reserveCapacity(batchSize * queryCount)
        for batchIndex in 0..<batchSize {
            for queryIndex in 0..<queryCount {
                let scoreBase = (batchIndex * queryCount + queryIndex) * candidateCount
                var bestCandidate = 0
                var bestScore = flatScores[scoreBase]
                for candidate in 1..<candidateCount where flatScores[scoreBase + candidate] > bestScore {
                    bestScore = flatScores[scoreBase + candidate]
                    bestCandidate = candidate
                }
                let maskBase =
                    ((batchIndex * queryCount + queryIndex) * candidateCount + bestCandidate) * pixelCount
                outMasks.append(contentsOf: flatMasks[maskBase..<(maskBase + pixelCount)])
                outScores.append(bestScore)
            }
        }
        return (outMasks, [batchSize, queryCount, height, width], outScores)
    }

    /// Trim phantom slots off a `[B, Q_full, H, W]` masks tensor + `[B, Q_full]` scores tensor,
    /// keeping only the leading `userQueryCount` queries per batch — the slots `buildPointTensors`
    /// fills with the user's queries (the rest are sentinel-padded for EfficientSAM's
    /// `invalid_points` embedding). Returns `[B, userQueryCount, H, W]` masks plus the matching
    /// shape and `[B, userQueryCount]` scores. Caller must ensure `userQueryCount ≤ Q`.
    /// No-op when `userQueryCount == Q` (segment-everything path).
    static func sliceUserQueries(
        flatMasks: [Float], flatScores: [Float], shape: [Int], userQueryCount: Int
    ) -> (masks: [Float], shape: [Int], scores: [Float]) {
        precondition(shape.count == 4, "sliceUserQueries expects [B, Q, H, W] shape")
        let batchSize = shape[0]
        let queryCount = shape[1]
        let height = shape[2]
        let width = shape[3]
        precondition(userQueryCount <= queryCount, "userQueryCount must be ≤ Q")
        if userQueryCount == queryCount {
            return (flatMasks, shape, flatScores)
        }
        let pixelCount = height * width
        var outMasks = [Float]()
        outMasks.reserveCapacity(batchSize * userQueryCount * pixelCount)
        var outScores = [Float]()
        outScores.reserveCapacity(batchSize * userQueryCount)
        for batchIndex in 0..<batchSize {
            let masksRowStart = batchIndex * queryCount * pixelCount
            outMasks.append(
                contentsOf: flatMasks[masksRowStart..<(masksRowStart + userQueryCount * pixelCount)])
            let scoresRowStart = batchIndex * queryCount
            outScores.append(
                contentsOf: flatScores[scoresRowStart..<(scoresRowStart + userQueryCount)])
        }
        return (outMasks, [batchSize, userQueryCount, height, width], outScores)
    }

    // MARK: - Point query resolution + validation

    /// Resolve the user's `PointQuery` against the model's static `[Q, P]` shape.
    ///
    /// - Empty `pointQuery.queries` → segment-everything: a `gridSide × gridSide` grid of
    ///   foreground points, one per query (`gridSide = sqrt(queryCount)`).
    /// - Non-empty queries are validated for structural correctness (each query has at least
    ///   one point with finite, in-bounds coordinates; box corners come paired; at most one
    ///   corner of each kind per query) and then for size against `queryCount` /
    ///   `pointsPerQuery`. Returns the queries as-is.
    static func resolveQueries(
        _ pointQuery: PointQuery,
        queryCount: Int,
        pointsPerQuery: Int,
        imageWidth: Float,
        imageHeight: Float
    ) throws -> [[PointQuery.Point]] {
        if pointQuery.queries.isEmpty {
            let gridSide = Int(Double(queryCount).squareRoot())
            guard gridSide * gridSide == queryCount else {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Segment-everything requires a perfect-square num_queries (got \(queryCount))."
                )
            }
            var grid: [[PointQuery.Point]] = []
            grid.reserveCapacity(queryCount)
            for row in 0..<gridSide {
                for col in 0..<gridSide {
                    let x = imageWidth * (Float(col) + 0.5) / Float(gridSide)
                    let y = imageHeight * (Float(row) + 0.5) / Float(gridSide)
                    grid.append([PointQuery.Point(x: x, y: y, label: .foreground)])
                }
            }
            return grid
        }
        try validate(
            queries: pointQuery.queries,
            queryCount: queryCount,
            pointsPerQuery: pointsPerQuery,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
        return pointQuery.queries
    }

    static func validate(
        queries: [[PointQuery.Point]],
        queryCount: Int,
        pointsPerQuery: Int,
        imageWidth: Float,
        imageHeight: Float
    ) throws {
        for (queryIndex, query) in queries.enumerated() {
            if query.isEmpty {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Query \(queryIndex) is empty. Each query must contain at least one point."
                )
            }
            for (pointIndex, point) in query.enumerated() {
                guard point.x.isFinite, point.y.isFinite else {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Query \(queryIndex) point \(pointIndex) has non-finite coordinates "
                            + "(x=\(point.x), y=\(point.y))."
                    )
                }
                if point.x < 0 || point.x > imageWidth || point.y < 0 || point.y > imageHeight {
                    throw SegmentationRuntimeError.invalidConfiguration(
                        "Query \(queryIndex) point \(pointIndex) at (\(point.x), \(point.y)) "
                            + "is outside image bounds (\(Int(imageWidth))×\(Int(imageHeight)))."
                    )
                }
            }
            let topLeftCount = query.filter { $0.label == .boxTopLeft }.count
            let bottomRightCount = query.filter { $0.label == .boxBottomRight }.count
            if topLeftCount > 1 {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Query \(queryIndex) has \(topLeftCount) box-top-left points; expected at most 1 per query."
                )
            }
            if bottomRightCount > 1 {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Query \(queryIndex) has \(bottomRightCount) box-bottom-right points; expected at most 1 per query."
                )
            }
            if topLeftCount != bottomRightCount {
                throw SegmentationRuntimeError.invalidConfiguration(
                    "Query \(queryIndex) has a box corner without its pair: "
                        + "\(topLeftCount) box-top-left, \(bottomRightCount) box-bottom-right. "
                        + "Box prompts require both corners."
                )
            }
        }
        if queries.count > queryCount {
            throw SegmentationRuntimeError.invalidConfiguration(
                "PointQuery has \(queries.count) queries but model expects ≤ \(queryCount). "
                    + "Re-export with --num-queries \(queries.count) (or higher)."
            )
        }
        for (queryIndex, query) in queries.enumerated() where query.count > pointsPerQuery {
            throw SegmentationRuntimeError.invalidConfiguration(
                "Query \(queryIndex) has \(query.count) points but model expects ≤ \(pointsPerQuery). "
                    + "Re-export with --num-pts \(query.count) (or higher)."
            )
        }
    }

    /// For each query, emit `[x0, y0, x1, y1]` normalized to `[0, 1]` if the query has
    /// both `.boxTopLeft` and `.boxBottomRight` points; otherwise emit zeros.
    /// Output is flat `[Q * 4]` (single batch — the engine fixes B at 1).
    static func extractBoxesFromPointQuery(_ pointQuery: PointQuery, imageSize: CGSize) -> [Float] {
        guard imageSize.width > 0, imageSize.height > 0, !pointQuery.queries.isEmpty else { return [] }
        let inverseWidth = 1.0 / Float(imageSize.width)
        let inverseHeight = 1.0 / Float(imageSize.height)
        var flatBoxes = [Float](repeating: 0, count: pointQuery.queries.count * 4)
        for (queryIndex, query) in pointQuery.queries.enumerated() {
            guard let topLeft = query.first(where: { $0.label == .boxTopLeft }),
                let bottomRight = query.first(where: { $0.label == .boxBottomRight })
            else { continue }
            flatBoxes[queryIndex * 4 + 0] = topLeft.x * inverseWidth
            flatBoxes[queryIndex * 4 + 1] = topLeft.y * inverseHeight
            flatBoxes[queryIndex * 4 + 2] = bottomRight.x * inverseWidth
            flatBoxes[queryIndex * 4 + 3] = bottomRight.y * inverseHeight
        }
        return flatBoxes
    }

    // MARK: - Static name-discovery helpers

    static func findImageInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("pixel") || l.contains("image")
        }
    }

    static func findTextInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            // Match input_id / token / text, but exclude text_features (a `detect` input that
            // also contains "text") so it isn't mistaken for the token input.
            return (l.contains("input_id") || l.contains("token") || l.contains("text"))
                && !l.contains("feat")
        }
    }

    static func findEmbeddingsInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("embed") || l.contains("text_feat")
        }
    }

    static func findPointsInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("point") && !l.contains("label")
        }
    }

    static func findPointLabelsInputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("point") && l.contains("label")
        }
    }

    static func findBackboneFeaturesName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("backbone") }
    }

    static func findTextFeaturesName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("text_feat") || l == "text_features"
        }
    }

    static func findMasksOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("mask") }
    }

    static func findBoxesOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("box") }
    }

    static func findLogitsOutputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("logit") && !l.contains("presence")
        }
    }

    static func findPresenceOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("presence") }
    }

    static func findIouScoresOutputName(in names: [String]) -> String? {
        names.first {
            let l = $0.lowercased()
            return l.contains("iou") || (l.contains("score") && !l.contains("logit"))
        }
    }

    static func findSemanticOutputName(in names: [String]) -> String? {
        names.first { $0.lowercased().contains("semantic") }
    }

    // MARK: - Token / embedding generators

    static func tokenValue(at idx: Int, sequenceLength: Int, batch: [[Int32]], eotTokenId: Int32) -> Int32 {
        let batchIndex = idx / sequenceLength
        let tokenIndex = idx % sequenceLength
        let tokens = batchIndex < batch.count ? batch[batchIndex] : []
        return tokenIndex < tokens.count ? tokens[tokenIndex] : eotTokenId
    }

    static func embeddingValue(at idx: Int, sequenceLength: Int, hiddenSize: Int, batch: [[[Float]]]) -> Float {
        let batchIndex = idx / (sequenceLength * hiddenSize)
        let sequenceIndex = (idx / hiddenSize) % sequenceLength
        let hiddenIndex = idx % hiddenSize
        let sequenceEmbeddings = batchIndex < batch.count ? batch[batchIndex] : []
        let tokenEmbedding =
            sequenceIndex < sequenceEmbeddings.count
            ? sequenceEmbeddings[sequenceIndex]
            : [Float](repeating: 0, count: hiddenSize)
        return hiddenIndex < tokenEmbedding.count ? tokenEmbedding[hiddenIndex] : 0
    }
}
