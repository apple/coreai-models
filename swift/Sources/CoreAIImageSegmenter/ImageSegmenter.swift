// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// High-level runner that combines tokenization, engine inference, and output decoding.
///
/// ```swift
/// // Text-guided (SAM3):
/// let runner = try ImageSegmenter(engine: sam3Engine, tokenizerFolder: url)
/// let segments = try await runner.segment(image: cgImage, prompt: "cat")
///
/// // Single click (EfficientSAM):
/// let runner = try ImageSegmenter(engine: efficientSamEngine)
/// let pq = PointQuery(points: [.init(x: 320, y: 240)])
/// let segments = try await runner.segment(image: cgImage, pointQuery: pq)
///
/// // Box prompt — one query with two points:
/// let box = PointQuery(points: [
///     .init(x: 100, y: 100, label: .boxTopLeft),
///     .init(x: 400, y: 300, label: .boxBottomRight),
/// ])
///
/// // Multiple independent prompts — Q queries, P points each:
/// let multi = PointQuery(queries: [
///     [.init(x: 100, y: 100)],
///     [.init(x: 300, y: 300)],
/// ])
/// ```
public struct ImageSegmenter {
    private let engine: CoreAISegmentationEngine
    private let tokenizer: CLIPTokenizer?

    /// Designated init: takes a pre-initialized engine and a tokenizer.
    ///
    /// `tokenizer` is required when `engine.supportsTextQuery` is `true` (throws otherwise),
    /// and ignored when the engine is point-only.
    public init(engine: CoreAISegmentationEngine, tokenizer: CLIPTokenizer? = nil) throws {
        self.engine = engine
        if engine.supportsTextQuery {
            guard let tokenizer else {
                throw SegmentationRuntimeError.modelLoadFailed(
                    "Text-capable engine requires a tokenizer; pass one via init(engine:tokenizerFolder:)."
                )
            }
            self.tokenizer = tokenizer
        } else {
            self.tokenizer = nil
        }
    }

    /// Convenience init that loads a CLIP tokenizer from `tokenizerFolder` (HF-format
    /// directory containing `tokenizer.json`).
    ///
    /// `tokenizerFolder` may be `nil` when the engine is point-only.
    public init(
        engine: CoreAISegmentationEngine,
        tokenizerFolder: URL?
    ) throws {
        let tokenizer: CLIPTokenizer?
        if engine.supportsTextQuery, let folder = tokenizerFolder {
            tokenizer = try CLIPTokenizer(folder: folder)
        } else {
            tokenizer = nil
        }
        try self.init(engine: engine, tokenizer: tokenizer)
    }

    /// Warm up the engine with a dummy forward pass to trigger kernel compilation.
    public func warmup() async throws {
        try await self.engine.warmup()
    }

    /// Segment `image` using the given `textQuery`.
    ///
    /// `.prompt` is tokenized here before being passed to the engine as `.tokens`.
    /// `.tokens` and `.embeddings` are forwarded to the engine directly.
    ///
    /// Throws `SegmentationRuntimeError.unsupportedEngine` if the loaded model does not
    /// accept text queries (e.g. EfficientSAM).
    ///
    /// - Parameters:
    ///   - image: Input image (any size; the engine resizes internally).
    ///   - textQuery: Text input — a raw prompt, pre-tokenized tokens, or pre-computed embeddings.
    ///   - parameters: Decoding parameters (threshold, max segments).
    /// - Returns: A `SegmentationResponse` with segments sorted by score descending,
    ///   and a `SemanticSegmentationMap` if the model exposes a semantic head.
    public func segment(
        image: CGImage,
        textQuery: TextQuery,
        parameters: SegmentationParameters = .default
    ) async throws -> SegmentationResponse {
        guard let tokenizer else {
            throw SegmentationRuntimeError.unsupportedEngine(
                "The loaded model does not support text queries. Use segment(image:pointQuery:) instead."
            )
        }
        let resolvedQuery: TextQuery
        if case .prompt(let text) = textQuery {
            let tokens = tokenizer.encode(text, contextLength: parameters.tokenizerContextLength)
            resolvedQuery = .tokens([tokens])
        } else {
            resolvedQuery = textQuery
        }
        let output = try await self.engine.segment(image: image, textQuery: resolvedQuery, parameters: parameters)
        let inputSize = CGSize(width: image.width, height: image.height)
        return SegmentationPostprocessor.decode(output: output, inputSize: inputSize, parameters: parameters)
    }

    /// Convenience overload for the common case of a raw text prompt.
    public func segment(
        image: CGImage,
        prompt: String,
        parameters: SegmentationParameters = .default
    ) async throws -> SegmentationResponse {
        try await segment(image: image, textQuery: .prompt(prompt), parameters: parameters)
    }

    /// Segment `image` using point (and optional box) prompts.
    ///
    /// Suitable for point-guided models such as EfficientSAM. Pass an empty `PointQuery`
    /// (or call with no arguments) to segment-everything: the engine substitutes a
    /// `gridSide × gridSide` grid of foreground points where `gridSide = sqrt(num_queries)`.
    ///
    /// Throws `SegmentationRuntimeError.unsupportedEngine` if the loaded model does not
    /// accept point queries (e.g. SAM3).
    ///
    /// - Parameters:
    ///   - image: Input image (any size; the engine resizes internally).
    ///   - pointQuery: Point/box prompts in input-image pixel coordinates. Empty = segment everything.
    ///   - parameters: Decoding parameters (threshold, max segments).
    /// - Returns: A `SegmentationResponse` with segments sorted by score descending.
    public func segment(
        image: CGImage,
        pointQuery: PointQuery = PointQuery(),
        parameters: SegmentationParameters = .default
    ) async throws -> SegmentationResponse {
        guard engine.supportsPointQuery else {
            throw SegmentationRuntimeError.unsupportedEngine(
                "The loaded model does not support point queries. Use segment(image:textQuery:) instead."
            )
        }
        let output = try await self.engine.segment(image: image, pointQuery: pointQuery, parameters: parameters)
        let inputSize = CGSize(width: image.width, height: image.height)
        return SegmentationPostprocessor.decode(output: output, inputSize: inputSize, parameters: parameters)
    }

    /// Convenience initializer that loads a `ModelBundle` from a directory path.
    ///
    /// The directory must contain a `metadata.json` declaring `kind: "segmenter"`
    /// and an `assets.main` pointing at the `.aimodel` file. A `tokenizer/`
    /// subdirectory is required only for text-capable engines (e.g. SAM3); it
    /// is ignored for point-only engines (e.g. EfficientSAM).
    ///
    /// ```swift
    /// let runner = try await ImageSegmenter(resourcesAt: "~/models/my-model")
    /// let segments = try await runner.segment(image: cgImage, prompt: "cat")
    /// ```
    public init(resourcesAt path: String, parameters: SegmentationParameters = .default)
        async throws
    {
        let bundle = try ModelBundle(from: path)
        guard bundle.kind == .segmenter else {
            throw ModelBundle.BundleError.kindMismatch(expected: .segmenter, got: bundle.kind)
        }
        try bundle.verifyAssetsExisting()
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let tokenizerFolder = bundle.bundlePath.appending(path: "tokenizer")

        let engine = try await CoreAISegmentationEngine(parameters: parameters, modelURL: modelURL)
        try self.init(engine: engine, tokenizerFolder: tokenizerFolder)
    }
}
