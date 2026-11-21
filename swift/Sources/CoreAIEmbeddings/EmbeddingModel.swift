// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// A model that turns texts and images into vectors of one space: `CoreAIEmbeddingModel` for one
/// owner, `QueuedEmbeddingModel` for many callers. Generic code takes `inout some EmbeddingModel & ~Copyable`.
public protocol EmbeddingModel: ~Copyable {
    /// Vectors are comparable only within one space.
    var space: EmbeddingSpace { get }
    /// The native vector width.
    var dimension: Int { get }
    /// The widths `embed` can return, native first.
    var matryoshkaDimensions: [Int] { get }
    var modalities: Set<EmbeddingModality> { get }
    /// `logits = logitScale * cosine`, for models trained with one (CLIP).
    var logitScale: Double? { get }

    /// Embeds `inputs`, texts and images in any mix, and returns one vector per input in order.
    ///
    /// - Parameter dimensions: one of `matryoshkaDimensions`; the native width when `nil`.
    mutating func embed(_ inputs: [EmbeddingInput], dimensions: Int?) async throws -> [Embedding]
}

extension EmbeddingModel where Self: ~Copyable {
    /// Embeds `inputs` at the native width.
    public mutating func embed(_ inputs: [EmbeddingInput]) async throws -> [Embedding] {
        try await embed(inputs, dimensions: nil)
    }

    /// Embeds `texts`.
    public mutating func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(texts.map { .text($0) }, dimensions: dimensions)
    }

    /// Embeds `images`.
    public mutating func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(images.map { .image($0) }, dimensions: dimensions)
    }

    /// Throws `unsupportedDimension` unless `dimensions` is `nil` or one of `matryoshkaDimensions`.
    func check(dimensions: Int?) throws {
        if let dimensions, !matryoshkaDimensions.contains(dimensions) {
            throw EmbeddingError.unsupportedDimension(dimensions, supported: matryoshkaDimensions)
        }
    }
}
