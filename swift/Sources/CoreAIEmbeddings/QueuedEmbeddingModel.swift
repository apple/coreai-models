// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreGraphics
import Foundation
import Synchronization

/// Serves one `CoreAIEmbeddingModel` to many callers.
///
/// ```swift
/// let queued = QueuedEmbeddingModel(try await CoreAIEmbeddingModel(resourcesAt: bundleURL))
/// async let a = queued.embed(texts: ["a cat"])
/// async let b = queued.embed(images: photos)
/// let (texts, images) = try await (a, b)
/// ```
///
/// Calls run in the order they arrive, each as its own batch: a call waits for the calls before it,
/// takes the model, runs and puts the model back, also when it throws. Merging the inputs of waiting
/// calls that share `dimensions` into the same program batches is a planned addition. A call
/// cancelled while it waits throws `CancellationError`.
public final class QueuedEmbeddingModel: EmbeddingModel, Sendable {
    public let space: EmbeddingSpace
    public let dimension: Int
    public let matryoshkaDimensions: [Int]
    public let modalities: Set<EmbeddingModality>
    public let logitScale: Double?

    /// The model between calls; `nil` while a call has it.
    private let model: Mutex<CoreAIEmbeddingModel?>
    private let queue: RequestQueue

    /// Serves `model`.
    public init(_ model: consuming CoreAIEmbeddingModel) {
        space = model.space
        dimension = model.dimension
        matryoshkaDimensions = model.matryoshkaDimensions
        modalities = model.modalities
        logitScale = model.logitScale
        self.model = Mutex(.some(model))
        queue = RequestQueue(maxDepth: .max)
    }

    /// `CoreAIEmbeddingModel.embed(_:dimensions:)`, after the calls before it.
    public func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding] {
        try await withModel { try await $0.embed(inputs, dimensions: dimensions) }
    }

    /// Embeds `texts`.
    public func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(texts.map { .text($0) }, dimensions: dimensions)
    }

    /// Embeds `images`.
    public func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(images.map { .image($0) }, dimensions: dimensions)
    }

    /// Calls waiting for the one that runs.
    var waitingCount: Int { queue.queuedCount }

    /// Waits for the calls before this one, then runs `body` on the model and puts it back.
    private func withModel<Result: Sendable>(
        _ body: (inout CoreAIEmbeddingModel) async throws -> Result
    ) async throws -> Result {
        let permit = try await queue.acquire()
        defer { permit.release() }
        var checkedOut = model.withLock { $0.take() }
        defer { model.withLock { swap(&$0, &checkedOut) } }
        return try await body(&checkedOut!)
    }
}
