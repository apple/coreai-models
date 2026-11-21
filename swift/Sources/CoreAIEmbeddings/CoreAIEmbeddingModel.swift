// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Runs a `kind: "embedding"` bundle for one owner: the calls are `mutating`, so two at once do
/// not compile; `QueuedEmbeddingModel` serves many callers. Each tower loads on first use, and
/// dropping the model releases its programs.
public struct CoreAIEmbeddingModel: EmbeddingModel, ~Copyable, Sendable {
    public let space: EmbeddingSpace
    public let dimension: Int
    public let matryoshkaDimensions: [Int]
    public let modalities: Set<EmbeddingModality>
    public let logitScale: Double?
    /// Whether a vector cut to a smaller width is L2-normalized again.
    let renormalizesAfterTruncation: Bool

    private let drivers: [EmbeddingModality: PipelineDriver]
    private var programs: ProgramCache
    private let text: TextPreparer

    /// Reads and checks the bundle at `url` and loads its tokenizer. Loads no program.
    ///
    /// - Throws: `ModelBundle.BundleError` for a missing or invalid asset, `DecodingError` or
    ///   `EmbeddingError` for a block that breaks the contract.
    public init(resourcesAt url: URL) async throws {
        let bundle = try EmbeddingBundle(at: url)
        try bundle.bundle.validateModelAssets()
        let text = try await InstrumentsProfiler.profile(
            InstrumentsProfiler.beginTokenizerLoad(id: url.lastPathComponent)
        ) {
            try await TextPreparer.load(from: bundle.tokenizerURL, config: bundle.block.text)
        }
        try self.init(bundle: bundle, programs: ProgramCache.coreAI, text: text)
    }

    /// A model over `bundle` with programs from `programs` and token ids from `text`.
    init(bundle: EmbeddingBundle, programs: @escaping ProgramCache.Factory, text: TextPreparer) throws {
        let block = bundle.block
        var drivers: [EmbeddingModality: PipelineDriver] = [:]
        for tower in block.towers {
            drivers[tower.modality] = try PipelineDriver(tower: tower, block: block)
        }
        space = block.space
        dimension = block.dimension
        matryoshkaDimensions = block.matryoshkaDimensions
        modalities = Set(block.towers.map(\.modality))
        logitScale = block.logitScale
        renormalizesAfterTruncation = block.renormalizeAfterTruncation
        self.drivers = drivers
        self.programs = ProgramCache(bundle: bundle.bundle, factory: programs)
        self.text = text
    }

    /// Texts go in after the bundle's default prompt (`default_prompt_name`), if any; a text longer
    /// than the bundle's token limit is cut, and its embedding says so in `status`.
    ///
    /// - Throws: `EmbeddingError.modalityNotInBundle`, `.unsupportedDimension`,
    ///   `.imageConversionFailed`, a program failure, or `CancellationError` when the task is
    ///   cancelled; cancellation takes effect before the next program call.
    public mutating func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding] {
        try await InstrumentsProfiler.profile(InstrumentsProfiler.beginEmbed(inputs: inputs.count)) {
            try await run(inputs, dimensions: dimensions)
        }
    }

    // MARK: - Private

    /// The body of `embed(_:dimensions:)`.
    private mutating func run(_ inputs: [EmbeddingInput], dimensions: Int?) async throws -> [Embedding] {
        try check(dimensions: dimensions)
        let textDriver = try inputs.contains(where: \.isText) ? driver(for: .text) : nil
        let imageDriver = try inputs.contains(where: { !$0.isText }) ? driver(for: .image) : nil
        var vectors = [[Float]](repeating: [], count: inputs.count)
        var statuses = [Embedding.Status](repeating: .complete, count: inputs.count)
        var texts: [(index: Int, tokens: [Int32])] = []
        var images: [(index: Int, image: RGB8Image)] = []
        for (index, input) in inputs.enumerated() {
            try Task.checkCancellation()
            switch input.content {
            case .text(let string):
                let (ids, count) = text.tokenize(string)
                if count > ids.count { statuses[index] = .truncated(tokens: count, limit: text.config.maxTokens) }
                texts.append((index, ids))
            case .image(let image):
                images.append((index, try RGB8Image(cgImage: image)))
            }
        }
        if let driver = textDriver {
            let rows = driver.rows(tokens: texts.map(\.tokens))
            for (vector, item) in zip(try await driver.run(rows, programs: &programs), texts) {
                vectors[item.index] = vector
            }
        }
        if let driver = imageDriver {
            try Task.checkCancellation()
            let rows = try driver.rows(images: images.map(\.image))
            for (vector, item) in zip(try await driver.run(rows, programs: &programs), images) {
                vectors[item.index] = vector
            }
        }
        return try zip(vectors, statuses).map { vector, status in
            guard vector.allSatisfy(\.isFinite) else {
                throw EmbeddingError.programFailed("embedding returned an unexpected output: non-finite values")
            }
            return Embedding(vector: vector, space: space, status: status)
                .cut(to: dimensions ?? dimension, renormalize: renormalizesAfterTruncation)
        }
    }

    private func driver(for modality: EmbeddingModality) throws -> PipelineDriver {
        guard let driver = drivers[modality] else { throw EmbeddingError.modalityNotInBundle(modality) }
        return driver
    }
}
