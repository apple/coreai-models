// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

// The reader for `kind: "embedding"` bundles: `python/src/coreai_models/embedding/contract.py`
// R1-R10, plus the EmbeddingGemma 2 keys listed in `embedding/embedding_gemma2/__init__.py`.
// Decoding is lenient: keys this runtime does not read are ignored, and towers of modalities it
// does not run (audio) are skipped, so a CLAP bundle runs text.

/// A `kind: "embedding"` bundle: the common `ModelBundle` plus the decoded `embedding` block.
struct EmbeddingBundle: Sendable {
    let bundle: ModelBundle
    let block: EmbeddingBlock

    /// Reads `metadata.json` from the bundle directory at `url`.
    ///
    /// - Throws: `ModelBundle.BundleError` for a missing or malformed `metadata.json` or another
    ///   kind, `DecodingError` or `EmbeddingError` when the `embedding` block breaks the contract.
    init(at url: URL) throws {
        try self.init(bundle: ModelBundle(at: url))
    }

    init(bundle: ModelBundle) throws {
        guard bundle.kind == .embedding else {
            throw ModelBundle.BundleError.kindMismatch(expected: .embedding, got: bundle.kind)
        }
        self.bundle = bundle
        self.block = try EmbeddingBlock(metadata: bundle.raw)
    }

    /// The bundled tokenizer folder.
    var tokenizerURL: URL { bundle.bundlePath.appending(path: "tokenizer") }
}

/// The `embedding` block (R2).
struct EmbeddingBlock: Sendable {
    var space: EmbeddingSpace
    var dimension: Int
    /// `logits = logitScale * cosine` (R9); `nil` for models without one.
    var logitScale: Double?
    var components: [String: Component]
    var towers: [Tower]
    var text: TextConfig
    var image: ImageConfig?
    /// Output widths, native first: `[dimension]` unless the bundle lists Matryoshka widths.
    var matryoshkaDimensions: [Int]
    /// Whether a vector cut to a smaller width is L2-normalized again.
    var renormalizeAfterTruncation: Bool

    /// Decodes the `embedding` block of `metadata`.
    init(metadata: Data) throws {
        struct Envelope: Decodable { let embedding: EmbeddingBlock }
        self = try JSONDecoder().decode(Envelope.self, from: metadata).embedding
    }
}

/// The space a vector lives in (R2). Vectors are comparable only within one space.
public struct EmbeddingSpace: Codable, Sendable, Hashable {
    public let id: String
    public let revision: String

    public init(id: String, revision: String) {
        self.id = id
        self.revision = revision
    }
}

/// A tower's input kind.
public enum EmbeddingModality: String, Sendable { case text, image }

/// The components a modality runs, in order (R4). The last one's `embedding` output is the vector.
struct Tower: Sendable {
    var modality: EmbeddingModality
    var pipeline: [String]
}

// MARK: - Decoding

extension EmbeddingBlock: Decodable {
    enum CodingKeys: String, CodingKey {
        case space, dimension, components, towers, text, image
        case logitScale = "logit_scale"
        case matryoshkaDimensions = "matryoshka_dimensions"
        case renormalizeAfterTruncation = "renormalize_after_truncation"
    }

    private struct RawTower: Decodable {
        let modality: String
        let pipeline: [String]
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        space = try c.decode(EmbeddingSpace.self, forKey: .space)
        dimension = try c.decode(Int.self, forKey: .dimension)
        logitScale = try c.decodeIfPresent(Double.self, forKey: .logitScale)
        components = try c.decode([String: Component].self, forKey: .components)
        towers = try c.decode([RawTower].self, forKey: .towers).compactMap { t in
            EmbeddingModality(rawValue: t.modality).map { Tower(modality: $0, pipeline: t.pipeline) }
        }
        text = try c.decode(TextConfig.self, forKey: .text)
        image = try c.decodeIfPresent(ImageConfig.self, forKey: .image)
        let widths = try c.decodeIfPresent([Int].self, forKey: .matryoshkaDimensions) ?? []
        matryoshkaDimensions = widths.isEmpty ? [dimension] : widths
        renormalizeAfterTruncation = try c.decodeIfPresent(Bool.self, forKey: .renormalizeAfterTruncation) ?? false
        // `Embedding.status` reports a cut at `max_tokens`, so every text function must hold it.
        for name in towers.first(where: { $0.modality == .text })?.pipeline ?? [] {
            if let longest = components[name]?.functions.compactMap(\.length?.upperBound).max(),
                text.maxTokens > longest
            {
                throw EmbeddingError.invalidMetadata(
                    "text.truncation.max_tokens \(text.maxTokens) exceeds \(name)'s longest function, \(longest) tokens"
                )
            }
        }
    }
}
