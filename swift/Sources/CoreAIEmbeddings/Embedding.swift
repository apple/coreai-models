// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import Foundation

/// One input's vector, in the space of the bundle that made it. `Codable`: the space and status
/// travel with the vector when it is stored.
public struct Embedding: Sendable, Equatable, Codable {
    public let vector: [Float]
    /// Vectors are comparable only within one space.
    public let space: EmbeddingSpace
    /// Whether the model saw the whole input.
    public let status: Status

    /// How much of its input an embedding covers.
    public enum Status: Sendable, Equatable, Codable {
        /// The model saw the whole input; always for images.
        case complete
        /// The text had `tokens` tokens, special tokens and default prompt included; the model saw
        /// the first `limit` of them (the bundle's `text.truncation.max_tokens`), with EOS kept last
        /// when the bundle says so.
        case truncated(tokens: Int, limit: Int)
    }

    public init(vector: [Float], space: EmbeddingSpace, status: Status = .complete) {
        self.vector = vector
        self.space = space
        self.status = status
    }

    public var dimension: Int { vector.count }

    /// Cosine similarity with `other`, from -1 to 1. Two zero vectors score 1; one scores 0. The
    /// statuses do not matter.
    ///
    /// - Throws: `EmbeddingError.incompatibleEmbeddings` when `other` is from another space or has
    ///   another width.
    public func cosine(_ other: Embedding) throws -> Float {
        guard space == other.space, dimension == other.dimension else {
            throw EmbeddingError.incompatibleEmbeddings(
                "\(space.id)@\(space.revision.prefix(8)) [\(dimension)] vs "
                    + "\(other.space.id)@\(other.space.revision.prefix(8)) [\(other.dimension)]")
        }
        return cosineSimilarity(vector, other.vector)
    }

    /// The first `width` values, L2-normalized again when `renormalize` is set; the status stays.
    func cut(to width: Int, renormalize: Bool) -> Embedding {
        var v = Array(vector.prefix(width))
        if width < vector.count && renormalize {
            let norm = vDSP.sumOfSquares(v).squareRoot()
            if norm > 0 { v = vDSP.divide(v, norm) }
        }
        return Embedding(vector: v, space: space, status: status)
    }
}
