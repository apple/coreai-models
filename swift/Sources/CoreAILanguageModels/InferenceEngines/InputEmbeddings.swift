// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation

/// Pre-computed embeddings ready for injection into an LLM decoder.
///
/// Used by multimodal engines to pass vision/audio embeddings into the
/// language model. The engine performs scatter-merge: replacing placeholder
/// token positions with these embeddings before the first forward pass.
public struct InputEmbeddings: Sendable {
    /// The embedding tensor, shape [batch, seq_len, hidden_dim].
    /// Scalar type matches the LLM's expected input (float16, bFloat16, etc.).
    public let embeddings: NDArray

    /// Positions in the token sequence where embeddings replace placeholders.
    public let embeddingPositions: Range<Int>

    public init(embeddings: NDArray, embeddingPositions: Range<Int>) throws {
        guard embeddings.shape.count == 3 else {
            throw InferenceRuntimeError.invalidArgument(
                "InputEmbeddings requires 3D embeddings [batch, seq_len, hidden_dim], "
                    + "got shape with \(embeddings.shape.count) dimensions")
        }
        self.embeddings = embeddings
        self.embeddingPositions = embeddingPositions
    }

    /// Number of embedding tokens (seq_len dimension).
    public var tokenCount: Int { embeddings.shape[1] }

    /// `parts` joined along the sequence dimension, in order: the input for a prompt with one
    /// image placeholder run per part. The parts must have batch size 1 and the same hidden size and
    /// scalar type.
    package static func concatenating(_ parts: [InputEmbeddings]) throws -> InputEmbeddings {
        guard let first = parts.first else {
            throw InferenceRuntimeError.invalidArgument("InputEmbeddings.concatenating needs at least one part")
        }
        if parts.count == 1 { return first }
        let hiddenDim = first.embeddings.shape[2]
        let scalarType = first.embeddings.scalarType
        for part in parts {
            let shape = part.embeddings.shape
            guard shape[0] == 1, shape[2] == hiddenDim, part.embeddings.scalarType == scalarType else {
                throw InferenceRuntimeError.invalidArgument(
                    "InputEmbeddings.concatenating: part of shape \(shape) and type \(part.embeddings.scalarType) "
                        + "does not match [1, _, \(hiddenDim)] \(scalarType)")
            }
        }
        let totalTokens = parts.reduce(0) { $0 + $1.tokenCount }
        var joined = NDArray(shape: [1, totalTokens, hiddenDim], scalarType: scalarType)
        switch scalarType {
        case .float16, .bfloat16:
            copy(parts, into: &joined, as: Float16.self)
        case .float32:
            copy(parts, into: &joined, as: Float.self)
        default:
            throw InferenceRuntimeError.invalidInputType(
                "InputEmbeddings.concatenating: unsupported embedding scalar type \(scalarType)")
        }
        return try InputEmbeddings(embeddings: joined, embeddingPositions: 0..<totalTokens)
    }

    /// Copies each part's elements into `destination`, one part after another.
    private static func copy<Scalar: BitwiseCopyable>(
        _ parts: [InputEmbeddings], into destination: inout NDArray, as _: Scalar.Type
    ) {
        var destinationView = destination.mutableView(as: Scalar.self)
        destinationView.withUnsafeMutablePointer { destinationPointer, _, _ in
            var offset = 0
            for part in parts {
                let count = part.tokenCount * part.embeddings.shape[2]
                part.embeddings.view(as: Scalar.self).withUnsafePointer { sourcePointer, _, _ in
                    (destinationPointer + offset).update(from: sourcePointer, count: count)
                }
                offset += count
            }
        }
    }
}
