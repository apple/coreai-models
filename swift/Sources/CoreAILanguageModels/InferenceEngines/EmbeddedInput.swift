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
public struct EmbeddedInput: Sendable {
    /// The embedding tensor, shape [batch, seq_len, hidden_dim].
    /// Scalar type matches the LLM's expected input (float16, bFloat16, etc.).
    public let embeddings: NDArray

    /// Positions in the token sequence where embeddings replace placeholders.
    public let embeddingPositions: Range<Int>

    public init(embeddings: NDArray, embeddingPositions: Range<Int>) throws {
        guard embeddings.shape.count == 3 else {
            throw InferenceRuntimeError.invalidArgument(
                "EmbeddedInput requires 3D embeddings [batch, seq_len, hidden_dim], "
                    + "got shape with \(embeddings.shape.count) dimensions")
        }
        self.embeddings = embeddings
        self.embeddingPositions = embeddingPositions
    }

    /// Number of embedding tokens (seq_len dimension).
    public var tokenCount: Int { embeddings.shape[1] }

    // TODO: Multi-turn support — allow multiple image regions per input,
    // persistent across generate() calls (keep in KV cache on reset).
}
