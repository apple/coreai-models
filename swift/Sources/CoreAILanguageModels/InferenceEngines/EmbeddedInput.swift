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
    /// The embedding tensor, shape [1, seq_len, hidden_dim] or [seq_len, hidden_dim].
    /// Scalar type matches the LLM's expected input (float16, bFloat16, etc.).
    public let embeddings: NDArray

    /// Positions in the token sequence where embeddings replace placeholders.
    public let embeddingPositions: Range<Int>

    /// The seq_len dimension, regardless of whether embeddings are 2D or 3D.
    public let tokenCount: Int

    public init(embeddings: NDArray, embeddingPositions: Range<Int>) {
        self.embeddings = embeddings
        self.embeddingPositions = embeddingPositions
        switch embeddings.shape.count {
        case 3...: self.tokenCount = embeddings.shape[1]
        case 2: self.tokenCount = embeddings.shape[0]
        default: self.tokenCount = 0
        }
    }

    /// The seq_len dimension of an NDArray with the same layout conventions.
    static func seqLen(of tensor: NDArray) -> Int {
        switch tensor.shape.count {
        case 3...: tensor.shape[1]
        case 2: tensor.shape[0]
        default: 0
        }
    }

    // TODO: Multi-turn support — allow multiple image regions per input,
    // persistent across generate() calls (keep in KV cache on reset).
}
