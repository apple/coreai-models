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
    /// The embedding tensor, typically shape [1, seq_len, hidden_dim].
    /// Scalar type matches the LLM's expected input (float16, bFloat16, etc.).
    public let embeddings: NDArray

    /// Positions in the token sequence where image embeddings replace placeholders.
    public let imageTokenPositions: Range<Int>

    public init(embeddings: NDArray, imageTokenPositions: Range<Int>) {
        self.embeddings = embeddings
        self.imageTokenPositions = imageTokenPositions
    }

    /// Number of embedding tokens (seq_len dimension).
    public var tokenCount: Int {
        embeddings.shape.count >= 2 ? embeddings.shape[1] : 0
    }

    // TODO: Multi-turn support — allow multiple image regions per input,
    // persistent across generate() calls (keep in KV cache on reset).
}
