// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics

/// Output from a text encoder — hidden states with optional pooled embedding.
public struct TextEncoderOutput: Sendable {
    /// Token-level embeddings [1, seq_len, hidden_dim].
    public let hiddenStates: NDArray
    /// Sentence-level embedding [1, hidden_dim]. Nil for single-output encoders (e.g. T5).
    public let pooledOutput: NDArray?

    public init(hiddenStates: NDArray, pooledOutput: NDArray? = nil) {
        self.hiddenStates = hiddenStates
        self.pooledOutput = pooledOutput
    }
}
