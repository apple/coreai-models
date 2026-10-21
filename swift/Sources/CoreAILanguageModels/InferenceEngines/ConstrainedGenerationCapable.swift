// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Tokenizers

/// An inference engine that supports GPU-accelerated grammar-constrained generation.
///
/// Conforming engines can apply xgrammar bitmasks directly in the GPU sampling path,
/// eliminating the need to transfer logits to CPU for masking. This protocol is used as
/// a capability signal for routing (e.g., choosing pipelined vs. sequential constrained
/// strategies) and for testability (mock conformers in unit tests).
///
/// The session lifecycle follows a checkout/checkin pattern:
/// 1. Call `getOrCreateConstrainedSession` to obtain a handle (cached or fresh)
/// 2. Pass the handle to `generateConstrained` which drives the GPU loop
/// 3. The engine returns the handle to its internal cache after generation completes
///
/// Conformers must:
/// - Honor `maxTokens` by finishing the stream within that limit
/// - Return the session handle to cache in a `defer` block (even on error/cancellation)
package protocol ConstrainedGenerationCapable: InferenceEngine {
    /// Obtain a constrained session handle, reusing a cached one if the schema matches.
    ///
    /// The internal cache slot is emptied on checkout — concurrent calls create independent sessions.
    func getOrCreateConstrainedSession(
        jsonSchema: String,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        stopTokenIds: [Int32]?
    ) throws -> ConstrainedSessionHandle

    /// Stream constrained token generation using GPU-side bitmask application.
    ///
    /// The engine resets state, prefills the prompt (unconstrained), then enters
    /// a semi-pipelined loop: inference overlaps bitmask computation, but sampling
    /// waits per token (grammar state is inherently sequential).
    ///
    /// The handle is returned to the engine's cache automatically when the Task completes.
    func generateConstrained(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        maxTokens: Int,
        session: ConstrainedSessionHandle
    ) throws -> InferenceTokenSequence
}
