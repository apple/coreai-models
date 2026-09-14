// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Shared chunked-prefill loop for the sequential engines.
///
/// Splits `tokens` into chunks of `chunkSize`, holding back the final `heldBack` tokens
/// (see `prefillHeldBackTokens`) for the logits-producing pass. Each chunk is handed to
/// `processChunk`, which returns that chunk's logit buffer — or `[]` when the chunk produces
/// no logits (e.g. a KV-only prefill-graph chunk). Returns the last-token logits.
///
/// - `CoreAISequentialEngine` passes `heldBack = prefillHeldBackTokens(hasPrefillGraph:)` and a
///   closure that routes non-held-back chunks through the prefill graph (returning `[]`) when one
///   exists, otherwise through `main`.
/// - `CoreAISequentialVLMEngine` passes `heldBack = 0` and a closure that always runs the decoder,
///   so the loop collapses to a plain chunked pass.
func runChunkedPrefill(
    tokens: ArraySlice<Int32>,
    chunkSize: Int,
    heldBack: Int,
    vocabSize: Int,
    processChunk: (_ chunk: ArraySlice<Int32>, _ isHeldBack: Bool) async throws -> [LogitsScalarType]
) async throws -> [LogitsScalarType] {
    let plan = prefillChunkSizes(tokenCount: tokens.count, chunkSize: chunkSize, heldBack: heldBack)
    let trailing = tokens.count - plan.reduce(0, +)
    let total = plan.count + (trailing > 0 ? 1 : 0)

    var lastLogits: [LogitsScalarType] = []
    var remaining = tokens
    var index = 0

    for size in plan {
        let chunkEnd = remaining.startIndex + size
        let chunk = remaining[remaining.startIndex..<chunkEnd]
        CLILogger.log("Prefill chunk \(index + 1)/\(total): \(chunk.count) tokens")
        let logits = try await processChunk(chunk, false)
        if !logits.isEmpty { lastLogits = logits }
        remaining = remaining[chunkEnd...]
        index += 1
    }

    // Held-back tail (the tokens `prefillChunkSizes` left for the logits-producing pass).
    if !remaining.isEmpty {
        CLILogger.log("Prefill chunk \(index + 1)/\(total): \(remaining.count) tokens (held back)")
        lastLogits = try await processChunk(remaining, true)
    }

    return lastTokenLogits(from: lastLogits, vocabSize: vocabSize)
}
