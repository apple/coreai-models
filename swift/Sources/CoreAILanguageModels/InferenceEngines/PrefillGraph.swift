// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Name of the optional prefill entrypoint. Exported beside `main` (see
/// `export/macos.py`) with the same inputs and states, but no LM head and no outputs:
/// it only fills the KV cache.
let prefillGraphFunctionName = "prefill"

// MARK: - Selection

/// Widest query the prefill graph is run with: `prefillChunkSize`, clamped to the context
/// and to at least one token so a degenerate config can't produce an empty or negative
/// chunk width.
func prefillQueryLength(prefillChunkSize: Int, maxContextLength: Int) -> Int {
    max(1, min(prefillChunkSize, maxContextLength))
}

/// Whether prefill should be chunked at all.
///
/// With a prefill graph, chunking is cheaper at any size: every chunk skips the LM head, so
/// there is no threshold to clear. Without one, prefill runs through `main`, where a chunk
/// costs a full LM head, so it only pays off past `chunkThreshold`.
func shouldChunkPrefill(tokenCount: Int, hasPrefillGraph: Bool, chunkThreshold: Int) -> Bool {
    hasPrefillGraph || tokenCount > chunkThreshold
}

/// Tokens the prefill plan must leave for `main`.
///
/// One with a prefill graph, which has no LM head and so cannot produce the logits that
/// seed sampling; none without one, where prefill runs on `main` and the trailing chunk
/// carries the logits itself.
func prefillHeldBackTokens(hasPrefillGraph: Bool) -> Int {
    hasPrefillGraph ? 1 : 0
}

/// Initial size of the logits buffer, in token rows.
///
/// With a prefill graph `main` only ever sees the one held-back token, so a prompt-sized
/// buffer -- hundreds of MB at large vocabularies -- would go unused. Without one `main`
/// serves prefill too and sees whole chunks, so it starts at the usual guess and grows.
func prefillLogitsInitialCapacity(hasPrefillGraph: Bool, averagePromptSize: Int) -> Int {
    hasPrefillGraph ? 1 : averagePromptSize
}

/// Sizes of the chunks prefill runs, in order, covering all but the held-back tokens.
///
/// `heldBack` comes from `prefillHeldBackTokens`. Returns an empty array when there is
/// nothing to prefill -- a prompt at or below `heldBack` is entirely the caller's to run.
func prefillChunkSizes(tokenCount: Int, chunkSize: Int, heldBack: Int) -> [Int] {
    let width = max(1, chunkSize)
    var remaining = max(0, tokenCount - max(0, heldBack))
    var sizes: [Int] = []
    while remaining > 0 {
        let chunk = min(width, remaining)
        sizes.append(chunk)
        remaining -= chunk
    }
    return sizes
}

// MARK: - Loading

/// Check a prefill descriptor against `main`, throwing if it can't be bound the same way.
///
/// Split out from `loadPrefillGraph` so the contract can be exercised without an asset.
func validatePrefillShape(
    prefillInputs: [String],
    prefillStates: [String],
    prefillOutputs: [String],
    mainInputs: [String],
    mainStates: [String],
    mainName: String
) throws {
    guard prefillInputs == mainInputs else {
        throw InferenceRuntimeError.invalidInputType(
            "'\(prefillGraphFunctionName)' graph inputs \(prefillInputs) do not match "
                + "'\(mainName)' inputs \(mainInputs)")
    }
    guard Set(prefillStates) == Set(mainStates) else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(prefillGraphFunctionName)' graph states \(prefillStates) do not match "
                + "'\(mainName)' states \(mainStates)")
    }
    guard prefillOutputs.isEmpty else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(prefillGraphFunctionName)' graph declares outputs \(prefillOutputs); "
                + "expected none. Re-export the model.")
    }
}

/// Load the prefill graph, or nil if the asset has none.
///
/// It must take the same inputs and states as `main` and declare no outputs, because that
/// is how callers bind it. A graph that disagrees is a stale asset, so this throws instead
/// of falling back.
func loadPrefillGraph(
    from model: AIModel,
    matching main: InferenceFunctionDescriptor,
    mainName: String
) throws -> InferenceFunction? {
    guard let prefill = model.functionDescriptor(for: prefillGraphFunctionName) else { return nil }

    try validatePrefillShape(
        prefillInputs: prefill.inputNames,
        prefillStates: prefill.stateNames,
        prefillOutputs: prefill.outputNames,
        mainInputs: main.inputNames,
        mainStates: main.stateNames,
        mainName: mainName)

    guard let loaded = try model.loadFunction(named: prefillGraphFunctionName) else {
        throw InferenceRuntimeError.genericError(
            "Cannot load function '\(prefillGraphFunctionName)'")
    }
    return loaded
}
