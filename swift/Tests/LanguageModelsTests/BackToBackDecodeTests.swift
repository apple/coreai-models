// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// Requests back to back on one engine, each ending at a stop sequence: the strategy releases the
/// engine's stream there rather than draining it, so the next request has to find the engine free
/// and its cache consistent. Needs a text model bundle named by `COREAI_TEST_DECODE_BUNDLE`, exported
/// without `--attention-mask batched` so the pipelined engine can load it.
@Suite(
    "Back-to-back requests after a stop sequence", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"] != nil))
struct BackToBackDecodeTests {
    private static let prompts = [
        "List three colours, one per line.", "List three colours, one per line.", "Name two planets.",
    ]

    private func load(_ variant: String) async throws -> (any InferenceEngine, any Tokenizer) {
        let bundle = try LanguageModelBundle(from: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"]!)
        let engine = try await EngineFactory.createEngine(
            bundle: bundle, options: EngineOptions(variant: variant, tensorData: bundle.tensorData))
        return (engine, try await bundle.loadTokenizer())
    }

    /// Greedy reply to `prompt`, up to `maxTokens`, ending at any of `stops`.
    private func reply(
        _ prompt: String, engine: any InferenceEngine, tokenizer: any Tokenizer, stops: [[Int32]], maxTokens: Int
    ) async throws -> String {
        let stream = try await VanillaDecodingStrategy().decode(
            from: .prompt(prompt), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: maxTokens), stopSequences: StopSequences(sequences: stops))
        var text = ""
        for try await result in stream { text += result.text }
        return text
    }

    @Test(
        "Replies after a released stream match a fresh engine's", arguments: ["coreai-pipelined", "coreai-sequential"])
    func backToBackMatchesFresh(variant: String) async throws {
        let (engine, tokenizer) = try await load(variant)
        // The fifth token of a reference reply is the stop, so every reply below ends early, on the
        // release path, with its prefix in the engine's cache.
        let reference = try await reply(Self.prompts[0], engine: engine, tokenizer: tokenizer, stops: [], maxTokens: 12)
        let referenceTokens = tokenizer.encode(text: reference, addSpecialTokens: false)
        let stop = Int32(try #require(referenceTokens.dropFirst(4).first))
        try await engine.reset()

        var reused: [String] = []
        for prompt in Self.prompts {
            reused.append(try await reply(prompt, engine: engine, tokenizer: tokenizer, stops: [[stop]], maxTokens: 64))
        }
        let (fresh, _) = try await load(variant)
        var fromFresh: [String] = []
        for prompt in Self.prompts {
            try await fresh.reset()
            fromFresh.append(
                try await reply(prompt, engine: fresh, tokenizer: tokenizer, stops: [[stop]], maxTokens: 64))
        }
        #expect(reused == fromFresh)
        #expect(reused[0] == reused[1])

        // The released generation gives the engine back once it winds down.
        for _ in 0..<50 where engine.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        #expect(!engine.isBusy)
    }
}

#endif
