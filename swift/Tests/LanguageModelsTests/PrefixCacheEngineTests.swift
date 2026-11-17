// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// Greedy replies from an engine that restores a shared prefix from its cache, against replies computed
/// without one. Needs a text model bundle: an attention-only model in `COREAI_TEST_DECODE_BUNDLE`, a
/// model with recurrent state in `COREAI_TEST_HYBRID_BUNDLE`, each exported so the pipelined engine
/// loads it.
@Suite("Prefix cache, through the pipelined engine", .serialized)
struct PrefixCacheEngineTests {
    /// A system prompt over two chunks long, then a question: prompts that share the first share chunks.
    private static func prompt(asking question: String) -> String {
        let colours = ["red", "blue", "green", "amber", "violet", "silver"]
        let records = (1..<80).map { i in
            "Record \(i): the crate labelled \((i * 37) % 997 + 100) holds \(colours[i % 6]) marbles."
        }
        return "You are a careful assistant. Answer from these records only. "
            + records.joined(separator: " ") + "\nQuestion: \(question)\nAnswer:"
    }

    private static let first = prompt(asking: "What colour are the marbles in Record 3?")
    private static let second = prompt(asking: "What is the label of the crate in Record 17?")

    private func load(_ path: String, prefixCacheBytes: Int) async throws -> (any InferenceEngine, any Tokenizer) {
        let bundle = try LanguageModelBundle(from: path)
        let engine = try await EngineFactory.createEngine(
            bundle: bundle,
            options: EngineOptions(
                variant: "coreai-pipelined", tensorData: bundle.tensorData, prefixCacheBytes: prefixCacheBytes))
        return (engine, try await bundle.loadTokenizer())
    }

    private func reply(_ prompt: String, engine: any InferenceEngine, tokenizer: any Tokenizer) async throws -> String {
        let stream = try await VanillaDecodingStrategy().decode(
            from: .prompt(prompt), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 16), stopSequences: StopSequences(sequences: []))
        var text = ""
        for try await result in stream { text += result.text }
        return text
    }

    /// Replies to `prompts` from one engine, in order, with what each restored.
    private func replies(
        _ prompts: [String], engine: any InferenceEngine, tokenizer: any Tokenizer
    ) async throws -> [(text: String, restored: Int)] {
        var out: [(text: String, restored: Int)] = []
        for prompt in prompts {
            out.append((try await reply(prompt, engine: engine, tokenizer: tokenizer), engine.lastPrefixHitCount))
        }
        return out
    }

    @Test(
        "An attention-only model's second prompt restores the shared chunks, and replies as a cold prefill does",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"] != nil))
    func attentionOnly() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"])
        let (cached, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30)
        try #require(cached.prefixCacheEnabled)
        let prompts = [Self.first, Self.second, Self.first]
        let withCache = try await replies(prompts, engine: cached, tokenizer: tokenizer)

        let (cold, _) = try await load(path, prefixCacheBytes: 0)
        #expect(!cold.prefixCacheEnabled)
        var withoutCache: [String] = []
        for prompt in prompts {
            try await cold.reset()
            withoutCache.append(try await reply(prompt, engine: cold, tokenizer: tokenizer))
        }
        #expect(withCache.map(\.text) == withoutCache)
        // The second and third diverge from the history inside the question, after the shared chunks.
        #expect(withCache[1].restored >= 2 * 256)
        #expect(withCache[2].restored >= 2 * 256)
    }

    @Test(
        "A hybrid model's second prompt resumes at a snapshot, and replies as a fresh engine does",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"] != nil))
    func hybrid() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"])
        let (cached, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30)
        try #require(cached.prefixCacheEnabled)
        #expect(cached.hasRecurrentState)
        let withCache = try await replies([Self.first, Self.second], engine: cached, tokenizer: tokenizer)
        #expect(withCache[1].restored >= 2 * 256)

        // Snapshots split prefill, which can tip a near-tie, so the reference is a fresh engine with the
        // cache on: it splits the second prompt's prefill at the same places.
        let (fresh, _) = try await load(path, prefixCacheBytes: 1 << 30)
        let reference = try await reply(Self.second, engine: fresh, tokenizer: tokenizer)
        #expect(withCache[1].text == reference)
    }
}

#endif
