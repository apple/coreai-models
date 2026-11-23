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

    private func load(
        _ path: String, prefixCacheBytes: Int, diskDirectory: URL? = nil
    ) async throws -> (any InferenceEngine, any Tokenizer) {
        let bundle = try LanguageModelBundle(from: path)
        let engine = try await EngineFactory.createEngine(
            bundle: bundle,
            options: EngineOptions(
                variant: "coreai-pipelined", tensorData: bundle.tensorData, prefixCacheBytes: prefixCacheBytes,
                prefixCacheDiskBytes: diskDirectory == nil ? 0 : 1 << 30, prefixCacheDirectory: diskDirectory))
        return (engine, try await bundle.loadTokenizer())
    }

    /// Waits for `engine`'s last generation to give the engine back.
    private func settle(_ engine: any InferenceEngine) async throws {
        for _ in 0..<250 where engine.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        #expect(!engine.isBusy, "the engine never gave itself back")
    }

    private func reply(_ prompt: String, engine: any InferenceEngine, tokenizer: any Tokenizer) async throws -> String {
        let stream = try await VanillaDecodingStrategy().decode(
            from: .prompt(prompt), tokenizer: tokenizer, inferenceEngine: engine, samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 16),
            decoderOptions: DecoderOptions(stopSequences: StopSequences(sequences: [])))
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
        #expect(withCache[1].restored >= 2 * CoreAIPipelinedEngine.prefixChunkTokens)
        #expect(withCache[2].restored >= 2 * CoreAIPipelinedEngine.prefixChunkTokens)
    }

    @Test(
        "A hybrid model's replies are the same with the cache off, on a miss, and resuming at a snapshot",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"] != nil))
    func hybrid() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"])
        let (cached, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30)
        try #require(cached.prefixCacheEnabled)
        #expect(cached.hasRecurrentState)
        // The first misses and snapshots; the second resumes at the snapshot inside the shared prompt.
        let withCache = try await replies([Self.first, Self.second], engine: cached, tokenizer: tokenizer)
        #expect(withCache[1].restored >= PrefixSnapshotPlan.gridTokens)

        let (cold, _) = try await load(path, prefixCacheBytes: 0)
        var withoutCache: [String] = []
        for prompt in [Self.first, Self.second] {
            try await cold.reset()
            withoutCache.append(try await reply(prompt, engine: cold, tokenizer: tokenizer))
        }
        #expect(withCache.map(\.text) == withoutCache)
    }

    @Test(
        "A second engine on the same disk directory restores what the first one computed",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"] != nil))
    func diskAcrossEngines() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"])
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrefixCacheEngineTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            let (first, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30, diskDirectory: directory)
            _ = try await reply(Self.first, engine: first, tokenizer: tokenizer)
            try await settle(first)
            let store = try #require((first as? CoreAIPipelinedEngine)?.prefixStore)
            store.flush()
            #expect(store.nodeCount >= 2)
        }
        // The first engine is gone, and with it its hold on the directory.
        let (second, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30, diskDirectory: directory)
        try #require((second as? CoreAIPipelinedEngine)?.prefixStore != nil)
        let restored = try await replies([Self.second], engine: second, tokenizer: tokenizer)
        #expect(restored[0].restored >= 2 * CoreAIPipelinedEngine.prefixChunkTokens)

        let (cold, _) = try await load(path, prefixCacheBytes: 0)
        #expect(restored[0].text == (try await reply(Self.second, engine: cold, tokenizer: tokenizer)))
    }

    @Test(
        "Shedding for memory gives back what it says, least recently used first, and the cache fills again",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"] != nil))
    func shedForMemory() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"])
        let (engine, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30)
        let pipelined = try #require(engine as? CoreAIPipelinedEngine)
        _ = try await reply(Self.first, engine: engine, tokenizer: tokenizer)
        try await settle(engine)
        let held = try #require(pipelined.prefixTree?.bytesHeld)
        #expect(held > 0)

        // One byte asked for takes one leaf.
        let one = engine.shedPrefixCache(atLeast: 1)
        #expect(one > 0 && one < held)
        #expect(pipelined.prefixTree?.bytesHeld == held - one)
        // More than it holds takes the rest.
        #expect(engine.shedPrefixCache(atLeast: .max) == held - one)
        #expect(pipelined.prefixTree?.nodeCount == 0)

        // The next prompt's chunks are published again: what the engine had published is forgotten too.
        _ = try await reply(Self.second, engine: engine, tokenizer: tokenizer)
        try await settle(engine)
        #expect((pipelined.prefixTree?.nodeCount ?? 0) >= 2)
    }

    @Test(
        "A hybrid model on a fresh engine resumes at a snapshot read from disk, and replies as a cold engine does",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"] != nil))
    func hybridDiskAcrossEngines() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_HYBRID_BUNDLE"])
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrefixCacheEngineTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            let (first, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30, diskDirectory: directory)
            _ = try await reply(Self.first, engine: first, tokenizer: tokenizer)
            try await settle(first)
            try #require((first as? CoreAIPipelinedEngine)?.prefixStore).flush()
        }
        let (second, tokenizer) = try await load(path, prefixCacheBytes: 1 << 30, diskDirectory: directory)
        let restored = try await replies([Self.second], engine: second, tokenizer: tokenizer)
        #expect(restored[0].restored >= PrefixSnapshotPlan.gridTokens)
        #expect(restored[0].restored % PrefixSnapshotPlan.gridTokens == 0)

        let (cold, _) = try await load(path, prefixCacheBytes: 0)
        #expect(restored[0].text == (try await reply(Self.second, engine: cold, tokenizer: tokenizer)))
    }

    @Test(
        "A disk budget without a memory cache or a directory is refused when the engine is made",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"] != nil))
    func unusableDiskBudgetIsRefused() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_DECODE_BUNDLE"])
        let bundle = try LanguageModelBundle(from: path)
        let directory = FileManager.default.temporaryDirectory
        for options in [
            EngineOptions(
                variant: "coreai-pipelined", tensorData: bundle.tensorData, prefixCacheDiskBytes: 1 << 20,
                prefixCacheDirectory: directory),
            EngineOptions(
                variant: "coreai-pipelined", tensorData: bundle.tensorData, prefixCacheBytes: 1 << 20,
                prefixCacheDiskBytes: 1 << 20),
        ] {
            await #expect(throws: InferenceRuntimeError.self) {
                _ = try await EngineFactory.createEngine(bundle: bundle, options: options)
            }
        }
    }
}

#endif
