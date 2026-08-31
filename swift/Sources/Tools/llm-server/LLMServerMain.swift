// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAILMCommon
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Hummingbird
import Tokenizers

@main
struct LLMServer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "llm-server",
        abstract: "Start an OpenAI-compatible HTTP server for LLM inference"
    )

    @Option(name: .customLong("model"), help: "Path to a model bundle directory")
    var modelPath: String

    @Option(help: "HTTP port to listen on")
    var port: Int = 8080

    @Option(name: .customLong("server-model-name"), help: "Model name in API responses (default: from metadata)")
    var serverModelName: String?

    @Option(help: "Maximum tokens to generate per request (default: 512)")
    var maxTokens: Int = 512

    @Option(help: "Default temperature (0.0 = greedy)")
    var temperature: Double = 0.7

    @Option(name: .customLong("top-k"), help: "Default Top-K sampling")
    var topK: Int?

    @Option(name: .customLong("top-p"), help: "Default Top-P sampling")
    var topP: Double?

    @Option(name: .customLong("min-p"), help: "Default Min-P sampling")
    var minP: Double?

    @Option(name: .customLong("variant"), help: "Engine variant: auto, coreai-pipelined, coreai-sequential")
    var inferenceEngineVariant: String = "default"

    @Option(name: .customLong("kv-cache-strategy"), help: "KV cache strategy: auto, growing, fixed_size")
    var kvCacheStrategy: KVCacheStrategy = .auto

    @Option(name: .customLong("kv-cache-initial-capacity"), help: "Initial KV cache capacity in tokens")
    var kvCacheInitialCapacity: Int?

    @Flag(name: .customLong("no-thinking"), help: "Disable thinking/reasoning (appends /no_think or sets template)")
    var noThinking: Bool = false

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    @Flag(help: "Enable verbose logging")
    var verbose: Bool = false

    func run() async throws {
        CLILogger.level = verbose ? 1 : 0

        let resolver = ModelPaths()
        guard let url = resolver.resolve(modelPath) else {
            print("Error: \(resolver.notFoundError(for: modelPath))")
            throw ExitCode.failure
        }

        let bundle = try LanguageBundle(from: url.path)
        try bundle.bundle.verify()

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundle.bundlePath)
            print("Cleared specialization cache for \(bundle.name) (\(cleared.count) component(s))")
        }

        let engineOptions = EngineOptions(
            variant: inferenceEngineVariant,
            kvCacheStrategy: kvCacheStrategy,
            kvCacheSize: kvCacheInitialCapacity
        )

        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let cacheHit = PreparedModel.isCached(at: modelURL)
        let assetLabel: String = modelURL.pathExtension == "aimodelc" ? "compiled" : "source"

        if !verbose {
            print("\n⏳ Preparing AI asset from \(assetLabel)...", terminator: "")
            fflush(stdout)
        }

        let modelLoadSpan = InstrumentsProfiler.beginModelLoad(name: bundle.name)

        let engineConfig = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let configData = try JSONEncoder().encode(engineConfig)
        let engine = try await EngineFactory.createEngine(
            config: configData,
            modelURL: modelURL,
            options: engineOptions
        )

        let tokenizer = try await bundle.loadTokenizer()

        let additionalEosTokenIds: [Int32]
        if let tokenizerDir = bundle.tokenizerPath {
            additionalEosTokenIds = LanguageConfig.additionalStopTokenIds(
                from: tokenizerDir, tokenizer: tokenizer)
        } else {
            additionalEosTokenIds = []
        }

        let samplingConfig = SamplingConfiguration(
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP
        )
        try await engine.warmup(queryLength: 1, sampling: samplingConfig)

        modelLoadSpan.end()
        await Task.yield()

        let modelName = serverModelName ?? bundle.name
        let supportsLogprobs = engine.supportsLogits

        if !verbose {
            let prepareElapsed = await PerformanceMetrics.shared.modelLoadTime
            let cacheSuffix = cacheHit ? " (cache hit)" : ""
            print(" done in \(String(format: "%.3f", prepareElapsed))s\(cacheSuffix)\n")
        }

        let config = ServerConfig(
            modelName: modelName,
            defaultMaxTokens: maxTokens,
            defaultTemperature: temperature,
            defaultTopP: topP,
            defaultTopK: topK,
            defaultMinP: minP,
            noThinking: noThinking,
            supportsLogprobs: supportsLogprobs,
            maxContextLength: bundle.maxContextLength,
            vocabSize: bundle.vocabSize,
            additionalEosTokenIds: additionalEosTokenIds
        )

        let state = ServerState(
            engine: engine,
            tokenizer: tokenizer,
            config: config
        )

        if verbose {
            print("Model: \(modelName)")
            print("  Engine: \(type(of: engine))")
            print("  Logprobs: \(supportsLogprobs ? "supported" : "not supported (use --variant coreai-sequential)")")
            print("  Context: \(bundle.maxContextLength) tokens")
            print("  No-thinking: \(noThinking)")
            let topKStr = topK.map { "\($0)" } ?? "nil"
            let topPStr = topP.map { "\($0)" } ?? "nil"
            print("  Sampling: temperature=\(temperature), topK=\(topKStr), topP=\(topPStr)")
        }

        print("Serving \(modelName) on http://127.0.0.1:\(port)")
        if verbose {
            print("  POST /v1/chat/completions   (generate_until)")
            if supportsLogprobs {
                print("  POST /v1/completions        (loglikelihood)")
            }
            print("  GET  /v1/models")
            print("  GET  /v1/stats")
            print("  GET  /health")
            print("  GET  /ready")
        }
        print("")

        try await startServer(state: state, port: port)
    }
}
