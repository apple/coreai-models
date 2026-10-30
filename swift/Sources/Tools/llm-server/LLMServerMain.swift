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
        abstract: "Serve a Core AI LLM bundle over the chat-completions and completions HTTP APIs.",
        discussion: """
            A prototype LLM server engine, to be used for evaluations and to serve as an example for \
            integrating Core AI into high-performance agentic workflows. It serves a single exported LLM bundle.

            Build and run:
              uv run coreai.llm.export qwen3-0.6b
              swift run -c release llm-server --model exports/qwen3_0_6b_4bit_dynamic

            Several requests at once: export with --attention-mask batched --dynamic-batch-size 8, then start \
            the server with --concurrent-requests auto. New requests join the batch as rows free up.

            Eval harnesses: point the harness at http://127.0.0.1:8080/v1. Generation uses \
            /v1/chat/completions; log-likelihood scoring uses /v1/completions with echo and logprobs. \
            --replay runs a JSONL file of chat ("request") and scoring ("completion") lines without HTTP.

            Endpoints:
              POST /v1/chat/completions  chat generation: streaming, tools, reasoning_effort
              POST /v1/completions       prompt scoring: echo, logprobs
              GET  /v1/models            the served model
              GET  /ready                readiness, KV-cache memory estimates and batch occupancy
              GET  /v1/stats             request and token counters
              GET  /health               liveness
            """
    )

    @Option(name: .customLong("model"), help: "Path to a model bundle directory")
    var modelPath: String

    @Option(help: "HTTP port to listen on")
    var port: Int = 8080

    @Option(name: .customLong("server-model-name"), help: "Model name in API responses (default: from metadata)")
    var serverModelName: String?

    @Option(help: "Maximum tokens to generate per request")
    var maxTokens: Int = 512

    @Option(help: "Default temperature (0.0 = greedy)")
    var temperature: Double = 0.7

    @Option(name: .customLong("top-k"), help: "Default Top-K sampling")
    var topK: Int?

    @Option(name: .customLong("top-p"), help: "Default Top-P sampling")
    var topP: Double?

    @Option(name: .customLong("min-p"), help: "Default Min-P sampling")
    var minP: Double?

    @Option(
        name: .customLong("replay"),
        help:
            "Process a JSONL request file and exit instead of serving over HTTP; runs the generation core directly. See CoreAILMCommon/ReplayTypes."
    )
    var replayPath: String?

    @Option(name: .customLong("replay-output"), help: "Write replay results JSONL here (default: stdout)")
    var replayOutput: String?

    @Flag(
        name: .customLong("replay-concurrent"),
        help: ArgumentHelp(
            "Run all --replay requests concurrently instead of one at a time. Use with --concurrent-requests N.")
    )
    var replayConcurrent: Bool = false

    @Option(name: .customLong("variant"), help: "Engine variant: auto, coreai-pipelined, coreai-sequential")
    var inferenceEngineVariant: String = "default"

    @Option(name: .customLong("kv-cache-strategy"), help: "KV cache strategy: auto, growing, fixed_size")
    var kvCacheStrategy: KVCacheStrategy = .auto

    @Option(name: .customLong("kv-cache-initial-capacity"), help: "Initial KV cache capacity in tokens")
    var kvCacheInitialCapacity: Int?

    @Option(
        name: .customLong("chunk-size"),
        help: ArgumentHelp(
            "Prefill chunk size in tokens (default: 2048; 4096 on macOS with >=64GB memory)", visibility: .hidden
        )
    )
    var chunkSize: Int?

    @Option(
        name: .customLong("chunk-threshold"),
        help: ArgumentHelp("Minimum prompt tokens to trigger chunking (default: 2x chunk size)", visibility: .hidden)
    )
    var chunkThreshold: Int?

    @Option(
        name: .customLong("max-queue-depth"),
        help: "Max requests queued beyond the concurrent ones before returning 429")
    var maxQueueDepth: Int = 16

    @Option(
        name: .customLong("concurrent-requests"),
        help: ArgumentHelp(
            "Max requests decoded together in one batch: a number, or auto for the model's max batch size. "
                + "Values above 1 enable the batch scheduler when the model supports batching, and are ignored "
                + "with a warning otherwise.")
    )
    var concurrentRequests: String = "1"

    @Flag(
        name: .customLong("no-thinking"),
        help: "Disable thinking/reasoning (alias for --default-reasoning-effort none)"
    )
    var noThinking: Bool = false

    @Option(
        name: .customLong("default-reasoning-effort"),
        help: "Default reasoning_effort when a request omits it (none, low, medium, high)."
    )
    var defaultReasoningEffort: String?

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    @Flag(help: "Enable verbose logging")
    var verbose: Bool = false

    func validate() throws {
        guard maxQueueDepth >= 0 else {
            throw ValidationError("--max-queue-depth must be >= 0 (got \(maxQueueDepth))")
        }
        guard ConcurrentRequests(concurrentRequests) != nil else {
            throw ValidationError("--concurrent-requests must be auto or a number >= 1 (got \(concurrentRequests))")
        }
        do {
            _ = try ReasoningEffort.resolveDefault(
                defaultReasoningEffort: defaultReasoningEffort, noThinking: noThinking)
        } catch {
            throw ValidationError("\(error)")
        }
    }

    func run() async throws {
        CLILogger.level = verbose ? 1 : 0

        let resolver = ModelPaths()
        guard let url = resolver.resolve(modelPath) else {
            print("Error: \(resolver.notFoundError(for: modelPath))")
            throw ExitCode.failure
        }

        let bundle = try LanguageModelBundle(from: url.path)
        try bundle.modelBundle.validateModelAssets()

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundle.bundlePath)
            print("Cleared specialization cache for \(bundle.name) (\(cleared.count) component(s))")
        }

        // Resolve chunking config with CLI flags taking precedence over metadata.json.
        // A nil result preserves the lower layers (deprecated env var, memory-based default).
        let resolvedChunkSize = chunkSize ?? bundle.language.prefillChunkSize
        let resolvedChunkThreshold = chunkThreshold ?? bundle.language.prefillChunkThreshold

        let engineOptions = EngineOptions(
            variant: inferenceEngineVariant,
            kvCacheStrategy: kvCacheStrategy,
            kvCacheSize: kvCacheInitialCapacity,
            prefillChunkSize: resolvedChunkSize,
            prefillChunkThreshold: resolvedChunkThreshold,
            tensorData: bundle.tensorData
        )

        let modelURL = try bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let cacheHit = PreparedModel.isCached(at: modelURL)

        if !verbose && replayPath == nil {
            print("\n⏳ Preparing AI asset from \(modelURL.lastPathComponent)...", terminator: "")
            fflush(stdout)
        }

        let modelLoadSpan = InstrumentsProfiler.beginModelLoad(name: bundle.name)

        let engineConfig = ModelConfig(bundle: bundle)
        let configData = try JSONEncoder().encode(engineConfig)
        let engine = try await EngineFactory.createEngine(
            config: configData,
            modelURL: modelURL,
            options: engineOptions
        )

        let tokenizer = try await bundle.loadTokenizer()

        var additionalEosTokenIds: Set<Int32>
        if let tokenizerDir = bundle.tokenizerPath {
            additionalEosTokenIds = LanguageConfig.additionalStopTokenIds(
                from: tokenizerDir, tokenizer: tokenizer)
        } else {
            additionalEosTokenIds = []
        }

        // Agentic models use <|eot|> as a turn boundary. It may not appear in
        // tokenizer_config.json, so add it explicitly (skipping the main EOS).
        let thinkingFormat = detectThinkingFormat(using: tokenizer)
        if let eotId = agenticEndOfTurnTokenId(thinkingFormat: thinkingFormat, tokenizer: tokenizer),
            eotId != tokenizer.eosTokenId.map({ Int32($0) })
        {
            additionalEosTokenIds.insert(eotId)
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

        if !verbose && replayPath == nil {
            let prepareElapsed = await PerformanceMetrics.shared.modelLoadTime
            let cacheSuffix = cacheHit ? " (cache hit)" : ""
            print(" done in \(String(format: "%.3f", prepareElapsed))s\(cacheSuffix)\n")
        }

        let resolvedReasoningDefault = try ReasoningEffort.resolveDefault(
            defaultReasoningEffort: defaultReasoningEffort, noThinking: noThinking)

        let config = ServerConfig(
            modelName: modelName,
            defaultMaxTokens: maxTokens,
            defaultTemperature: temperature,
            defaultTopP: topP,
            defaultTopK: topK,
            defaultMinP: minP,
            defaultReasoningEffort: resolvedReasoningDefault,
            supportsLogprobs: supportsLogprobs,
            maxContextLength: bundle.maxContextLength,
            vocabSize: bundle.vocabSize,
            additionalEosTokenIds: additionalEosTokenIds,
            maxQueueDepth: maxQueueDepth
        )

        // The scheduler holds up to --concurrent-requests plus --max-queue-depth requests, then rejects.
        var batchScheduler: BatchScheduler? = nil
        let requested = ConcurrentRequests(concurrentRequests) ?? .count(1)
        let sequentialEngine = engine as? CoreAISequentialEngine
        let supportsBatching = sequentialEngine?.supportsBatching ?? false
        let rows: Int
        do {
            rows = try requested.resolve(supportsBatching: supportsBatching, maxBatchSize: bundle.maxBatchSize)
        } catch {
            switch error {
            case .exceedsMaxBatchSize(let count, let maxBatchSize):
                throw ValidationError(
                    "--concurrent-requests \(count) exceeds the asset's max batch size \(maxBatchSize). "
                        + "Re-export with --dynamic-batch-size \(count) or higher, or lower --concurrent-requests."
                )
            }
        }
        if rows > 1, let sequentialEngine {
            batchScheduler = try BatchScheduler(
                engine: sequentialEngine, maxConcurrent: rows, maxQueued: maxQueueDepth)
            print("Batched serving: \(rows) concurrent requests (asset max batch \(bundle.maxBatchSize))")
        } else if requested != .count(1), !supportsBatching {
            print(
                "⚠️  --concurrent-requests \(concurrentRequests): serving one request at a time. Batched serving needs "
                    + "the coreai-sequential engine and an asset exported with --attention-mask batched.")
        }

        let state = ServerState(
            engine: engine,
            tokenizer: tokenizer,
            config: config,
            batchServing: batchScheduler.map { (scheduler: $0, modelConfig: engineConfig) }
        )

        if let replayPath {
            try await ReplayRunner.run(
                inputPath: replayPath, outputPath: replayOutput, state: state,
                concurrent: replayConcurrent)
            return
        }

        if verbose {
            print("Model: \(modelName)")
            print("  Engine: \(type(of: engine))")
            print("  Logprobs: \(supportsLogprobs ? "supported" : "not supported (use --variant coreai-sequential)")")
            print("  Context: \(bundle.maxContextLength) tokens")
            print("  Default reasoning effort: \(config.defaultReasoningEffort ?? "template default")")
            print("  Max queue depth: \(maxQueueDepth)")
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
