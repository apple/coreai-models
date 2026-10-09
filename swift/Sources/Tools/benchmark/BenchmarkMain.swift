// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Based on mlx-lm benchmark (https://github.com/ml-explore/mlx-lm)

import ArgumentParser
import CoreAI
import CoreAILanguageModels
import CoreAIShared
import Foundation

@main
struct Main {
    static func main() async throws {
        await LLMBenchmark.main()
    }
}

struct LLMBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "llm-benchmark",
        abstract: "LLM inference benchmark for CoreAI models"
    )

    @Option(name: .customLong("model"), help: "Path to a model bundle directory")
    var model: String

    @Option(name: [.customShort("p"), .customLong("prompt-tokens")], help: "Length of prompt")
    var promptTokens: Int = 512

    @Option(name: [.customShort("g"), .customLong("generation-tokens")], help: "Length of completion")
    var generationTokens: Int = 1024

    @Option(name: [.customShort("n"), .customLong("num-trials")], help: "Number of timing trials")
    var numTrials: Int = 5

    @Option(name: .long, help: "Random seed for synthetic prompt")
    var seed: UInt64 = 0

    @Option(name: .customLong("output-json"), help: "Write summary JSON to file")
    var outputJson: String?

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

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    @Option(name: .customLong("inference-engine-variant"), help: "Engine variant (default auto). Use coreai-sequential to benchmark a fixed batch=N (attn_mask) graph.")
    var inferenceEngineVariant: String = "default"

    @Option(
        name: .customLong("batch-size"),
        help:
            "Request-batch size to benchmark (rows run together). Default: auto-detect from the graph's pinned dim0. Set explicitly to drive a DYNAMIC-batch graph (whose dim0 is symbolic and auto-detects as 1). Requires --inference-engine-variant coreai-sequential and a batch-capable graph."
    )
    var batchSize: Int?

    func validate() throws {
        if promptTokens < 1 { throw ValidationError("--prompt-tokens must be >= 1") }
        if generationTokens < 1 { throw ValidationError("--generation-tokens must be >= 1") }
        if numTrials < 1 { throw ValidationError("--num-trials must be >= 1") }
        if let b = batchSize, b < 1 { throw ValidationError("--batch-size must be >= 1") }
        if !FileManager.default.fileExists(atPath: model) {
            throw ValidationError("Model path not found: \(model)")
        }
    }

    func run() async throws {
        #if DEBUG
        print("Note: built in Debug mode. For more reliable results, build with -c release.")
        #endif

        let bundle = try LanguageModelBundle(from: model)
        try bundle.modelBundle.validateModelAssets()
        let vocabSize = bundle.vocabSize

        let modelURL = try bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundle.bundlePath)
            print("\n🗑️  Cleared specialization cache for \(bundle.name) (\(cleared.count) component(s))")
        }

        // Detect an existing cached specialization before loading so we can annotate the load
        // time below. This only inspects the cache; it never triggers specialization.
        let cacheHit = PreparedModel.isCached(at: modelURL)

        let engineConfig = ModelConfig(bundle: bundle)
        let configData = try JSONEncoder().encode(engineConfig)
        print("\n⏳ Preparing AI asset from \(modelURL.lastPathComponent)...", terminator: "")
        fflush(stdout)
        // Resolve chunking config with CLI flags taking precedence over metadata.json.
        // A nil result preserves the lower layers (deprecated env var, memory-based default).
        let resolvedChunkSize = chunkSize ?? bundle.language.prefillChunkSize
        let resolvedChunkThreshold = chunkThreshold ?? bundle.language.prefillChunkThreshold
        let engineOptions = EngineOptions(
            variant: inferenceEngineVariant,
            prefillChunkSize: resolvedChunkSize,
            prefillChunkThreshold: resolvedChunkThreshold,
            tensorData: bundle.tensorData
        )
        let prepareStart = SuspendingClock.now
        let engine = try await EngineFactory.createEngine(
            config: configData,
            modelURL: modelURL,
            options: engineOptions
        )
        let prepareSeconds = (SuspendingClock.now - prepareStart).inSeconds
        let cacheSuffix = cacheHit ? " (cache hit)" : ""
        print(" done in \(fmt(prepareSeconds))s\(cacheSuffix)")

        let prompt = randomPrompt(vocabSize: vocabSize, count: promptTokens, seed: seed)
        let sampling = SamplingConfiguration(temperature: 0)

        // Batch size: an explicit --batch-size wins (needed for a dynamic-batch graph, whose dim0
        // is symbolic and auto-detects as 1); otherwise auto-detect a fixed batch=N graph from the
        // pinned input_ids leading dim.
        let detectedBatch = (engine as? CoreAISequentialEngine)?.declaredBatchSize ?? 1
        let effectiveBatch = batchSize ?? detectedBatch
        if effectiveBatch > bundle.maxBatchSize {
            throw ValidationError(
                "--batch-size \(effectiveBatch) exceeds the asset's max batch size \(bundle.maxBatchSize). "
                    + "Re-export with --dynamic-batch-size >= \(effectiveBatch), or lower --batch-size.")
        }
        if effectiveBatch > 1 {
            guard engine is CoreAISequentialEngine else {
                throw ValidationError(
                    "batched benchmarking (batch=\(effectiveBatch)) requires "
                        + "--inference-engine-variant coreai-sequential")
            }
            let source = batchSize != nil ? "requested" : "auto-detected from model"
            print("🧵 Batch size: \(effectiveBatch) (\(source); benchmarking \(effectiveBatch) rows together)")
        }

        // One trial via the batched path when batch > 1, else the single path.
        func oneTrial() async throws -> TrialResult {
            if effectiveBatch > 1, let seq = engine as? CoreAISequentialEngine {
                return try await runBatchedTrial(engine: seq, prompt: prompt, batchSize: effectiveBatch)
            }
            return try await runTrial(engine: engine, prompt: prompt, sampling: sampling)
        }

        // Warmup
        print("\n⚙️  Warming up engine...", terminator: "")
        fflush(stdout)
        let warmupStart = SuspendingClock.now
        _ = try await oneTrial()
        let warmupSeconds = (SuspendingClock.now - warmupStart).inSeconds
        print(" done in \(fmt(warmupSeconds))s")

        // Timed trials
        print("\n🔄 Benchmarking with \(promptTokens) prompt tokens, \(generationTokens) generation tokens\n")
        var trials: [TrialResult] = []

        for i in 0..<numTrials {
            let r = try await oneTrial()
            trials.append(r)
            if i > 0 { print() }
            print("🧪 Trial \(i + 1)")
            print("⚡ Prompt:     \(fmt(r.promptTps)) tokens/sec")
            print("🏃 Generation: \(fmt(r.genTps)) tokens/sec")
        }

        let n = Double(trials.count)
        let avgPrompt = trials.map(\.promptTps).reduce(0, +) / n
        let avgGen = trials.map(\.genTps).reduce(0, +) / n
        print("\n📊 Benchmark Summary:")
        print(String(repeating: "=", count: 50))
        print("Prepare:    \(fmt(prepareSeconds))s\(cacheSuffix)")
        print("Warmup:     \(fmt(warmupSeconds))s")
        if effectiveBatch > 1 {
            print("Batch:      \(effectiveBatch) rows (prompt/generation below are aggregate across rows)")
        }
        print("Prompt:     \(fmt(avgPrompt)) tokens/sec")
        print("Generation: \(fmt(avgGen)) tokens/sec")
        print(String(repeating: "=", count: 50))

        if let path = outputJson {
            let report = BenchmarkReport(
                model: bundle.name,
                promptTokens: promptTokens,
                generationTokens: generationTokens,
                numTrials: numTrials,
                batchSize: effectiveBatch,
                prepareSeconds: prepareSeconds,
                cacheHit: cacheHit,
                warmupSeconds: warmupSeconds,
                trials: trials,
                averages: BenchmarkReport.Averages(
                    promptTps: avgPrompt, generationTps: avgGen)
            )
            try writeJSON(report: report, path: path)
        }
    }

    // MARK: - Trial

    private func runTrial(
        engine: any InferenceEngine,
        prompt: [Int32],
        sampling: SamplingConfiguration
    ) async throws -> TrialResult {
        // Brief pause for engine to finish prior async work before reset
        try? await Task.sleep(for: .milliseconds(50))
        try await engine.reset()

        let options = InferenceOptions(maxTokens: generationTokens, includeLogits: false)
        let start = SuspendingClock.now
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: sampling, inferenceOptions: options
        )

        var promptTime: Double = 0
        var genStart = SuspendingClock.now
        var count = 0

        for try await _ in stream {
            if promptTime == 0 {
                let now = SuspendingClock.now
                promptTime = seconds(from: start, to: now)
                genStart = now
            }
            count += 1
        }

        let genTime = seconds(from: genStart, to: .now)
        let promptTps = promptTime > 0 ? Double(prompt.count) / promptTime : 0
        let decodeCount = max(0, count - 1)
        let genTps = genTime > 0 ? Double(decodeCount) / genTime : 0

        return TrialResult(promptTps: promptTps, genTps: genTps)
    }

    /// One batched trial: run `batchSize` copies of the prompt together through the fixed-batch
    /// graph (all greedy, no EOS so every row generates `generationTokens`). Prompt/generation
    /// tok/s are AGGREGATE across the batch (total tokens processed per second).
    private func runBatchedTrial(
        engine: CoreAISequentialEngine,
        prompt: [Int32],
        batchSize: Int
    ) async throws -> TrialResult {
        try? await Task.sleep(for: .milliseconds(50))
        let rows = Array(repeating: prompt, count: batchSize)
        let configs = Array(repeating: SamplingConfiguration(temperature: 0), count: batchSize)
        let r = try await engine.lockstepBatchedGenerate(
            promptRows: rows, configs: configs, maxNewTokens: generationTokens, eosTokenIds: [])
        let generatedPerRow = r.tokens.first?.count ?? 0
        let promptTps = r.prefillSeconds > 0 ? Double(batchSize * promptTokens) / r.prefillSeconds : 0
        let genTps = r.decodeSeconds > 0 ? Double(batchSize * generatedPerRow) / r.decodeSeconds : 0
        return TrialResult(promptTps: promptTps, genTps: genTps)
    }

    // MARK: - Helpers

    private func randomPrompt(vocabSize: Int, count: Int, seed: UInt64) -> [Int32] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var out = [Int32]()
        out.reserveCapacity(count)
        let v = UInt64(vocabSize)
        for _ in 0..<count {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            out.append(Int32(z % v))
        }
        return out
    }

    private func seconds(from start: SuspendingClock.Instant, to end: SuspendingClock.Instant) -> Double {
        (end - start).inSeconds
    }

    private func fmt(_ v: Double) -> String {
        String(format: "%.3f", v)
    }

    // MARK: - JSON output

    private func writeJSON(report: BenchmarkReport, path: String) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        print("Wrote: \(path)")
    }
}

// MARK: - Codable Report

struct TrialResult: Codable {
    let promptTps: Double
    let genTps: Double
}

struct BenchmarkReport: Codable {
    let model: String
    let promptTokens: Int
    let generationTokens: Int
    let numTrials: Int
    let batchSize: Int
    let prepareSeconds: Double
    let cacheHit: Bool
    let warmupSeconds: Double
    let trials: [TrialResult]
    let averages: Averages

    struct Averages: Codable {
        let promptTps: Double
        let generationTps: Double
    }
}
