// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Based on mlx-lm benchmark (https://github.com/ml-explore/mlx-lm)

import ArgumentParser
import CoreAILanguageModels
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

    func validate() throws {
        if promptTokens < 1 { throw ValidationError("--prompt-tokens must be >= 1") }
        if generationTokens < 1 { throw ValidationError("--generation-tokens must be >= 1") }
        if numTrials < 1 { throw ValidationError("--num-trials must be >= 1") }
        if !FileManager.default.fileExists(atPath: model) {
            throw ValidationError("Model path not found: \(model)")
        }
    }

    func run() async throws {
        #if DEBUG
        print("Note: built in Debug mode. For more reliable results, build with -c release.")
        #endif

        let bundle = try LanguageBundle(from: model)
        let vocabSize = bundle.vocabSize

        let engineConfig = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let configData = try JSONEncoder().encode(engineConfig)
        print("\n⏳ Preparing AI asset...")
        let engine = try await EngineFactory.createEngine(
            config: configData,
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        )

        let prompt = randomPrompt(vocabSize: vocabSize, count: promptTokens, seed: seed)
        let sampling = SamplingConfiguration(temperature: 0)

        // Warmup
        print("\n⚙️  Warming up engine...")
        _ = try await runTrial(engine: engine, prompt: prompt, sampling: sampling)

        // Timed trials
        print("\n🔄 Benchmarking with \(promptTokens) prompt tokens, \(generationTokens) generation tokens\n")
        var trials: [TrialResult] = []

        for i in 0..<numTrials {
            let r = try await runTrial(engine: engine, prompt: prompt, sampling: sampling)
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
        print("Prompt:     \(fmt(avgPrompt)) tokens/sec")
        print("Generation: \(fmt(avgGen)) tokens/sec")
        print(String(repeating: "=", count: 50))

        if let path = outputJson {
            let report = BenchmarkReport(
                model: bundle.name,
                promptTokens: promptTokens,
                generationTokens: generationTokens,
                numTrials: numTrials,
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
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
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
    let trials: [TrialResult]
    let averages: Averages

    struct Averages: Codable {
        let promptTps: Double
        let generationTps: Double
    }
}
