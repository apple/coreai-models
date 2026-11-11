// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import CoreAIShared
import Foundation

// MARK: - Concurrent requests

/// `--concurrency N`: N requests in flight at once through the continuous-batching scheduler, the
/// loop llm-server serves `--concurrent-requests` with. Unlike `--batch-size`, which steps identical
/// rows in lockstep, each request has its own prompt, joins as a row frees, and is timed on its own.
extension LLMBenchmark {
    func runConcurrent(
        engine: CoreAISequentialEngine,
        bundle: LanguageModelBundle,
        prepareSeconds: Double,
        cacheHit: Bool
    ) async throws {
        let cacheSuffix = cacheHit ? " (cache hit)" : ""
        let width = Swift.min(concurrency, bundle.maxBatchSize)
        if width < concurrency {
            print(
                "\nNote: --concurrency \(concurrency) is past the asset's max batch size "
                    + "\(bundle.maxBatchSize); \(width) run at once and the rest queue behind them.")
        }
        let scheduler = try BatchScheduler(engine: engine, maxConcurrent: width)

        var promptIndex: UInt64 = 0
        func requests() -> [BatchRequest] {
            (0..<concurrency).map { _ in
                defer { promptIndex += 1 }
                // No stop tokens, so every request generates exactly `--generation-tokens`.
                return BatchRequest(
                    promptTokens: randomPrompt(
                        vocabSize: bundle.vocabSize, count: promptTokens, seed: seed &+ promptIndex),
                    sampling: SamplingConfiguration(temperature: 0), eosTokenIds: [], maxTokens: generationTokens)
            }
        }

        // Warmup at the timed shape, so every decode width the trials reach is compiled.
        print("\n⚙️  Warming up engine...", terminator: "")
        fflush(stdout)
        let warmupStart = SuspendingClock.now
        _ = try await scheduler.generateBatch(requests())
        let warmupSeconds = (SuspendingClock.now - warmupStart).inSeconds
        print(" done in \(fmt(warmupSeconds))s")

        print(
            "\n🔄 Benchmarking with \(promptTokens) prompt tokens, \(generationTokens) generation "
                + "tokens, \(concurrency) requests at once\n")
        var trials: [TrialResult] = []
        for i in 0..<numTrials {
            let trial = TrialResult(metrics: try await scheduler.generateBatch(requests()).metrics)
            trials.append(trial)
            if i > 0 { print() }
            print("🧪 Trial \(i + 1)")
            print("⚡ Prompt:     \(fmt(trial.promptTps)) tokens/sec, all requests together")
            print("⏱️  TTFT:       \(fmt(trial.meanTtftSeconds ?? 0))s mean")
            print("🏃 Generation: \(fmt(trial.perRequestGenTps ?? 0)) tokens/sec per request")
            print("📈 Aggregate:  \(fmt(trial.genTps)) tokens/sec decode")
        }

        let n = Double(trials.count)
        func mean(_ value: (TrialResult) -> Double?) -> Double? {
            let values = trials.compactMap(value)
            return values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
        }
        let averages = BenchmarkReport.Averages(
            promptTps: trials.map(\.promptTps).reduce(0, +) / n,
            generationTps: trials.map(\.genTps).reduce(0, +) / n,
            perRequestGenerationTps: mean(\.perRequestGenTps),
            endToEndTps: mean(\.endToEndTps),
            meanTtftSeconds: mean(\.meanTtftSeconds))
        print("\n📊 Benchmark Summary:")
        print(String(repeating: "=", count: 50))
        print("Prepare:    \(fmt(prepareSeconds))s\(cacheSuffix)")
        print("Warmup:     \(fmt(warmupSeconds))s")
        print("Requests:   \(concurrency) at once, \(width) resident")
        print("Prompt:     \(fmt(averages.promptTps)) tokens/sec, all requests together")
        print("TTFT:       \(fmt(averages.meanTtftSeconds ?? 0))s mean")
        print("Generation: \(fmt(averages.perRequestGenerationTps ?? 0)) tokens/sec per request")
        print("Aggregate:  \(fmt(averages.generationTps)) tokens/sec decode")
        print("End-to-end: \(fmt(averages.endToEndTps ?? 0)) tokens/sec, wall clock")
        print(String(repeating: "=", count: 50))

        if let path = outputJson {
            let report = BenchmarkReport(
                model: bundle.name,
                promptTokens: promptTokens,
                generationTokens: generationTokens,
                numTrials: numTrials,
                batchSize: width,
                concurrency: concurrency,
                prepareSeconds: prepareSeconds,
                cacheHit: cacheHit,
                warmupSeconds: warmupSeconds,
                trials: trials,
                averages: averages
            )
            try writeJSON(report: report, path: path)
        }
    }
}

extension TrialResult {
    /// One concurrent trial's numbers, from ``BatchGenerationMetrics``. `genTps` is the decode
    /// throughput across requests, a total, as `--batch-size` reports one.
    init(metrics: BatchGenerationMetrics) {
        self.init(
            promptTps: metrics.promptTokensPerSecond,
            genTps: metrics.decodeTokensPerSecond,
            perRequestGenTps: metrics.meanRequestTokensPerSecond,
            endToEndTps: metrics.endToEndTokensPerSecond,
            meanTtftSeconds: metrics.meanTimeToFirstToken)
    }
}
