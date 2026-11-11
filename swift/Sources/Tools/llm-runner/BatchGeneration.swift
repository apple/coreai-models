// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - Batch mode

/// `--prompt-list` mode: several prompts generating at once through the continuous-batching
/// scheduler, as llm-server runs concurrent requests.
///
/// One decode call carries a token from every running prompt, so the gain is in aggregate throughput;
/// per-request tokens/sec is reported next to it.
extension LLMRunner {
    /// The non-blank lines of `path`, each one prompt. Throws when there are none.
    static func readPromptList(_ path: String) throws -> [String] {
        let contents = try String(contentsOfFile: path, encoding: .utf8)
        // Any newline, so a file saved with CRLF line endings splits too.
        let prompts = contents.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !prompts.isEmpty else {
            throw ValidationError("--prompt-list \(path) has no prompts: every line is blank")
        }
        return prompts
    }

    /// Prompts generating at once: `requested`, or every prompt, capped at what the model's batch
    /// holds. Throws when `requested` is past the model's max batch size.
    static func batchWidth(requested: Int?, promptCount: Int, maxBatchSize: Int) throws -> Int {
        if let requested, requested > maxBatchSize {
            throw ValidationError(
                "--concurrent-requests \(requested) exceeds the model's max batch size \(maxBatchSize). "
                    + "Re-export with --dynamic-batch-size \(requested) or more.")
        }
        return Swift.min(requested ?? promptCount, promptCount, maxBatchSize)
    }

    func runBatchGeneration(
        promptListPath: String,
        inferenceEngine: any InferenceEngine,
        bundle: LanguageModelBundle,
        tokenizer: any Tokenizer,
        samplingConfiguration: SamplingConfiguration,
        additionalEosTokenIds: Set<Int32>
    ) async throws {
        guard let engine = inferenceEngine as? CoreAISequentialEngine, engine.supportsBatching else {
            throw ValidationError(
                "--prompt-list needs the coreai-sequential engine and a model exported with "
                    + "--attention-mask batched --dynamic-batch-size N")
        }
        let prompts = try Self.readPromptList(promptListPath)
        let width = try Self.batchWidth(
            requested: concurrentRequests, promptCount: prompts.count, maxBatchSize: bundle.maxBatchSize)

        // End-of-sequence ids end a reply as `eos`; --stop-tokens end it as a stop sequence.
        var eosTokenIds = additionalEosTokenIds
        if let eos = tokenizer.eosTokenId { eosTokenIds.insert(Int32(eos)) }
        let userStops = try validateAndEncodeStopTokens(stopTokens: stopTokens, tokenizer: tokenizer)
            .sequences.filter { !($0.count == 1 && eosTokenIds.contains($0[0])) }
        let stopSequences = StopSequences(sequences: userStops)
        let decodeStopText: @Sendable ([Int32]) -> String = { tokenizer.decode(tokens: $0.map(Int.init)) }

        var requests: [BatchRequest] = []
        var promptTokenTotal = 0
        for text in prompts {
            let input: Input = applyChatTemplate ? .prompt(text) : .rawText(text)
            let tokens = try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            let room = engine.config.maxContextLength - tokens.count
            guard room > 0 else {
                throw ValidationError(
                    "prompt \(requests.count) is \(tokens.count) tokens, which fills the model's "
                        + "\(engine.config.maxContextLength)-token context")
            }
            promptTokenTotal += tokens.count
            requests.append(
                BatchRequest(
                    promptTokens: tokens.map { Int32($0) }, sampling: samplingConfiguration,
                    eosTokenIds: eosTokenIds, maxTokens: Swift.min(maxTokens, room),
                    stopSequences: stopSequences, decodeStopText: decodeStopText))
        }

        print("Batch: \(requests.count) prompts, up to \(width) at once")
        print("Prompt tokens: \(promptTokenTotal) across \(requests.count) requests\n")

        let scheduler = try BatchScheduler(engine: engine, maxConcurrent: width)
        let inferenceID = InstrumentsProfiler.beginInference(
            promptTokens: promptTokenTotal, maxTokens: maxTokens * requests.count)
        let (results, metrics) = try await scheduler.generateBatch(requests)
        InstrumentsProfiler.endInference(generatedTokens: metrics.generatedTokens, signpostID: inferenceID)

        for (index, result) in results.enumerated() {
            let text = tokenizer.decode(tokens: result.tokens.map { Int($0) })
            print("[\(index)] \(prompts[index])")
            print("    \(text.replacingOccurrences(of: "\n", with: "\n    "))")
            print(
                "    \(result.tokens.count) tokens · \(Self.describe(result.stopReason))"
                    + " · TTFT \(Self.seconds(result.timeToFirstToken))"
                    + " · \(Self.rate(result.tokensPerSecond)) tok/s")
        }

        print("\nAggregate")
        print("  wall           \(Self.seconds(metrics.wallSeconds))")
        print("  generated      \(metrics.generatedTokens) tokens")
        print("  decode         \(Self.rate(metrics.decodeTokensPerSecond)) tok/s across requests")
        print("  end-to-end     \(Self.rate(metrics.endToEndTokensPerSecond)) tok/s, prefill and queueing in")
        print("  per request    \(Self.rate(metrics.meanRequestTokensPerSecond)) tok/s mean")
    }

    private static func seconds(_ value: Double) -> String { String(format: "%.2fs", value) }
    private static func rate(_ value: Double) -> String { String(format: "%.1f", value) }

    static func describe(_ stopReason: StopReason) -> String {
        switch stopReason {
        case .maxTokens: return "max tokens"
        case .eos: return "eos"
        case .stopSequence(let text): return "stop '\(text)'"
        case .cancelled: return "cancelled"
        case .error: return "error"
        }
    }
}
