// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation

/// Drives the server's generation core over a JSONL request file instead of HTTP.
///
/// Each line carries either a chat body or a loglikelihood body. Chat lines run through the same
/// cores the HTTP handler uses: `runChatCompletion` for `stream:false`, `runStreamingLoop` for
/// `stream:true` (deltas folded back into the result). Loglikelihood lines run through
/// `runLoglikelihood`, the teacher-forced scorer behind `/v1/completions`. Requests run in
/// arrival-timestamp order. The engine is single-active, so requests run sequentially, matching the
/// server. Interleaving sessions in the input reproduces the same per-session cache behavior. Each
/// result is one JSONL line with per-request instrumentation.
enum ReplayRunner {
    static func run(inputPath: String, outputPath: String?, state: ServerState) async throws {
        let text = try String(contentsOfFile: inputPath, encoding: .utf8)
        let requests = try ReplayIO.parseRequests(text)

        var lines: [String] = []
        lines.reserveCapacity(requests.count)
        for req in requests {
            let sessionID = req.session ?? "default"
            let result: ReplayResult
            if let loglikelihood = req.loglikelihood {
                result = await runLoglikelihoodLine(req: req, body: loglikelihood, state: state)
            } else if let chat = req.request {
                do {
                    if chat.stream == true {
                        result = try await runStreamed(req, chat: chat, sessionID: sessionID, state: state)
                    } else {
                        result = try await runNonStreamed(req, chat: chat, sessionID: sessionID, state: state)
                    }
                } catch {
                    result = ReplayResult(
                        id: req.id, session: req.session, t: req.t,
                        choices: nil, systemFingerprint: nil,
                        promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
                        ttftMs: 0, totalMs: 0, decodeTps: 0,
                        streamed: chat.stream == true, error: "\(error)")
                }
            } else {
                // The decoder guarantees one body; keep a defensive branch anyway.
                result = ReplayResult(
                    id: req.id, session: req.session, t: req.t,
                    choices: nil, systemFingerprint: nil,
                    promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
                    ttftMs: 0, totalMs: 0, decodeTps: 0, error: "empty replay request")
            }
            lines.append(try ReplayIO.encodeResult(result))
        }

        let output = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
        if let outputPath {
            try output.write(toFile: outputPath, atomically: true, encoding: .utf8)
        } else {
            print(output, terminator: "")
        }
    }

    /// Non-streaming path: the same `runChatCompletion` core the HTTP handler uses.
    private static func runNonStreamed(
        _ req: ReplayRequest, chat: ChatCompletionRequest, sessionID: String, state: ServerState
    ) async throws -> ReplayResult {
        let outcome = try await runChatCompletion(
            chatRequest: chat, state: state, sessionID: sessionID)
        let genSeconds = max(0, outcome.totalSeconds - outcome.ttftSeconds)
        let decodeTps = genSeconds > 0 ? Double(outcome.genTokenCount) / genSeconds : 0
        return ReplayResult(
            id: req.id, session: req.session, t: req.t,
            choices: outcome.response.choices,
            systemFingerprint: outcome.response.systemFingerprint,
            promptTokens: outcome.promptTokenCount,
            completionTokens: outcome.genTokenCount,
            prefixReuseTokens: outcome.prefixReuseTokens,
            ttftMs: outcome.ttftSeconds * 1000,
            totalMs: outcome.totalSeconds * 1000,
            decodeTps: decodeTps,
            streamed: false)
    }

    /// Streaming path: drives the real `runStreamingLoop` (the SSE handler's core) and folds the
    /// emitted deltas back into a result, so `stream:true` requests exercise incremental parsing.
    private static func runStreamed(
        _ req: ReplayRequest, chat: ChatCompletionRequest, sessionID: String, state: ServerState
    ) async throws -> ReplayResult {
        let prepared = try await prepareStreaming(chatRequest: chat, state: state, sessionID: sessionID)
        var aggregator = ReplayStreamAggregator()
        let outcome = try await runStreamingLoop(
            prepared: prepared, chatRequest: chat, state: state,
            emit: { aggregator.consume($0) })
        let genSeconds = max(0, outcome.totalSeconds - outcome.ttftSeconds)
        let decodeTps = genSeconds > 0 ? Double(outcome.genTokenCount) / genSeconds : 0
        return ReplayResult(
            id: req.id, session: req.session, t: req.t,
            choices: [aggregator.choice(finishReason: outcome.finishReason)],
            systemFingerprint: outcome.systemFingerprint,
            promptTokens: outcome.promptTokenCount,
            completionTokens: outcome.genTokenCount,
            prefixReuseTokens: outcome.prefixReuseTokens,
            ttftMs: outcome.ttftSeconds * 1000,
            totalMs: outcome.totalSeconds * 1000,
            decodeTps: decodeTps,
            streamed: true)
    }

    /// Scores a single loglikelihood prompt (one window per line) and surfaces its per-token
    /// logprobs. Requires the sequential variant, same as the HTTP `/v1/completions` path.
    private static func runLoglikelihoodLine(
        req: ReplayRequest, body: CompletionRequest, state: ServerState
    ) async -> ReplayResult {
        guard state.config.supportsLogprobs else {
            return ReplayResult(
                id: req.id, session: req.session, t: req.t,
                choices: nil, systemFingerprint: nil,
                promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
                ttftMs: 0, totalMs: 0, decodeTps: 0,
                error: "Logprobs not supported. Use --variant coreai-sequential")
        }
        let t0 = SuspendingClock().now
        do {
            let response = try await runLoglikelihood(req: body, state: state)
            let elapsed = SuspendingClock().now - t0
            let totalMs =
                (Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18) * 1000
            let logprobs = response.choices.first?.logprobs
            return ReplayResult(
                id: req.id, session: req.session, t: req.t,
                choices: nil, systemFingerprint: nil,
                promptTokens: logprobs?.tokens.count ?? 0,
                completionTokens: 0, prefixReuseTokens: 0,
                ttftMs: 0, totalMs: totalMs, decodeTps: 0,
                tokens: logprobs?.tokens,
                tokenLogprobs: logprobs?.tokenLogprobs,
                textOffset: logprobs?.textOffset)
        } catch {
            return ReplayResult(
                id: req.id, session: req.session, t: req.t,
                choices: nil, systemFingerprint: nil,
                promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
                ttftMs: 0, totalMs: 0, decodeTps: 0, error: "\(error)")
        }
    }
}
