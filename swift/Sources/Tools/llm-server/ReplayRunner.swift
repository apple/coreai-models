// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation

/// Drives the server's generation core over a JSONL request file instead of HTTP.
///
/// Requests run in arrival-timestamp order through the same cores the HTTP handler uses:
/// `runChatCompletion` for `stream:false`, `runStreamingLoop` for `stream:true` (deltas folded
/// back into the result). The engine is single-active, so requests run sequentially, matching the
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
            do {
                if req.isStreaming {
                    result = try await runStreamed(req, sessionID: sessionID, state: state)
                } else {
                    result = try await runNonStreamed(req, sessionID: sessionID, state: state)
                }
            } catch {
                result = ReplayResult(
                    id: req.id, session: req.session, t: req.t,
                    choices: nil, systemFingerprint: nil,
                    promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
                    ttftMs: 0, totalMs: 0, decodeTps: 0,
                    streamed: req.isStreaming, error: "\(error)")
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
    private static func runNonStreamed(_ req: ReplayRequest, sessionID: String, state: ServerState) async throws
        -> ReplayResult
    {
        let outcome = try await runChatCompletion(
            chatRequest: req.request, state: state, sessionID: sessionID)
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
    private static func runStreamed(_ req: ReplayRequest, sessionID: String, state: ServerState) async throws
        -> ReplayResult
    {
        let prepared = try await prepareStreaming(chatRequest: req.request, state: state, sessionID: sessionID)
        var aggregator = ReplayStreamAggregator()
        let outcome = try await runStreamingLoop(
            prepared: prepared, chatRequest: req.request, state: state,
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
}
