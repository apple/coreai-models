// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import Foundation

/// Drives the server's generation core over a JSONL request file instead of HTTP.
///
/// Requests run in arrival-timestamp order through the same cores the HTTP handler uses:
/// `runChatCompletion` for `stream:false`, `runStreamingLoop` for `stream:true` (deltas folded
/// back into the result). By default requests run one at a time. With `concurrent`, all requests
/// start at once, and requests over the queue limits produce queue-full error lines. Output lines
/// keep arrival-timestamp order in both modes. Interleaving sessions in the input reproduces the
/// same per-session cache behavior. Each result is one JSONL line with per-request instrumentation.
enum ReplayRunner {
    static func run(
        inputPath: String, outputPath: String?, state: ServerState, concurrent: Bool = false
    ) async throws {
        let text = try String(contentsOfFile: inputPath, encoding: .utf8)
        let requests = try ReplayIO.parseRequests(text)

        var lines = [String](repeating: "", count: requests.count)
        if concurrent {
            try await withThrowingTaskGroup(of: (Int, String).self) { group in
                for (i, req) in requests.enumerated() {
                    group.addTask { (i, await oneLine(req, state: state)) }
                }
                for try await (i, line) in group { lines[i] = line }
            }
        } else {
            for (i, req) in requests.enumerated() {
                lines[i] = await oneLine(req, state: state)
            }
        }

        let output = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
        if let outputPath {
            try output.write(toFile: outputPath, atomically: true, encoding: .utf8)
        } else {
            print(output, terminator: "")
        }
    }

    /// Runs `req` while holding the access for its lane and returns its encoded result line. An
    /// error, including queue-full, produces an error result line.
    private static func oneLine(_ req: ReplayRequest, state: ServerState) async -> String {
        let sessionID = req.session ?? "default"
        let result: ReplayResult
        do {
            let lane = BatchServingRouter.lane(
                hasResponseSchema: req.request.responseFormat?.extractedSchema != nil)
            let access = try await state.acquireAccess(lane: lane)
            defer { access.release() }

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
        return (try? ReplayIO.encodeResult(result)) ?? "{\"error\":\"encode failed\"}"
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
        let started = try await startStreamingDecode(prepared: prepared, chatRequest: req.request, state: state)
        let outcome = try await runStreamingLoop(
            started: started, prepared: prepared, chatRequest: req.request, state: state,
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
