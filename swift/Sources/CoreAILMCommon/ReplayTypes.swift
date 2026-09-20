// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - Replay format

/// One replayed chat request. Wraps the exact `ChatCompletionRequest` the HTTP server
/// decodes, plus a session id and an arrival timestamp, so a JSONL file can drive the real
/// generation stack without an HTTP surface. One JSON object per line:
///
///     {"id":"r1","session":"A","t":0.0,"request":{"messages":[{"role":"user","content":"hi"}],"seed":7,"max_tokens":64}}
public struct ReplayRequest: Decodable, Sendable {
    /// Caller-supplied id, echoed on the result. Optional.
    public let id: String?
    /// Session identity — maps to `X-Session-ID`. Defaults to "default" when absent.
    public let session: String?
    /// Arrival timestamp in seconds. Used to order requests; optional.
    public let t: Double?
    /// The chat request body, identical to what the HTTP endpoint accepts.
    public let request: ChatCompletionRequest

    public init(id: String?, session: String?, t: Double?, request: ChatCompletionRequest) {
        self.id = id
        self.session = session
        self.t = t
        self.request = request
    }

    /// Whether this request asks for the streaming path.
    public var isStreaming: Bool { request.stream == true }
}

/// Per-request result with instrumentation, emitted as one JSON object per line.
public struct ReplayResult: Encodable, Sendable {
    public let id: String?
    public let session: String?
    public let t: Double?
    public let content: String?
    /// Reasoning trace, when the model emits one (mirrors `reasoning_content` in the response).
    public let reasoningContent: String?
    /// Tool calls, when the request triggers tool-calling.
    public let toolCalls: [ToolCall]?
    public let finishReason: String?
    public let systemFingerprint: String?
    public let promptTokens: Int
    public let completionTokens: Int
    public let prefixReuseTokens: Int
    public let ttftMs: Double
    public let totalMs: Double
    public let decodeTps: Double
    /// True when the streaming path produced this result (request had `stream:true`).
    public let streamed: Bool
    public let error: String?

    public init(
        id: String?, session: String?, t: Double?,
        content: String?, finishReason: String?, systemFingerprint: String?,
        promptTokens: Int, completionTokens: Int, prefixReuseTokens: Int,
        ttftMs: Double, totalMs: Double, decodeTps: Double,
        reasoningContent: String? = nil, toolCalls: [ToolCall]? = nil,
        streamed: Bool = false, error: String? = nil
    ) {
        self.id = id
        self.session = session
        self.t = t
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.systemFingerprint = systemFingerprint
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.prefixReuseTokens = prefixReuseTokens
        self.ttftMs = ttftMs
        self.totalMs = totalMs
        self.decodeTps = decodeTps
        self.streamed = streamed
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case id, session, t, content, streamed, error
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
        case finishReason = "finish_reason"
        case systemFingerprint = "system_fingerprint"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case prefixReuseTokens = "prefix_reuse_tokens"
        case ttftMs = "ttft_ms"
        case totalMs = "total_ms"
        case decodeTps = "decode_tps"
    }
}

/// Folds streaming SSE deltas back into a final content/reasoning/tool-call triple, the way a
/// client reassembles a stream. Lets `--replay` drive the real streaming path and record its
/// aggregate output. The server emits each tool call as one whole delta (name + full args), so
/// deltas are appended as-is.
public struct ReplayStreamAggregator {
    private var content = ""
    private var reasoning = ""
    private var calls: [ToolCall] = []

    public init() {}

    public mutating func consume(_ delta: ChatCompletionChunk.Delta) {
        if let c = delta.content { content += c }
        if let r = delta.reasoningContent { reasoning += r }
        if let deltas = delta.toolCalls {
            for tc in deltas {
                calls.append(
                    ToolCall(
                        id: tc.id ?? "",
                        function: .init(name: tc.function?.name ?? "", arguments: tc.function?.arguments ?? "")))
            }
        }
    }

    public var aggregatedContent: String? { content.isEmpty ? nil : content }
    public var aggregatedReasoning: String? { reasoning.isEmpty ? nil : reasoning }
    public var aggregatedToolCalls: [ToolCall]? { calls.isEmpty ? nil : calls }
}

/// JSONL parsing/serialization for the replay format. Blank lines and `#`-prefixed comment
/// lines are ignored on input; requests are returned in arrival-timestamp order (stable for
/// equal or absent `t`).
public enum ReplayIO {
    public static func parseRequests(_ text: String) throws -> [ReplayRequest] {
        let decoder = JSONDecoder()
        var indexed: [(Int, ReplayRequest)] = []
        for (i, rawLine) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let req = try decoder.decode(ReplayRequest.self, from: Data(line.utf8))
            indexed.append((i, req))
        }
        // Stable sort by t (absent t sorts as 0), preserving input order within ties.
        return
            indexed
            .sorted { lhs, rhs in
                let lt = lhs.1.t ?? 0
                let rt = rhs.1.t ?? 0
                if lt != rt { return lt < rt }
                return lhs.0 < rhs.0
            }
            .map(\.1)
    }

    public static func encodeResult(_ result: ReplayResult) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(result), as: UTF8.self)
    }
}
