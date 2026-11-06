// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - Replay format

/// One line of a `--replay` file: a request body, a session id, and an arrival time.
///
/// A line has either a `request` (a `/v1/chat/completions` body) or a `completion` (a
/// `/v1/completions` body, to score prompt logprobs):
///
///     {"id":"r1","session":"A","t":0.0,"request":{"messages":[{"role":"user","content":"hi"}],"seed":7,"max_tokens":64}}
///     {"id":"d0","completion":{"prompt":[[1,2,3],[3,4,5]],"logprobs":0}}
public struct ReplayRequest: Decodable, Sendable {
    public enum Body: Sendable {
        case chat(ChatCompletionRequest)
        case completion(CompletionRequest)
    }

    /// Caller-supplied id, echoed on the result. Optional.
    public let id: String?
    /// Session identity — maps to `X-Session-ID`. Defaults to "default" when absent.
    public let session: String?
    /// Arrival timestamp in seconds. Used to order requests; optional.
    public let t: Double?
    public let body: Body

    public init(id: String?, session: String?, t: Double?, request: ChatCompletionRequest) {
        self.init(id: id, session: session, t: t, body: .chat(request))
    }

    public init(id: String?, session: String?, t: Double?, body: Body) {
        self.id = id
        self.session = session
        self.t = t
        self.body = body
    }

    enum CodingKeys: String, CodingKey {
        case id, session, t, request, completion
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id)
        session = try container.decodeIfPresent(String.self, forKey: .session)
        t = try container.decodeIfPresent(Double.self, forKey: .t)
        let chat = try container.decodeIfPresent(ChatCompletionRequest.self, forKey: .request)
        let completion = try container.decodeIfPresent(CompletionRequest.self, forKey: .completion)
        switch (chat, completion) {
        case (let chat?, nil):
            body = .chat(chat)
        case (nil, let completion?):
            body = .completion(completion)
        default:
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: [],
                    debugDescription: "A replay line needs exactly one of 'request' or 'completion'"))
        }
    }

    /// True for a chat body with `stream: true`.
    public var isStreaming: Bool {
        if case .chat(let request) = body { return request.stream == true }
        return false
    }
}

/// The `choices` of a replay result, in the shape of the matching HTTP response.
public enum ReplayChoices: Encodable, Sendable {
    case chat([ChatCompletionResponse.Choice])
    case completion([CompletionResponse.CompletionChoice])

    public func encode(to encoder: any Swift.Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .chat(let choices): try container.encode(choices)
        case .completion(let choices): try container.encode(choices)
        }
    }
}

/// One line of `--replay` output: the response `choices`, plus token counts and timing. On error,
/// `choices` is nil and `error` is set.
///
///     {"id":"r1","session":"A","t":0.0,"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"prompt_tokens":3,"completion_tokens":1,"ttft_ms":12.5}
public struct ReplayResult: Encodable, Sendable {
    public let id: String?
    public let session: String?
    public let t: Double?
    public let choices: ReplayChoices?
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
        choices: [ChatCompletionResponse.Choice]?, systemFingerprint: String?,
        promptTokens: Int, completionTokens: Int, prefixReuseTokens: Int,
        ttftMs: Double, totalMs: Double, decodeTps: Double,
        streamed: Bool = false, error: String? = nil
    ) {
        self.id = id
        self.session = session
        self.t = t
        self.choices = choices.map { .chat($0) }
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

    /// Result of a `completion` line. `promptTokens` counts the scored tokens; `ttftMs` is `totalMs`.
    public init(
        id: String?, session: String?, t: Double?,
        completionChoices: [CompletionResponse.CompletionChoice], promptTokens: Int, totalMs: Double
    ) {
        self.id = id
        self.session = session
        self.t = t
        self.choices = .completion(completionChoices)
        self.systemFingerprint = nil
        self.promptTokens = promptTokens
        self.completionTokens = 0
        self.prefixReuseTokens = 0
        self.ttftMs = totalMs
        self.totalMs = totalMs
        self.decodeTps = 0
        self.streamed = false
        self.error = nil
    }

    enum CodingKeys: String, CodingKey {
        case id, session, t, choices, streamed, error
        case systemFingerprint = "system_fingerprint"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case prefixReuseTokens = "prefix_reuse_tokens"
        case ttftMs = "ttft_ms"
        case totalMs = "total_ms"
        case decodeTps = "decode_tps"
    }
}

/// Folds streaming SSE deltas back into a single response `Choice`, the way a client reassembles
/// a stream. Lets `--replay` drive the real streaming path and record its aggregate output using
/// the same `choices` shape as the non-streaming path. The server emits each tool call as one
/// whole delta (name + full args), so deltas are appended as-is.
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

    /// Fold the deltas into a single assistant `Choice` (index 0) carrying the final finish
    /// reason, matching the shape the non-streaming path records.
    public func choice(finishReason: String?) -> ChatCompletionResponse.Choice {
        let message = ChatCompletionResponse.ResponseMessage(
            role: "assistant",
            content: content.isEmpty ? nil : content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: calls.isEmpty ? nil : calls)
        return ChatCompletionResponse.Choice(index: 0, message: message, finishReason: finishReason)
    }
}

/// Errors surfaced while reading a replay JSONL file.
public enum ReplayError: Error, CustomStringConvertible {
    case malformedLine(line: Int, underlying: Error)

    public var description: String {
        switch self {
        case .malformedLine(let line, let underlying):
            return "malformed JSON on line \(line): \(underlying)"
        }
    }
}

/// JSONL parsing/serialization for the replay format. Blank lines and `#`-prefixed comment
/// lines are ignored on input; requests are returned in arrival-timestamp order (stable for
/// equal or absent `t`).
public enum ReplayIO {
    public static func parseRequests(_ text: String) throws -> [ReplayRequest] {
        let decoder = JSONDecoder()
        var indexed: [(Int, ReplayRequest)] = []
        let text = text.hasPrefix("\u{FEFF}") ? String(text.dropFirst()) : text
        // `\r\n` is a single Swift Character, so split on any Unicode newline rather than "\n".
        for (i, rawLine) in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
            .enumerated()
        {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { continue }
            do {
                indexed.append((i, try decoder.decode(ReplayRequest.self, from: Data(line.utf8))))
            } catch {
                throw ReplayError.malformedLine(line: i + 1, underlying: error)
            }
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
