// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import CoreAIShared
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NIOFoundationCompat
import Tokenizers

func startServer(state: ServerState, port: Int) async throws {
    let router = Router()

    router.get("/health") { _, _ in
        Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(string: #"{"status":"ok"}"#))
        )
    }

    router.get("/v1/models") { _, _ in
        let response = ModelsResponse(
            data: [
                ModelsResponse.ModelInfo(
                    id: state.config.modelName,
                    created: Int(Date().timeIntervalSince1970),
                    ownedBy: "coreai"
                )
            ]
        )
        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    router.post("/v1/chat/completions") { request, _ in
        let sessionID =
            HTTPField.Name("X-Session-ID").flatMap { request.headers[$0] } ?? "default"
        return try await handleChatCompletionsRoute(request: request, state: state, sessionID: sessionID)
    }

    router.post("/v1") { request, _ in
        try await handleAutoRoute(request: request, state: state)
    }

    router.post("/v1/completions") { request, _ in
        try await handleCompletionsRoute(request: request, state: state)
    }

    router.get("/ready") { _, _ in
        let ready = state.readySnapshot()
        let data = try JSONEncoder().encode(ready)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    router.get("/v1/stats") { _, _ in
        let stats = state.statsSnapshot()
        let data = try JSONEncoder().encode(stats)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    let app = Application(
        router: router,
        configuration: .init(
            address: .hostname("127.0.0.1", port: port)
        ))
    try await app.run()
}

// MARK: - Auto-detect Route (lm-eval posts to base_url directly)

private func handleAutoRoute(request: Request, state: ServerState) async throws -> Response {
    let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
    let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any]

    if json?["prompt"] != nil {
        return try await handleCompletionsFromBody(body: body, state: state)
    } else {
        return try await handleChatCompletionsFromBody(body: body, state: state)
    }
}

private func handleChatCompletionsFromBody(body: ByteBuffer, state: ServerState, sessionID: String? = nil) async throws
    -> Response
{
    guard state.tryAcquire() else {
        let err = ErrorResponse(error: .init(message: "Server is busy.", type: "server_error", code: "busy"))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .tooManyRequests, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
    let chatRequest: ChatCompletionRequest
    do {
        chatRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: body)
    } catch {
        state.release()
        let err = ErrorResponse(error: .init(message: "\(error)", type: "invalid_request_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .badRequest, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
    do {
        let shouldStream = chatRequest.stream ?? false
        if shouldStream {
            return try await handleStreamingRequest(chatRequest: chatRequest, state: state, sessionID: sessionID)
        } else {
            let response = try await handleNonStreamingRequest(
                chatRequest: chatRequest, state: state, sessionID: sessionID)
            state.release()
            return response
        }
    } catch let error as ServerError {
        state.release()
        let status: HTTPResponse.Status = error.isBadRequest ? .badRequest : .internalServerError
        let err = ErrorResponse(error: .init(message: "\(error)", type: "invalid_request_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: status, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data))
        )
    } catch {
        state.release()
        let err = ErrorResponse(error: .init(message: "\(error)", type: "server_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .internalServerError, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
}

// MARK: - Route Handler

private func handleChatCompletionsRoute(request: Request, state: ServerState, sessionID: String? = nil) async throws
    -> Response
{
    let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
    return try await handleChatCompletionsFromBody(body: body, state: state, sessionID: sessionID)
}

// MARK: - Non-Streaming

private func handleNonStreamingRequest(chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil)
    async throws -> Response
{
    let requestMaxTokens = chatRequest.maxCompletionTokens ?? chatRequest.maxTokens ?? state.config.defaultMaxTokens
    guard requestMaxTokens > 0 else {
        throw ServerError.badRequest("max_tokens must be positive")
    }
    let requestID = RequestID.next()
    let created = Int(Date().timeIntervalSince1970)

    let samplingConfig = state.makeSamplingConfig(
        temperature: chatRequest.temperature,
        topP: chatRequest.topP,
        topK: chatRequest.topK,
        minP: nil
    )

    let promptTokens = tokenizeMessages(chatRequest.messages, tools: chatRequest.tools, state: state)
    let stopSequences = buildStopSequences(from: chatRequest, state: state)
    let input: Input = .tokens(promptTokens)

    guard promptTokens.count < state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt (\(promptTokens.count) tokens) exceeds context length (\(state.config.maxContextLength))")
    }

    CLILogger.log(
        "[\(requestID)] messages: \(chatRequest.messages.count), tokens: \(promptTokens.count), max_tokens: \(requestMaxTokens)",
        component: "Server")

    let promptTokensInt32 = promptTokens.map { Int32($0) }
    let prefixReused = await state.prepareForRequest(sessionID: sessionID, promptTokens: promptTokensInt32)
    if prefixReused > 0 {
        CLILogger.log("[\(requestID)] prefix reuse: \(prefixReused) tokens cached", component: "Server")
    }
    let t0 = SuspendingClock().now

    let strategy: any DecodingStrategy
    if let schema = chatRequest.responseFormat?.extractedSchema {
        CLILogger.log("[\(requestID)] constrained generation (json_schema)", component: "Server")
        strategy = ConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: state.config.vocabSize)
    } else {
        strategy = VanillaDecodingStrategy()
    }

    let stream = try await strategy.decode(
        from: input,
        tokenizer: state.tokenizer,
        inferenceEngine: state.engine,
        samplingConfiguration: samplingConfig,
        options: InferenceOptions(maxTokens: requestMaxTokens),
        stopSequences: stopSequences
    )

    var genTokenCount = 0
    var parts: [String] = []
    var promptSeconds: Double = 0
    for try await result in stream {
        if genTokenCount == 0 {
            let ttft = SuspendingClock().now - t0
            promptSeconds = Double(ttft.components.seconds) + Double(ttft.components.attoseconds) / 1e18
        }
        parts.append(result.text)
        genTokenCount += 1
    }
    let text = parts.joined()

    let elapsed = SuspendingClock().now - t0
    let totalSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let genSeconds = totalSeconds - promptSeconds
    let cleaned = stripThinkingTags(text)

    // Parse tool calls if the model supports them and tools were requested
    var responseContent: String? = cleaned
    var responseToolCalls: [ToolCall]? = nil
    var finishReason = genTokenCount >= requestMaxTokens ? "length" : "stop"

    if chatRequest.tools != nil, var parser = state.makeToolCallParser() {
        let events = parser.consume(cleaned) + parser.flush()
        var textParts: [String] = []
        var toolCalls: [ToolCall] = []
        for event in events {
            switch event {
            case .text(let t):
                textParts.append(t)
            case .toolCall(let id, let name, let argsJSON):
                toolCalls.append(ToolCall(id: id, function: .init(name: name, arguments: argsJSON)))
            }
        }
        if !toolCalls.isEmpty {
            responseToolCalls = toolCalls
            if finishReason != "length" {
                finishReason = "tool_calls"
            }
            let remaining = textParts.joined().trimmingCharacters(in: .whitespacesAndNewlines)
            responseContent = remaining.isEmpty ? nil : remaining
            state.recordToolCalls(toolCalls.map(\.function.name))
        }
    }

    let prefillTps = promptSeconds > 0 ? Double(promptTokens.count) / promptSeconds : 0
    let genTps = genSeconds > 0 ? Double(genTokenCount) / genSeconds : 0
    var logLine =
        "\(ts()) [\(requestID)] \(promptTokens.count)t prefill \(String(format: "%.1f", prefillTps)) t/s, \(genTokenCount)t gen \(String(format: "%.1f", genTps)) t/s (\(String(format: "%.2f", totalSeconds))s)"
    if let calls = responseToolCalls {
        logLine += " → \(calls.count) tool call(s)"
        if CLILogger.level > 0 {
            logLine += ": \(calls.map(\.function.name).joined(separator: ", "))"
        }
    }
    print(logLine)
    state.stats.record(
        promptTokens: promptTokens.count, genTokens: genTokenCount, promptSeconds: promptSeconds,
        genSeconds: genSeconds, totalSeconds: totalSeconds, toolCalls: responseToolCalls?.count ?? 0)
    state.recordPromptTokens(promptTokensInt32)

    let response = ChatCompletionResponse(
        id: requestID,
        object: "chat.completion",
        created: created,
        model: state.config.modelName,
        choices: [
            .init(
                index: 0,
                message: .init(role: "assistant", content: responseContent, toolCalls: responseToolCalls),
                finishReason: finishReason
            )
        ],
        usage: ChatCompletionResponse.Usage(
            promptTokens: promptTokens.count,
            completionTokens: genTokenCount,
            totalTokens: promptTokens.count + genTokenCount
        )
    )

    let data = try JSONEncoder().encode(response)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: data))
    )
}

// MARK: - Streaming (SSE)

private func handleStreamingRequest(chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil)
    async throws -> Response
{
    let requestMaxTokens = chatRequest.maxCompletionTokens ?? chatRequest.maxTokens ?? state.config.defaultMaxTokens
    guard requestMaxTokens > 0 else {
        throw ServerError.badRequest("max_tokens must be positive")
    }
    let requestID = RequestID.next()
    let created = Int(Date().timeIntervalSince1970)

    let samplingConfig = state.makeSamplingConfig(
        temperature: chatRequest.temperature,
        topP: chatRequest.topP,
        topK: chatRequest.topK,
        minP: nil
    )

    let promptTokens = tokenizeMessages(chatRequest.messages, tools: chatRequest.tools, state: state)
    let stopSequences = buildStopSequences(from: chatRequest, state: state)
    let input: Input = .tokens(promptTokens)

    guard promptTokens.count < state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt (\(promptTokens.count) tokens) exceeds context length (\(state.config.maxContextLength))")
    }

    CLILogger.log(
        "[\(requestID)] stream, messages: \(chatRequest.messages.count), tokens: \(promptTokens.count), max_tokens: \(requestMaxTokens)",
        component: "Server")

    let promptTokensInt32 = promptTokens.map { Int32($0) }
    let prefixReused = await state.prepareForRequest(sessionID: sessionID, promptTokens: promptTokensInt32)
    if prefixReused > 0 {
        CLILogger.log("[\(requestID)] prefix reuse: \(prefixReused) tokens cached", component: "Server")
    }

    let responseBody = ResponseBody { writer in
        defer { state.release() }
        do {
            let encoder = JSONEncoder()
            let genStart = SuspendingClock().now

            let roleChunk = ChatCompletionChunk(
                id: requestID, object: "chat.completion.chunk", created: created, model: state.config.modelName,
                choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
            )
            if let data = try? encoder.encode(roleChunk), let json = String(data: data, encoding: .utf8) {
                try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
            }

            let strategy: any DecodingStrategy
            if let schema = chatRequest.responseFormat?.extractedSchema {
                strategy = ConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: state.config.vocabSize)
            } else {
                strategy = VanillaDecodingStrategy()
            }

            let tokenStream = try await strategy.decode(
                from: input,
                tokenizer: state.tokenizer,
                inferenceEngine: state.engine,
                samplingConfiguration: samplingConfig,
                options: InferenceOptions(maxTokens: requestMaxTokens),
                stopSequences: stopSequences
            )

            var thinkParser = ThinkTagParser()
            var toolParser = state.makeToolCallParser()
            var tokenCount = 0
            var hasToolCalls = false
            var toolCallIndex = 0
            var toolCallNames: [String] = []
            var promptSeconds: Double = 0

            // Emit a single SSE chunk (text content or tool call delta)
            func emitChunk(_ delta: ChatCompletionChunk.Delta) async throws {
                let chunk = ChatCompletionChunk(
                    id: requestID, object: "chat.completion.chunk", created: created,
                    model: state.config.modelName,
                    choices: [.init(index: 0, delta: delta, finishReason: nil)])
                if let data = try? encoder.encode(chunk),
                    let json = String(data: data, encoding: .utf8)
                {
                    try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
                }
            }

            // Process tool parser events: emit text as content, tool calls as deltas
            func emitToolEvents(_ events: [ToolCallParser.Event]) async throws {
                for event in events {
                    switch event {
                    case .text(let t) where !t.isEmpty:
                        try await emitChunk(.init(role: nil, content: t))
                    case .toolCall(let id, let name, let argsJSON):
                        hasToolCalls = true
                        toolCallNames.append(name)
                        let tcDelta = ToolCallDelta(
                            index: toolCallIndex, id: id, type: "function",
                            function: .init(name: name, arguments: argsJSON))
                        try await emitChunk(.init(role: nil, content: nil, toolCalls: [tcDelta]))
                        toolCallIndex += 1
                    default: break
                    }
                }
            }

            // Process text through the tool parser (or emit directly if no tool support)
            func emitText(_ text: String) async throws {
                guard !text.isEmpty else { return }
                if var tp = toolParser {
                    let events = tp.consume(text)
                    toolParser = tp
                    try await emitToolEvents(events)
                } else {
                    try await emitChunk(.init(role: nil, content: text))
                }
            }

            for try await result in tokenStream {
                if tokenCount == 0 {
                    let ttft = SuspendingClock().now - genStart
                    promptSeconds = Double(ttft.components.seconds) + Double(ttft.components.attoseconds) / 1e18
                }
                tokenCount += 1
                for event in thinkParser.consume(result.text) {
                    if case .text(let delta) = event {
                        try await emitText(delta)
                    }
                }
            }

            // Flush think parser
            for event in thinkParser.flush() {
                if case .text(let delta) = event {
                    try await emitText(delta)
                }
            }

            // Flush tool parser
            if var tp = toolParser {
                try await emitToolEvents(tp.flush())
                toolParser = tp
            }

            let finishReason = tokenCount >= requestMaxTokens ? "length" : (hasToolCalls ? "tool_calls" : "stop")
            let doneChunk = ChatCompletionChunk(
                id: requestID, object: "chat.completion.chunk", created: created, model: state.config.modelName,
                choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
            )
            if let data = try? encoder.encode(doneChunk), let json = String(data: data, encoding: .utf8) {
                try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
            }
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))

            let elapsed = SuspendingClock().now - genStart
            let totalSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            let genSeconds = totalSeconds - promptSeconds
            let prefillTps = promptSeconds > 0 ? Double(promptTokens.count) / promptSeconds : 0
            let genTps = genSeconds > 0 ? Double(tokenCount) / genSeconds : 0
            var logLine =
                "\(ts()) [\(requestID)] stream: \(promptTokens.count)t prefill \(String(format: "%.1f", prefillTps)) t/s, \(tokenCount)t gen \(String(format: "%.1f", genTps)) t/s (\(String(format: "%.2f", totalSeconds))s) [\(finishReason)]"
            if !toolCallNames.isEmpty {
                logLine += " → \(toolCallNames.count) tool call(s)"
                if CLILogger.level > 0 {
                    logLine += ": \(toolCallNames.joined(separator: ", "))"
                }
            }
            print(logLine)
            state.stats.record(
                promptTokens: promptTokens.count, genTokens: tokenCount, promptSeconds: promptSeconds,
                genSeconds: genSeconds, totalSeconds: totalSeconds, toolCalls: toolCallNames.count)
            state.recordPromptTokens(promptTokensInt32)
            if !toolCallNames.isEmpty {
                state.recordToolCalls(toolCallNames)
            }

            try await writer.finish(nil)
        } catch {
            print("\(ts()) [\(requestID)] stream error: \(error)")
            try? await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            try? await writer.finish(nil)
        }
    }

    return Response(
        status: .ok,
        headers: [
            .contentType: "text/event-stream",
            .init("Cache-Control")!: "no-cache",
            .init("Connection")!: "keep-alive",
        ],
        body: responseBody
    )
}

// MARK: - Helpers

private func tokenizeMessages(
    _ messages: [ChatMessage], tools: [ToolDefinition]? = nil, state: ServerState
) -> [Int] {
    var templateMessages: [[String: any Sendable]] = []
    for msg in messages {
        var dict: [String: any Sendable] = ["role": msg.role]

        if msg.role == "tool" {
            dict["content"] = msg.content.textContent
            if let id = msg.toolCallId { dict["tool_call_id"] = id }
        } else if msg.role == "assistant", let calls = msg.toolCalls, !calls.isEmpty {
            let callDicts: [[String: any Sendable]] = calls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    ] as [String: any Sendable],
                ]
            }
            dict["tool_calls"] = callDicts
            dict["content"] = msg.content.textContent
        } else {
            var content = msg.content.textContent
            if msg.role == "system" && state.config.noThinking {
                content += "\n/no_think"
            }
            dict["content"] = content
        }
        templateMessages.append(dict)
    }

    if state.config.noThinking && !messages.contains(where: { $0.role == "system" }) {
        templateMessages.insert(["role": "system", "content": "/no_think"], at: 0)
    }

    let toolSpecs: [[String: any Sendable]]? = tools?.map { tool in
        var funcDict: [String: any Sendable] = [
            "name": tool.function.name,
            "description": tool.function.description ?? "",
        ]
        if let params = tool.function.parameters {
            funcDict["parameters"] = jsonValueToSendable(params)
        }
        let spec: [String: any Sendable] = [
            "type": tool.type,
            "function": funcDict,
        ]
        return spec
    }

    do {
        let tokens = try state.tokenizer.applyChatTemplate(
            messages: templateMessages, tools: toolSpecs)
        return tokens
    } catch {
        CLILogger.log("applyChatTemplate failed: \(error)", component: "Server")
    }

    let text = templateMessages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
        .joined(separator: "\n")
    return state.tokenizer.encode(text: text)
}

private func jsonValueToSendable(_ value: JSONValue) -> any Sendable {
    switch value {
    case .string(let s): return s
    case .number(let n): return n
    case .bool(let b): return b
    case .null: return Optional<String>.none as any Sendable
    case .object(let obj): return obj.mapValues { jsonValueToSendable($0) } as [String: any Sendable]
    case .array(let arr): return arr.map { jsonValueToSendable($0) } as [any Sendable]
    }
}

private func buildStopSequences(from request: ChatCompletionRequest, state: ServerState) -> StopSequences {
    var additionalSequences: [[Int32]] = []
    if let stop = request.stop {
        for s in stop {
            let tokens = state.tokenizer.encode(text: s).map { Int32($0) }
            if !tokens.isEmpty {
                additionalSequences.append(tokens)
            }
        }
    }
    return StopSequences(
        for: state.tokenizer,
        additionalSequences: additionalSequences,
        additionalEosTokenIds: state.config.additionalEosTokenIds
    )
}

private func stripThinkingTags(_ text: String) -> String {
    ThinkTagParser.stripCompleted(from: text)
}

private func ts() -> String {
    let now = Date()
    let calendar = Calendar.current
    let h = calendar.component(.hour, from: now)
    let m = calendar.component(.minute, from: now)
    let s = calendar.component(.second, from: now)
    return String(format: "%02d:%02d:%02d", h, m, s)
}
