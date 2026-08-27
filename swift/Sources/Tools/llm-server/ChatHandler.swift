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

    let promptTokens = tokenizeMessages(chatRequest.messages, state: state)
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
    for try await result in stream {
        parts.append(result.text)
        genTokenCount += 1
    }
    let text = parts.joined()

    let elapsed = SuspendingClock().now - t0
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let cleaned = stripThinkingTags(text)
    let finishReason = genTokenCount >= requestMaxTokens ? "length" : "stop"

    let tokPerSec = seconds > 0 ? Double(genTokenCount) / seconds : 0
    print(
        "[\(requestID)] \(promptTokens.count)t → \(genTokenCount)t in \(String(format: "%.2f", seconds))s (\(String(format: "%.1f", tokPerSec)) tok/s)"
    )
    state.stats.record(
        promptTokens: promptTokens.count, genTokens: genTokenCount, promptSeconds: 0, genSeconds: seconds,
        totalSeconds: seconds)
    state.recordPromptTokens(promptTokensInt32)

    let response = ChatCompletionResponse(
        id: requestID,
        object: "chat.completion",
        created: created,
        model: state.config.modelName,
        choices: [
            .init(
                index: 0,
                message: .init(role: "assistant", content: cleaned),
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

    let promptTokens = tokenizeMessages(chatRequest.messages, state: state)
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
            var tokenCount = 0

            for try await result in tokenStream {
                tokenCount += 1
                let events = thinkParser.consume(result.text)
                for event in events {
                    if case .text(let delta) = event, !delta.isEmpty {
                        let chunk = ChatCompletionChunk(
                            id: requestID, object: "chat.completion.chunk", created: created,
                            model: state.config.modelName,
                            choices: [.init(index: 0, delta: .init(role: nil, content: delta), finishReason: nil)]
                        )
                        if let data = try? encoder.encode(chunk), let json = String(data: data, encoding: .utf8) {
                            try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
                        }
                    }
                }
            }

            // Flush any remaining buffered text
            let finalEvents = thinkParser.flush()
            for event in finalEvents {
                if case .text(let delta) = event, !delta.isEmpty {
                    let chunk = ChatCompletionChunk(
                        id: requestID, object: "chat.completion.chunk", created: created,
                        model: state.config.modelName,
                        choices: [.init(index: 0, delta: .init(role: nil, content: delta), finishReason: nil)]
                    )
                    if let data = try? encoder.encode(chunk), let json = String(data: data, encoding: .utf8) {
                        try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }
            }

            let finishReason = tokenCount >= requestMaxTokens ? "length" : "stop"
            let doneChunk = ChatCompletionChunk(
                id: requestID, object: "chat.completion.chunk", created: created, model: state.config.modelName,
                choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
            )
            if let data = try? encoder.encode(doneChunk), let json = String(data: data, encoding: .utf8) {
                try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
            }
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))

            let elapsed = SuspendingClock().now - genStart
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            let tokPerSec = seconds > 0 ? Double(tokenCount) / seconds : 0
            print(
                "[\(requestID)] stream: \(promptTokens.count)t → \(tokenCount)t in \(String(format: "%.2f", seconds))s (\(String(format: "%.1f", tokPerSec)) tok/s) [\(finishReason)]"
            )
            state.stats.record(
                promptTokens: promptTokens.count, genTokens: tokenCount, promptSeconds: 0, genSeconds: seconds,
                totalSeconds: seconds)
            state.recordPromptTokens(promptTokensInt32)

            try await writer.finish(nil)
        } catch {
            print("[\(requestID)] stream error: \(error)")
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

private func tokenizeMessages(_ messages: [ChatMessage], state: ServerState) -> [Int] {
    var templateMessages: [[String: any Sendable]] = []
    for msg in messages {
        var content = msg.content.textContent
        if msg.role == "system" && state.config.noThinking {
            content += "\n/no_think"
        }
        templateMessages.append(["role": msg.role, "content": content])
    }

    if state.config.noThinking && !messages.contains(where: { $0.role == "system" }) {
        templateMessages.insert(["role": "system", "content": "/no_think"], at: 0)
    }

    if let tokens = try? state.tokenizer.applyChatTemplate(messages: templateMessages) {
        return tokens
    }

    let text = templateMessages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }.joined(separator: "\n")
    return state.tokenizer.encode(text: text)
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
