// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import CoreAIShared
import CoreGraphics
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NIOFoundationCompat
import Tokenizers

func startServer(state: ServerState, port: Int) async throws {
    let router = Router()

    router.get("/health") { _, _ in
        let data = try JSONEncoder().encode(HealthResponse(status: "ok", modalities: state.modalities))
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    router.get("/v1/models") { _, _ in
        let response = ModelsResponse(
            data: [
                ModelsResponse.ModelInfo(
                    id: state.config.modelName,
                    created: Int(Date().timeIntervalSince1970),
                    ownedBy: "coreai",
                    capabilities: ModelsResponse.ModelInfo.capabilities(vision: state.modalities.vision)
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
        // 503 while refusing for memory, so a load balancer or supervisor stops routing here.
        return Response(
            status: ready.deviceMemory?.refusing == nil ? .ok : .serviceUnavailable,
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

    // Process lifecycle is left to the supervising process. TODO: any in-process
    // idle-exit must first drain in-flight engine work (no drain hook today).
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
    let chatRequest: ChatCompletionRequest
    do {
        chatRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: body)
    } catch {
        let err = ErrorResponse(error: .init(message: "\(error)", type: "invalid_request_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .badRequest, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }

    let lane = BatchServingRouter.lane(
        hasResponseSchema: chatRequest.responseFormat?.extractedSchema != nil, hasImages: chatRequest.hasImageInput)
    let access: RequestAccess
    do {
        access = try await state.acquireAccess(lane: lane)
    } catch let refusal as MemoryRefusal {
        return try outOfMemoryResponse(refusal.reason)
    } catch let error as ServerError {
        let err = ErrorResponse(
            error: .init(
                message: error.localizedDescription, type: "server_error", code: "queue_full"))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .tooManyRequests, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }

    do {
        let shouldStream = chatRequest.stream ?? false
        if shouldStream {
            // Ownership of the access passes to the streaming body closure.
            return try await handleStreamingRequest(
                chatRequest: chatRequest, state: state, sessionID: sessionID, access: access)
        } else {
            let response = try await handleNonStreamingRequest(
                chatRequest: chatRequest, state: state, sessionID: sessionID)
            access.release()
            return response
        }
    } catch let error as InferenceRuntimeError where error.isQueueFull {
        // Thrown by a full batch scheduler on submit, on both the streaming and non-streaming paths.
        access.release()
        let err = ErrorResponse(
            error: .init(message: error.localizedDescription, type: "server_error", code: "queue_full"))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .tooManyRequests, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    } catch let error as ServerError {
        access.release()
        let status: HTTPResponse.Status = error.isBadRequest ? .badRequest : .internalServerError
        let err = ErrorResponse(
            error: .init(message: error.localizedDescription, type: "invalid_request_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: status, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data))
        )
    } catch {
        access.release()
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

/// Instrumented outcome of a non-streaming chat generation.
struct ChatCompletionOutcome {
    let response: ChatCompletionResponse
    let prefixReuseTokens: Int
    let ttftSeconds: Double
    let totalSeconds: Double
    let genTokenCount: Int
    let promptTokenCount: Int
}

/// The full non-streaming chat generation core, shared by the HTTP handler and `--replay`.
/// Returns the assembled response plus per-request instrumentation.
func runChatCompletion(chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil)
    async throws -> ChatCompletionOutcome
{
    let prepared = try await prepareStreaming(
        chatRequest: chatRequest, state: state, sessionID: sessionID, streaming: false)
    let requestID = prepared.requestID
    let requestMaxTokens = prepared.requestMaxTokens
    let promptTokens = prepared.promptTokensInt32
    let prefixReused = prepared.prefixReuseTokens
    let created = Int(Date().timeIntervalSince1970)
    let started = try await startStreamingDecode(prepared: prepared, chatRequest: chatRequest, state: state)
    let t0 = started.start
    let stream = started.tokens

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
    genTokenCount = reportedTokenCount(of: stream, results: genTokenCount)
    let text = parts.joined()

    let elapsed = SuspendingClock().now - t0
    let totalSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let genSeconds = totalSeconds - promptSeconds

    // Raw mode: return unprocessed model output (no thinking strip, no tool parsing)
    var responseContent: String? = text
    var responseReasoningContent: String? = nil
    var responseToolCalls: [ToolCall]? = nil
    var finishReason = try reportedFinishReason(of: stream, generated: genTokenCount, maxTokens: requestMaxTokens)

    if chatRequest.raw != true {
        // Separate reasoning from text using the streaming parser on the full output
        var thinkParser = ThinkTagParser(
            format: state.thinkingFormat, startsInReasoning: prepared.promptOpensReasoning)
        let events = thinkParser.consume(text) + thinkParser.flush()
        var textParts: [String] = []
        var reasoningParts: [String] = []
        for event in events {
            switch event {
            case .text(let t):
                textParts.append(t)
            case .reasoning(let r):
                reasoningParts.append(r)
            }
        }
        let cleaned = textParts.joined()
        responseContent = cleaned
        if !reasoningParts.isEmpty {
            responseReasoningContent = reasoningParts.joined()
        }

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
    logSummary(logLine)
    state.stats.record(
        promptTokens: promptTokens.count, genTokens: genTokenCount, promptSeconds: promptSeconds,
        genSeconds: genSeconds, totalSeconds: totalSeconds, toolCalls: responseToolCalls?.count ?? 0)
    state.recordPromptTokens(promptTokens)
    state.recordImageEmbeddings(prepared.images?.byKey ?? [:], sessionID: sessionID)

    let response = ChatCompletionResponse(
        id: requestID,
        object: "chat.completion",
        created: created,
        model: state.config.modelName,
        choices: [
            .init(
                index: 0,
                message: .init(
                    role: "assistant", content: responseContent,
                    reasoningContent: responseReasoningContent,
                    toolCalls: responseToolCalls
                ),
                finishReason: finishReason
            )
        ],
        usage: ChatCompletionResponse.Usage(
            promptTokens: promptTokens.count,
            completionTokens: genTokenCount,
            totalTokens: promptTokens.count + genTokenCount
        ),
        systemFingerprint: state.systemFingerprint
    )

    return ChatCompletionOutcome(
        response: response,
        prefixReuseTokens: prefixReused,
        ttftSeconds: promptSeconds,
        totalSeconds: totalSeconds,
        genTokenCount: genTokenCount,
        promptTokenCount: promptTokens.count
    )
}

private func handleNonStreamingRequest(chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil)
    async throws -> Response
{
    let outcome = try await runChatCompletion(chatRequest: chatRequest, state: state, sessionID: sessionID)
    let data = try JSONEncoder().encode(outcome.response)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: data))
    )
}

// MARK: - Streaming (SSE)

/// Inputs prepared for a chat generation. Split out so bad-request guards throw before
/// any SSE framing is emitted (and reused by `--replay`).
struct PreparedStream {
    let requestID: String
    let sessionID: String?
    let promptTokenCount: Int
    let promptTokensInt32: [Int32]
    let input: Input
    /// Nil for a request without images.
    let images: PromptImages?
    let samplingConfig: SamplingConfiguration
    let stopSequences: StopSequences
    let requestMaxTokens: Int
    let prefixReuseTokens: Int
    /// The chat template ended the prompt inside a reasoning block.
    let promptOpensReasoning: Bool
}

/// Instrumented outcome of a streaming chat generation.
struct StreamingOutcome {
    let prefixReuseTokens: Int
    let ttftSeconds: Double
    let totalSeconds: Double
    let genTokenCount: Int
    let promptTokenCount: Int
    let finishReason: String
    let systemFingerprint: String?
}

/// Preflight for a chat generation, streaming or not: tokenize, encode images, validate, and prime the
/// prefix cache. Throws `ServerError.badRequest` for oversized prompts, non-positive max_tokens,
/// out-of-range sampling values and unacceptable images so callers can map to HTTP 400 before
/// opening the stream.
func prepareStreaming(
    chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil, streaming: Bool = true
) async throws -> PreparedStream {
    let requestMaxTokens = chatRequest.maxCompletionTokens ?? chatRequest.maxTokens ?? state.config.defaultMaxTokens
    guard requestMaxTokens > 0 else {
        throw ServerError.badRequest("max_tokens must be positive")
    }
    let requestID = RequestID.next()

    let samplingConfig = try state.makeSamplingConfig(
        temperature: chatRequest.temperature,
        topP: chatRequest.topP,
        topK: chatRequest.topK,
        minP: nil,
        seed: chatRequest.seed
    )

    let reasoningEffort = ReasoningEffort.resolve(
        request: chatRequest.reasoningEffort, default: state.config.defaultReasoningEffort)
    let prompt = try await makeChatPrompt(
        chatRequest, reasoningEffort: reasoningEffort, state: state, sessionID: sessionID)
    let promptTokens = prompt.tokens
    let stopSequences = buildStopSequences(from: chatRequest, state: state)
    try requireSchemaSupport(chatRequest, state: state)
    let input: Input = .tokens(promptTokens)

    guard promptTokens.count < state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt (\(promptTokens.count) tokens) exceeds context length (\(state.config.maxContextLength))")
    }

    CLILogger.log(
        "[\(requestID)] \(streaming ? "stream, " : "")messages: \(chatRequest.messages.count), tokens: \(promptTokens.count), max_tokens: \(requestMaxTokens)",
        component: "Server")

    let promptTokensInt32 = promptTokens.map { Int32($0) }
    let prefixReused = await state.prepareForRequest(sessionID: sessionID, promptTokens: promptTokensInt32)
    if prefixReused > 0 {
        CLILogger.log("[\(requestID)] prefix reuse: \(prefixReused) tokens cached", component: "Server")
    }

    return PreparedStream(
        requestID: requestID, sessionID: sessionID, promptTokenCount: promptTokens.count,
        promptTokensInt32: promptTokensInt32, input: input, images: prompt.images, samplingConfig: samplingConfig,
        stopSequences: stopSequences,
        requestMaxTokens: requestMaxTokens, prefixReuseTokens: prefixReused,
        promptOpensReasoning: ThinkTagParser.promptOpensReasoning(
            promptTokens, tokenizer: state.tokenizer, format: state.thinkingFormat))
}

/// Token stream and start time of a streaming decode. On the batched lane the request is already in
/// the batch scheduler. Iterate `tokens` from one task only.
struct StartedDecode: @unchecked Sendable {
    let tokens: any AsyncSequence<GenerationResult, Error>
    let start: SuspendingClock.Instant
}

/// Starts decoding `prepared` on the engine for the request's lane, with constrained decoding when
/// `chatRequest` has a JSON schema. A prompt with images decodes with their embeddings on the
/// multimodal engine (the `.fallbackBatchOne` lane), which reuses the prefix in front of the first
/// image. Throws `InferenceRuntimeError.queueFull` when the batch scheduler is full.
func startStreamingDecode(
    prepared: PreparedStream, chatRequest: ChatCompletionRequest, state: ServerState
) async throws -> StartedDecode {
    let start = SuspendingClock().now
    if let images = prepared.images, let vision = state.vision {
        let tokens = try await VanillaDecodingStrategy().decode(
            embeddings: images.joined, tokens: prepared.promptTokensInt32, tokenizer: state.tokenizer,
            inferenceEngine: vision.engine, samplingConfiguration: prepared.samplingConfig,
            options: InferenceOptions(maxTokens: prepared.requestMaxTokens), stopSequences: prepared.stopSequences)
        return StartedDecode(tokens: tokens, start: start)
    }

    let strategy: any DecodingStrategy
    let schema = chatRequest.responseFormat?.extractedSchema
    if let schema {
        CLILogger.log("[\(prepared.requestID)] constrained generation (json_schema)", component: "Server")
        strategy = jsonSchemaStrategy(schema, state: state)
    } else {
        strategy = VanillaDecodingStrategy()
    }

    let lane = BatchServingRouter.lane(hasResponseSchema: schema != nil, hasImages: chatRequest.hasImageInput)
    let tokens = try await strategy.decode(
        from: prepared.input,
        tokenizer: state.tokenizer,
        inferenceEngine: state.engineForRequest(lane: lane),
        samplingConfiguration: prepared.samplingConfig,
        options: InferenceOptions(maxTokens: prepared.requestMaxTokens),
        stopSequences: prepared.stopSequences
    )
    return StartedDecode(tokens: tokens, start: start)
}

/// Iterates `started.tokens`, splits reasoning and tool calls out of the text unless the request is
/// raw, and passes each `Delta` to `emit`. Records the request in `state.stats` and returns its
/// timing, token counts and finish reason. Emits no role chunk, finish chunk or `[DONE]` frame.
func runStreamingLoop(
    started: StartedDecode, prepared: PreparedStream, chatRequest: ChatCompletionRequest,
    state: ServerState, emit: (ChatCompletionChunk.Delta) async throws -> Void
) async throws -> StreamingOutcome {
    let genStart = started.start
    let tokenStream = started.tokens

    var thinkParser = ThinkTagParser(
        format: state.thinkingFormat, startsInReasoning: prepared.promptOpensReasoning)
    var toolParser = state.makeToolCallParser()
    var tokenCount = 0
    var hasToolCalls = false
    var toolCallIndex = 0
    var toolCallNames: [String] = []
    var promptSeconds: Double = 0
    let isRaw = chatRequest.raw == true

    // Process tool parser events: emit text as content, tool calls as deltas
    func emitToolEvents(_ events: [ToolCallParser.Event]) async throws {
        for event in events {
            switch event {
            case .text(let t) where !t.isEmpty:
                try await emit(.init(role: nil, content: t))
            case .toolCall(let id, let name, let argsJSON):
                hasToolCalls = true
                toolCallNames.append(name)
                let tcDelta = ToolCallDelta(
                    index: toolCallIndex, id: id, type: "function",
                    function: .init(name: name, arguments: argsJSON))
                try await emit(.init(role: nil, content: nil, toolCalls: [tcDelta]))
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
            try await emit(.init(role: nil, content: text))
        }
    }

    for try await result in tokenStream {
        if tokenCount == 0 {
            let ttft = SuspendingClock().now - genStart
            promptSeconds = Double(ttft.components.seconds) + Double(ttft.components.attoseconds) / 1e18
        }
        tokenCount += 1
        if isRaw {
            try await emit(.init(role: nil, content: result.text))
        } else {
            for event in thinkParser.consume(result.text) {
                switch event {
                case .text(let delta):
                    try await emitText(delta)
                case .reasoning(let delta) where !delta.isEmpty:
                    try await emit(.init(role: nil, content: nil, reasoningContent: delta))
                default:
                    break
                }
            }
        }
    }

    if !isRaw {
        for event in thinkParser.flush() {
            switch event {
            case .text(let delta):
                try await emitText(delta)
            case .reasoning(let delta) where !delta.isEmpty:
                try await emit(.init(role: nil, content: nil, reasoningContent: delta))
            default:
                break
            }
        }

        if var tp = toolParser {
            try await emitToolEvents(tp.flush())
            toolParser = tp
        }
    }

    tokenCount = reportedTokenCount(of: tokenStream, results: tokenCount)
    let ended = try reportedFinishReason(of: tokenStream, generated: tokenCount, maxTokens: prepared.requestMaxTokens)
    let finishReason = hasToolCalls && ended == "stop" ? "tool_calls" : ended

    let elapsed = SuspendingClock().now - genStart
    let totalSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let genSeconds = totalSeconds - promptSeconds
    let prefillTps = promptSeconds > 0 ? Double(prepared.promptTokenCount) / promptSeconds : 0
    let genTps = genSeconds > 0 ? Double(tokenCount) / genSeconds : 0
    var logLine =
        "\(ts()) [\(prepared.requestID)] stream: \(prepared.promptTokenCount)t prefill \(String(format: "%.1f", prefillTps)) t/s, \(tokenCount)t gen \(String(format: "%.1f", genTps)) t/s (\(String(format: "%.2f", totalSeconds))s) [\(finishReason)]"
    if !toolCallNames.isEmpty {
        logLine += " → \(toolCallNames.count) tool call(s)"
        if CLILogger.level > 0 {
            logLine += ": \(toolCallNames.joined(separator: ", "))"
        }
    }
    logSummary(logLine)
    state.stats.record(
        promptTokens: prepared.promptTokenCount, genTokens: tokenCount, promptSeconds: promptSeconds,
        genSeconds: genSeconds, totalSeconds: totalSeconds, toolCalls: toolCallNames.count)
    state.recordPromptTokens(prepared.promptTokensInt32)
    state.recordImageEmbeddings(prepared.images?.byKey ?? [:], sessionID: prepared.sessionID)
    if !toolCallNames.isEmpty {
        state.recordToolCalls(toolCallNames)
    }

    return StreamingOutcome(
        prefixReuseTokens: prepared.prefixReuseTokens,
        ttftSeconds: promptSeconds,
        totalSeconds: totalSeconds,
        genTokenCount: tokenCount,
        promptTokenCount: prepared.promptTokenCount,
        finishReason: finishReason,
        systemFingerprint: state.systemFingerprint
    )
}

/// Starts a streaming chat completion and returns its SSE response, which releases `access` when
/// the stream ends. On a throw the caller still owns `access`. Throws `ServerError.badRequest` for
/// a bad request and `InferenceRuntimeError.queueFull` when the batch scheduler is full.
private func handleStreamingRequest(
    chatRequest: ChatCompletionRequest, state: ServerState, sessionID: String? = nil, access: RequestAccess
)
    async throws -> Response
{
    let prepared = try await prepareStreaming(chatRequest: chatRequest, state: state, sessionID: sessionID)
    let started = try await startStreamingDecode(prepared: prepared, chatRequest: chatRequest, state: state)
    let created = Int(Date().timeIntervalSince1970)

    let responseBody = ResponseBody { writer in
        // If this closure never runs, the permit and lease release on deinit.
        defer { access.release() }
        do {
            let encoder = JSONEncoder()

            func writeDelta(_ delta: ChatCompletionChunk.Delta, finishReason: String? = nil) async throws {
                let chunk = ChatCompletionChunk(
                    id: prepared.requestID, object: "chat.completion.chunk", created: created,
                    model: state.config.modelName,
                    choices: [.init(index: 0, delta: delta, finishReason: finishReason)],
                    systemFingerprint: state.systemFingerprint)
                if let data = try? encoder.encode(chunk), let json = String(data: data, encoding: .utf8) {
                    try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
                }
            }

            try await writeDelta(.init(role: "assistant", content: nil))

            let outcome = try await runStreamingLoop(
                started: started, prepared: prepared, chatRequest: chatRequest, state: state,
                emit: { try await writeDelta($0) })

            try await writeDelta(.init(role: nil, content: nil), finishReason: outcome.finishReason)
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        } catch {
            print("\(ts()) [\(prepared.requestID)] stream error: \(error)")
            try? await writer.write(ByteBuffer(string: streamErrorFrame(error)))
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

/// A chat request's prompt tokens and the images in them.
struct ChatPrompt {
    let tokens: [Int]
    /// Nil for a request without images.
    let images: PromptImages?
}

/// The embeddings of a prompt's images.
struct PromptImages {
    /// The embeddings of every image placeholder in the prompt, in order.
    let joined: InputEmbeddings
    /// The embeddings of each image of the prompt.
    let byKey: [ImageKey: InputEmbeddings]
}

/// Renders `chatRequest` through the chat template. For a request with images, encodes each image
/// that `sessionID` has not cached and expands its placeholder to the image's tokens. Throws
/// `ServerError.badRequest` for images the model does not accept or the server cannot read.
func makeChatPrompt(
    _ chatRequest: ChatCompletionRequest, reasoningEffort: String?, state: ServerState, sessionID: String?
) async throws -> ChatPrompt {
    func render(_ images: ImagePartRendering) -> [Int] {
        tokenizeMessages(
            chatRequest.messages, tools: chatRequest.tools, reasoningEffort: reasoningEffort,
            tokenizer: state.tokenizer, toolCallDetection: state.toolCallDetection, images: images)
    }
    if let vision = state.vision {
        let placeholder = ImagePlaceholders.text(imageTokenId: vision.config.imageTokenId, tokenizer: state.tokenizer)
        guard !chatRequest.messages.contains(where: { $0.content.textContent.contains(placeholder) }) else {
            throw ServerError.badRequest("message text must not contain the image token \(placeholder)")
        }
    }
    guard chatRequest.hasImageInput else { return ChatPrompt(tokens: render(.omitted), images: nil) }
    guard let vision = state.vision else {
        throw ServerError.badRequest("model \(state.config.modelName) does not accept image input")
    }
    guard chatRequest.responseFormat?.extractedSchema == nil else {
        throw ServerError.badRequest("response_format json_schema is not supported with image input")
    }
    let imageTokenId = vision.config.imageTokenId
    let placeholder = ImagePlaceholders.text(imageTokenId: imageTokenId, tokenizer: state.tokenizer)

    let images = try ImageInput.images(in: chatRequest.messages)
    var embeddings = state.cachedImageEmbeddings(sessionID: sessionID).filter { key, _ in
        images.contains { $0.key == key }
    }
    for image in images where embeddings[image.key] == nil {
        embeddings[image.key] = try await vision.engine.encodeImage(cgImage: image.image)
    }
    let ordered = images.compactMap { embeddings[$0.key] }

    let infos = images.map {
        vision.config.includeImageInfo
            ? ImagePlaceholders.imageInfoText(width: $0.image.width, height: $0.image.height) : ""
    }
    // The template's own image parts first, then the image token in the message text.
    let renderings: [ImagePartRendering] = [.parts(infos: infos), .text(infos.map { placeholder + "\n" + $0 })]
    let expanded = renderings.lazy.compactMap {
        ImagePlaceholders.expand(
            render($0).map { Int32($0) }, imageTokenId: imageTokenId, tokenCounts: ordered.map(\.tokenCount))
    }.first
    guard let expanded else {
        throw InferenceRuntimeError.invalidState("The chat template did not place one \(placeholder) per image")
    }
    return ChatPrompt(
        tokens: expanded.map { Int($0) },
        images: PromptImages(joined: try InputEmbeddings.concatenating(ordered), byKey: embeddings))
}

/// How `tokenizeMessages` passes image content parts to the chat template.
enum ImagePartRendering {
    /// Image parts are left out.
    case omitted
    /// Every message's content is a list of `{"type": "text", "text": ...}` and `{"type": "image"}`
    /// parts, as vision-language chat templates take it, with `infos[i]` as a text part after the
    /// i-th image when not empty.
    case parts(infos: [String])
    /// The i-th image part is replaced by `texts[i]` in the message text.
    case text([String])
}

/// A chat request's messages and tools rendered through the model's chat template, as tokens, with
/// image parts passed as `images` says.
///
/// Internal, not private, so tests can render a request through a real template.
func tokenizeMessages(
    _ messages: [ChatMessage], tools: [ToolDefinition]? = nil,
    reasoningEffort: String? = nil, tokenizer: any Tokenizer, toolCallDetection: ToolCallDetection?,
    images: ImagePartRendering = .omitted
) -> [Int] {
    var imageIndex = 0
    /// The template value of `content` followed by `suffix`.
    func render(_ content: MessageContent, appending suffix: String = "") -> any Sendable {
        guard case .parts(let parts) = content else {
            if case .parts = images { return [["type": "text", "text": content.textContent + suffix]] }
            return content.textContent + suffix
        }
        switch images {
        case .omitted:
            return content.textContent + suffix
        case .parts(let infos):
            var rendered: [[String: any Sendable]] = []
            for part in parts {
                guard case .text(let text) = part else {
                    rendered.append(["type": "image"])
                    if imageIndex < infos.count, !infos[imageIndex].isEmpty {
                        rendered.append(["type": "text", "text": infos[imageIndex]])
                    }
                    imageIndex += 1
                    continue
                }
                rendered.append(["type": "text", "text": text])
            }
            return suffix.isEmpty ? rendered : rendered + [["type": "text", "text": suffix]]
        case .text(let texts):
            var text = ""
            var previousIsText = false
            for part in parts {
                if case .text(let t) = part {
                    text += (previousIsText ? " " : "") + t
                    previousIsText = true
                } else if imageIndex < texts.count {
                    text += (previousIsText ? "\n" : "") + texts[imageIndex]
                    imageIndex += 1
                    previousIsText = false
                }
            }
            return text + suffix
        }
    }

    var templateMessages: [[String: any Sendable]] = []
    // Resolved effort is the single source of truth: `none` also injects the legacy `/no_think`
    // literal for models (e.g. Qwen3) that honor it in the system prompt.
    let noThink = ReasoningEffort.disablesThinking(reasoningEffort)
    for msg in messages {
        var dict: [String: any Sendable] = ["role": msg.role]

        if msg.role == "tool" {
            dict["content"] = render(msg.content)
            if let id = msg.toolCallId { dict["tool_call_id"] = id }
        } else if msg.role == "assistant", let calls = msg.toolCalls, !calls.isEmpty {
            dict["tool_calls"] = calls.map { call in
                TemplateValues.toolCall(id: call.id, name: call.function.name, arguments: call.function.arguments)
            }
            dict["content"] = render(msg.content)
        } else {
            dict["content"] = render(msg.content, appending: msg.role == "system" && noThink ? "\n/no_think" : "")
        }
        templateMessages.append(dict)
    }

    if noThink && !messages.contains(where: { $0.role == "system" }) {
        templateMessages.insert(["role": "system", "content": "/no_think"], at: 0)
    }

    let toolSpecs: [[String: any Sendable]]? = tools?.map { tool in
        var funcDict: [String: any Sendable] = [
            "name": tool.function.name,
            "description": tool.function.description ?? "",
        ]
        if let params = tool.function.parameters {
            funcDict["parameters"] = params.templateValue
        }
        let spec: [String: any Sendable] = [
            "type": tool.type,
            "function": funcDict,
        ]
        return spec
    }

    // Phi-family templates read tools from a system message's `tools` key, not the top-level
    // `tools` variable. Attach the specs there (and synthesize a system message if none exists)
    // only for that dialect, so top-level-`tools` families (e.g. Qwen3) are not perturbed.
    templateMessages = injectToolsIntoSystemMessageIfNeeded(
        templateMessages, toolSpecs: toolSpecs, detection: toolCallDetection)

    do {
        let additionalContext = ReasoningEffort.templateContext(reasoningEffort)
        let tokens = try tokenizer.applyChatTemplate(
            messages: templateMessages, tools: toolSpecs,
            additionalContext: additionalContext.isEmpty ? nil : additionalContext)
        return tokens
    } catch {
        // `makeChatPrompt` renders the request again with the image token in the text.
        if case .parts = images { return [] }
        // Shown without --verbose, once per kind of failure: the plain-text prompt below is not what
        // the model was trained on.
        CLILogger.warnOnce(
            "chat template: \(error)",
            "Warning: the chat template failed (\(error)); using a plain-text prompt instead",
            component: "Server")
    }

    let text = templateMessages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
        .joined(separator: "\n")
    return tokenizer.encode(text: text)
}

/// 503 `out_of_memory` for a request refused because the process is past the device's memory
/// recommendation, rather than a server fault. `Retry-After` because the refusal clears once the
/// running requests finish and the process is back under the line.
func outOfMemoryResponse(_ reason: String) throws -> Response {
    let err = ErrorResponse(
        error: .init(
            message: reason + "; try again once running requests finish", type: "server_error",
            code: "out_of_memory"))
    let data = try JSONEncoder().encode(err)
    return Response(
        status: .serviceUnavailable,
        headers: [.contentType: "application/json", .retryAfter: outOfMemoryRetryAfterSeconds],
        body: .init(byteBuffer: ByteBuffer(data: data)))
}

/// Seconds a client refused for memory is told to wait, long enough for a running request to finish
/// a few hundred tokens.
let outOfMemoryRetryAfterSeconds = "5"

/// `usage.completion_tokens` for a stream that has ended, given the `results` it yielded.
///
/// The stream's own count when it reports one: it includes the tokens that added no text and so
/// yielded no result, and the token that ended the reply. A stream that doesn't is counted a result
/// at a time.
func reportedTokenCount(of stream: any AsyncSequence<GenerationResult, Error>, results: Int) -> Int {
    guard let count = (stream as? any DecodeOutcomeReporting)?.tokensGenerated, count > 0 else { return results }
    return count
}

/// `finish_reason` for a stream that has ended.
///
/// Taken from the stream's stop reason when it reports one, so a reply that ended on a stop sequence
/// or the end of sequence says `stop` and one cut off by the limit says `length`, whatever the count.
/// A stream that ended on an error or was cancelled without throwing throws here, rather than
/// passing a truncated reply off as finished. A stream that reports nothing falls back to the count.
func reportedFinishReason(
    of stream: any AsyncSequence<GenerationResult, Error>, generated: Int, maxTokens: Int
) throws -> String {
    switch (stream as? any DecodeOutcomeReporting)?.stopReason {
    case .maxTokens:
        return "length"
    case .eos, .stopSequence:
        return "stop"
    case .error:
        throw InferenceRuntimeError.invalidState("generation ended on an error after \(generated) token(s)")
    case .cancelled:
        throw InferenceRuntimeError.invalidState("generation was cancelled after \(generated) token(s)")
    case nil:
        return generated >= maxTokens ? "length" : "stop"
    }
}

/// The strategy for a `json_schema` request. An engine that masks the logits on the GPU itself (the
/// pipelined engine) gets its own path; any other engine returns each step's logits for the strategy
/// to mask.
func jsonSchemaStrategy(_ schema: String, state: ServerState) -> any DecodingStrategy {
    if state.engineForRequest(lane: .fallbackBatchOne) is any ConstrainedGenerationCapable {
        return PipelinedConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: state.config.vocabSize)
    }
    return ConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: state.config.vocabSize)
}

/// Throws a 400 for a `json_schema` the engine can't serve: one that neither returns logits nor masks
/// them on the GPU. Checked before any generation, so a streaming request is refused before its
/// response starts rather than after.
func requireSchemaSupport(_ chatRequest: ChatCompletionRequest, state: ServerState) throws {
    guard chatRequest.responseFormat?.extractedSchema != nil,
        !state.engineForRequest(lane: .fallbackBatchOne).supportsGuidedGeneration
    else { return }
    throw ServerError.badRequest(
        "response_format json_schema needs an engine that returns logits or masks them on the GPU, and this "
            + "server's does neither; serve the model with --variant coreai-sequential for guided generation")
}

/// An SSE frame carrying `error`, for a stream that fails after its response has started, so the
/// client sees the failure instead of a reply that just stops.
func streamErrorFrame(_ error: any Error) -> String {
    let err = ErrorResponse(error: .init(message: "\(error)", type: "server_error", code: nil))
    let json = (try? JSONEncoder().encode(err)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    return "data: \(json)\n\n"
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

// Per-request diagnostics go to stderr so a `--replay` stdout redirect captures clean JSONL.
private func logSummary(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

private func ts() -> String {
    let now = Date()
    let calendar = Calendar.current
    let h = calendar.component(.hour, from: now)
    let m = calendar.component(.minute, from: now)
    let s = calendar.component(.second, from: now)
    return String(format: "%02d:%02d:%02d", h, m, s)
}
