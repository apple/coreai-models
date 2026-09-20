// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import Foundation
import Tokenizers

// Vision-language chat handling. The multimodal engine generates from image embeddings plus a
// token prompt rather than a plain token sequence, so it does not flow through DecodingStrategy.
// Output still runs through the same think-tag / tool-call parsing as the text path, so reasoning
// models return a clean answer. Shared by the HTTP handler and --replay via runChatCompletion.

/// Split raw model output into content / reasoning / tool calls, mirroring the text path.
/// Records tool calls on `state` as a side effect.
func parseAssistantOutput(
    fullText: String,
    genTokenCount: Int,
    requestMaxTokens: Int,
    chatRequest: ChatCompletionRequest,
    state: ServerState
) -> (content: String?, reasoning: String?, toolCalls: [ToolCall]?, finishReason: String) {
    var responseContent: String? = fullText
    var responseReasoningContent: String? = nil
    var responseToolCalls: [ToolCall]? = nil
    var finishReason = genTokenCount >= requestMaxTokens ? "length" : "stop"

    guard chatRequest.raw != true else {
        return (responseContent, responseReasoningContent, responseToolCalls, finishReason)
    }

    var thinkParser = ThinkTagParser(format: state.thinkingFormat)
    let events = thinkParser.consume(fullText) + thinkParser.flush()
    var textParts: [String] = []
    var reasoningParts: [String] = []
    for event in events {
        switch event {
        case .text(let t): textParts.append(t)
        case .reasoning(let r): reasoningParts.append(r)
        }
    }
    let cleaned = textParts.joined()
    responseContent = cleaned
    if !reasoningParts.isEmpty {
        responseReasoningContent = reasoningParts.joined()
    }

    if chatRequest.tools != nil, var parser = state.makeToolCallParser() {
        let toolEvents = parser.consume(cleaned) + parser.flush()
        var remainingParts: [String] = []
        var toolCalls: [ToolCall] = []
        for event in toolEvents {
            switch event {
            case .text(let t): remainingParts.append(t)
            case .toolCall(let id, let name, let argsJSON):
                toolCalls.append(ToolCall(id: id, function: .init(name: name, arguments: argsJSON)))
            }
        }
        if !toolCalls.isEmpty {
            responseToolCalls = toolCalls
            if finishReason != "length" { finishReason = "tool_calls" }
            let remaining = remainingParts.joined().trimmingCharacters(in: .whitespacesAndNewlines)
            responseContent = remaining.isEmpty ? nil : remaining
            state.recordToolCalls(toolCalls.map(\.function.name))
        }
    }

    return (responseContent, responseReasoningContent, responseToolCalls, finishReason)
}

/// The non-streaming VLM generation core, shared by the HTTP handler and --replay.
/// Encodes the request image, prefills through the embedding path, and assembles the response.
func runVLMCompletion(chatRequest: ChatCompletionRequest, state: ServerState) async throws -> ChatCompletionOutcome {
    guard let engine = state.multimodalEngine, let visionConfig = state.config.visionConfig else {
        throw ServerError.badRequest("this server is not serving a vision-language model")
    }
    guard let image = VLMChatSupport.lastImage(in: chatRequest.messages) else {
        throw ServerError.badRequest(
            "no decodable image found; provide an image_url content part with a base64 data URL or local path")
    }
    // Reasoning VLMs (e.g. muse) emit a long analysis channel before the final answer, so an
    // unspecified budget uses generous headroom to reach the final channel rather than truncating.
    let requestMaxTokens: Int
    if let explicit = chatRequest.maxCompletionTokens ?? chatRequest.maxTokens {
        requestMaxTokens = explicit
    } else {
        requestMaxTokens = max(state.config.defaultMaxTokens, 768)
    }
    guard requestMaxTokens > 0 else {
        throw ServerError.badRequest("max_tokens must be positive")
    }
    let requestID = RequestID.next()
    let created = Int(Date().timeIntervalSince1970)
    let samplingConfig = state.makeSamplingConfig(
        temperature: chatRequest.temperature, topP: chatRequest.topP, topK: chatRequest.topK, minP: nil,
        seed: chatRequest.seed)

    // Reset per request: image embeddings prefill from a clean state (no prefix reuse for VLM yet).
    try await engine.reset()
    let embeddedInput = try await engine.encodeImage(cgImage: image)
    let promptTokens = VLMChatSupport.buildPromptTokens(
        messages: chatRequest.messages,
        imageTokenCount: embeddedInput.tokenCount,
        imageTokenId: visionConfig.imageTokenId,
        tokenizer: state.tokenizer)
    guard promptTokens.count < state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt (\(promptTokens.count) tokens) exceeds context length (\(state.config.maxContextLength))")
    }

    var eosTokenIds = Set<Int32>()
    if let eos = state.tokenizer.eosTokenId { eosTokenIds.insert(Int32(eos)) }
    eosTokenIds.formUnion(state.config.additionalEosTokenIds)

    let t0 = SuspendingClock().now
    let stream = try await engine.generate(
        with: embeddedInput, tokens: promptTokens, samplingConfiguration: samplingConfig,
        inferenceOptions: InferenceOptions(maxTokens: requestMaxTokens))

    var generated: [Int] = []
    var previousText = ""
    var parts: [String] = []
    var promptSeconds: Double = 0
    for try await output in stream {
        if generated.isEmpty {
            let ttft = SuspendingClock().now - t0
            promptSeconds = Double(ttft.components.seconds) + Double(ttft.components.attoseconds) / 1e18
        }
        if eosTokenIds.contains(output.tokenId) { break }
        generated.append(Int(output.tokenId))
        let full = state.tokenizer.decode(tokens: generated)
        parts.append(String(full.dropFirst(previousText.count)))
        previousText = full
    }
    let text = parts.joined()
    let genTokenCount = generated.count

    let elapsed = SuspendingClock().now - t0
    let totalSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let genSeconds = totalSeconds - promptSeconds

    let parsed = parseAssistantOutput(
        fullText: text, genTokenCount: genTokenCount, requestMaxTokens: requestMaxTokens,
        chatRequest: chatRequest, state: state)

    let prefillTps = promptSeconds > 0 ? Double(promptTokens.count) / promptSeconds : 0
    let genTps = genSeconds > 0 ? Double(genTokenCount) / genSeconds : 0
    print(
        "\(ts()) [\(requestID)] vlm: \(promptTokens.count)t prefill \(String(format: "%.1f", prefillTps)) t/s, \(genTokenCount)t gen \(String(format: "%.1f", genTps)) t/s (\(String(format: "%.2f", totalSeconds))s) [\(parsed.finishReason)]"
    )
    state.stats.record(
        promptTokens: promptTokens.count, genTokens: genTokenCount, promptSeconds: promptSeconds,
        genSeconds: genSeconds, totalSeconds: totalSeconds, toolCalls: parsed.toolCalls?.count ?? 0)

    let response = ChatCompletionResponse(
        id: requestID, object: "chat.completion", created: created, model: state.config.modelName,
        choices: [
            .init(
                index: 0,
                message: .init(
                    role: "assistant", content: parsed.content, reasoningContent: parsed.reasoning,
                    toolCalls: parsed.toolCalls),
                finishReason: parsed.finishReason)
        ],
        usage: ChatCompletionResponse.Usage(
            promptTokens: promptTokens.count, completionTokens: genTokenCount,
            totalTokens: promptTokens.count + genTokenCount),
        systemFingerprint: state.systemFingerprint)

    return ChatCompletionOutcome(
        response: response,
        prefixReuseTokens: 0,
        ttftSeconds: promptSeconds,
        totalSeconds: totalSeconds,
        genTokenCount: genTokenCount,
        promptTokenCount: promptTokens.count)
}

private func ts() -> String {
    let now = Date()
    let c = Calendar.current
    return String(
        format: "%02d:%02d:%02d", c.component(.hour, from: now), c.component(.minute, from: now),
        c.component(.second, from: now))
}
