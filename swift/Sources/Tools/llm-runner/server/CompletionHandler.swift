// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Hummingbird
import NIOCore
import NIOFoundationCompat
import Tokenizers

// MARK: - Completions Request (Legacy /v1/completions)

struct CompletionRequest: Decodable, Sendable {
    let model: String?
    let prompt: [String]
    let maxTokens: Int?
    let temperature: Double?
    let echo: Bool?
    let logprobs: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, echo, logprobs
        case maxTokens = "max_tokens"
    }

    init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        echo = try container.decodeIfPresent(Bool.self, forKey: .echo)
        logprobs = try container.decodeIfPresent(Int.self, forKey: .logprobs)

        // prompt can be: string, [string], [int] (token IDs), [[int]] (batched token IDs)
        if let s = try? container.decode(String.self, forKey: .prompt) {
            prompt = [s]
        } else if let arr = try? container.decode([String].self, forKey: .prompt) {
            prompt = arr
        } else if let tokenIds = try? container.decode([Int].self, forKey: .prompt) {
            prompt = ["__TOKEN_IDS__:\(tokenIds.map(String.init).joined(separator: ","))"]
        } else if let batchedIds = try? container.decode([[Int]].self, forKey: .prompt) {
            prompt = batchedIds.map { "__TOKEN_IDS__:\($0.map(String.init).joined(separator: ","))" }
        } else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: [CodingKeys.prompt],
                    debugDescription: "Expected string, [string], [int], or [[int]] for 'prompt'"
                ))
        }
    }
}

// MARK: - Completions Response

struct CompletionResponse: Encodable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]

    struct CompletionChoice: Encodable, Sendable {
        let index: Int
        let text: String
        let logprobs: LogprobsResult?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, text, logprobs
            case finishReason = "finish_reason"
        }
    }

    struct LogprobsResult: Encodable, Sendable {
        let tokens: [String]
        let tokenLogprobs: [Double?]
        let topLogprobs: [[String: Double]?]
        let textOffset: [Int]

        enum CodingKeys: String, CodingKey {
            case tokens
            case tokenLogprobs = "token_logprobs"
            case topLogprobs = "top_logprobs"
            case textOffset = "text_offset"
        }
    }
}

// MARK: - Completions Handler

func handleCompletionsRoute(request: Request, state: ServerState) async throws -> Response {
    let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
    return try await handleCompletionsFromBody(body: body, state: state)
}

// MARK: - Loglikelihood Implementation

func handleLoglikelihood(req: CompletionRequest, state: ServerState) async throws -> Response {
    let requestID = RequestID.next()
    let created = Int(Date().timeIntervalSince1970)
    let wantsEcho = req.echo ?? false

    guard req.prompt.count <= 1024 else {
        throw ServerError.badRequest("prompt batch too large (\(req.prompt.count)); maximum 1024")
    }
    let topN = req.logprobs ?? 1

    let t0 = ContinuousClock.now
    var choices: [CompletionResponse.CompletionChoice] = []

    for (idx, promptText) in req.prompt.enumerated() {
        let choice = try await processOnePrompt(
            promptText: promptText, index: idx,
            wantsEcho: wantsEcho, topN: topN, state: state
        )
        choices.append(choice)
    }

    let elapsed = ContinuousClock.now - t0
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let totalTokens = req.prompt.reduce(0) { acc, p in
        if p.hasPrefix("__TOKEN_IDS__:") {
            return acc + p.dropFirst("__TOKEN_IDS__:".count).split(separator: ",").count
        }
        return acc + state.tokenizer.encode(text: p).count
    }
    let tokPerSec = seconds > 0 ? Double(totalTokens) / seconds : 0
    print(
        "[\(requestID)] logprobs: \(req.prompt.count) prompts, \(totalTokens) tokens, \(String(format: "%.2f", seconds))s (\(String(format: "%.0f", tokPerSec)) tok/s)"
    )

    let response = CompletionResponse(
        id: requestID, object: "text_completion", created: created,
        model: state.config.modelName, choices: choices
    )

    let data = try JSONEncoder().encode(response)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: data))
    )
}

private func processOnePrompt(
    promptText: String, index: Int,
    wantsEcho: Bool, topN: Int,
    state: ServerState
) async throws -> CompletionResponse.CompletionChoice {
    let allTokens: [Int32]
    if promptText.hasPrefix("__TOKEN_IDS__:") {
        let idsStr = String(promptText.dropFirst("__TOKEN_IDS__:".count))
        allTokens = try idsStr.split(separator: ",").map { sub in
            guard let id = Int32(sub) else {
                throw ServerError.badRequest("Invalid token ID: \(sub)")
            }
            return id
        }
    } else {
        allTokens = state.tokenizer.encode(text: promptText).map { Int32($0) }
    }

    guard allTokens.count >= 2 else {
        return .init(index: index, text: "", logprobs: nil, finishReason: "stop")
    }

    guard allTokens.count <= state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt length \(allTokens.count) exceeds max context \(state.config.maxContextLength)")
    }

    // lm-eval sends echo=true, max_tokens=1 and slices [ctxlen:-1].
    // The :-1 discards the last entry (the "generated" token).
    // We append one padding token so :-1 drops it instead of real data.
    let paddingToken = allTokens[0]
    let continuation = Array(allTokens.dropFirst()) + [paddingToken]

    let options = InferenceOptions(
        maxTokens: continuation.count,
        includeLogits: true,
        forcedContinuation: continuation
    )

    try await state.engine.reset()

    let stream = try await state.engine.generate(
        with: [allTokens[0]],
        samplingConfiguration: .greedy,
        inferenceOptions: options
    )

    var allLogits: [[LogitsScalarType]] = []
    for try await output in stream {
        if let logits = output.logits {
            allLogits.append(logits)
        }
    }

    // Compute per-token logprobs (including padding token at end)
    let probs = LogProbabilities.compute(logits: allLogits, targets: continuation, topK: topN)

    var tokens: [String] = []
    var tokenLogprobs: [Double?] = []
    var topLogprobsList: [[String: Double]?] = []
    var textOffsets: [Int] = []
    var currentOffset = 0

    // First token has no logprob (prompt start)
    let firstTokenStr = state.tokenizer.decode(tokens: [Int(allTokens[0])])
    tokens.append(firstTokenStr)
    tokenLogprobs.append(nil)
    topLogprobsList.append(nil)
    textOffsets.append(0)
    currentOffset += firstTokenStr.utf8.count

    for entry in probs.entries {
        let tokenStr = state.tokenizer.decode(tokens: [Int(entry.tokenId)])
        tokens.append(tokenStr)
        tokenLogprobs.append(entry.value)
        textOffsets.append(currentOffset)
        currentOffset += tokenStr.utf8.count

        if topN > 0 {
            var topMap: [String: Double] = [:]
            for alt in entry.alternatives {
                topMap[state.tokenizer.decode(tokens: [Int(alt.tokenId)])] = alt.value
            }
            topLogprobsList.append(topMap)
        } else {
            topLogprobsList.append(nil)
        }
    }

    let logprobsResult = CompletionResponse.LogprobsResult(
        tokens: tokens,
        tokenLogprobs: tokenLogprobs,
        topLogprobs: topLogprobsList,
        textOffset: textOffsets
    )

    let responseText: String
    if wantsEcho {
        responseText = state.tokenizer.decode(tokens: allTokens.map { Int($0) })
    } else {
        responseText = ""
    }

    return .init(
        index: index,
        text: responseText,
        logprobs: logprobsResult,
        finishReason: "stop"
    )
}
