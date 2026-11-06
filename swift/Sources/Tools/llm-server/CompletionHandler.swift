// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Hummingbird
import NIOCore
import NIOFoundationCompat
import Tokenizers

// MARK: - Route Entry

func handleCompletionsRoute(request: Request, state: ServerState) async throws -> Response {
    let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
    return try await handleCompletionsFromBody(body: body, state: state)
}

func handleCompletionsFromBody(body: ByteBuffer, state: ServerState) async throws -> Response {
    guard state.config.supportsLogprobs else {
        let err = ErrorResponse(
            error: .init(
                message: logprobsUnsupportedMessage, type: "server_error", code: "unsupported"))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .notImplemented, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
    let access: RequestAccess
    do {
        access = try await state.acquireScoringAccess()
    } catch let error as ServerError {
        let err = ErrorResponse(
            error: .init(
                message: error.localizedDescription, type: "server_error", code: "queue_full"))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .tooManyRequests, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
    defer { access.release() }
    let req: CompletionRequest
    do {
        req = try JSONDecoder().decode(CompletionRequest.self, from: body)
    } catch {
        print("[SERVER] Completions decode error: \(error)")
        let err = ErrorResponse(error: .init(message: "\(error)", type: "invalid_request_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .badRequest, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
    do {
        return try await handleLoglikelihood(req: req, state: state)
    } catch {
        print("[SERVER] Completions error: \(error)")
        let err = ErrorResponse(error: .init(message: "\(error)", type: "server_error", code: nil))
        let data = try JSONEncoder().encode(err)
        return Response(
            status: .internalServerError, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data)))
    }
}

// MARK: - Loglikelihood Implementation

let logprobsUnsupportedMessage = "Logprobs not supported. Use --variant coreai-sequential"

private func handleLoglikelihood(req: CompletionRequest, state: ServerState) async throws -> Response {
    let outcome = try await runLoglikelihood(req: req, state: state)
    let data = try JSONEncoder().encode(outcome.response)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(data: data))
    )
}

/// A scoring response with its token count and timing.
struct LoglikelihoodOutcome {
    let response: CompletionResponse
    /// Tokens over all prompts.
    let promptTokenCount: Int
    let totalSeconds: Double
}

/// Scores the prompts in `req`, for `/v1/completions` and `--replay`. Call it while holding the
/// scoring access, on an engine with `supportsLogprobs`.
func runLoglikelihood(req: CompletionRequest, state: ServerState) async throws -> LoglikelihoodOutcome {
    let requestID = RequestID.next()
    let created = Int(Date().timeIntervalSince1970)
    let wantsEcho = req.echo ?? false

    guard req.prompts.count <= 1024 else {
        throw ServerError.badRequest("prompt batch too large (\(req.prompts.count)); maximum 1024")
    }
    let topN = min(req.logprobs ?? 1, 20)

    let t0 = SuspendingClock().now
    var choices: [CompletionResponse.CompletionChoice] = []

    let prepared = try req.prompts.map { try preparePrompt($0, state: state) }
    let scorable = prepared.compactMap { $0 }
    var scores: [LogProbabilities] = []
    if let scheduler = state.scoringScheduler {
        scores = try await scheduler.score(
            scorable.map { ScoreRequest(tokens: $0.tokens, targets: $0.targets, topK: topN) })
    } else {
        for prompt in scorable {
            scores.append(try await scoreOnEngine(prompt, topN: topN, state: state))
        }
    }

    var nextScore = scores.makeIterator()
    for (index, prompt) in prepared.enumerated() {
        guard let prompt, let probs = nextScore.next() else {
            choices.append(.init(index: index, text: "", logprobs: nil, finishReason: "stop"))
            continue
        }
        choices.append(
            makeChoice(prompt, probs: probs, index: index, wantsEcho: wantsEcho, topN: topN, state: state))
    }

    let elapsed = SuspendingClock().now - t0
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let totalTokens = req.prompts.reduce(0) { acc, p in
        switch p {
        case .tokenIds(let ids): return acc + ids.count
        case .text(let s): return acc + state.tokenizer.encode(text: s).count
        }
    }
    let tokPerSec = seconds > 0 ? Double(totalTokens) / seconds : 0
    print(
        "[\(requestID)] logprobs: \(req.prompts.count) prompts, \(totalTokens) tokens, \(String(format: "%.2f", seconds))s (\(String(format: "%.0f", tokPerSec)) tok/s)"
    )

    let response = CompletionResponse(
        id: requestID, object: "text_completion", created: created,
        model: state.config.modelName, choices: choices
    )
    return LoglikelihoodOutcome(response: response, promptTokenCount: totalTokens, totalSeconds: seconds)
}

/// Tokenized prompt to score, with the target token each position predicts.
private struct PreparedPrompt {
    let tokens: [Int32]
    /// `tokens.dropFirst()` followed by `tokens[0]` as padding. Same count as `tokens`.
    let targets: [Int32]
}

/// Tokenizes `prompt` into tokens and targets. Returns nil when it has fewer than two tokens.
/// Throws `ServerError.badRequest` when it has more than `state.config.maxContextLength` tokens.
private func preparePrompt(_ prompt: Prompt, state: ServerState) throws -> PreparedPrompt? {
    let allTokens: [Int32]
    switch prompt {
    case .tokenIds(let ids):
        allTokens = ids
    case .text(let text):
        allTokens = state.tokenizer.encode(text: text).map { Int32($0) }
    }

    guard allTokens.count >= 2 else { return nil }

    guard allTokens.count <= state.config.maxContextLength else {
        throw ServerError.badRequest(
            "Prompt length \(allTokens.count) exceeds max context \(state.config.maxContextLength)")
    }
    return PreparedPrompt(tokens: allTokens, targets: Array(allTokens.dropFirst()) + [allTokens[0]])
}

/// Scores `prompt` on the `.fallbackBatchOne` engine after a reset, by feeding `tokens[0]` and
/// forcing `targets`. Reports the top `topN` alternatives per position.
private func scoreOnEngine(_ prompt: PreparedPrompt, topN: Int, state: ServerState) async throws
    -> LogProbabilities
{
    let options = InferenceOptions.eval(
        maxTokens: prompt.targets.count,
        forcedContinuation: prompt.targets
    )

    let scoringEngine = state.engineForRequest(lane: .fallbackBatchOne)
    try await scoringEngine.reset()

    let stream = try await scoringEngine.generate(
        with: [prompt.tokens[0]],
        samplingConfiguration: .greedy,
        inferenceOptions: options
    )

    var allLogits: [[LogitsScalarType]] = []
    for try await output in stream {
        if let logits = output.logits {
            allLogits.append(logits)
        }
    }
    return LogProbabilities.compute(logits: allLogits, targets: prompt.targets, topK: topN)
}

private func makeChoice(
    _ prompt: PreparedPrompt, probs: LogProbabilities, index: Int, wantsEcho: Bool, topN: Int,
    state: ServerState
) -> CompletionResponse.CompletionChoice {
    let allTokens = prompt.tokens

    var tokens: [String] = []
    var tokenLogprobs: [Double?] = []
    var topLogprobsList: [[String: Double]?] = []
    var textOffsets: [Int] = []
    var currentOffset = 0

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
