// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Synchronization
import Tokenizers

// MARK: - Server Configuration

struct ServerConfig: Sendable {
    let modelName: String
    let defaultMaxTokens: Int
    let defaultTemperature: Double
    let defaultTopP: Double?
    let defaultTopK: Int?
    let defaultMinP: Double?
    let noThinking: Bool
    let supportsLogprobs: Bool
    let maxContextLength: Int
    let vocabSize: Int?
    let additionalEosTokenIds: [Int32]
}

// MARK: - Server Stats

final class ServerStats: @unchecked Sendable {
    private let lock = Mutex<State>(State())

    private struct State {
        var totalRequests: Int = 0
        var totalPromptTokens: Int = 0
        var totalGenTokens: Int = 0
        var totalPromptSeconds: Double = 0
        var totalGenSeconds: Double = 0
        var totalSeconds: Double = 0
        var totalToolCalls: Int = 0
        var lastPrintTime: ContinuousClock.Instant = .now
        var requestsSinceLastPrint: Int = 0
    }

    func record(
        promptTokens: Int, genTokens: Int, promptSeconds: Double, genSeconds: Double, totalSeconds: Double,
        toolCalls: Int = 0
    ) {
        let shouldPrint = lock.withLock { s -> Bool in
            s.totalRequests += 1
            s.totalPromptTokens += promptTokens
            s.totalGenTokens += genTokens
            s.totalPromptSeconds += promptSeconds
            s.totalGenSeconds += genSeconds
            s.totalSeconds += totalSeconds
            s.totalToolCalls += toolCalls
            s.requestsSinceLastPrint += 1

            let elapsed = ContinuousClock.now - s.lastPrintTime
            if elapsed > .seconds(60) && s.requestsSinceLastPrint > 0 {
                s.lastPrintTime = .now
                s.requestsSinceLastPrint = 0
                return true
            }
            return false
        }

        if shouldPrint {
            printSummary()
        }
    }

    func printSummary() {
        let s = lock.withLock { $0 }
        let avgGenTokPerSec = s.totalGenSeconds > 0 ? Double(s.totalGenTokens) / s.totalGenSeconds : 0
        let avgPromptTokPerSec = s.totalPromptSeconds > 0 ? Double(s.totalPromptTokens) / s.totalPromptSeconds : 0
        let overhead = s.totalSeconds - s.totalPromptSeconds - s.totalGenSeconds
        let toolLine = s.totalToolCalls > 0 ? "\nTool calls: \(s.totalToolCalls)" : ""

        print(
            """

            Server Stats (\(s.totalRequests) requests):
            ==================================================
            Prefill:    \(s.totalPromptTokens) tokens, \(String(format: "%.1f", s.totalPromptSeconds))s (\(String(format: "%.1f", avgPromptTokPerSec)) tok/s)
            Generation: \(s.totalGenTokens) tokens, \(String(format: "%.1f", s.totalGenSeconds))s (\(String(format: "%.1f", avgGenTokPerSec)) tok/s)
            Overhead:   \(String(format: "%.1f", overhead))s (\(String(format: "%.1f", overhead / Double(max(1, s.totalRequests))))s/req)\(toolLine)
            ==================================================
            """)
    }

    func buildResponse(
        prefixHitRate: Double, prefixHits: Int, prefixMisses: Int,
        totalToolCalls: Int, topTools: [ToolCallStat]
    ) -> ServerStatsResponse {
        let s = lock.withLock { $0 }
        return ServerStatsResponse(
            totalRequests: s.totalRequests,
            totalPromptTokens: s.totalPromptTokens,
            totalGenTokens: s.totalGenTokens,
            avgPrefillTokPerSec: s.totalPromptSeconds > 0 ? Double(s.totalPromptTokens) / s.totalPromptSeconds : 0,
            avgDecodeTokPerSec: s.totalGenSeconds > 0 ? Double(s.totalGenTokens) / s.totalGenSeconds : 0,
            prefixHitRate: prefixHitRate,
            prefixHits: prefixHits,
            prefixMisses: prefixMisses,
            totalToolCalls: totalToolCalls,
            topTools: topTools.isEmpty ? nil : topTools
        )
    }
}

// MARK: - Server State

final class ServerState: @unchecked Sendable {
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let config: ServerConfig
    let stats = ServerStats()
    let toolCallMarkers: (open: String, close: String)?
    private let _state = Mutex<InternalState>(InternalState())

    private struct InternalState {
        var generating: Bool = false
        var lastSessionID: String? = nil
        var lastPromptTokens: [Int32] = []
        var prefixHits: Int = 0
        var prefixMisses: Int = 0
        var toolCallCounts: [String: Int] = [:]
        var totalToolCalls: Int = 0
    }

    init(engine: any InferenceEngine, tokenizer: any Tokenizer, config: ServerConfig) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.toolCallMarkers = detectToolCallMarkers(using: tokenizer)
    }

    var supportsToolCalling: Bool { toolCallMarkers != nil }

    func makeToolCallParser() -> ToolCallParser? {
        guard let markers = toolCallMarkers else { return nil }
        return ToolCallParser(openMarker: markers.open, closeMarker: markers.close)
    }

    func tryAcquire() -> Bool {
        _state.withLock { s in
            guard !s.generating else { return false }
            s.generating = true
            return true
        }
    }

    func release() {
        _state.withLock { $0.generating = false }
    }

    /// Prepare engine for a new request. Returns the number of prefix tokens reused.
    func prepareForRequest(sessionID: String?, promptTokens: [Int32]) async -> Int {
        let action = _state.withLock { s -> PrepareAction in
            guard let sid = sessionID, sid == s.lastSessionID else {
                s.lastSessionID = sessionID
                s.lastPromptTokens = []
                s.prefixMisses += 1
                return .reset
            }

            let cached = s.lastPromptTokens
            var match = 0
            let limit = min(cached.count, promptTokens.count)
            while match < limit && cached[match] == promptTokens[match] {
                match += 1
            }

            if match == 0 {
                s.prefixMisses += 1
                return .reset
            }

            s.prefixHits += 1
            return .reuse(prefixLength: match)
        }

        switch action {
        case .reset:
            return 0
        case .reuse(let prefixLength):
            return prefixLength
        }
    }

    /// Record the tokens that were processed (call after generate completes).
    func recordPromptTokens(_ tokens: [Int32]) {
        _state.withLock { $0.lastPromptTokens = tokens }
    }

    /// Record tool calls made in a response.
    func recordToolCalls(_ names: [String]) {
        _state.withLock { s in
            s.totalToolCalls += names.count
            for name in names {
                s.toolCallCounts[name, default: 0] += 1
            }
        }
    }

    /// Stats snapshot for /v1/stats endpoint.
    func statsSnapshot() -> ServerStatsResponse {
        let s = _state.withLock { s in
            (
                prefixHits: s.prefixHits, prefixMisses: s.prefixMisses,
                toolCallCounts: s.toolCallCounts, totalToolCalls: s.totalToolCalls
            )
        }
        let hitTotal = s.prefixHits + s.prefixMisses
        let hitRate = hitTotal > 0 ? Double(s.prefixHits) / Double(hitTotal) : 0
        let topTools = s.toolCallCounts.sorted { $0.value > $1.value }.prefix(5)
            .map { ToolCallStat(name: $0.key, count: $0.value) }
        return stats.buildResponse(
            prefixHitRate: hitRate, prefixHits: s.prefixHits, prefixMisses: s.prefixMisses,
            totalToolCalls: s.totalToolCalls, topTools: topTools)
    }

    private enum PrepareAction {
        case reset
        case reuse(prefixLength: Int)
    }

    /// Readiness snapshot for /ready endpoint.
    func readySnapshot() -> ReadyResponse {
        let busy = _state.withLock { $0.generating }
        let usedTokens = engine.processedTokenCount
        let maxTokens = config.maxContextLength
        let utilization = maxTokens > 0 ? Double(usedTokens) / Double(maxTokens) : 0
        return ReadyResponse(
            status: "ready",
            model: config.modelName,
            maxContextLength: maxTokens,
            busy: busy,
            cache: .init(
                usedTokens: usedTokens,
                maxTokens: maxTokens,
                utilization: utilization
            ),
            toolCalling: supportsToolCalling
        )
    }

    func makeSamplingConfig(
        temperature: Double?,
        topP: Double?,
        topK: Int?,
        minP: Double?
    ) -> SamplingConfiguration {
        let temp = temperature ?? config.defaultTemperature
        if temp == 0 {
            return .greedy
        }
        return SamplingConfiguration(
            temperature: temp,
            topK: topK ?? config.defaultTopK,
            topP: topP ?? config.defaultTopP,
            minP: minP ?? config.defaultMinP
        )
    }
}

// MARK: - Request ID Generator

enum RequestID {
    private static let counter = Mutex<Int>(0)

    static func next() -> String {
        let n = counter.withLock { val -> Int in
            val += 1
            return val
        }
        return "coreai-\(n)"
    }
}

// MARK: - Server Errors

enum ServerError: Error, LocalizedError {
    case badRequest(String)

    var isBadRequest: Bool {
        if case .badRequest = self { return true }
        return false
    }

    var errorDescription: String? {
        switch self {
        case .badRequest(let msg): return msg
        }
    }
}
