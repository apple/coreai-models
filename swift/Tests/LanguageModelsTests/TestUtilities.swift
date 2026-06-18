// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization
import TestUtilities
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Mock Inference Engine

/// Mock inference engine for testing the unified generation API.
///
/// Features:
/// - Returns tokens from a configurable sequence (cycles if exhausted)
/// - Logits: when requested, returns a distribution with the target token having high probability.
///   If vocabSize is nil, logits are not supported (returns nil even when requested).
/// - Tracks inference call count for assertions
/// - Tracks whether reset() was called
/// - Configurable maxContextLength for testing bounds
class MockEngine: InferenceEngine, @unchecked Sendable {
    struct MockConfig: Codable, InferenceConfiguration {
        var maxContextLength: Int
    }

    let config: MockConfig
    let tokenSequence: [Int32]
    let vocabSize: Int?
    let loadedModelURL: URL?
    var supportsLogits: Bool { vocabSize != nil }

    /// Tracks how many times generate() produced a token
    private(set) var inferenceCallCount: Int = 0

    /// Tracks whether reset() was called
    private(set) var resetCalled: Bool = false

    /// Number of tokens the engine has processed in the current session.
    var processedTokenCount: Int = 0

    /// Token history for implicit prefix caching
    var history = TokenHistory()
    private(set) var lastPrefixHitCount: Int = 0

    // Generation lifecycle
    private let _activeToken = Mutex<GenerationToken?>(nil)

    var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    /// Clear the engine's active token if it matches the given token.
    func clearTokenIfActive(_ token: GenerationToken) {
        _activeToken.withLock { if $0 === token { $0 = nil } }
    }

    init(
        tokens: [Int32] = [10, 20, 30, 40, 50],
        maxContextLength: Int = 4096,
        vocabSize: Int? = 100,
        modelURL: URL? = nil
    ) {
        self.config = MockConfig(maxContextLength: maxContextLength)
        self.tokenSequence = tokens
        self.vocabSize = vocabSize
        self.loadedModelURL = modelURL
    }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        let token = GenerationToken()
        _activeToken.withLock { $0 = token }

        // Implicit prefix caching: resolve input against history
        let (commonPrefix, resolvedNewTokens) = history.resolve(input: input)
        lastPrefixHitCount = commonPrefix

        if commonPrefix < processedTokenCount {
            // Input diverged — rewind
            processedTokenCount = commonPrefix
            history.truncate(to: commonPrefix)
        }

        // Simulate prefill: count new tokens beyond what's already cached.
        let newInputTokens = Array(resolvedNewTokens)
        processedTokenCount += newInputTokens.count
        history.append(contentsOf: newInputTokens[...])

        return GenerationSequence(
            engine: self,
            input: input,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    struct GenerationSequence: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = Error

        let engine: MockEngine
        let input: [TokenId]
        let inferenceOptions: InferenceOptions
        let generationToken: GenerationToken

        let stopReasonStore = StopReasonStore()

        var stopReason: StopReason? { stopReasonStore.stopReason }

        func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        func makeAsyncIterator() -> Iterator {
            Iterator(
                engine: engine,
                input: input,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore,
                generationToken: generationToken
            )
        }

        struct Iterator: AsyncIteratorProtocol {
            typealias Element = InferenceOutput
            typealias Failure = Error

            let engine: MockEngine
            let returnsLogits: Bool
            let forcedContinuation: [TokenId]?
            let maxTokens: Int
            let stopReasonStore: StopReasonStore
            let generationToken: GenerationToken

            var step: Int = 0
            var finished: Bool = false

            init(
                engine: MockEngine,
                input: [TokenId],
                inferenceOptions: InferenceOptions,
                stopReasonStore: StopReasonStore,
                generationToken: GenerationToken
            ) {
                self.engine = engine
                self.returnsLogits = inferenceOptions.includeLogits
                self.forcedContinuation = inferenceOptions.forcedContinuation
                self.stopReasonStore = stopReasonStore
                self.generationToken = generationToken
                if let forced = inferenceOptions.forcedContinuation {
                    self.maxTokens = forced.count
                } else {
                    self.maxTokens = Swift.min(
                        inferenceOptions.maxTokens ?? Int.max,
                        Swift.max(0, engine.config.maxContextLength - input.count)
                    )
                }
            }

            mutating func next() async throws -> InferenceOutput? {
                if finished { return nil }

                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    return nil
                }

                guard step < maxTokens else {
                    stopReasonStore.setIfUnset(.maxTokens)
                    finishAndRelease()
                    return nil
                }

                do {
                    try Task.checkCancellation()

                    let idx = engine.inferenceCallCount % engine.tokenSequence.count
                    let sequenceToken = engine.tokenSequence[idx]
                    engine.inferenceCallCount += 1
                    engine.processedTokenCount += 1

                    let nextToken = forcedContinuation?[step] ?? sequenceToken

                    // Track generated token in history
                    engine.history.append(nextToken)

                    let logits: [Float16]?
                    if returnsLogits, let vocabSize = engine.vocabSize {
                        var dist = [Float16](repeating: Float16(-10.0), count: vocabSize)
                        let tokenIdx = Int(nextToken)
                        if tokenIdx >= 0 && tokenIdx < vocabSize {
                            dist[tokenIdx] = Float16(10.0)
                        }
                        logits = dist
                    } else {
                        logits = nil
                    }

                    step += 1
                    return InferenceOutput(tokenId: nextToken, logits: logits)
                } catch is CancellationError {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    throw CancellationError()
                } catch {
                    stopReasonStore.set(.error)
                    finishAndRelease()
                    throw error
                }
            }

            private mutating func finishAndRelease() {
                guard !finished else { return }
                finished = true
                engine.clearTokenIfActive(generationToken)
            }
        }
    }

    func cancel() async throws {
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
    }

    func reset(to tokenIndex: Int) async throws {
        try await cancel()
        resetCalled = true
        processedTokenCount = tokenIndex
        if tokenIndex == 0 {
            inferenceCallCount = 0
            history.clear()
        } else {
            history.truncate(to: tokenIndex)
        }
    }

    func cleanup() async throws {}
}

// MARK: - ByteLevel Encoding Extension

/// ByteLevel encoding for tokenizers that use it (GPT-2, Qwen, etc.)
/// Maps control characters to their ByteLevel Unicode equivalents
extension String {
    /// Apply ByteLevel encoding - maps control characters to ByteLevel equivalents
    ///
    /// Mapping (matches GPT-2/Qwen tokenizers):
    /// - `\n` (0x0A) → `Ċ` (U+010A)
    /// - ` ` (0x20) → `Ġ` (U+0120)
    /// - `\t` (0x09) → `ĉ` (U+0109)
    func byteleveled() -> String {
        var result = self
        result = result.replacingOccurrences(of: "\n", with: "Ċ")
        result = result.replacingOccurrences(of: " ", with: "Ġ")
        result = result.replacingOccurrences(of: "\t", with: "ĉ")
        return result
    }
}

// MARK: - Temp File Helper

/// Helper for creating temp files that auto-cleanup
struct TempFile {
    let url: URL

    init(extension ext: String = "json") {
        let tempDir = FileManager.default.temporaryDirectory
        url = tempDir.appendingPathComponent("test_\(UUID().uuidString).\(ext)")
    }

    var path: String { url.path }

    func cleanup() {
        try? FileManager.default.removeItem(at: url)
    }

    func readData() throws -> Data {
        try Data(contentsOf: url)
    }

    func readString() throws -> String {
        try String(contentsOf: url, encoding: .utf8)
    }
}

/// Mock tokenizer that simulates BPE-style token merging.
/// When space + uppercase letter appears, it merges them into a single token.
/// - encode("Answer: B") → [..., 200] where 200 is " B" merged token
/// - decode([200]) → " B"
struct MergingMockTokenizer: Tokenizer, Sendable {
    // MARK: - Properties
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eos>" }
    var eosTokenId: Int? { 2 }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { 0 }

    // MARK: - Core encoding/decoding

    func encode(text: String) -> [Int] {
        var tokens: [Int] = []
        let chars = Array(text)
        var i = 0

        while i < chars.count {
            // Check for space + uppercase letter merge
            if i + 1 < chars.count && chars[i] == " " {
                let nextChar = chars[i + 1]
                if nextChar.isUppercase && nextChar.isASCII {
                    // Merge space + letter into special token (200 + letter offset)
                    let letterOffset = Int(nextChar.asciiValue! - Character("A").asciiValue!)
                    tokens.append(200 + letterOffset)
                    i += 2
                    continue
                }
            }

            // Regular byte encoding
            if let ascii = chars[i].asciiValue {
                tokens.append(Int(ascii))
            } else {
                tokens.append(0)  // Unknown
            }
            i += 1
        }

        return tokens
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text)
    }

    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        var result = ""
        for token in tokens {
            if token >= 200 && token < 226 {
                // Merged space + letter token
                let letter = Character(UnicodeScalar(UInt8(token - 200) + Character("A").asciiValue!))
                result += " " + String(letter)
            } else if token >= 0 && token < 128 {
                result += String(Character(UnicodeScalar(UInt8(token))))
            }
        }
        return result
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        decode(tokens: tokens)
    }

    func tokenize(text: String) -> [String] {
        // For simplicity, just return characters
        text.map { String($0) }
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token.count == 2 && token.first == " " {
            if let letter = token.last, letter.isUppercase && letter.isASCII {
                return 200 + Int(letter.asciiValue! - Character("A").asciiValue!)
            }
        }
        return token.utf8.first.map { Int($0) }
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id >= 200 && id < 226 {
            let letter = Character(UnicodeScalar(UInt8(id - 200) + Character("A").asciiValue!))
            return " " + String(letter)
        }
        guard (0...255).contains(id) else { return nil }
        return String(decoding: [UInt8(id)], as: UTF8.self)
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    // MARK: - Chat template methods

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        let combined = messages.compactMap { $0["content"] as? String }.joined(separator: " ")
        return encode(text: combined)
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument) throws
        -> [Int]
    {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        let combined = messages.compactMap { $0["content"] }.joined(separator: " ")
        return encode(text: combined)
    }
}
