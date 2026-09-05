// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing
import Tokenizers

@testable import CoreAILanguageModels

/// Tokenizer that returns `unknownTokenId` for absent tokens, reproducing the
/// swift-transformers behaviour on GPT-2-lineage models (SmolLM2, etc.).
private struct UnkFallbackTokenizer: Tokenizer {
    let knownTokens: [String: Int]
    let reverseMap: [Int: String]

    init(tokens: [String: Int]) {
        self.knownTokens = tokens
        self.reverseMap = Dictionary(uniqueKeysWithValues: tokens.map { ($0.value, $0.key) })
    }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<|endoftext|>" }
    var eosTokenId: Int? { knownTokens["<|endoftext|>"] }
    var unknownToken: String? { "<|endoftext|>" }
    var unknownTokenId: Int? { knownTokens["<|endoftext|>"] }

    func convertTokenToId(_ token: String) -> Int? {
        knownTokens[token] ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        reverseMap[id]
    }

    func encode(text: String) -> [Int] { [] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokens: [Int]) -> String { "" }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func tokenize(text: String) -> [String] { [] }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { convertTokenToId($0) } }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { convertIdToToken($0) } }
    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [[String: String]]) throws -> [Int] { [] }
}

@Suite("Marker probe: unk-token round-trip guard")
struct VocabProbeTests {

    @Test("unk-only tokenizer: probes fall back, no false capabilities")
    func unkOnlyTokenizerRejects() {
        let tok = UnkFallbackTokenizer(tokens: ["<|endoftext|>": 0])

        // convertTokenToId returns 0 (unk) for everything — vocabContains must reject
        #expect(tok.convertTokenToId("<|eom|>") == 0)
        #expect(!tok.vocabContains("<|eom|>"))

        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tok)
        guard case .tagPair(let open, _) = format else {
            Issue.record("Expected tagPair fallback, got agentic"); return
        }
        #expect(open == "<think>")
        #expect(detectToolCallMarkers(using: tok) == nil)
    }

    @Test("genuine tokens: agentic and tool markers detected")
    func genuineTokensDetected() {
        let tok = UnkFallbackTokenizer(tokens: [
            "<|endoftext|>": 0, "<|eom|>": 1, "<|eot|>": 2, "<|message|>": 3,
            "<tool_call>": 4, "</tool_call>": 5,
        ])

        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tok)
        guard case .agentic = format else {
            Issue.record("Expected agentic format"); return
        }
        #expect(detectToolCallMarkers(using: tok)?.open == "<tool_call>")
    }
}
