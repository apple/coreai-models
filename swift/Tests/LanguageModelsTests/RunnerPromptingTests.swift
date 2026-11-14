// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("RunnerPrompting")
struct RunnerPromptingTests {
    private static let agenticVocab: [String: Int] = [
        "<eos>": 2, "<|eom|>": 7, "<|eot|>": 8, "<|message|>": 9,
    ]

    @Test("The agentic end-of-turn token joins the additional stop tokens")
    func agenticEndOfTurnIncluded() {
        let tokenizer = MockTokenizer(vocab: Self.agenticVocab)
        #expect(RunnerPrompting.additionalStopTokenIds(tokenizerDirectory: nil, tokenizer: tokenizer) == [8])
    }

    @Test("A non-agentic vocabulary adds no stop tokens")
    func nonAgenticAddsNothing() {
        let tokenizer = MockTokenizer(vocab: ["<eos>": 2])
        #expect(RunnerPrompting.additionalStopTokenIds(tokenizerDirectory: nil, tokenizer: tokenizer).isEmpty)
    }

    @Test("The stop set is the end-of-sequence token plus the additional tokens")
    func stopSetIncludesEos() {
        let tokenizer = MockTokenizer(vocab: Self.agenticVocab)
        let additional = RunnerPrompting.additionalStopTokenIds(tokenizerDirectory: nil, tokenizer: tokenizer)
        #expect(RunnerPrompting.stopTokenIds(tokenizer: tokenizer, additional: additional) == [2, 8])
    }

    @Test("Declared end-of-sequence IDs join the stop tokens, without the main end-of-sequence token")
    func declaredIdsIncluded() {
        let tokenizer = MockTokenizer(vocab: ["<eos>": 2, "<|tool_response>": 50, "<end_of_turn>": 106])
        let ids = RunnerPrompting.additionalStopTokenIds(tokenizerDirectory: nil, tokenizer: tokenizer, declared: [2, 50, 106])
        #expect(ids == [50, 106])
    }

    @Test("A bundle's eos_token_ids are decoded and join its stop tokens")
    func bundleEosTokenIds() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "RunnerPromptingTests-\(UUID().uuidString)/m")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let metadata = """
            {"metadata_version": "0.2", "kind": "llm", "name": "m", "assets": {"main": "model.aimodel"},
             "language": {"tokenizer": "x/y", "vocab_size": 200, "max_context_length": 64,
                          "eos_token_ids": [2, 50, 106]}}
            """
        try metadata.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        let bundle = try LanguageModelBundle(at: dir)
        #expect(bundle.language.eosTokenIds == [2, 50, 106])
        let tokenizer = MockTokenizer(vocab: ["<eos>": 2])
        #expect(bundle.additionalStopTokenIds(tokenizer: tokenizer) == [50, 106])
        #expect(RunnerPrompting.additionalStopTokenIds(bundle: bundle, tokenizer: tokenizer) == [50, 106])
    }

    @Test("A bundle without eos_token_ids declares none")
    func bundleWithoutEosTokenIds() throws {
        let json = #"{"tokenizer": "x/y", "vocab_size": 10, "max_context_length": 8}"#
        let config = try JSONDecoder().decode(LanguageConfig.self, from: Data(json.utf8))
        #expect(config.eosTokenIds.isEmpty)
    }
}
