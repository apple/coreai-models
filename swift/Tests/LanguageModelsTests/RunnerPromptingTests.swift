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
}
