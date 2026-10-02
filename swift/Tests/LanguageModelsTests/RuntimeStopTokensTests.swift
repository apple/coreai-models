// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Tokenizer.runtimeStopTokens (runtime terminating set)

/// `Tokenizer.runtimeStopTokens(additional:)` is the runtime union both adapters
/// check each generated token against: the tokenizer's main EOS, a base-vocab
/// `<|im_end|>`, and the IDs resolved at load.
///
/// The config-derived turn-end scan lives in
/// `LanguageConfig.additionalStopTokenIds` and is covered by
/// `AdditionalStopTokensTests`. The text adapter's agentic `<|eot|>` fold happens
/// inline in `CoreAILanguageModel.init`, so it has no standalone unit test.
@Suite("Tokenizer.runtimeStopTokens")
struct RuntimeStopTokensTests {
    private static let vocab: [String: Int] = [
        "<eos>": 2,
        "<|im_end|>": 4,
    ]

    private static func tokenizer(vocab: [String: Int] = vocab) -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    @Test("Main EOS is always included")
    func includesMainEos() {
        let stopTokens = Self.tokenizer(vocab: [:]).runtimeStopTokens(additional: [])
        #expect(stopTokens == [2])
    }

    @Test("Additional stop tokens resolved at load are unioned into the stop set")
    func unionsAdditionalStopTokens() {
        // Gemma's <end_of_turn> (106) and Phi's <|end|> (200020) only reach the
        // stop set via the additional IDs resolved at load.
        let stopTokens = Self.tokenizer(vocab: [:]).runtimeStopTokens(additional: [106, 200_020])
        #expect(stopTokens == [2, 106, 200_020])
    }

    @Test("Additional stop tokens combine with the main EOS without duplication")
    func unionsWithMainEos() {
        // 2 is the main EOS and 4 is base-vocab <|im_end|>; passing them as
        // additional IDs must not duplicate them.
        let stopTokens = Self.tokenizer().runtimeStopTokens(additional: [2, 4, 106])
        #expect(stopTokens == [2, 4, 106])
    }

    @Test("Base-vocab <|im_end|> is folded in even with no additional IDs")
    func foldsBaseVocabImEnd() {
        // <|im_end|> (4) is in the base vocab but not passed as an additional ID.
        // runtimeStopTokens folds it in regardless of the tokenizer directory.
        let stopTokens = Self.tokenizer().runtimeStopTokens(additional: [])
        #expect(stopTokens == [2, 4])
    }

    @Test("Base-vocab <|im_end|> is not added when absent from the vocab")
    func skipsImEndWhenAbsent() {
        let stopTokens = Self.tokenizer(vocab: ["<eos>": 2]).runtimeStopTokens(additional: [])
        #expect(stopTokens == [2])
    }

    @Test("Empty additional stop tokens leave only the main EOS")
    func emptyAdditionalStopTokensNoOp() {
        let stopTokens = Self.tokenizer(vocab: ["<eos>": 2]).runtimeStopTokens(additional: [])
        #expect(stopTokens == [2])
    }
}
