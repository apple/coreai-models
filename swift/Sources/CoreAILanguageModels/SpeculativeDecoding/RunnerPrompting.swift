// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Stop-token construction shared by the command-line runner's generation paths, kept apart from
/// argument handling so the paths build identical inputs and the logic can be tested.
package enum RunnerPrompting {
    /// Stop tokens beyond the end-of-sequence token: those named in the tokenizer configuration
    /// plus, for agentic vocabularies, the end-of-turn token.
    package static func additionalStopTokenIds(
        tokenizerDirectory: URL?,
        tokenizer: any Tokenizer
    ) -> Set<Int32> {
        var ids: Set<Int32> = []
        if let tokenizerDirectory {
            ids = LanguageConfig.additionalStopTokenIds(from: tokenizerDirectory, tokenizer: tokenizer)
        }
        // The end-of-turn token may be absent from tokenizer_config.json.
        let format = detectThinkingFormat(using: tokenizer)
        if let eot = agenticEndOfTurnTokenId(thinkingFormat: format, tokenizer: tokenizer),
            eot != tokenizer.eosTokenId.map({ Int32($0) })
        {
            ids.insert(eot)
        }
        return ids
    }

    /// The end-of-sequence token together with `additional`.
    package static func stopTokenIds(
        tokenizer: any Tokenizer,
        additional: Set<Int32>
    ) -> Set<Int32> {
        var ids = additional
        if let eos = tokenizer.eosTokenId { ids.insert(Int32(eos)) }
        return ids
    }
}
