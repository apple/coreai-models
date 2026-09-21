// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Shared stop-token resolution for the text and VLM language-model adapters.
enum StopTokens {
    /// EOS-like token IDs beyond the tokenizer's main EOS, resolved at load:
    /// turn-end tokens from the tokenizer config, `<|im_end|>`, and an optional
    /// agentic `<|eot|>`.
    static func additionalIds(
        bundle: LanguageBundle,
        tokenizer: any Tokenizer,
        agenticEOT: String? = nil
    ) -> [Int32] {
        var ids: [Int32] = []
        if let tokenizerDir = bundle.tokenizerPath {
            ids = LanguageConfig.additionalStopTokenIds(from: tokenizerDir, tokenizer: tokenizer)
        }
        // Folded in at load (not runtime) so the text path also stops on a base-vocab <|im_end|>.
        fold("<|im_end|>", into: &ids, tokenizer: tokenizer)
        // Agentic models stop on <|eot|> so the runner doesn't loop self->user turns.
        if let agenticEOT {
            fold(agenticEOT, into: &ids, tokenizer: tokenizer)
        }
        return ids
    }

    /// Full terminating set: main EOS plus the load-time additional IDs.
    static func set(tokenizer: any Tokenizer, additional: [Int32]) -> Set<Int32> {
        var stop = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stop.insert(Int32(eos)) }
        stop.formUnion(additional)
        return stop
    }

    /// Append `token`'s vocab ID to `ids` when it resolves and isn't already present.
    private static func fold(_ token: String, into ids: inout [Int32], tokenizer: any Tokenizer) {
        guard tokenizer.vocabContains(token), let id = tokenizer.convertTokenToId(token) else {
            return
        }
        let id32 = Int32(id)
        if !ids.contains(id32) {
            ids.append(id32)
        }
    }
}
