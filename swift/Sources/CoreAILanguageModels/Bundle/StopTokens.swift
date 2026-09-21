// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Stop-token resolution shared by the text and VLM language-model adapters.
///
/// Keeps load-time EOS-like ID discovery (`additionalIds`) and the runtime
/// terminating set (`set`) in one place so the two adapters can't drift.
enum StopTokens {
    /// Load-time: EOS-like token IDs beyond the tokenizer's main EOS.
    ///
    /// Union of:
    ///  - turn-end tokens resolved from tokenizer_config.json / tokenizer.json
    ///    (`LanguageConfig.additionalStopTokenIds`),
    ///  - `<|im_end|>` when present in the vocab,
    ///  - an optional agentic `<|eot|>` token, when provided and in the vocab.
    ///
    /// De-duplicated. Returns an empty list when the bundle ships no embedded
    /// tokenizer directory.
    static func additionalIds(
        bundle: LanguageBundle,
        tokenizer: any Tokenizer,
        agenticEOT: String? = nil
    ) -> [Int32] {
        var ids: [Int32] = []
        if let tokenizerDir = bundle.tokenizerPath {
            ids = LanguageConfig.additionalStopTokenIds(from: tokenizerDir, tokenizer: tokenizer)
        }
        // Qwen-family turn end. Resolving <|im_end|> at load (not runtime) is the
        // unification: the text path now stops on a base-vocab <|im_end|> too.
        fold("<|im_end|>", into: &ids, tokenizer: tokenizer)
        // Agentic models: stop on <|eot|> (end of user-facing turn) so the runner
        // doesn't loop through repeated self->user cycles.
        if let agenticEOT {
            fold(agenticEOT, into: &ids, tokenizer: tokenizer)
        }
        return ids
    }

    /// Runtime: full terminating set = main EOS ∪ additional load-time IDs.
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
