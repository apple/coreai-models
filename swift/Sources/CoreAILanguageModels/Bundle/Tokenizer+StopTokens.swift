// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

extension Tokenizer {
    /// The full set of token IDs that terminate generation: the tokenizer's main
    /// EOS (when present) unioned with the load-time `additional` IDs resolved by
    /// `LanguageConfig.additionalStopTokenIds`.
    ///
    /// Both the text and VLM adapters check each generated token against this set.
    func runtimeStopTokens(additional: Set<Int32>) -> Set<Int32> {
        var stop = additional
        if let eos = eosTokenId { stop.insert(Int32(eos)) }
        return stop
    }
}
