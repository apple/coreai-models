// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Turns text into token ids as the `text` block says (R6).
struct TextPreparer: Sendable {
    /// Token ids of a string, special tokens included.
    typealias Encode = @Sendable (String) -> [Int]

    let config: TextConfig
    let encode: Encode

    /// Loads the bundled tokenizer from `folder`: `CoreAIShared.CLIPTokenizer` for a
    /// `CLIPTokenizer` class in `tokenizer_config.json`, else swift-transformers' `AutoTokenizer`.
    /// CLIP text is NFC-normalized first, as the `tokenizer.json` normalizer does.
    static func load(from folder: URL, config: TextConfig) async throws -> TextPreparer {
        if try tokenizerClass(in: folder)?.hasPrefix("CLIPTokenizer") == true {
            let clip = try CLIPTokenizer(folder: folder)
            return TextPreparer(config: config) { text in
                clip.encodeUntruncated(text.precomposedStringWithCanonicalMapping).map(Int.init)
            }
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        return TextPreparer(config: config) { tokenizer.encode(text: $0) }
    }

    /// `tokenizer_class` of `folder/tokenizer_config.json`, if the file has one.
    static func tokenizerClass(in folder: URL) throws -> String? {
        let url = folder.appending(path: "tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        return json?["tokenizer_class"] as? String
    }

    /// Token ids of the default prompt plus `text`, truncated to `maxTokens` with EOS kept last,
    /// and the token count before truncation.
    func tokenize(_ text: String) -> (ids: [Int32], count: Int) {
        var ids = encode(config.prompt + text).map(Int32.init)
        let count = ids.count
        if count > config.maxTokens {
            ids.removeLast(count - config.maxTokens)
            if config.keepEos && config.addEosToken { ids[ids.count - 1] = config.eosTokenId }
        }
        return (ids, count)
    }
}
