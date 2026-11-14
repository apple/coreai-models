// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Tokenizers

/// Image placeholders in a prompt rendered through a tokenizer's chat template.
///
/// Put `text(imageTokenId:tokenizer:)` in the message text where each image goes, render the
/// messages through the chat template, then expand each placeholder token to its image's
/// soft-token count:
///
/// ```swift
/// let placeholder = ImagePlaceholders.text(imageTokenId: vision.imageTokenId, tokenizer: tokenizer)
/// let rendered = try tokenizer.applyChatTemplate(
///     messages: [["role": "user", "content": "\(placeholder)\nDescribe the image."]])
/// let prompt = ImagePlaceholders.expand(
///     rendered.map(Int32.init), imageTokenId: vision.imageTokenId, tokenCounts: [embeddings.tokenCount])
/// ```
package enum ImagePlaceholders {
    /// The image token's text, `<image>` when the tokenizer has no text for it.
    package static func text(imageTokenId: Int32, tokenizer: any Tokenizer) -> String {
        tokenizer.convertIdToToken(Int(imageTokenId)) ?? "<image>"
    }

    /// The line that puts an image's size in front of the prompt text, for models whose vision
    /// config sets `include_image_info`.
    package static func imageInfoText(width: Int, height: Int) -> String {
        "Image: \(width)x\(height)\n"
    }

    /// `tokens` with its i-th `imageTokenId` expanded to `tokenCounts[i]` copies. Nil when `tokens`
    /// holds a different number of image tokens than `tokenCounts`.
    package static func expand(_ tokens: [Int32], imageTokenId: Int32, tokenCounts: [Int]) -> [Int32]? {
        guard tokens.count(where: { $0 == imageTokenId }) == tokenCounts.count else { return nil }
        var counts = tokenCounts.makeIterator()
        return tokens.flatMap { token in
            token == imageTokenId ? Array(repeating: imageTokenId, count: counts.next() ?? 0) : [token]
        }
    }
}
