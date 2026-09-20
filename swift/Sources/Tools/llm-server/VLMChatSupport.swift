// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAILanguageModels
import CoreGraphics
import Foundation
import ImageIO
import Tokenizers

// Image handling and prompt construction for the vision-language chat path.
enum VLMChatSupport {
    // MARK: - Image decode

    /// Decode the last image found in the messages (most recent user turn wins).
    /// Returns nil if no decodable image is present.
    static func lastImage(in messages: [ChatMessage]) -> CGImage? {
        for message in messages.reversed() {
            for urlString in message.content.imageDataURLs.reversed() {
                if let image = decodeImage(from: urlString) {
                    return image
                }
            }
        }
        return nil
    }

    /// Whether any message carries an image part.
    static func hasImage(in messages: [ChatMessage]) -> Bool {
        messages.contains { !$0.content.imageDataURLs.isEmpty }
    }

    /// Decode an OpenAI `image_url` string into a CGImage.
    ///
    /// Supported: `data:` URLs (base64 or percent-encoded) and local file paths
    /// (`file://` or a bare path). Remote `http(s)` URLs are not fetched.
    static func decodeImage(from urlString: String) -> CGImage? {
        let data: Data?
        if urlString.hasPrefix("data:") {
            data = decodeDataURL(urlString)
        } else if urlString.hasPrefix("http://") || urlString.hasPrefix("https://") {
            data = nil
        } else if urlString.hasPrefix("file://"), let url = URL(string: urlString) {
            data = try? Data(contentsOf: url)
        } else {
            data = try? Data(contentsOf: URL(fileURLWithPath: urlString))
        }
        guard let data,
            let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            return nil
        }
        return image
    }

    /// Parse the payload of a `data:[<mediatype>][;base64],<data>` URL into bytes.
    private static func decodeDataURL(_ urlString: String) -> Data? {
        guard let comma = urlString.firstIndex(of: ",") else { return nil }
        let header = urlString[..<comma]
        let payload = String(urlString[urlString.index(after: comma)...])
        if header.contains(";base64") {
            return Data(base64Encoded: payload)
        }
        return payload.removingPercentEncoding?.data(using: .utf8)
    }

    // MARK: - Prompt construction

    /// Build the prompt token sequence for a VLM turn: apply the chat template with the
    /// image placeholder embedded in the final user message, then expand that single
    /// placeholder into `imageTokenCount` copies so the engine can scatter-merge the
    /// visual embeddings. Falls back to a USER/ASSISTANT layout when the tokenizer has no
    /// usable chat template.
    static func buildPromptTokens(
        messages: [ChatMessage],
        imageTokenCount: Int,
        imageTokenId: Int32,
        tokenizer: any Tokenizer
    ) -> [Int32] {
        let imageToken = tokenizer.convertIdToToken(Int(imageTokenId)) ?? "<image>"

        if let expanded = templatedTokens(
            messages: messages,
            imageToken: imageToken,
            imageTokenId: imageTokenId,
            imageTokenCount: imageTokenCount,
            tokenizer: tokenizer
        ) {
            return expanded
        }

        // Fallback: no chat template (or it dropped the placeholder).
        let userText = messages.last(where: { $0.role == "user" })?.content.textContent ?? ""
        var tokens = tokenizer.encode(text: "USER: ", addSpecialTokens: true).map { Int32($0) }
        tokens.append(contentsOf: [Int32](repeating: imageTokenId, count: imageTokenCount))
        let suffix = "\n" + userText + "\nASSISTANT:"
        tokens.append(contentsOf: tokenizer.encode(text: suffix, addSpecialTokens: false).map { Int32($0) })
        return tokens
    }

    /// Apply the chat template with the image token prepended to the last user message and
    /// expand the placeholder. Returns nil when the template is absent or never emits the
    /// placeholder (caller falls back).
    private static func templatedTokens(
        messages: [ChatMessage],
        imageToken: String,
        imageTokenId: Int32,
        imageTokenCount: Int,
        tokenizer: any Tokenizer
    ) -> [Int32]? {
        var templateMessages: [[String: any Sendable]] = []
        var injected = false
        // Prepend the image token to the most recent user message (walk from the end).
        let lastUserIndex = messages.lastIndex(where: { $0.role == "user" })
        for (index, message) in messages.enumerated() {
            var content = message.content.textContent
            if index == lastUserIndex, !injected {
                content = "\(imageToken)\n\(content)"
                injected = true
            }
            templateMessages.append(["role": message.role, "content": content])
        }
        guard injected else { return nil }

        guard let tokens = try? tokenizer.applyChatTemplate(messages: templateMessages) else {
            return nil
        }
        return expandPlaceholder(
            in: tokens.map { Int32($0) }, imageTokenId: imageTokenId, imageTokenCount: imageTokenCount)
    }

    /// Replace the first `imageTokenId` with `imageTokenCount` copies and drop any extra
    /// single occurrences. Returns nil if the placeholder never appears.
    private static func expandPlaceholder(
        in tokens: [Int32], imageTokenId: Int32, imageTokenCount: Int
    ) -> [Int32]? {
        var result: [Int32] = []
        result.reserveCapacity(tokens.count + imageTokenCount)
        var expanded = false
        for token in tokens {
            if token == imageTokenId {
                if !expanded {
                    result.append(contentsOf: [Int32](repeating: imageTokenId, count: imageTokenCount))
                    expanded = true
                }
                continue
            }
            result.append(token)
        }
        return expanded ? result : nil
    }
}
