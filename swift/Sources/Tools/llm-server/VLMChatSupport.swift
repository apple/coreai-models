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

/// Policy governing local filesystem access for `image_url` content parts.
///
/// A client can put an arbitrary path in `image_url` (e.g. `/etc/hosts` or a
/// `file://` URL). This policy decides whether the server is willing to read it.
enum FileAccessPolicy: String, Sendable, CaseIterable {
    /// Reject all local filesystem access (bare paths and `file://` URLs). Only
    /// `data:` and `http(s)://` image URLs are accepted. This is the default.
    case off
    /// Allow local files only when the canonicalized real path resolves inside the
    /// server's current working directory subtree. Absolute paths outside the CWD,
    /// `..` traversal, and symlink escapes are rejected.
    case subdirs
}

// Image handling and prompt construction for the vision-language chat path.
enum VLMChatSupport {
    // MARK: - Image decode

    /// Decode the last image found in the messages (most recent user turn wins).
    /// Returns nil if no decodable image is present. Throws when a local file
    /// reference is present but forbidden by `fileAccess`.
    static func lastImage(in messages: [ChatMessage], fileAccess: FileAccessPolicy) throws -> CGImage? {
        for message in messages.reversed() {
            for urlString in message.content.imageDataURLs.reversed() {
                if let image = try decodeImage(from: urlString, fileAccess: fileAccess) {
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

    /// Decode an `image_url` string into a CGImage.
    ///
    /// Supported: `data:` URLs (base64 or percent-encoded). Local file references
    /// (`file://` or a bare path) are gated by `fileAccess`; remote `http(s)` URLs
    /// are not fetched. Throws `ServerError.badRequest` when a local reference is
    /// forbidden by the policy.
    static func decodeImage(from urlString: String, fileAccess: FileAccessPolicy) throws -> CGImage? {
        let data: Data?
        if urlString.hasPrefix("data:") {
            data = decodeDataURL(urlString)
        } else if urlString.hasPrefix("http://") || urlString.hasPrefix("https://") {
            data = nil
        } else {
            // Local file reference (`file://` URL or bare path): enforce the policy.
            let fileURL = try resolveLocalFile(urlString, policy: fileAccess)
            data = try? Data(contentsOf: fileURL)
        }
        guard let data,
            let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            return nil
        }
        return image
    }

    /// Resolve a local filesystem reference (a bare path or a `file://` URL) to a
    /// URL that the policy permits reading, or throw `ServerError.badRequest`.
    ///
    /// Under `.subdirs` the reference is canonicalized (symlinks resolved) and must
    /// land strictly inside `baseDirectory`'s real path; absolute paths outside the
    /// tree, `..` traversal, and symlink escapes are rejected.
    static func resolveLocalFile(
        _ urlString: String,
        policy: FileAccessPolicy,
        baseDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) throws -> URL {
        let rawPath: String
        if urlString.hasPrefix("file://") {
            guard let url = URL(string: urlString), url.isFileURL else {
                throw ServerError.badRequest("invalid file URL: \(urlString)")
            }
            rawPath = url.path
        } else {
            rawPath = urlString
        }

        switch policy {
        case .off:
            throw ServerError.badRequest(
                "local file access is disabled; start the server with --file-access subdirs to read files "
                    + "under the working directory, or pass the image as a data: URL")
        case .subdirs:
            let base = baseDirectory.standardizedFileURL.resolvingSymlinksInPath()
            let candidate = URL(fileURLWithPath: rawPath, relativeTo: base)
                .standardizedFileURL.resolvingSymlinksInPath()
            // Require a real-path prefix on a path boundary (so `/cwdX` cannot pass as
            // a child of `/cwd`) and forbid the base directory itself.
            let basePath = base.path
            let boundary = basePath.hasSuffix("/") ? basePath : basePath + "/"
            guard candidate.path != basePath, candidate.path.hasPrefix(boundary) else {
                throw ServerError.badRequest(
                    "file access denied: \(urlString) resolves outside the server working directory")
            }
            return candidate
        }
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
