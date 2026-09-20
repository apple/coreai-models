// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import llm_server

@Suite("VLMChatSupport")
struct VLMChatSupportTests {
    // 1x1 transparent PNG.
    private static let png1x1Base64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    @Test("decodeImage decodes a base64 data URL")
    func decodesDataURL() {
        let url = "data:image/png;base64,\(Self.png1x1Base64)"
        let image = VLMChatSupport.decodeImage(from: url)
        #expect(image != nil)
        #expect(image?.width == 1)
        #expect(image?.height == 1)
    }

    @Test("decodeImage returns nil for remote http(s) URLs")
    func rejectsRemoteURL() {
        #expect(VLMChatSupport.decodeImage(from: "https://example.com/cat.png") == nil)
        #expect(VLMChatSupport.decodeImage(from: "http://example.com/cat.png") == nil)
    }

    @Test("buildPromptTokens fallback expands the image placeholder to imageTokenCount copies")
    func fallbackExpandsPlaceholder() throws {
        // MockTokenizer has no usable chat template, so buildPromptTokens takes the
        // USER/ASSISTANT fallback that appends imageTokenCount image tokens directly.
        let tokenizer = MockTokenizer()
        let imageTokenId: Int32 = 200_092
        let imageTokenCount = 4
        let message = try JSONDecoder().decode(
            ChatMessage.self,
            from: Data(#"{"role":"user","content":"What is in this image?"}"#.utf8))

        let tokens = VLMChatSupport.buildPromptTokens(
            messages: [message],
            imageTokenCount: imageTokenCount,
            imageTokenId: imageTokenId,
            tokenizer: tokenizer
        )

        let placeholders = tokens.filter { $0 == imageTokenId }.count
        #expect(placeholders == imageTokenCount)
        #expect(tokens.count > imageTokenCount)
    }
}
