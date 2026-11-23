// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import Testing

/// The opt-in `return_token_ids` request field and the `token_ids` / `prompt_token_ids` it adds.
struct ServerAPITokenIdsTests {
    private func decodeRequest(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    private func encodeToString(_ value: some Encodable) throws -> String {
        String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
    }

    @Test("return_token_ids decodes, and is nil when absent")
    func requestField() throws {
        let asked = try decodeRequest(#"{"messages":[{"role":"user","content":"hi"}],"return_token_ids":true}"#)
        #expect(asked.returnTokenIds == true)
        let plain = try decodeRequest(#"{"messages":[{"role":"user","content":"hi"}]}"#)
        #expect(plain.returnTokenIds == nil)
    }

    private func response(tokenIds: [Int32]?, promptTokenIds: [Int32]?) -> ChatCompletionResponse {
        ChatCompletionResponse(
            id: "coreai-1", model: "test-model",
            choices: [
                .init(
                    index: 0, message: .init(role: "assistant", content: "hi"), finishReason: "stop",
                    tokenIds: tokenIds)
            ],
            promptTokenIds: promptTokenIds)
    }

    @Test("The response leaves the token ID fields out unless they were asked for")
    func responseOmitsByDefault() throws {
        let json = try encodeToString(response(tokenIds: nil, promptTokenIds: nil))
        #expect(!json.contains("token_ids"))
    }

    @Test("The response carries token_ids on the choice and prompt_token_ids at the top")
    func responseCarriesIds() throws {
        let data = try JSONEncoder().encode(response(tokenIds: [5, 6, 2], promptTokenIds: [1, 2, 3]))
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["prompt_token_ids"] as? [Int] == [1, 2, 3])
        let choice = try #require((json["choices"] as? [[String: Any]])?.first)
        #expect(choice["token_ids"] as? [Int] == [5, 6, 2])
    }

    @Test("A streaming chunk carries its token_ids and, when set, prompt_token_ids; neither by default")
    func chunk() throws {
        let plain = ChatCompletionChunk(
            id: "c", model: "m", choices: [.init(index: 0, delta: .init(content: "hi"), finishReason: nil)])
        #expect(!(try encodeToString(plain)).contains("token_ids"))

        let withIds = ChatCompletionChunk(
            id: "c", model: "m",
            choices: [.init(index: 0, delta: .init(role: "assistant"), finishReason: nil, tokenIds: [])],
            promptTokenIds: [1, 2])
        let json = try encodeToString(withIds)
        #expect(json.contains(#""prompt_token_ids":[1,2]"#))
        #expect(json.contains(#""token_ids":[]"#))
    }
}
