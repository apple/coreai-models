// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Server API Types")
struct ServerAPITypesTests {
    // MARK: - ChatCompletionRequest Decoding

    @Test("Basic chat request decodes all fields")
    func basicChatRequest() throws {
        let json = """
            {"model":"qwen3","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"temperature":0.7,"stream":false}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.model == "qwen3")
        #expect(request.messages.count == 1)
        #expect(request.messages[0].role == "user")
        #expect(request.maxTokens == 50)
        #expect(request.temperature == 0.7)
        #expect(request.stream == false)
    }

    @Test("Stop field as string decodes")
    func stopAsString() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"stop":"END"}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.stop == ["END"])
    }

    @Test("Stop field as array decodes")
    func stopAsArray() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"stop":["END","STOP"]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.stop == ["END", "STOP"])
    }

    @Test("response_format with json_schema decodes")
    func responseFormatJsonSchema() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object"}}}}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.responseFormat?.type == "json_schema")
    }

    @Test("Multimodal message content with image_url")
    func multimodalContent() throws {
        let json = """
            {"messages":[{"role":"user","content":[{"type":"text","text":"What?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.messages.count == 1)
        if case .parts(let parts) = request.messages[0].content {
            #expect(parts.count == 2)
        }
    }

    // MARK: - Response Encoding

    @Test("ChatCompletionResponse encodes required fields")
    func chatResponseEncodes() throws {
        let response = ChatCompletionResponse(
            id: "coreai-1",
            model: "qwen3_4b",
            choices: [
                ChatCompletionResponse.Choice(
                    index: 0,
                    message: ChatCompletionResponse.ResponseMessage(role: "assistant", content: "Hello!"),
                    finishReason: "stop"
                )
            ]
        )
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["id"] as? String == "coreai-1")
        #expect(obj["object"] as? String == "chat.completion")
        #expect(obj["model"] as? String == "qwen3_4b")
    }

    @Test("ChatCompletionChunk encodes streaming fields")
    func chunkEncodes() throws {
        let chunk = ChatCompletionChunk(
            id: "coreai-1",
            model: "qwen3_4b",
            choices: [
                ChatCompletionChunk.ChunkChoice(
                    index: 0, delta: ChatCompletionChunk.Delta(content: "Hi"), finishReason: nil
                )
            ]
        )
        let data = try JSONEncoder().encode(chunk)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["object"] as? String == "chat.completion.chunk")
    }

    @Test("ModelsResponse encodes")
    func modelsResponse() throws {
        let response = ModelsResponse(data: [
            ModelsResponse.ModelInfo(id: "qwen3_4b", created: 1_700_000_000, ownedBy: "coreai")
        ])
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let models = (obj["data"] as? [[String: Any]]) ?? []
        #expect(models.count == 1)
        #expect(models[0]["id"] as? String == "qwen3_4b")
    }

    @Test("ErrorResponse encodes")
    func errorResponse() throws {
        let response = ErrorResponse(error: ErrorResponse.ErrorDetail(message: "bad request", type: "invalid_request"))
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let err = obj["error"] as! [String: Any]
        #expect(err["message"] as? String == "bad request")
    }
}
