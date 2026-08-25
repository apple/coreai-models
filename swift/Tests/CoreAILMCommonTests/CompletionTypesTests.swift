// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Completion API Types")
struct CompletionTypesTests {
    // MARK: - CompletionRequest Decoding

    @Test("Single string prompt decodes")
    func singleStringPrompt() throws {
        let json = """
            {"prompt":"Hello world","max_tokens":10}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.prompts.count == 1)
        if case .text(let s) = request.prompts[0] { #expect(s == "Hello world") }
        #expect(request.maxTokens == 10)
    }

    @Test("Array of strings prompt decodes")
    func arrayStringPrompt() throws {
        let json = """
            {"prompt":["Hello","World"],"temperature":0.5}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.prompts.count == 2)
        if case .text(let s) = request.prompts[0] { #expect(s == "Hello") }
        if case .text(let s) = request.prompts[1] { #expect(s == "World") }
        #expect(request.temperature == 0.5)
    }

    @Test("Token ID array prompt decodes")
    func tokenIdPrompt() throws {
        let json = """
            {"prompt":[1,2,3]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.prompts.count == 1)
        if case .tokenIds(let ids) = request.prompts[0] {
            #expect(ids == [1, 2, 3])
        } else {
            Issue.record("Expected .tokenIds")
        }
    }

    @Test("Batched token ID arrays prompt decodes")
    func batchedTokenIdPrompt() throws {
        let json = """
            {"prompt":[[1,2],[3,4,5]]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.prompts.count == 2)
        if case .tokenIds(let ids) = request.prompts[0] { #expect(ids == [1, 2]) }
        if case .tokenIds(let ids) = request.prompts[1] { #expect(ids == [3, 4, 5]) }
    }

    @Test("Echo and logprobs fields decode")
    func echoAndLogprobs() throws {
        let json = """
            {"prompt":"test","echo":true,"logprobs":5}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.echo == true)
        #expect(request.logprobs == 5)
    }

    @Test("Missing optional fields default to nil")
    func optionalFieldsNil() throws {
        let json = """
            {"prompt":"test"}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionRequest.self, from: json)
        #expect(request.model == nil)
        #expect(request.maxTokens == nil)
        #expect(request.temperature == nil)
        #expect(request.echo == nil)
        #expect(request.logprobs == nil)
    }

    // MARK: - CompletionResponse Encoding

    @Test("CompletionResponse encodes all fields")
    func responseEncodes() throws {
        let response = CompletionResponse(
            id: "coreai-1",
            object: "text_completion",
            created: 1_700_000_000,
            model: "qwen3_4b",
            choices: [
                .init(index: 0, text: "world", logprobs: nil, finishReason: "stop")
            ]
        )
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["id"] as? String == "coreai-1")
        #expect(obj["object"] as? String == "text_completion")
        #expect(obj["model"] as? String == "qwen3_4b")
        let choices = obj["choices"] as! [[String: Any]]
        #expect(choices.count == 1)
        #expect(choices[0]["text"] as? String == "world")
        #expect(choices[0]["finish_reason"] as? String == "stop")
    }

    @Test("LogprobsResult encodes with snake_case keys")
    func logprobsResultEncodes() throws {
        let logprobs = CompletionResponse.LogprobsResult(
            tokens: ["Hello", " world"],
            tokenLogprobs: [-0.5, -1.2],
            topLogprobs: [["Hello": -0.5, "Hi": -1.0], [" world": -1.2]],
            textOffset: [0, 5]
        )
        let choice = CompletionResponse.CompletionChoice(
            index: 0, text: "Hello world", logprobs: logprobs, finishReason: "stop"
        )
        let response = CompletionResponse(
            id: "test", object: "text_completion", created: 0, model: "m", choices: [choice]
        )
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let choices = obj["choices"] as! [[String: Any]]
        let lp = choices[0]["logprobs"] as! [String: Any]
        #expect(lp["token_logprobs"] != nil)
        #expect(lp["top_logprobs"] != nil)
        #expect(lp["text_offset"] != nil)
    }

    // MARK: - JSONValue

    @Test("JSONValue bool vs number ordering")
    func jsonValueBoolFirst() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"stream":true}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.stream == true)
    }

    @Test("JSONValue false does not become 0")
    func jsonValueFalseNotZero() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"stream":false}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.stream == false)
    }

    @Test("JSONValue encodes nested object")
    func jsonValueNestedObject() throws {
        let value = JSONValue.object([
            "type": .string("object"),
            "required": .array([.string("name")]),
            "count": .number(3),
            "nullable": .null,
            "flag": .bool(true),
        ])
        let data = try JSONEncoder().encode(value)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["type"] as? String == "object")
        #expect(obj["count"] as? Double == 3.0)
        #expect(obj["flag"] as? Bool == true)
    }

    @Test("JSONValue round-trips through encode/decode")
    func jsonValueRoundTrip() throws {
        let value = JSONValue.object([
            "nested": .object(["inner": .string("deep")]),
            "list": .array([.number(1), .bool(false), .null]),
        ])
        let data = try JSONEncoder().encode(value)
        let decoded = try JSONDecoder().decode(JSONValue.self, from: data)
        if case .object(let obj) = decoded,
            case .object(let inner) = obj["nested"],
            case .string(let s) = inner["inner"]
        {
            #expect(s == "deep")
        } else {
            Issue.record("Round-trip failed")
        }
    }

    // MARK: - ResponseFormat

    @Test("extractedSchema returns schema string for json_schema type")
    func responseFormatExtractsSchema() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"test","schema":{"type":"object","properties":{"x":{"type":"integer"}}}}}}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        let schema = request.responseFormat?.extractedSchema
        #expect(schema != nil)
        #expect(schema!.contains("integer"))
    }

    @Test("extractedSchema returns {} for json_object type")
    func responseFormatJsonObject() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_object"}}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.responseFormat?.extractedSchema == "{}")
    }

    @Test("extractedSchema returns nil for text type")
    func responseFormatText() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"text"}}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.responseFormat?.extractedSchema == nil)
    }

    // MARK: - MessageContent

    @Test("textContent extracts text from parts")
    func messageContentTextFromParts() throws {
        let json = """
            {"messages":[{"role":"user","content":[{"type":"text","text":"Hello"},{"type":"text","text":"World"}]}]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.messages[0].content.textContent == "Hello World")
    }

    @Test("imageDataURLs extracts URLs from parts")
    func messageContentImageURLs() throws {
        let json = """
            {"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},{"type":"text","text":"describe"}]}]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.messages[0].content.imageDataURLs == ["data:image/png;base64,AAA"])
    }

    @Test("Empty content string decodes gracefully")
    func emptyContentString() throws {
        let json = """
            {"messages":[{"role":"system","content":""}]}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.messages[0].content.textContent == "")
    }

    // MARK: - ChatCompletionResponse Usage

    @Test("Usage encodes with snake_case keys")
    func usageSnakeCase() throws {
        let response = ChatCompletionResponse(
            id: "coreai-1",
            model: "m",
            choices: [],
            usage: .init(promptTokens: 10, completionTokens: 20, totalTokens: 30)
        )
        let data = try JSONEncoder().encode(response)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let usage = obj["usage"] as! [String: Any]
        #expect(usage["prompt_tokens"] as? Int == 10)
        #expect(usage["completion_tokens"] as? Int == 20)
        #expect(usage["total_tokens"] as? Int == 30)
    }

    // MARK: - max_completion_tokens

    @Test("max_completion_tokens decodes (newer OpenAI field)")
    func maxCompletionTokens() throws {
        let json = """
            {"messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":100}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.maxCompletionTokens == 100)
        #expect(request.maxTokens == nil)
    }

    // MARK: - Invalid Input

    @Test("Invalid prompt type throws DecodingError")
    func invalidPromptThrows() throws {
        let json = """
            {"prompt":{"invalid":"object"}}
            """.data(using: .utf8)!
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(CompletionRequest.self, from: json)
        }
    }
}
