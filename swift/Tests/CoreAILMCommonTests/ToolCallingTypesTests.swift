// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Tool Calling Types")
struct ToolCallingTypesTests {
    @Test("Request with tools decodes")
    func requestWithTools() throws {
        let json = """
            {
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                    }
                }],
                "tool_choice": "auto"
            }
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.tools?.count == 1)
        #expect(request.tools?[0].function.name == "get_weather")
        #expect(request.toolChoice == .auto)
    }

    @Test("ToolChoice decodes all variants")
    func toolChoiceVariants() throws {
        let cases: [(String, ToolChoice)] = [
            ("\"auto\"", .auto),
            ("\"none\"", .none),
            ("\"required\"", .required),
            ("{\"type\":\"function\",\"function\":{\"name\":\"foo\"}}", .function(name: "foo")),
        ]
        for (json, expected) in cases {
            let decoded = try JSONDecoder().decode(ToolChoice.self, from: json.data(using: .utf8)!)
            #expect(decoded == expected)
        }
    }

    @Test("ToolChoice round-trips through encode/decode")
    func toolChoiceRoundTrip() throws {
        let choices: [ToolChoice] = [.auto, .none, .required, .function(name: "bar")]
        for choice in choices {
            let data = try JSONEncoder().encode(choice)
            let decoded = try JSONDecoder().decode(ToolChoice.self, from: data)
            #expect(decoded == choice)
        }
    }

    @Test("Assistant message with tool_calls decodes")
    func messageWithToolCalls() throws {
        let json = """
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\\"location\\":\\"SF\\"}"}
                }]
            }
            """.data(using: .utf8)!
        let msg = try JSONDecoder().decode(ChatMessage.self, from: json)
        #expect(msg.role == "assistant")
        #expect(msg.toolCalls?.count == 1)
        #expect(msg.toolCalls?[0].function.name == "get_weather")
        #expect(msg.toolCalls?[0].id == "call_abc123")
    }

    @Test("Tool result message decodes")
    func toolResultMessage() throws {
        let json = """
            {"role": "tool", "content": "72 degrees", "tool_call_id": "call_abc123"}
            """.data(using: .utf8)!
        let msg = try JSONDecoder().decode(ChatMessage.self, from: json)
        #expect(msg.role == "tool")
        #expect(msg.content.textContent == "72 degrees")
        #expect(msg.toolCallId == "call_abc123")
    }

    @Test("ResponseMessage encodes null content explicitly")
    func responseMessageNullContent() throws {
        let msg = ChatCompletionResponse.ResponseMessage(
            role: "assistant", content: nil,
            toolCalls: [ToolCall(id: "call_1", function: .init(name: "foo", arguments: "{}"))])
        let data = try JSONEncoder().encode(msg)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj.keys.contains("content"))
        #expect(obj["content"] is NSNull)
        let calls = obj["tool_calls"] as? [[String: Any]]
        #expect(calls?.count == 1)
    }

    @Test("Streaming delta with tool_calls encodes")
    func streamingToolCallDelta() throws {
        let delta = ChatCompletionChunk.Delta(
            role: nil, content: nil,
            toolCalls: [
                ToolCallDelta(
                    index: 0, id: "call_1", type: "function",
                    function: .init(name: "get_weather", arguments: nil))
            ])
        let data = try JSONEncoder().encode(delta)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let calls = obj["tool_calls"] as? [[String: Any]]
        #expect(calls?.count == 1)
        #expect(calls?[0]["index"] as? Int == 0)
    }
}
