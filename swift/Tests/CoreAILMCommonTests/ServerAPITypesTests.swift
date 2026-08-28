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

    // MARK: - Stats Response

    @Test("ServerStatsResponse encodes snake_case keys")
    func statsResponseEncodes() throws {
        let stats = ServerStatsResponse(
            totalRequests: 10, totalPromptTokens: 500, totalGenTokens: 200,
            avgPrefillTokPerSec: 1000.0, avgDecodeTokPerSec: 50.0,
            prefixHitRate: 0.8, prefixHits: 8, prefixMisses: 2,
            totalToolCalls: 3, topTools: [ToolCallStat(name: "get_weather", count: 3)]
        )
        let data = try JSONEncoder().encode(stats)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["total_requests"] as? Int == 10)
        #expect(obj["avg_prefill_tok_per_sec"] as? Double == 1000.0)
        #expect(obj["prefix_hit_rate"] as? Double == 0.8)
        #expect(obj["total_tool_calls"] as? Int == 3)
        let tools = obj["top_tools"] as? [[String: Any]]
        #expect(tools?.count == 1)
        #expect(tools?[0]["name"] as? String == "get_weather")
    }

    @Test("ServerStatsResponse round-trips through JSON")
    func statsResponseRoundTrip() throws {
        let stats = ServerStatsResponse(
            totalRequests: 5, totalPromptTokens: 100, totalGenTokens: 50,
            avgPrefillTokPerSec: 500.0, avgDecodeTokPerSec: 25.0,
            prefixHitRate: 0.6, prefixHits: 3, prefixMisses: 2
        )
        let data = try JSONEncoder().encode(stats)
        let decoded = try JSONDecoder().decode(ServerStatsResponse.self, from: data)
        #expect(decoded.totalRequests == 5)
        #expect(decoded.prefixHitRate == 0.6)
        #expect(decoded.totalToolCalls == 0)
        #expect(decoded.topTools == nil)
    }

    @Test("ServerStatsResponse decodes from external JSON")
    func statsResponseDecodes() throws {
        let json = """
            {"total_requests":3,"total_prompt_tokens":100,"total_gen_tokens":50,
             "avg_prefill_tok_per_sec":500,"avg_decode_tok_per_sec":25,
             "prefix_hit_rate":0.5,"prefix_hits":1,"prefix_misses":1,
             "total_tool_calls":0}
            """.data(using: .utf8)!
        let stats = try JSONDecoder().decode(ServerStatsResponse.self, from: json)
        #expect(stats.totalRequests == 3)
        #expect(stats.avgDecodeTokPerSec == 25.0)
    }

    // MARK: - Ready Response

    @Test("ReadyResponse encodes with nested cache status")
    func readyResponseEncodes() throws {
        let ready = ReadyResponse(
            status: "ready", model: "qwen3-4b", maxContextLength: 32768,
            busy: false,
            cache: .init(usedTokens: 1024, maxTokens: 32768, utilization: 0.03125),
            toolCalling: true
        )
        let data = try JSONEncoder().encode(ready)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        #expect(obj["status"] as? String == "ready")
        #expect(obj["model"] as? String == "qwen3-4b")
        #expect(obj["max_context_length"] as? Int == 32768)
        #expect(obj["busy"] as? Bool == false)
        #expect(obj["tool_calling"] as? Bool == true)
        let cache = obj["cache"] as? [String: Any]
        #expect(cache?["used_tokens"] as? Int == 1024)
        #expect(cache?["max_tokens"] as? Int == 32768)
        #expect(cache?["utilization"] as? Double == 0.03125)
    }

    @Test("ReadyResponse round-trips through JSON")
    func readyResponseRoundTrip() throws {
        let ready = ReadyResponse(
            status: "ready", model: "test-model", maxContextLength: 4096,
            busy: true,
            cache: .init(usedTokens: 2048, maxTokens: 4096, utilization: 0.5),
            toolCalling: false
        )
        let data = try JSONEncoder().encode(ready)
        let decoded = try JSONDecoder().decode(ReadyResponse.self, from: data)
        #expect(decoded.status == "ready")
        #expect(decoded.busy == true)
        #expect(decoded.cache.usedTokens == 2048)
        #expect(decoded.cache.utilization == 0.5)
        #expect(decoded.toolCalling == false)
    }

    @Test("ReadyResponse decodes from external JSON")
    func readyResponseDecodes() throws {
        let json = """
            {"status":"ready","model":"qwen3","max_context_length":8192,
             "busy":false,"cache":{"used_tokens":0,"max_tokens":8192,"utilization":0.0},
             "tool_calling":true}
            """.data(using: .utf8)!
        let ready = try JSONDecoder().decode(ReadyResponse.self, from: json)
        #expect(ready.model == "qwen3")
        #expect(ready.maxContextLength == 8192)
        #expect(ready.cache.usedTokens == 0)
        #expect(ready.toolCalling == true)
    }
}
