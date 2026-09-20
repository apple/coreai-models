// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import Testing

/// Coverage for the shared replay JSONL format: request decoding (reusing
/// `ChatCompletionRequest`), multi-line parsing with ordering, and result encoding.
struct ReplayTypesTests {
    @Test("A replay request line decodes, wrapping a ChatCompletionRequest")
    func requestDecodes() throws {
        let line = #"""
            {"id":"r1","session":"A","t":1.5,"request":{"messages":[{"role":"user","content":"hi"}],"seed":7,"max_tokens":32}}
            """#
        let req = try JSONDecoder().decode(ReplayRequest.self, from: Data(line.utf8))
        #expect(req.id == "r1")
        #expect(req.session == "A")
        #expect(req.t == 1.5)
        #expect(req.request.seed == 7)
        #expect(req.request.maxTokens == 32)
        #expect(req.request.messages.count == 1)
    }

    @Test("parseRequests skips blanks and comments and orders by timestamp")
    func parseOrdersAndSkips() throws {
        let jsonl = """
            # a comment
            {"session":"B","t":2.0,"request":{"messages":[{"role":"user","content":"second"}]}}

            {"session":"A","t":1.0,"request":{"messages":[{"role":"user","content":"first"}]}}
            """
        let reqs = try ReplayIO.parseRequests(jsonl)
        #expect(reqs.count == 2)
        #expect(reqs[0].session == "A")  // t=1.0 sorts before t=2.0
        #expect(reqs[1].session == "B")
    }

    @Test("Requests with equal or absent timestamps keep input order")
    func stableOrderOnTies() throws {
        let jsonl = """
            {"id":"x","request":{"messages":[{"role":"user","content":"a"}]}}
            {"id":"y","request":{"messages":[{"role":"user","content":"b"}]}}
            """
        let reqs = try ReplayIO.parseRequests(jsonl)
        #expect(reqs.map(\.id) == ["x", "y"])
    }

    @Test("A result encodes with snake_case instrumentation keys")
    func resultEncodes() throws {
        let result = ReplayResult(
            id: "r1", session: "A", t: 0,
            content: "hello", finishReason: "stop", systemFingerprint: "fp_z",
            promptTokens: 3, completionTokens: 2, prefixReuseTokens: 1,
            ttftMs: 12.5, totalMs: 30.0, decodeTps: 66.6)
        let json = try ReplayIO.encodeResult(result)
        #expect(json.contains(#""finish_reason":"stop""#))
        #expect(json.contains(#""system_fingerprint":"fp_z""#))
        #expect(json.contains(#""prefix_reuse_tokens":1"#))
        #expect(json.contains(#""ttft_ms":12.5"#))
        #expect(!json.contains("\n"))  // single JSONL line
    }

    @Test("An error result omits content but records the error")
    func errorResultEncodes() throws {
        let result = ReplayResult(
            id: nil, session: "A", t: nil,
            content: nil, finishReason: nil, systemFingerprint: nil,
            promptTokens: 0, completionTokens: 0, prefixReuseTokens: 0,
            ttftMs: 0, totalMs: 0, decodeTps: 0, error: "boom")
        let json = try ReplayIO.encodeResult(result)
        #expect(json.contains(#""error":"boom""#))
    }

    @Test("A result carries reasoning_content, tool_calls, and streamed when present")
    func resultEncodesToolCallsAndReasoning() throws {
        let result = ReplayResult(
            id: "r1", session: "A", t: 0,
            content: nil, finishReason: "tool_calls", systemFingerprint: nil,
            promptTokens: 5, completionTokens: 4, prefixReuseTokens: 0,
            ttftMs: 1, totalMs: 2, decodeTps: 3,
            reasoningContent: "let me think",
            toolCalls: [ToolCall(id: "c1", function: .init(name: "get_weather", arguments: #"{"city":"SF"}"#))],
            streamed: true)
        let json = try ReplayIO.encodeResult(result)
        #expect(json.contains(#""reasoning_content":"let me think""#))
        #expect(json.contains(#""finish_reason":"tool_calls""#))
        #expect(json.contains(#""streamed":true"#))
        #expect(json.contains(#""name":"get_weather""#))
        #expect(json.contains(#""tool_calls""#))
    }

    @Test("isStreaming reflects the request's stream field")
    func isStreamingReflectsRequest() throws {
        let streaming = #"{"request":{"messages":[{"role":"user","content":"hi"}],"stream":true}}"#
        let nonStreaming = #"{"request":{"messages":[{"role":"user","content":"hi"}],"stream":false}}"#
        let absent = #"{"request":{"messages":[{"role":"user","content":"hi"}]}}"#
        #expect(try JSONDecoder().decode(ReplayRequest.self, from: Data(streaming.utf8)).isStreaming)
        #expect(!(try JSONDecoder().decode(ReplayRequest.self, from: Data(nonStreaming.utf8)).isStreaming))
        #expect(!(try JSONDecoder().decode(ReplayRequest.self, from: Data(absent.utf8)).isStreaming))
    }

    @Test("ReplayStreamAggregator folds content, reasoning, and tool-call deltas")
    func aggregatorFoldsDeltas() {
        var agg = ReplayStreamAggregator()
        agg.consume(.init(role: nil, content: nil, reasoningContent: "think"))
        agg.consume(.init(role: nil, content: "Hello, "))
        agg.consume(.init(role: nil, content: "world"))
        agg.consume(
            .init(
                role: nil, content: nil,
                toolCalls: [
                    ToolCallDelta(index: 0, id: "c1", type: "function", function: .init(name: "f", arguments: "{}"))
                ]))
        #expect(agg.aggregatedContent == "Hello, world")
        #expect(agg.aggregatedReasoning == "think")
        #expect(agg.aggregatedToolCalls?.count == 1)
        #expect(agg.aggregatedToolCalls?.first?.id == "c1")
        #expect(agg.aggregatedToolCalls?.first?.function.name == "f")
    }

    @Test("An empty aggregator yields nil aggregates")
    func aggregatorEmpty() {
        let agg = ReplayStreamAggregator()
        #expect(agg.aggregatedContent == nil)
        #expect(agg.aggregatedReasoning == nil)
        #expect(agg.aggregatedToolCalls == nil)
    }
}
