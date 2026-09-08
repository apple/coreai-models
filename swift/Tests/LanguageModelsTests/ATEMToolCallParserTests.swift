// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("ToolCallParser — ATEM format")
struct ATEMToolCallParserTests {
    private func makeParser() -> ToolCallParser {
        ToolCallParser(
            openMarker: "<atem:function_calls>",
            closeMarker: "</atem:function_calls>",
            format: .atem
        )
    }

    @Test("Single invoke with one parameter")
    func singleInvoke() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="get_weather">
            <atem:parameter name="city">Tokyo</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(parseJSON(calls[0].argsJSON)["city"] as? String == "Tokyo")
    }

    @Test("Multiple parameters")
    func multipleParams() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="search_flights">
            <atem:parameter name="origin">SFO</atem:parameter>
            <atem:parameter name="destination">NRT</atem:parameter>
            <atem:parameter name="date">2026-09-15</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 1)
        let args = parseJSON(calls[0].argsJSON)
        #expect(args["origin"] as? String == "SFO")
        #expect(args["destination"] as? String == "NRT")
        #expect(args["date"] as? String == "2026-09-15")
    }

    @Test("Multiple invocations in one block")
    func multipleInvocations() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="get_weather">
            <atem:parameter name="city">Tokyo</atem:parameter>
            </atem:invoke>
            <atem:invoke name="get_time">
            <atem:parameter name="timezone">Asia/Tokyo</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 2)
        #expect(calls[0].name == "get_weather")
        #expect(calls[1].name == "get_time")
    }

    @Test("Value coercion: int, double, bool true, bool false")
    func valueCoercion() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="configure">
            <atem:parameter name="count">72</atem:parameter>
            <atem:parameter name="ratio">45.5</atem:parameter>
            <atem:parameter name="enabled">true</atem:parameter>
            <atem:parameter name="verbose">false</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let args = parseJSON(extractToolCalls(events)[0].argsJSON)
        #expect(args["count"] as? Int == 72)
        #expect(args["ratio"] as? Double == 45.5)
        #expect(args["enabled"] as? Bool == true)
        #expect(args["verbose"] as? Bool == false)
    }

    @Test("No parameters yields empty args")
    func noParameters() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="ping">
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].argsJSON == "{}")
    }

    @Test("Text before tool call block emitted separately")
    func textBeforeToolCall() {
        var parser = makeParser()
        let input = """
            Let me check.<atem:function_calls>
            <atem:invoke name="get_weather">
            <atem:parameter name="city">Tokyo</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let textParts = events.compactMap { if case .text(let t) = $0 { return t } else { return nil } }
        #expect(textParts.joined().contains("Let me check"))
        #expect(extractToolCalls(events).count == 1)
    }

    @Test("Streaming: markers split across consume calls")
    func streamingSplitMarkers() {
        var parser = makeParser()
        var events: [ToolCallParser.Event] = []
        events += parser.consume("text<atem:func")
        events += parser.consume("tion_calls>\n<atem:invoke name=\"get_weather\">\n")
        events += parser.consume("<atem:parameter name=\"city\">Tokyo</atem:para")
        events += parser.consume("meter>\n</atem:invoke>\n</atem:function_calls>")
        events += parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 1)
        #expect(parseJSON(calls[0].argsJSON)["city"] as? String == "Tokyo")
    }

    @Test("Each tool call gets a unique ID")
    func uniqueCallIds() {
        var parser = makeParser()
        let input = """
            <atem:function_calls>
            <atem:invoke name="a"><atem:parameter name="x">1</atem:parameter></atem:invoke>
            <atem:invoke name="b"><atem:parameter name="x">2</atem:parameter></atem:invoke>
            </atem:function_calls>
            """
        let events = parser.consume(input) + parser.flush()
        let ids = extractToolCalls(events).map(\.id)
        #expect(ids.count == 2)
        #expect(ids[0] != ids[1])
    }

    @Test("JSON format unaffected by ATEM addition")
    func jsonFormatRegression() {
        var parser = ToolCallParser(format: .json)
        let input = #"<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>"#
        let events = parser.consume(input) + parser.flush()
        let calls = extractToolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
    }

    // MARK: - Helpers

    private struct ToolCallInfo {
        let id: String
        let name: String
        let argsJSON: String
    }

    private func extractToolCalls(_ events: [ToolCallParser.Event]) -> [ToolCallInfo] {
        events.compactMap { event in
            if case .toolCall(let id, let name, let argsJSON) = event {
                return ToolCallInfo(id: id, name: name, argsJSON: argsJSON)
            }
            return nil
        }
    }

    private func parseJSON(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return obj
    }
}
