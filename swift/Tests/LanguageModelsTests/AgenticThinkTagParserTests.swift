// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("ThinkTagParser — Agentic mode")
struct AgenticThinkTagParserTests {
    private func makeParser() -> ThinkTagParser {
        ThinkTagParser(
            format: .agentic(
                selfMarker: "to=self<|message|>",
                userMarker: "to=user<|message|>",
                endOfMessage: "<|eom|>",
                endOfTurn: "<|eot|>"
            ))
    }

    private func reasoning(_ events: [ThinkTagParser.Event]) -> String {
        events.compactMap { if case .reasoning(let t) = $0 { return t } else { return nil } }.joined()
    }

    private func text(_ events: [ThinkTagParser.Event]) -> String {
        events.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined()
    }

    @Test("First turn: reasoning then user response, no protocol leaks")
    func firstTurnReasoningThenUser() {
        var parser = makeParser()
        let input =
            " to=self<|message|>I need to think.<|eom|><|start|>assistant to=user<|message|>The answer is 4."
        let events = parser.consume(input) + parser.flush()
        #expect(reasoning(events).contains("I need to think"))
        #expect(text(events) == "The answer is 4.")
    }

    @Test("First turn: reasoning then tool call passes ATEM through as text")
    func firstTurnReasoningThenToolCall() {
        var parser = makeParser()
        let atem =
            "<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Tokyo</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
        let input = " to=self<|message|>Need weather.<|eom|><|start|>assistant to=get_weather<|message|>\(atem)"
        let events = parser.consume(input) + parser.flush()
        #expect(reasoning(events) == "Need weather.")
        #expect(text(events).contains("<atem:function_calls>"))
        #expect(text(events).contains("get_weather"))
    }

    @Test("Continuation turn: bare eom then reasoning then user response")
    func continuationTurn() {
        var parser = makeParser()
        let input =
            "<|eom|><|start|>assistant to=self<|message|>Got weather data.<|eom|><|start|>assistant to=user<|message|>It's 28C in Tokyo."
        let events = parser.consume(input) + parser.flush()
        #expect(reasoning(events) == "Got weather data.")
        #expect(text(events) == "It's 28C in Tokyo.")
    }

    @Test("Streaming: routing header split across consume calls")
    func streamingSplitHeader() {
        var parser = makeParser()
        var events: [ThinkTagParser.Event] = []
        events += parser.consume(" to=self<|message|>think<|eom|>")
        events += parser.consume("<|start|>")
        events += parser.consume("assistant ")
        events += parser.consume("to=user")
        events += parser.consume("<|message|>")
        events += parser.consume("done")
        events += parser.flush()
        #expect(text(events) == "done")
    }

    @Test("Streaming: partial header held back, not leaked")
    func streamingNoPartialLeak() {
        var parser = makeParser()
        var events: [ThinkTagParser.Event] = []
        events += parser.consume(" to=self<|message|>R<|eom|>")
        events += parser.consume("<|sta")
        events += parser.consume("rt|>assistan")
        events += parser.consume("t to=user<|message|>OK")
        events += parser.flush()
        #expect(text(events) == "OK")
    }

    @Test("Generic tool target routes as text")
    func genericToolTarget() {
        var parser = makeParser()
        let input =
            " to=self<|message|>plan<|eom|><|start|>assistant to=search_web<|message|>tool content here"
        let events = parser.consume(input) + parser.flush()
        #expect(text(events) == "tool content here")
    }

    @Test("stripReasoning extracts only user-facing text")
    func stripReasoningStaticMethod() {
        let format = ThinkTagParser.Format.agentic(
            selfMarker: "to=self<|message|>",
            userMarker: "to=user<|message|>",
            endOfMessage: "<|eom|>",
            endOfTurn: "<|eot|>"
        )
        let raw =
            " to=self<|message|>internal thoughts<|eom|><|start|>assistant to=user<|message|>Hello!"
        let result = ThinkTagParser.stripReasoning(from: raw, format: format)
        #expect(result == "Hello!")
    }
}
