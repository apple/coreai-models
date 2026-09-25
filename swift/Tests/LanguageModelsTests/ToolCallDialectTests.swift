// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

private func firstToolCall(_ events: [ToolCallParser.Event]) -> (name: String, args: String)? {
    for case .toolCall(_, let name, let args) in events { return (name, args) }
    return nil
}

private func runParser(_ input: String, open: String, close: String, format: ToolCallParser.Format)
    -> [ToolCallParser.Event]
{
    var parser = ToolCallParser(openMarker: open, closeMarker: close, format: format)
    return parser.consume(input) + parser.flush()
}

@Suite("Tool-call dialects: Phi + Qwen3-Coder")
struct ToolCallDialectTests {
    // Phi emits JSON inside <|tool_call|> markers, which the detector must now recognize.
    @Test("detects the <|tool_call|> marker family (Phi)")
    func detectsPhiMarkers() {
        let tok = MockTokenizer(vocab: ["<|tool_call|>": 1, "<|/tool_call|>": 2])
        let d = detectToolCallFormat(using: tok)
        #expect(d?.openMarker == "<|tool_call|>")
        #expect(d?.closeMarker == "<|/tool_call|>")
        if case .json = d?.format {} else { Issue.record("expected .json format for Phi markers") }
        // Phi renders tools from the system message.
        #expect(d?.toolsInSystemMessage == true)
    }

    // Existing JSON <tool_call> detection must be unchanged (Qwen3).
    @Test("still detects <tool_call> as JSON (Qwen3 unaffected)")
    func detectsJSONMarkers() {
        let tok = MockTokenizer(vocab: ["<tool_call>": 1, "</tool_call>": 2])
        let d = detectToolCallFormat(using: tok)
        #expect(d?.openMarker == "<tool_call>")
        // Qwen3 uses top-level tools, not a system-message key.
        #expect(d?.toolsInSystemMessage == false)
    }

    // Tool injection is scoped to the Phi dialect via applyToolsToSystemMessage.
    @Test("applyToolsToSystemMessage attaches to an existing system message")
    func toolsAttachToExistingSystem() {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "hi"],
            ["role": "user", "content": "q"],
        ]
        let out = applyToolsToSystemMessage(messages, toolsJSON: "[TOOLS]")
        #expect(out.count == 2)
        #expect((out[0]["role"] as? String) == "system")
        #expect((out[0]["tools"] as? String) == "[TOOLS]")
        #expect((out[0]["content"] as? String) == "hi")
    }

    @Test("applyToolsToSystemMessage synthesizes a system message when none exists")
    func toolsSynthesizeSystem() {
        let messages: [[String: any Sendable]] = [["role": "user", "content": "q"]]
        let out = applyToolsToSystemMessage(messages, toolsJSON: "[TOOLS]")
        #expect(out.count == 2)
        #expect((out[0]["role"] as? String) == "system")
        #expect((out[0]["tools"] as? String) == "[TOOLS]")
        #expect((out[1]["role"] as? String) == "user")
    }

    // Shared guard: injects only when specs present and the dialect reads system-message tools.
    @Test("injectToolsIntoSystemMessageIfNeeded injects when flag + specs present")
    func injectHelperInjects() {
        let messages: [[String: any Sendable]] = [["role": "user", "content": "q"]]
        let specs: [[String: any Sendable]] = [
            ["type": "function", "function": ["name": "add", "description": "adds"]]
        ]
        let detection = ToolCallDetection(
            openMarker: "<|tool_call|>", closeMarker: "<|/tool_call|>", format: .json,
            toolsInSystemMessage: true)
        let out = injectToolsIntoSystemMessageIfNeeded(messages, toolSpecs: specs, detection: detection)
        #expect(out.count == 2)
        #expect((out[0]["role"] as? String) == "system")
        #expect((out[0]["tools"] as? String) == toolsJSONForSystemMessage(specs))
    }

    @Test("injectToolsIntoSystemMessageIfNeeded is a no-op when flag false or specs nil")
    func injectHelperNoOp() {
        let messages: [[String: any Sendable]] = [["role": "user", "content": "q"]]
        let specs: [[String: any Sendable]] = [["type": "function", "function": ["name": "add"]]]
        // Flag false (Qwen3-style): untouched.
        let qwen = ToolCallDetection(
            openMarker: "<tool_call>", closeMarker: "</tool_call>", format: .json,
            toolsInSystemMessage: false)
        let outFlagFalse = injectToolsIntoSystemMessageIfNeeded(messages, toolSpecs: specs, detection: qwen)
        #expect(outFlagFalse.count == 1)
        #expect(outFlagFalse.first?["tools"] == nil)
        // No specs: untouched even when the flag is set.
        let phi = ToolCallDetection(
            openMarker: "<|tool_call|>", closeMarker: "<|/tool_call|>", format: .json,
            toolsInSystemMessage: true)
        let outNoSpecs = injectToolsIntoSystemMessageIfNeeded(messages, toolSpecs: nil, detection: phi)
        #expect(outNoSpecs.count == 1)
        #expect(outNoSpecs.first?["tools"] == nil)
        // No detection: untouched.
        let outNoDetection = injectToolsIntoSystemMessageIfNeeded(messages, toolSpecs: specs, detection: nil)
        #expect(outNoDetection.count == 1)
    }

    // Qwen3-Coder XML function body via the explicit .xmlFunction format.
    @Test("parses Qwen3-Coder <function=…> XML (explicit format)")
    func parsesXMLFunctionExplicit() {
        let input =
            "<tool_call><function=add>"
            + "<parameter=a>1</parameter><parameter=b>2</parameter>"
            + "</function></tool_call>"
        let call = firstToolCall(runParser(input, open: "<tool_call>", close: "</tool_call>", format: .xmlFunction))
        #expect(call?.name == "add")
        #expect(call?.args == #"{"a":1,"b":2}"#)
    }

    // XML params must coerce to their JSON-schema types, not stringify.
    @Test("XML function params coerce numeric/bool/array (not stringified)")
    func xmlFunctionParamCoercion() {
        let input =
            "<tool_call><function=configure>"
            + "<parameter=count>5</parameter>"
            + "<parameter=ratio>1.5</parameter>"
            + "<parameter=enabled>true</parameter>"
            + "<parameter=tags>[\"a\",\"b\"]</parameter>"
            + "</function></tool_call>"
        let call = firstToolCall(runParser(input, open: "<tool_call>", close: "</tool_call>", format: .xmlFunction))
        #expect(call?.args == #"{"count":5,"enabled":true,"ratio":1.5,"tags":["a","b"]}"#)
    }

    // JSON-first routing: a JSON call whose argument text contains `<function=` must parse as
    // JSON, not be misrouted to the XML fallback.
    @Test("JSON call with <function= in arg text parses as JSON (not misrouted)")
    func jsonArgContainingFunctionTagNotMisrouted() {
        let input = #"<tool_call>{"name":"write","arguments":{"code":"<function=foo>"}}</tool_call>"#
        let call = firstToolCall(runParser(input, open: "<tool_call>", close: "</tool_call>", format: .json))
        #expect(call?.name == "write")
        #expect(call?.args == #"{"code":"<function=foo>"}"#)
    }

    // Qwen3-Coder shares Qwen3's <tool_call> markers, so it is detected as .json; the JSON
    // parser must fall back to XML rather than dropping the call.
    @Test("JSON path falls back to XML for <function=…> bodies (Qwen3-Coder)")
    func jsonFallsBackToXML() {
        let input = "<tool_call><function=lookup><parameter=q>swift</parameter></function></tool_call>"
        let call = firstToolCall(runParser(input, open: "<tool_call>", close: "</tool_call>", format: .json))
        #expect(call?.name == "lookup")
        #expect(call?.args == #"{"q":"swift"}"#)
    }

    // The plain JSON body must still parse (Qwen3 regression guard).
    @Test("JSON <tool_call>{…} still parses (Qwen3 regression)")
    func jsonStillParses() {
        let input = #"<tool_call>{"name":"add","arguments":{"a":1}}</tool_call>"#
        let call = firstToolCall(runParser(input, open: "<tool_call>", close: "</tool_call>", format: .json))
        #expect(call?.name == "add")
        #expect(call?.args == #"{"a":1}"#)
    }

    // toolsJSONForSystemMessage serializes with sorted keys so server and FM paths match.
    @Test("toolsJSONForSystemMessage serializes tool specs with sorted keys")
    func toolsJSONSortedKeys() {
        let specs: [[String: any Sendable]] = [
            ["type": "function", "function": ["name": "add", "description": "adds"]]
        ]
        let json = toolsJSONForSystemMessage(specs)
        #expect(json == #"[{"function":{"description":"adds","name":"add"},"type":"function"}]"#)
    }

    // FM path: a Phi-style detection folds tools into a synthesized system message.
    @Test("FM makeTokens injects tools into the system message for Phi")
    func fmInjectsToolsForPhi() {
        let tok = CapturingTokenizer()
        let detection = ToolCallDetection(
            openMarker: "<|tool_call|>", closeMarker: "<|/tool_call|>", format: .json,
            toolsInSystemMessage: true)
        _ = CoreAILanguageModel.CoreAIExecutor.makeTokens(
            from: [.prompt(makePrompt("hi"))],
            using: tok,
            tools: [makeToolDef()],
            toolCallDetection: detection)
        let system = tok.capturedMessages.first { ($0["role"] as? String) == "system" }
        #expect(system != nil)
        #expect((system?["tools"] as? String) == toolsJSONForSystemMessage(tok.capturedTools ?? []))
        #expect(tok.capturedTools?.isEmpty == false)
    }

    // FM path: a Qwen3-style detection leaves tools at the top level only.
    @Test("FM makeTokens keeps tools top-level for Qwen3")
    func fmTopLevelForQwen3() {
        let tok = CapturingTokenizer()
        let detection = ToolCallDetection(
            openMarker: "<tool_call>", closeMarker: "</tool_call>", format: .json,
            toolsInSystemMessage: false)
        _ = CoreAILanguageModel.CoreAIExecutor.makeTokens(
            from: [.prompt(makePrompt("hi"))],
            using: tok,
            tools: [makeToolDef()],
            toolCallDetection: detection)
        let system = tok.capturedMessages.first { ($0["role"] as? String) == "system" }
        #expect(system == nil)
        #expect(tok.capturedTools?.isEmpty == false)
    }

    private func makePrompt(_ text: String) -> Transcript.Prompt {
        Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: text))])
    }

    private func makeToolDef() -> Transcript.ToolDefinition {
        let schema = try! GenerationSchema(
            root: DynamicGenerationSchema(name: "args", properties: []), dependencies: [])
        return Transcript.ToolDefinition(name: "lookup", description: "look things up", parameters: schema)
    }
}
