// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Testing
import Tokenizers

@testable import CoreAILanguageModels

/// Records the messages/tools passed to `applyChatTemplate` so the FM-path tool
/// injection can be inspected. A class so the nonmutating protocol method can store.
private final class CapturingTokenizer: Tokenizer, @unchecked Sendable {
    var capturedMessages: [Message] = []
    var capturedTools: [ToolSpec]?
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func encode(text: String) -> [Int] { [] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokens: [Int]) -> String { "" }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func tokenize(text: String) -> [String] { [] }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { [] }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { [] }
    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        capturedMessages = messages
        capturedTools = tools
        return [1]
    }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws
        -> [Int]
    { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool,
        maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool,
        maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(messages: [[String: String]]) throws -> [Int] { [] }
}

/// Minimal tokenizer whose `vocabContains` is exact (round-trips only the known tokens).
private struct DialectTokenizer: Tokenizer {
    let known: [String: Int]
    let reverse: [Int: String]
    init(_ tokens: [String: Int]) {
        known = tokens
        reverse = Dictionary(uniqueKeysWithValues: tokens.map { ($0.value, $0.key) })
    }
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    func convertTokenToId(_ token: String) -> Int? { known[token] }
    func convertIdToToken(_ id: Int) -> String? { reverse[id] }
    func encode(text: String) -> [Int] { [] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokens: [Int]) -> String { "" }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func tokenize(text: String) -> [String] { [] }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { convertTokenToId($0) } }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { convertIdToToken($0) } }
    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws
        -> [Int]
    { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool,
        maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool,
        maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(messages: [[String: String]]) throws -> [Int] { [] }
}

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
        let tok = DialectTokenizer(["<|tool_call|>": 1, "<|/tool_call|>": 2])
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
        let tok = DialectTokenizer(["<tool_call>": 1, "</tool_call>": 2])
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

    // Content-based routing: a .json-declared parser (Qwen3-Coder) maps <function=…> to .xmlFunction.
    @Test("effectiveToolCallFormat routes <function=…> bodies to .xmlFunction")
    func effectiveFormatRoutesXML() {
        #expect(effectiveToolCallFormat(declared: .json, body: "<function=add></function>") == .xmlFunction)
        #expect(effectiveToolCallFormat(declared: .json, body: #"{"name":"add"}"#) == .json)
        #expect(effectiveToolCallFormat(declared: .atem, body: "<function=add>") == .atem)
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
        #expect((system?["tools"] as? String)?.contains("lookup") == true)
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
