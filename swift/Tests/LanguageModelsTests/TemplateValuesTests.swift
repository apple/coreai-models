// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Jinja
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("Past tool calls as chat templates take them")
struct TemplateValuesTests {
    // MARK: Past tool calls through each kind of template

    /// Renders `template` over one assistant message holding a past call with `arguments`, built
    /// and converted as llm-server and the tokenizer do.
    private func renderPastCall(_ template: String, arguments: String) throws -> String {
        let message: [String: any Sendable] = [
            "role": "assistant",
            "tool_calls": [TemplateValues.toolCall(id: "c1", name: "search", arguments: arguments)],
        ]
        return try Template(template).render(["m": try Value(any: message)])
    }

    private static let pastArguments = #"{"query": "files", "limit": 3}"#

    @Test("A template that writes each argument gets them all (Qwen3.5)")
    func itemsTemplate() throws {
        let template = "{% for k, v in m.tool_calls[0].function.arguments|items %}<{{ k }}={{ v }}>{% endfor %}"
        #expect(try renderPastCall(template, arguments: Self.pastArguments) == "<limit=3><query=files>")
    }

    @Test("A template that refuses anything but a mapping renders the call")
    func mappingOnlyTemplate() throws {
        let template = """
            {%- set a = m.tool_calls[0].function.arguments -%}
            {%- if a is not mapping -%}{{ raise_exception('arguments must be a dict') }}{%- endif -%}
            {%- for k, v in a.items() -%}<{{ k }}={{ v }}>{%- endfor -%}
            """
        #expect(try renderPastCall(template, arguments: Self.pastArguments) == "<limit=3><query=files>")
    }

    @Test("A template that writes the arguments with tojson writes them once, not as a string (Qwen2.5)")
    func tojsonTemplate() throws {
        let template = "{{ m.tool_calls[0].function.arguments | tojson }}"
        #expect(try renderPastCall(template, arguments: Self.pastArguments) == #"{"limit": 3, "query": "files"}"#)
    }

    @Test("A template that also takes a string writes the object as json.dumps does (Qwen3), keys sorted")
    func stringOrTojsonTemplate() throws {
        let template = """
            {%- set a = m.tool_calls[0].function.arguments -%}
            {%- if a is string -%}{{ a }}{%- else -%}{{ a | tojson }}{%- endif -%}
            """
        #expect(
            try renderPastCall(template, arguments: #"{"query":"files","limit":3}"#)
                == #"{"limit": 3, "query": "files"}"#)
    }

    @Test("Arguments that are JSON but not an object are that value, as json.loads reads them")
    func nonObjectArgumentsAreTheirValue() throws {
        let template = "{{ m.tool_calls[0].function.arguments | tojson }}"
        #expect(try renderPastCall(template, arguments: "[1, 2]") == "[1, 2]")
        #expect(try renderPastCall(template, arguments: "5") == "5")
    }

    @Test("Arguments that are not JSON reach the template as the string they came as")
    func nonJSONArgumentsStayAString() throws {
        let template = "{{ m.tool_calls[0].function.arguments | tojson }}"
        #expect(try renderPastCall(template, arguments: "not json") == #""not json""#)
    }

    @Test("No arguments at all are an empty object, which a mapping-only template accepts")
    func emptyArgumentsAreAnEmptyObject() throws {
        let template = """
            {%- set a = m.tool_calls[0].function.arguments -%}
            {%- if a is not mapping -%}{{ raise_exception('arguments must be a dict') }}{%- endif -%}
            {{- a | tojson -}}
            """
        for empty in ["", " ", "\n"] {
            #expect(try renderPastCall(template, arguments: empty) == "{}")
        }
    }

    @Test("The library's makeTokens passes a past call's arguments to the template as an object")
    func makeTokensPassesAnObject() throws {
        let tokenizer = CapturingTokenizer()
        let call = Transcript.ToolCall(
            id: "c1", toolName: "lookup", arguments: try GeneratedContent(json: #"{"q":"swift"}"#))
        _ = CoreAILanguageModel.CoreAIExecutor.makeTokens(
            from: [.toolCalls(Transcript.ToolCalls([call]))], using: tokenizer)

        let assistant = try #require(tokenizer.capturedMessages.first { ($0["role"] as? String) == "assistant" })
        let calls = try #require(assistant["tool_calls"] as? [any Sendable])
        guard case .object(let call)? = calls.first as? Value, case .object(let function)? = call["function"] else {
            Issue.record("the past call is not an object")
            return
        }
        #expect(function["name"] == .string("lookup"))
        #expect(function["arguments"] == .object(["q": .string("swift")] as OrderedDictionary<String, Value>))
    }

    @Test("makeTokens renders a past call through a real template, the same way every time")
    func makeTokensRendersThroughATemplate() async throws {
        let template = """
            {%- for m in messages -%}
            {%- for tc in m.tool_calls or [] %}{{ tc.function | tojson }}{% for k, v in tc.function.arguments|items %}<{{ k }}>{% endfor %}{% endfor -%}
            {%- endfor -%}
            """
        let directory = try ByteLevelTokenizerFixture.directory(chatTemplate: template)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        var renderings = Set<String>()
        for _ in 0..<20 {
            // A fresh GeneratedContent each time, whose own key order varies from one to the next.
            let arguments = try GeneratedContent(json: #"{"zeta":"z","alpha":2,"mid":{"b":true,"a":null}}"#)
            let call = Transcript.ToolCall(id: "c1", toolName: "lookup", arguments: arguments)
            let tokens = CoreAILanguageModel.CoreAIExecutor.makeTokens(
                from: [.toolCalls(Transcript.ToolCalls([call]))], using: tokenizer)
            renderings.insert(tokenizer.decode(tokens: tokens))
        }
        #expect(
            renderings == [
                #"{"name": "lookup", "arguments": {"alpha": 2, "mid": {"a": null, "b": true}, "zeta": "z"}}"#
                    + "<alpha><mid><zeta>"
            ])
    }

    @Test("A past call's function is written name first, as Mistral's template tojsons it")
    func toolCallEnvelopeOrder() throws {
        let template = "{{ m.tool_calls[0] | tojson }}"
        #expect(
            try renderPastCall(template, arguments: Self.pastArguments)
                == #"{"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"limit": 3, "query": "files"}}}"#
        )
    }

    @Test("Arguments nested deeper than the cap stay a string rather than recursing that deep")
    func deepArgumentsStayAString() {
        func nested(_ depth: Int) -> String {
            #"{"a":"# + String(repeating: "[", count: depth - 1) + String(repeating: "]", count: depth - 1) + "}"
        }
        #expect(TemplateValues.arguments(nested(TemplateValues.maxArgumentDepth)) != nil)
        #expect(TemplateValues.arguments(nested(TemplateValues.maxArgumentDepth + 1)) == nil)
        #expect(TemplateValues.arguments(nested(200_000)) == nil)
        // Brackets inside strings are not nesting.
        #expect(TemplateValues.nestingDepth(of: #"{"a":"[[[{\"[","b":[1]}"#) == 2)
    }

    // swift-jinja 2.5.1 or later, which these guard: an earlier release would break both.

    @Test("The same tools render the same way every time, as json.dumps writes them")
    func toolsRenderTheSameEveryTime() throws {
        let template = try Template("{{ tool | tojson }}")
        var renderings = Set<String>()
        for _ in 0..<20 {
            // Built fresh each time, as llm-server builds each request's tools.
            let tool: [String: Any] = [
                "type": "function",
                "function": [
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": [
                        "type": "object", "properties": ["path": ["type": "string"]],
                        "required": ["path"],
                    ],
                ],
            ]
            renderings.insert(try template.render(["tool": Value(any: tool)]))
        }
        #expect(renderings.count == 1)
        #expect(
            renderings.first
                == #"{"function": {"description": "Write a file", "name": "write_file", "parameters": {"properties": {"path": {"type": "string"}}, "required": ["path"], "type": "object"}}, "type": "function"}"#
        )
    }

    @Test("A tool result opens its turn, which needs loop.previtem (Qwen3.5's template)")
    func loopKnowsThePreviousItem() throws {
        let template = try Template(
            "{% for m in messages %}{% if loop.previtem and loop.previtem.role != 'tool' %}<open>{% endif %}"
                + "{{ m.role }}{% if not loop.last and loop.nextitem.role != 'tool' %}<close>{% endif %}{% endfor %}"
        )
        let messages: [Value] = ["assistant", "tool", "tool", "user"].map { .object(["role": .string($0)]) }
        #expect(
            try template.render(["messages": .array(messages)]) == "assistant<open>tooltool<close>user")
    }
}
