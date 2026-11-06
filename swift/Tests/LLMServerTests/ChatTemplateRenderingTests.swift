// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import llm_server

@Suite("A chat request through a real chat template")
struct ChatTemplateRenderingTests {
    /// Writes tools and each past call's `function` with `tojson` (as Mistral's template does), and
    /// each argument one by one (as Qwen3.5's does).
    private static let template = """
        {%- if tools %}[TOOLS]{{ tools | tojson }}
        {% endif -%}
        {%- for m in messages -%}
        {%- if m.tool_calls -%}
        {%- for tc in m.tool_calls %}[CALL]{{ tc.function | tojson }}{% for k, v in tc.function.arguments|items %}<{{ k }}={{ v }}>{% endfor %}
        {% endfor -%}
        {%- else %}[{{ m.role }}]{{ m.content }}
        {% endif -%}
        {%- endfor -%}
        """

    /// The prompt llm-server sends for `requestJSON`, as text.
    private func render(_ requestJSON: String) async throws -> String {
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(requestJSON.utf8))
        let directory = try ByteLevelTokenizerFixture.directory(chatTemplate: Self.template)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let tokens = tokenizeMessages(
            request.messages, tools: request.tools, tokenizer: tokenizer, toolCallDetection: nil)
        return tokenizer.decode(tokens: tokens)
    }

    @Test("Tools and a past call render as Python's Jinja2 does with the same values")
    func toolsAndPastCall() async throws {
        let request = #"""
            {"model": "m",
             "messages": [
               {"role": "user", "content": "Find files"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": "{\"query\": \"files\", \"limit\": 3}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}],
             "tools": [{"type": "function", "function": {"name": "search", "description": "Find files",
               "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 2.5}}}}}]}
            """#
        // The schema's `minimum` an integer, not 1.0; the call's function name first; its arguments
        // an object, keys sorted as swift-jinja decodes a `Value`. Jinja2 renders exactly this for the
        // same template, given those values.
        #expect(
            try await render(request) == """
                [TOOLS][{"function": {"description": "Find files", "name": "search", "parameters": \
                {"properties": {"limit": {"maximum": 2.5, "minimum": 1, "type": "integer"}}, "type": "object"}}, \
                "type": "function"}]
                [user]Find files
                [CALL]{"name": "search", "arguments": {"limit": 3, "query": "files"}}<limit=3><query=files>[tool]a.txt

                """)
    }
}
