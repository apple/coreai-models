// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Reasoning Effort")
struct ReasoningEffortTests {
    @Test("An absent or empty effort injects no template context")
    func emptyInjectsNothing() {
        #expect(ReasoningEffort.templateContext(nil).isEmpty)
        #expect(ReasoningEffort.templateContext("").isEmpty)
    }

    @Test("none disables thinking")
    func noneDisablesThinking() {
        let context = ReasoningEffort.templateContext("none")
        #expect((context["enable_thinking"] as? Bool) == false)
    }

    @Test("A level binds each reasoning variable")
    func levelBindsVariables() {
        for level in ["low", "medium", "high"] {
            let context = ReasoningEffort.templateContext(level)
            #expect((context["reasoning_effort"] as? String) == level)
            #expect((context["reasoning_strength"] as? String) == level)
            #expect((context["enable_thinking"] as? Bool) == true)
        }
    }

    @Test("A request value takes precedence over the default")
    func precedence() {
        #expect(ReasoningEffort.resolve(request: "high", default: "low") == "high")
        #expect(ReasoningEffort.resolve(request: nil, default: "low") == "low")
        #expect(ReasoningEffort.resolve(request: nil, default: nil) == nil)
    }

    @Test("reasoning_effort decodes from a chat completion request")
    func requestDecodes() throws {
        let json = """
            {"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}
            """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: json)
        #expect(request.reasoningEffort == "high")
    }
}
