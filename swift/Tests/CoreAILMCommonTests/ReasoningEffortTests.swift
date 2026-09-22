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
            #expect((context["enable_thinking"] as? Bool) == true)
            #expect(context["reasoning_strength"] == nil)
        }
    }

    @Test("A request value takes precedence over the default")
    func precedence() {
        #expect(ReasoningEffort.resolve(request: "high", default: "low") == "high")
        #expect(ReasoningEffort.resolve(request: nil, default: "low") == "low")
        #expect(ReasoningEffort.resolve(request: nil, default: nil) == nil)
    }

    @Test("disablesThinking is true only for the canonical none")
    func disablesThinking() {
        #expect(ReasoningEffort.disablesThinking("none"))
        #expect(ReasoningEffort.disablesThinking("None"))
        #expect(ReasoningEffort.disablesThinking(" none "))
        #expect(!ReasoningEffort.disablesThinking("low"))
        #expect(!ReasoningEffort.disablesThinking(nil))
        #expect(!ReasoningEffort.disablesThinking(""))
    }

    @Test("--no-thinking folds into the reasoning default as none")
    func resolveDefaultNoThinking() throws {
        #expect(try ReasoningEffort.resolveDefault(reasoningDefault: nil, noThinking: true) == "none")
        #expect(try ReasoningEffort.resolveDefault(reasoningDefault: nil, noThinking: false) == nil)
        #expect(try ReasoningEffort.resolveDefault(reasoningDefault: "low", noThinking: false) == "low")
        // --no-thinking plus an explicit none default is consistent, not a conflict.
        #expect(try ReasoningEffort.resolveDefault(reasoningDefault: "none", noThinking: true) == "none")
    }

    @Test("--no-thinking with a non-none default is rejected")
    func resolveDefaultContradiction() {
        #expect(throws: ReasoningEffortError.self) {
            try ReasoningEffort.resolveDefault(reasoningDefault: "low", noThinking: true)
        }
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
