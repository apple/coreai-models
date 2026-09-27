// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Context truncation policy")
struct ContextTruncationTests {
    /// Builds a ChatMessage from role + content via the public JSON decoder.
    private func message(_ role: String, _ content: String) -> ChatMessage {
        let json = #"{"role":"\#(role)","content":"\#(content)"}"#
        return try! JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
    }

    /// A render closure whose token count equals the summed character length of kept messages.
    /// Content values are irrelevant to the policy — only the token count matters.
    private func render(_ messages: [ChatMessage]) -> [Int] {
        let count = messages.reduce(0) { $0 + $1.content.textContent.count }
        return Array(repeating: 0, count: count)
    }

    private func text(_ n: Int) -> String { String(repeating: "x", count: n) }

    // MARK: - off (default)

    @Test("off returns rendered tokens when within budget")
    func offFits() {
        let messages = [message("user", text(10))]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .off, maxContextLength: 100, render: render)
        #expect(outcome == .ok(render(messages)))
    }

    @Test("off overflows (errors) when over budget")
    func offOverflow() {
        let messages = [message("user", text(150))]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .off, maxContextLength: 100, render: render)
        #expect(outcome == .overflow(promptTokens: 150, budget: 100))
    }

    // MARK: - auto

    @Test("auto is a no-op when within budget")
    func autoFits() {
        let messages = [message("system", text(5)), message("user", text(5))]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .auto, maxContextLength: 100, render: render)
        #expect(outcome == .ok(render(messages)))
    }

    @Test("auto drops oldest droppable, keeps system and last")
    func autoDropsOldest() {
        // system(5) + user(60) + assistant(60) + user(30, last) = 155 > 100.
        // Drop oldest droppable (user 60) -> system(5)+assistant(60)+user(30) = 95 < 100.
        let messages = [
            message("system", text(5)), message("user", text(60)),
            message("assistant", text(60)), message("user", text(30)),
        ]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .auto, maxContextLength: 100, render: render)
        guard case .ok(let tokens) = outcome else {
            Issue.record("expected .ok, got \(outcome)")
            return
        }
        #expect(tokens.count == 95)
    }

    // MARK: - tokensAt(N)

    @Test("tokensAt caps to N below maxContext")
    func tokensAtCaps() {
        // budget = min(50, 100) = 50. Drop both 60-token turns -> system(5)+user(30) = 35 < 50.
        let messages = [
            message("system", text(5)), message("user", text(60)),
            message("assistant", text(60)), message("user", text(30)),
        ]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .tokensAt(50), maxContextLength: 100, render: render)
        guard case .ok(let tokens) = outcome else {
            Issue.record("expected .ok, got \(outcome)")
            return
        }
        #expect(tokens.count == 35)
    }

    // MARK: - safety net

    @Test("auto token-trims when system plus last still overflow")
    func safetyNetTrims() {
        // system(80) + user(60, last) = 140 > 100, nothing droppable -> keep-last trim to budget-1.
        let messages = [message("system", text(80)), message("user", text(60))]
        let outcome = ContextTruncation.apply(
            messages: messages, truncation: .auto, maxContextLength: 100, render: render)
        guard case .ok(let tokens) = outcome else {
            Issue.record("expected .ok, got \(outcome)")
            return
        }
        #expect(tokens.count == 99)
    }
}
