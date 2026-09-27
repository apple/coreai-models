// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Applies a context-overflow policy to a chat message list at the message boundary.
///
/// The `render` closure turns a candidate message list into its rendered token array (chat template
/// applied), so this policy is independent of any tokenizer and unit-testable in isolation.
public enum ContextTruncation {
    public enum Outcome: Equatable {
        /// The token array to feed the engine.
        case ok([Int])
        /// The prompt does not fit and truncation is off; the caller should reject with an error.
        case overflow(promptTokens: Int, budget: Int)
    }

    public static func apply(
        messages: [ChatMessage],
        truncation: Truncation,
        maxContextLength: Int,
        render: ([ChatMessage]) -> [Int]
    ) -> Outcome {
        let full = render(messages)
        switch truncation {
        case .off:
            return full.count < maxContextLength
                ? .ok(full)
                : .overflow(promptTokens: full.count, budget: maxContextLength)
        case .auto:
            return fit(messages, budget: maxContextLength, full: full, render: render)
        case .tokensAt(let n):
            return fit(messages, budget: min(max(n, 0), maxContextLength), full: full, render: render)
        }
    }

    /// Drops oldest droppable messages (everything except system messages and the newest turn) until
    /// the rendered prompt fits `budget`. Falls back to a keep-last token trim if even the pinned set
    /// overflows, so a truncation request never errors.
    private static func fit(
        _ messages: [ChatMessage], budget: Int, full: [Int], render: ([ChatMessage]) -> [Int]
    ) -> Outcome {
        if full.count < budget { return .ok(full) }

        let lastIndex = messages.count - 1
        var dropped = Set<Int>()
        for index in messages.indices where index != lastIndex && messages[index].role != "system" {
            dropped.insert(index)
            let kept = messages.indices.filter { !dropped.contains($0) }.map { messages[$0] }
            let tokens = render(kept)
            if tokens.count < budget { return .ok(tokens) }
        }

        let kept = messages.indices.filter { !dropped.contains($0) }.map { messages[$0] }
        let tokens = render(kept)
        if tokens.count < budget { return .ok(tokens) }
        return .ok(Array(tokens.suffix(max(budget - 1, 0))))
    }
}
