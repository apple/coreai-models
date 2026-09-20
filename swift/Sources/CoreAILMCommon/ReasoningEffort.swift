// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Resolves a reasoning-effort value into chat-template keyword arguments, which are then passed to
/// `applyChatTemplate(additionalContext:)`.
///
/// Reasoning models expose their thinking budget through different chat-template variables (for
/// example `reasoning_effort`, `reasoning_strength`, or a boolean `enable_thinking`). Binding the
/// canonical value to each known variable lets one request field drive them all: a chat template
/// reads only the variables it references, so setting the others alongside is safe.
public enum ReasoningEffort {
    /// Canonical value that requests no reasoning.
    public static let none = "none"

    /// Maps a canonical effort value to chat-template keyword arguments.
    ///
    /// - `nil` or empty returns an empty dictionary, so the template keeps its own default.
    /// - `"none"` sets `enable_thinking` to `false` for templates that support disabling reasoning.
    /// - any level (for example `low`, `medium`, `high`) binds the level to `reasoning_effort` and
    ///   `reasoning_strength`, and sets `enable_thinking` to `true`.
    public static func templateContext(_ effort: String?) -> [String: any Sendable] {
        guard let value = effort?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return [:]
        }
        if value.lowercased() == none {
            return ["enable_thinking": false]
        }
        return [
            "reasoning_effort": value,
            "reasoning_strength": value,
            "enable_thinking": true,
        ]
    }

    /// Resolves the effort for a request: the per-request value takes precedence, then the server
    /// default, then `nil` (leaving the template default in place).
    public static func resolve(request: String?, default defaultEffort: String?) -> String? {
        request ?? defaultEffort
    }
}
