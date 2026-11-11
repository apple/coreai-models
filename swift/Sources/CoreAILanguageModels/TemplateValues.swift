// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Jinja

/// Values for chat templates that need more than the request's own types.
package enum TemplateValues {
    /// A past tool call as an entry in a chat template's `tool_calls`: `id`, `type`, then
    /// `function` with `name` before `arguments`, in OpenAI's order. A Swift dictionary would reach
    /// the template with its keys sorted, and Mistral's template writes `function` with `tojson`.
    ///
    /// The arguments are the value they encode. Clients send them as a JSON string, as the OpenAI
    /// API does, but chat templates are written for an object: some write each argument (Qwen3.5's
    /// `arguments|items`), some refuse anything that is not a mapping, and some `tojson` it, which
    /// would encode a string twice (Qwen2.5, gpt-oss). An object's keys come sorted, as swift-jinja
    /// decodes a `Value`, so the same arguments render the same way on every request, which is what
    /// reusing a prompt's prefix needs. No arguments at all (an empty string, as some clients send
    /// for a tool that takes none) are an empty object. Arguments that are not JSON, or that nest
    /// deeper than ``maxArgumentDepth``, stay the string they came as.
    package static func toolCall(id: String, name: String, arguments: String) -> Value {
        let parsed = arguments.allSatisfy(\.isWhitespace) ? .object([:]) : Self.arguments(arguments)
        let function: OrderedDictionary<String, Value> = [
            "name": .string(name), "arguments": parsed ?? .string(arguments),
        ]
        let call: OrderedDictionary<String, Value> = [
            "id": .string(id), "type": .string("function"), "function": .object(function),
        ]
        return .object(call)
    }

    /// The deepest nesting of arrays and objects decoded. Decoding recurses once per level, and
    /// templates render on Swift concurrency threads, whose stacks are small; real tool arguments
    /// nest a few levels.
    static let maxArgumentDepth = 128

    /// A tool call's `arguments` string as the value it encodes. Nil when it is not JSON or nests
    /// deeper than ``maxArgumentDepth``.
    static func arguments(_ json: String) -> Value? {
        guard nestingDepth(of: json) <= maxArgumentDepth else { return nil }
        return try? JSONDecoder().decode(Value.self, from: Data(json.utf8))
    }

    /// How deeply `json`'s arrays and objects nest, not counting brackets inside strings.
    static func nestingDepth(of json: String) -> Int {
        var depth = 0
        var deepest = 0
        var inString = false
        var escaped = false
        for byte in json.utf8 {
            if inString {
                if escaped {
                    escaped = false
                } else if byte == UInt8(ascii: "\\") {
                    escaped = true
                } else if byte == UInt8(ascii: "\"") {
                    inString = false
                }
                continue
            }
            switch byte {
            case UInt8(ascii: "\""): inString = true
            case UInt8(ascii: "["), UInt8(ascii: "{"):
                depth += 1
                deepest = max(deepest, depth)
            case UInt8(ascii: "]"), UInt8(ascii: "}"): depth -= 1
            default: break
            }
        }
        return deepest
    }
}
