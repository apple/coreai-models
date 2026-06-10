// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Streaming parser that detects tool call blocks in the model's token stream.
struct ToolCallParser {
    enum Event {
        case text(String)
        case toolCall(id: String, name: String, argsJSON: String)
    }

    private let openMarker: String
    private let closeMarker: String
    private var buffer: String = ""
    private var isInsideToolCall: Bool = false

    init(openMarker: String = "<tool_call>", closeMarker: String = "</tool_call>") {
        self.openMarker = openMarker
        self.closeMarker = closeMarker
    }

    mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        return drain(isFinal: false)
    }

    /// Emit any pending buffered content as final events.
    ///
    /// Required at end of stream — without it, text held back to wait for a
    /// possible marker match is silently lost. An unclosed `<tool_call>` block
    /// at EOS is dropped (malformed JSON is not useful to surface as text).
    /// Exception: newline-terminated formats (e.g. Mistral's `[TOOL_CALLS]`)
    /// have no trailing close token, so the buffered content is parsed on EOS.
    mutating func flush() -> [Event] {
        drain(isFinal: true)
    }

    private mutating func drain(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while let range = buffer.range(of: isInsideToolCall ? closeMarker : openMarker) {
            processMarker(at: range, events: &events)
            isInsideToolCall.toggle()
        }
        processRemainder(of: &events, isFinal: isFinal)
        return events
    }

    private mutating func processMarker(at range: Range<String.Index>, events: inout [Event]) {
        let before = String(buffer[buffer.startIndex..<range.lowerBound])
        if isInsideToolCall {
            events.append(contentsOf: parseToolCalls(from: before))
        } else if !before.isEmpty {
            events.append(.text(before))
        }
        buffer = String(buffer[range.upperBound...])
    }

    private mutating func processRemainder(of events: inout [Event], isFinal: Bool) {
        if isInsideToolCall {
            guard isFinal else { return }
            // Newline-terminated formats (e.g. Mistral) have no dedicated close
            // token — the block ends at EOS. Try to parse what we have.
            // For tag-pair formats, an unclosed block is malformed: drop it.
            if closeMarker == "\n" {
                events.append(contentsOf: parseToolCalls(from: buffer))
            }
            buffer = ""
        } else {
            let safe = isFinal ? buffer.endIndex : lastSafeIndex(for: openMarker)
            if safe > buffer.startIndex {
                let toEmit = String(buffer[buffer.startIndex..<safe])
                if !toEmit.isEmpty { events.append(.text(toEmit)) }
                buffer = String(buffer[safe...])
            }
        }
    }

    /// Single object `{"name":..}` (Qwen3) or array `[{"name":..},..]` (Mistral).
    private func parseToolCalls(from json: String) -> [Event] {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else { return [] }

        if trimmed.hasPrefix("["),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        {
            return array.compactMap { makeToolCallEvent(from: $0) }
        }

        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return makeToolCallEvent(from: obj).map { [$0] } ?? []
        }

        return []
    }

    private func makeToolCallEvent(from obj: [String: Any]) -> Event? {
        guard let name = obj["name"] as? String else { return nil }

        let argsJSON: String
        if let argsDict = obj["arguments"] as? [String: Any],
            let argsData = try? JSONSerialization.data(withJSONObject: argsDict),
            let argsStr = String(data: argsData, encoding: .utf8)
        {
            argsJSON = argsStr
        } else if let argsStr = obj["arguments"] as? String {
            argsJSON = argsStr
        } else {
            argsJSON = "{}"
        }

        return .toolCall(id: UUID().uuidString, name: name, argsJSON: argsJSON)
    }

    /// Rightmost index such that the suffix from there to end-of-buffer is NOT
    /// a non-empty prefix of `tag`. Same implementation as `ThinkTagParser`.
    private func lastSafeIndex(for tag: String) -> String.Index {
        let maxHold = tag.count - 1
        guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
        let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
        for offset in 0..<buffer.distance(from: holdStart, to: buffer.endIndex) {
            let idx = buffer.index(holdStart, offsetBy: offset)
            if tag.starts(with: buffer[idx...]) {
                return idx
            }
        }
        return buffer.endIndex
    }
}
