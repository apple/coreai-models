// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Streaming parser that segments a model's text deltas into plain text and
/// reasoning content emitted inside chain-of-thought markers.
///
/// Two formats are supported:
///
/// **Tag-pair**: symmetric open/close markers wrap
/// reasoning content inline: `<think>reasoning</think>response`.
///
/// **Agentic**: multi-turn message routing where reasoning
/// is emitted as `to=self` messages and responses as `to=user` messages,
/// delimited by message boundary tokens.
public struct ThinkTagParser {
    public enum Event {
        case text(String)
        case reasoning(String)
    }

    /// Format configuration for the parser.
    enum Format {
        /// Symmetric open/close tag pair (e.g. `<think>`/`</think>`).
        case tagPair(open: String, close: String)

        /// Agentic message routing with role-based delimiters.
        /// - `selfMarker`: string that begins a reasoning segment (e.g. "to=self<|message|>")
        /// - `userMarker`: string that begins a user-facing segment (e.g. "to=user<|message|>")
        /// - `endOfMessage`: terminates a reasoning segment (e.g. "<|eom|>")
        /// - `endOfTurn`: terminates a user-facing segment (e.g. "<|eot|>")
        case agentic(selfMarker: String, userMarker: String, endOfMessage: String, endOfTurn: String)
    }

    private let format: Format
    private var buffer: String = ""
    private var insideThink: Bool = false

    public init(open: String = "<think>", close: String = "</think>") {
        self.format = .tagPair(open: open, close: close)
    }

    init(format: Format) {
        self.format = format
        if case .agentic = format {
            self.insideThink = true
        }
    }

    public mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        switch format {
        case .tagPair:
            return drainTagPair(isFinal: false)
        case .agentic:
            return drainAgentic(isFinal: false)
        }
    }

    public mutating func flush() -> [Event] {
        switch format {
        case .tagPair:
            return drainTagPair(isFinal: true)
        case .agentic:
            return drainAgentic(isFinal: true)
        }
    }

    // MARK: - Tag-pair mode

    private mutating func drainTagPair(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            let marker = insideThink ? closeMarkerForTagPair : openMarkerForTagPair
            let makeEvent: (String) -> Event = insideThink ? { .reasoning($0) } : { .text($0) }

            if let range = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<range.lowerBound])
                if !before.isEmpty { events.append(makeEvent(before)) }
                buffer = String(buffer[range.upperBound...])
                insideThink.toggle()
            } else {
                let safe = isFinal ? buffer.endIndex : lastSafeIndex(forTag: marker)
                if safe > buffer.startIndex {
                    let toEmit = String(buffer[buffer.startIndex..<safe])
                    if !toEmit.isEmpty { events.append(makeEvent(toEmit)) }
                    buffer = String(buffer[safe...])
                }
                return events
            }
        }
    }

    private var openMarkerForTagPair: String {
        if case .tagPair(let open, _) = format { return open }
        return ""
    }

    private var closeMarkerForTagPair: String {
        if case .tagPair(_, let close) = format { return close }
        return ""
    }

    // MARK: - Agentic mode

    private mutating func drainAgentic(isFinal: Bool) -> [Event] {
        guard case .agentic(let selfMarker, let userMarker, let eom, let eot) = format else {
            return []
        }

        var events: [Event] = []
        while true {
            // Entry markers may arrive across consume() boundaries — strip them
            // at the top of each iteration before searching for end markers.
            if buffer.hasPrefix(selfMarker) {
                buffer = String(buffer.dropFirst(selfMarker.count))
                insideThink = true
            } else if buffer.hasPrefix(userMarker) {
                buffer = String(buffer.dropFirst(userMarker.count))
                insideThink = false
            }

            if insideThink {
                if let range = buffer.range(of: eom) {
                    let before = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !before.isEmpty { events.append(.reasoning(before)) }
                    buffer = String(buffer[range.upperBound...])
                    insideThink = false
                    // Consume the following entry marker if present
                    if buffer.hasPrefix(selfMarker) {
                        buffer = String(buffer.dropFirst(selfMarker.count))
                        insideThink = true
                    } else if buffer.hasPrefix(userMarker) {
                        buffer = String(buffer.dropFirst(userMarker.count))
                    }
                } else if let range = buffer.range(of: userMarker) {
                    let before = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !before.isEmpty { events.append(.reasoning(before)) }
                    buffer = String(buffer[range.upperBound...])
                    insideThink = false
                } else {
                    let holdBack = max(eom.count, userMarker.count) - 1
                    return emitSafe(events: &events, holdBack: isFinal ? 0 : holdBack, asReasoning: true)
                }
            } else {
                if let range = buffer.range(of: eot) {
                    let before = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !before.isEmpty { events.append(.text(before)) }
                    buffer = String(buffer[range.upperBound...])
                    insideThink = true
                    // Consume the following entry marker if present
                    if buffer.hasPrefix(userMarker) {
                        buffer = String(buffer.dropFirst(userMarker.count))
                        insideThink = false
                    } else if buffer.hasPrefix(selfMarker) {
                        buffer = String(buffer.dropFirst(selfMarker.count))
                    }
                } else if let range = buffer.range(of: selfMarker) {
                    let before = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !before.isEmpty { events.append(.text(before)) }
                    buffer = String(buffer[range.upperBound...])
                    insideThink = true
                } else {
                    let holdBack = max(eot.count, selfMarker.count) - 1
                    return emitSafe(events: &events, holdBack: isFinal ? 0 : holdBack, asReasoning: false)
                }
            }
        }
    }

    private mutating func emitSafe(events: inout [Event], holdBack: Int, asReasoning: Bool) -> [Event] {
        let safeEnd: String.Index
        if holdBack <= 0 || buffer.isEmpty {
            safeEnd = buffer.endIndex
        } else {
            safeEnd = buffer.index(buffer.endIndex, offsetBy: -min(holdBack, buffer.count))
        }
        if safeEnd > buffer.startIndex {
            let toEmit = String(buffer[buffer.startIndex..<safeEnd])
            if !toEmit.isEmpty {
                events.append(asReasoning ? .reasoning(toEmit) : .text(toEmit))
            }
            buffer = String(buffer[safeEnd...])
        }
        return events
    }

    // MARK: - Helpers

    private func lastSafeIndex(forTag tag: String) -> String.Index {
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

    /// Strip all completed thinking blocks from a full string.
    /// Unclosed blocks at the end are also removed.
    public static func stripCompleted(
        from text: String, open: String = "<think>", close: String = "</think>"
    ) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        var remaining = text[...]
        while let startRange = remaining.range(of: open) {
            result.append(contentsOf: remaining[remaining.startIndex..<startRange.lowerBound])
            if let endRange = remaining.range(of: close, range: startRange.upperBound..<remaining.endIndex) {
                remaining = remaining[endRange.upperBound...]
            } else {
                // Unclosed block — discard everything from here
                return result
            }
        }
        result.append(contentsOf: remaining)
        return result
    }
}
