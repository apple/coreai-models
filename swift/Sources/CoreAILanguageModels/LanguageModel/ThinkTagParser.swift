// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Streaming parser that segments a model's text deltas into plain text and
/// reasoning content emitted inside chain-of-thought markers.
///
/// Reasoning-capable models like Qwen3 and DeepSeek-R1 emit chain-of-thought
/// as inline markup mixed into the regular text stream — most commonly
/// `<think>...</think>`. Without intercepting it, the markup leaks into the
/// user-visible response. This parser routes the body of each thinking block
/// as `.reasoning` events and everything else as `.text` events, so the
/// executor can dispatch them to the right FoundationModels channel event
/// (top-level `.reasoning(...)` vs `.response(...).appendText`).
///
/// The marker pair is configurable at init so the same parser works for
/// models with different conventions. Defaults are `<think>`/`</think>`.
/// Caller is responsible for picking the right pair for a given tokenizer
/// (see `CoreAIExecutor.detectThinkingMarkers`).
///
/// Feed `delta` strings (incremental detokenizer output) via `consume(_:)`
/// and call `flush()` once at end of stream. The parser internally holds
/// back at most `closeMarker.count - 1` characters of trailing buffer so a
/// marker that straddles two deltas isn't truncated mid-match.
struct ThinkTagParser {
    enum Event {
        case text(String)
        case reasoning(String)
    }

    private let openMarker: String
    private let closeMarker: String

    private var buffer: String = ""
    private var insideThink: Bool = false

    init(open: String = "<think>", close: String = "</think>") {
        self.openMarker = open
        self.closeMarker = close
    }

    mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        return drain(isFinal: false)
    }

    /// Emit any pending buffered content as a final event. Required at end of
    /// stream — without it, content held back to wait for a possible marker
    /// match is silently lost. Stream-end content gets routed by current
    /// mode: in-think content becomes `.reasoning`, plain text becomes
    /// `.text`.
    mutating func flush() -> [Event] {
        drain(isFinal: true)
    }

    private mutating func drain(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            let marker = insideThink ? closeMarker : openMarker
            let makeEvent: (String) -> Event = insideThink ? { .reasoning($0) } : { .text($0) }

            if let range = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<range.lowerBound])
                if !before.isEmpty { events.append(makeEvent(before)) }
                buffer = String(buffer[range.upperBound...])
                insideThink.toggle()
            } else {
                // `isFinal == true` (called from `flush()`): no need to hold back
                // a partial-marker suffix; emit the entire buffer. Otherwise:
                // hold back at most `marker.count - 1` characters in case the
                // next delta completes the marker.
                let safe = isFinal ? buffer.endIndex : lastSafeIndex(in: buffer, forTag: marker)
                if safe > buffer.startIndex {
                    let toEmit = String(buffer[buffer.startIndex..<safe])
                    if !toEmit.isEmpty { events.append(makeEvent(toEmit)) }
                    buffer = String(buffer[safe...])
                }
                return events
            }
        }
    }
}
