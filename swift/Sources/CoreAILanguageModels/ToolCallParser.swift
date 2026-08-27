// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Streaming parser that detects tool call blocks in the model's token stream.
public struct ToolCallParser: Sendable {
    public enum Event {
        case text(String)
        case toolCall(id: String, name: String, argsJSON: String)
    }

    public enum Format: Sendable {
        case json
        case atem
    }

    private let openMarker: String
    private let closeMarker: String
    private let format: Format
    private var buffer: String = ""
    private var isInsideToolCall: Bool = false

    public init(
        openMarker: String = "<tool_call>",
        closeMarker: String = "</tool_call>",
        format: Format = .json
    ) {
        self.openMarker = openMarker
        self.closeMarker = closeMarker
        self.format = format
    }

    public mutating func consume(_ delta: String) -> [Event] {
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
    public mutating func flush() -> [Event] {
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
            if closeMarker == "\n" {
                events.append(contentsOf: parseToolCalls(from: buffer))
            }
            buffer = ""
        } else {
            let safe = isFinal ? buffer.endIndex : lastSafeIndex(in: buffer, forTag: openMarker)
            if safe > buffer.startIndex {
                let toEmit = String(buffer[buffer.startIndex..<safe])
                if !toEmit.isEmpty { events.append(.text(toEmit)) }
                buffer = String(buffer[safe...])
            }
        }
    }

    private func parseToolCalls(from content: String) -> [Event] {
        switch format {
        case .json:
            return parseJSONToolCalls(from: content)
        case .atem:
            return parseATEMToolCalls(from: content)
        }
    }

    // MARK: - JSON Format

    /// Single object `{"name":..}` (Qwen3) or array `[{"name":..},..]` (Mistral).
    private func parseJSONToolCalls(from json: String) -> [Event] {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else { return [] }

        if trimmed.hasPrefix("["),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        {
            return array.compactMap { makeJSONToolCallEvent(from: $0) }
        }

        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return makeJSONToolCallEvent(from: obj).map { [$0] } ?? []
        }

        return []
    }

    private func makeJSONToolCallEvent(from obj: [String: Any]) -> Event? {
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

        let callId = "call_\(UUID().uuidString.prefix(8).lowercased())"
        return .toolCall(id: callId, name: name, argsJSON: argsJSON)
    }

    // MARK: - ATEM Format

    /// Parse `<atem:invoke name="fn"><atem:parameter name="k">v</atem:parameter></atem:invoke>` blocks.
    private func parseATEMToolCalls(from xml: String) -> [Event] {
        let invokePattern = "<atem:invoke\\s+name=\"([^\"]+)\"[^>]*>(.*?)</atem:invoke>"
        guard let invokeRegex = try? NSRegularExpression(pattern: invokePattern, options: .dotMatchesLineSeparators)
        else { return [] }

        let nsString = xml as NSString
        let matches = invokeRegex.matches(in: xml, range: NSRange(location: 0, length: nsString.length))

        return matches.compactMap { match -> Event? in
            guard match.numberOfRanges >= 3 else { return nil }
            let name = nsString.substring(with: match.range(at: 1))
            let body = nsString.substring(with: match.range(at: 2))
            let args = parseATEMParameters(from: body)
            let argsJSON: String
            if args.isEmpty {
                argsJSON = "{}"
            } else if let data = try? JSONSerialization.data(withJSONObject: args),
                let str = String(data: data, encoding: .utf8)
            {
                argsJSON = str
            } else {
                argsJSON = "{}"
            }
            let callId = "call_\(UUID().uuidString.prefix(8).lowercased())"
            return .toolCall(id: callId, name: name, argsJSON: argsJSON)
        }
    }

    private func parseATEMParameters(from body: String) -> [String: Any] {
        let paramPattern = "<atem:parameter\\s+name=\"([^\"]+)\">(.*?)</atem:parameter>"
        guard let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: .dotMatchesLineSeparators)
        else { return [:] }

        let nsBody = body as NSString
        let matches = paramRegex.matches(in: body, range: NSRange(location: 0, length: nsBody.length))

        var result: [String: Any] = [:]
        for match in matches {
            guard match.numberOfRanges >= 3 else { continue }
            let key = nsBody.substring(with: match.range(at: 1))
            let rawValue = nsBody.substring(with: match.range(at: 2))
            result[key] = coerceATEMValue(rawValue)
        }
        return result
    }

    private func coerceATEMValue(_ raw: String) -> Any {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == "true" { return true }
        if trimmed == "false" { return false }
        if let intVal = Int(trimmed) { return intVal }
        if let doubleVal = Double(trimmed), trimmed.contains(".") { return doubleVal }
        return trimmed
    }
}

// MARK: - Tool Call Marker Detection

public struct ToolCallDetection: Sendable {
    public let openMarker: String
    public let closeMarker: String
    public let format: ToolCallParser.Format
}

/// Probes a tokenizer's vocabulary for known tool-call special tokens.
public func detectToolCallFormat(using tokenizer: any Tokenizer) -> ToolCallDetection? {
    // ATEM special tokens (if a future tokenizer adds them)
    if tokenizer.convertTokenToId("<atem:function_calls>") != nil,
        tokenizer.convertTokenToId("</atem:function_calls>") != nil
    {
        return ToolCallDetection(
            openMarker: "<atem:function_calls>",
            closeMarker: "</atem:function_calls>",
            format: .atem
        )
    }

    // ATEM via agentic model signature: <|eom|> + <|eot|> + <|start|> + <|message|>
    // indicate a Meta agentic model that emits ATEM tags as regular text.
    if tokenizer.convertTokenToId("<|eom|>") != nil,
        tokenizer.convertTokenToId("<|eot|>") != nil,
        tokenizer.convertTokenToId("<|start|>") != nil,
        tokenizer.convertTokenToId("<|message|>") != nil
    {
        return ToolCallDetection(
            openMarker: "<atem:function_calls>",
            closeMarker: "</atem:function_calls>",
            format: .atem
        )
    }

    let tagPairs: [(open: String, close: String)] = [
        ("<tool_call>", "</tool_call>"),
        ("<function_calls>", "</function_calls>"),
    ]
    for pair in tagPairs
    where tokenizer.convertTokenToId(pair.open) != nil
        && tokenizer.convertTokenToId(pair.close) != nil
    {
        return ToolCallDetection(openMarker: pair.open, closeMarker: pair.close, format: .json)
    }
    if tokenizer.convertTokenToId("[TOOL_CALLS]") != nil {
        return ToolCallDetection(openMarker: "[TOOL_CALLS]", closeMarker: "\n", format: .json)
    }
    return nil
}

// MARK: - Thinking Format Detection

/// Probes a tokenizer's vocabulary for thinking/reasoning markers.
public func detectThinkingFormat(using tokenizer: any Tokenizer) -> ThinkTagParser.Format {
    if tokenizer.convertTokenToId("<|eom|>") != nil,
        tokenizer.convertTokenToId("<|eot|>") != nil,
        tokenizer.convertTokenToId("<|message|>") != nil
    {
        return .agentic(
            selfMarker: "to=self<|message|>",
            userMarker: "to=user<|message|>",
            endOfMessage: "<|eom|>",
            endOfTurn: "<|eot|>"
        )
    }

    let candidates: [(open: String, close: String)] = [
        ("<think>", "</think>"),
        ("<|reasoning_start|>", "<|reasoning_end|>"),
    ]
    for pair in candidates {
        if tokenizer.convertTokenToId(pair.open) != nil,
            tokenizer.convertTokenToId(pair.close) != nil
        {
            return .tagPair(open: pair.open, close: pair.close)
        }
    }
    return .tagPair(open: "<think>", close: "</think>")
}
