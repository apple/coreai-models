// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import TestUtilities
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// Bare-essentials coverage for `ThinkTagParser`. Pins the four behaviors
/// most likely to regress: full passthrough on non-reasoning models, a
/// complete in-one-chunk block, marker straddling two consumes (the
/// streaming buffer), and the end-of-stream flush path.
@Suite("ThinkTagParser — Tag-pair mode")
struct ThinkTagParserTests {
    @Test("No markers in stream — all input is emitted as .text")
    func passthroughWhenNoMarkers() {
        var parser = ThinkTagParser()
        let events = parser.consume("Hello, world!") + parser.flush()
        #expect(eventStrings(events, kind: .text) == ["Hello, world!"])
        #expect(eventStrings(events, kind: .reasoning).isEmpty)
    }

    @Test("Full block in one consume — text/reasoning/text split correctly")
    func fullBlockInOneConsume() {
        var parser = ThinkTagParser()
        let events = parser.consume("before<think>thoughts</think>after") + parser.flush()
        #expect(eventStrings(events, kind: .text) == ["before", "after"])
        #expect(eventStrings(events, kind: .reasoning) == ["thoughts"])
    }

    @Test("Marker split across consumes — buffer holds back partial match")
    func markerStraddlesTwoConsumes() {
        var parser = ThinkTagParser()
        // First chunk ends in "<thi" — a prefix of the open marker. Parser must
        // hold it (no .text("<thi") leaks) until the next chunk disambiguates.
        var events = parser.consume("before<thi")
        #expect(eventStrings(events, kind: .text) == ["before"])
        events += parser.consume("nk>thoughts</think>after")
        events += parser.flush()
        #expect(eventStrings(events, kind: .text) == ["before", "after"])
        #expect(eventStrings(events, kind: .reasoning) == ["thoughts"])
    }

    @Test("Unclosed <think> at EOS — flush drains held buffer as .reasoning")
    func unclosedThinkAtEndOfStream() {
        var parser = ThinkTagParser()
        let events = parser.consume("<think>unterminated thoughts") + parser.flush()
        #expect(eventStrings(events, kind: .text).isEmpty)
        #expect(eventStrings(events, kind: .reasoning) == ["unterminated thoughts"])
    }

    // MARK: - Helpers

    private enum EventKind { case text, reasoning }

    private func eventStrings(_ events: [ThinkTagParser.Event], kind: EventKind) -> [String] {
        events.compactMap { event in
            switch (event, kind) {
            case (.text(let s), .text): return s
            case (.reasoning(let s), .reasoning): return s
            default: return nil
            }
        }
    }
}

@Suite("ThinkTagParser — Agentic mode")
struct ThinkTagParserAgenticTests {
    private let format = ThinkTagParser.Format.agentic(
        selfMarker: "to=self<|message|>",
        userMarker: "to=user<|message|>",
        endOfMessage: "<|eom|>",
        endOfTurn: "<|eot|>"
    )

    @Test("Single reasoning + response turn")
    func singleTurn() {
        var parser = ThinkTagParser(format: format)
        let input = "thinking here<|eom|>visible response<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["thinking here"])
        #expect(eventStrings(events, kind: .text) == ["visible response"])
    }

    @Test("Multiple reasoning segments before response")
    func multipleReasoningSegments() {
        var parser = ThinkTagParser(format: format)
        let input = "step 1<|eom|>to=self<|message|>step 2<|eom|>to=user<|message|>answer<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["step 1", "step 2"])
        #expect(eventStrings(events, kind: .text) == ["answer"])
    }

    @Test("Starts in reasoning mode (agentic default)")
    func startsInReasoning() {
        var parser = ThinkTagParser(format: format)
        let events = parser.consume("initial thought") + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["initial thought"])
        #expect(eventStrings(events, kind: .text).isEmpty)
    }

    @Test("Switch from user back to self (multi-turn)")
    func multiTurnSwitching() {
        var parser = ThinkTagParser(format: format)
        let input =
            "thought<|eom|>to=user<|message|>reply 1<|eot|>to=self<|message|>more thought<|eom|>to=user<|message|>reply 2<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["thought", "more thought"])
        #expect(eventStrings(events, kind: .text) == ["reply 1", "reply 2"])
    }

    @Test("Marker split across consumes")
    func markerStraddlesTwoConsumes() {
        var parser = ThinkTagParser(format: format)
        // Split "<|eom|>" across two chunks
        var events = parser.consume("thinking<|eo")
        #expect(eventStrings(events, kind: .reasoning).isEmpty)
        events += parser.consume("m|>to=user<|message|>response<|eot|>")
        events += parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["thinking"])
        #expect(eventStrings(events, kind: .text) == ["response"])
    }

    @Test("Token-by-token streaming")
    func tokenByToken() {
        var parser = ThinkTagParser(format: format)
        let input = "think<|eom|>to=user<|message|>hi<|eot|>"
        var events: [ThinkTagParser.Event] = []
        for char in input {
            events += parser.consume(String(char))
        }
        events += parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["think"])
        #expect(eventStrings(events, kind: .text) == ["hi"])
    }

    @Test("User marker directly transitions from reasoning (no eom)")
    func userMarkerDirectTransition() {
        var parser = ThinkTagParser(format: format)
        let input = "reasoning content to=user<|message|>visible<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["reasoning content "])
        #expect(eventStrings(events, kind: .text) == ["visible"])
    }

    @Test("Empty reasoning segment")
    func emptyReasoning() {
        var parser = ThinkTagParser(format: format)
        let input = "<|eom|>to=user<|message|>just response<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning).isEmpty)
        #expect(eventStrings(events, kind: .text) == ["just response"])
    }

    @Test("Unclosed reasoning at EOS — flush drains as .reasoning")
    func unclosedReasoningAtEOS() {
        var parser = ThinkTagParser(format: format)
        let events = parser.consume("partial thought without end") + parser.flush()
        let allReasoning = eventStrings(events, kind: .reasoning).joined()
        #expect(allReasoning == "partial thought without end")
        #expect(eventStrings(events, kind: .text).isEmpty)
    }

    @Test("Repeated eom without intervening content — malformed, best-effort")
    func repeatedEomNoContent() {
        var parser = ThinkTagParser(format: format)
        // Double <|eom|> is malformed — second one leaks as text prefix since
        // the parser is already in user mode and doesn't recognize it
        let input = "<|eom|><|eom|>to=user<|message|>response<|eot|>"
        let events = parser.consume(input) + parser.flush()
        let allText = eventStrings(events, kind: .text).joined()
        #expect(allText.contains("response"))
    }

    @Test("Markers embedded in content don't confuse parser")
    func markersInContent() {
        var parser = ThinkTagParser(format: format)
        // The text "discuss eom behavior" shouldn't trigger on "eom" substring
        let input = "discuss eom behavior<|eom|>to=user<|message|>ok<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["discuss eom behavior"])
        #expect(eventStrings(events, kind: .text) == ["ok"])
    }

    @Test("Very long reasoning segment")
    func longReasoningSegment() {
        var parser = ThinkTagParser(format: format)
        let longText = String(repeating: "reasoning step. ", count: 100)
        let input = "\(longText)<|eom|>to=user<|message|>done<|eot|>"
        let events = parser.consume(input) + parser.flush()
        let allReasoning = eventStrings(events, kind: .reasoning).joined()
        #expect(allReasoning == longText)
        #expect(eventStrings(events, kind: .text) == ["done"])
    }

    @Test("Newlines and special characters in content")
    func specialCharsInContent() {
        var parser = ThinkTagParser(format: format)
        let input = "line1\nline2\ttab<|eom|>to=user<|message|>hello 🌍<|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning) == ["line1\nline2\ttab"])
        #expect(eventStrings(events, kind: .text) == ["hello 🌍"])
    }

    @Test("Empty buffer — consume empty string produces no events")
    func emptyConsume() {
        var parser = ThinkTagParser(format: format)
        let events = parser.consume("") + parser.flush()
        #expect(events.isEmpty)
    }

    @Test("Only end markers, no content")
    func onlyMarkers() {
        var parser = ThinkTagParser(format: format)
        let input = "<|eom|>to=user<|message|><|eot|>to=self<|message|><|eom|>to=user<|message|><|eot|>"
        let events = parser.consume(input) + parser.flush()
        #expect(eventStrings(events, kind: .reasoning).isEmpty)
        #expect(eventStrings(events, kind: .text).isEmpty)
    }

    // MARK: - Helpers

    private enum EventKind { case text, reasoning }

    private func eventStrings(_ events: [ThinkTagParser.Event], kind: EventKind) -> [String] {
        events.compactMap { event in
            switch (event, kind) {
            case (.text(let s), .text): return s
            case (.reasoning(let s), .reasoning): return s
            default: return nil
            }
        }
    }
}

@Suite("ThinkTagParser — Format detection")
struct ThinkTagParserDetectionTests {
    @Test("Tokenizer with think tags detects tag-pair format")
    func detectsTagPairForThinkTokens() {
        let tokenizer = MockTokenizer(vocab: [
            "<think>": 100, "</think>": 101,
            "<eos>": 2,
        ])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .tagPair(let open, let close) = format else {
            Issue.record("Expected .tagPair, got \(format)")
            return
        }
        #expect(open == "<think>")
        #expect(close == "</think>")
    }

    @Test("Agentic tokenizer (eom+eot+message) detects agentic format")
    func detectsAgenticFormat() {
        let tokenizer = MockTokenizer(vocab: [
            "<|eom|>": 200, "<|eot|>": 201,
            "<|message|>": 202,
            "<eos>": 2,
        ])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .agentic(let selfM, let userM, let eom, let eot) = format else {
            Issue.record("Expected .agentic, got \(format)")
            return
        }
        #expect(selfM == "to=self<|message|>")
        #expect(userM == "to=user<|message|>")
        #expect(eom == "<|eom|>")
        #expect(eot == "<|eot|>")
    }

    @Test("Tokenizer with reasoning_start/end detects that variant")
    func detectsReasoningStartEnd() {
        let tokenizer = MockTokenizer(vocab: [
            "<|reasoning_start|>": 300, "<|reasoning_end|>": 301,
            "<eos>": 2,
        ])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .tagPair(let open, let close) = format else {
            Issue.record("Expected .tagPair, got \(format)")
            return
        }
        #expect(open == "<|reasoning_start|>")
        #expect(close == "<|reasoning_end|>")
    }

    @Test("Plain tokenizer (no special tokens) falls back to <think>")
    func fallbackToThinkTags() {
        let tokenizer = MockTokenizer(vocab: ["<eos>": 2])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .tagPair(let open, let close) = format else {
            Issue.record("Expected .tagPair, got \(format)")
            return
        }
        #expect(open == "<think>")
        #expect(close == "</think>")
    }

    @Test("Agentic format takes priority over tag-pair when both present")
    func agenticPriorityOverTagPair() {
        let tokenizer = MockTokenizer(vocab: [
            "<think>": 100, "</think>": 101,
            "<|eom|>": 200, "<|eot|>": 201,
            "<|message|>": 202,
            "<eos>": 2,
        ])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .agentic = format else {
            Issue.record("Expected .agentic (priority), got \(format)")
            return
        }
    }

    @Test("eom+eot without message token falls back to tag-pair")
    func eomEotWithoutMessageFallsBack() {
        let tokenizer = MockTokenizer(vocab: [
            "<|eom|>": 200, "<|eot|>": 201,
            "<eos>": 2,
        ])
        let format = CoreAILanguageModel.CoreAIExecutor.detectThinkingFormat(using: tokenizer)
        guard case .tagPair = format else {
            Issue.record("Expected .tagPair fallback, got \(format)")
            return
        }
    }
}

#endif

// MARK: - stripCompleted (no platform restriction — pure string processing)

@Suite("ThinkTagParser.stripCompleted")
struct ThinkTagStripTests {
    @Test("Strips completed thinking blocks and unclosed tails")
    func stripVariants() {
        #expect(ThinkTagParser.stripCompleted(from: "<think>reason</think>Answer") == "Answer")
        #expect(ThinkTagParser.stripCompleted(from: "Hello<think>still...") == "Hello")
        #expect(ThinkTagParser.stripCompleted(from: "<think>all</think>") == "")
        #expect(ThinkTagParser.stripCompleted(from: "No tags here") == "No tags here")
        #expect(ThinkTagParser.stripCompleted(from: "") == "")
    }

    @Test("Custom markers")
    func customMarkers() {
        #expect(ThinkTagParser.stripCompleted(from: "<r>x</r>Y", open: "<r>", close: "</r>") == "Y")
    }

    @Test("Preserves whitespace")
    func preservesWhitespace() {
        #expect(ThinkTagParser.stripCompleted(from: "Hello ") == "Hello ")
        #expect(ThinkTagParser.stripCompleted(from: "<think>x</think>Answer ") == "Answer ")
    }
}
