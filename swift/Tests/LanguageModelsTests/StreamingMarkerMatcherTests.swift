// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("StreamingMarkerMatcher")
struct StreamingMarkerMatcherTests {
    @Test("No partial match — entire buffer is safe")
    func noMatch() {
        let buffer = "hello world"
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == buffer.endIndex)
    }

    @Test("Single-char prefix held back")
    func singleCharHoldback() {
        let buffer = "hello<"
        let expected = buffer.index(buffer.startIndex, offsetBy: 5)
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == expected)
    }

    @Test("Multi-char partial prefix held back")
    func multiCharHoldback() {
        let buffer = "some text<thi"
        let expected = buffer.index(buffer.startIndex, offsetBy: 9)
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == expected)
    }

    @Test("Maximum holdback — entire buffer is a prefix of tag")
    func maxHoldback() {
        let buffer = "<thin"
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == buffer.startIndex)
    }

    @Test("Empty buffer returns endIndex")
    func emptyBuffer() {
        let buffer = ""
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == buffer.endIndex)
    }

    @Test("Buffer is just '<' — held back as prefix")
    func bufferIsSingleOpenAngle() {
        let buffer = "<"
        #expect(lastSafeIndex(in: buffer, forTag: "<think>") == buffer.startIndex)
    }

    @Test("'>' and '!' are not prefixes — safe to emit")
    func nonPrefixSpecialChars() {
        let gt = ">"
        let bang = "!"
        #expect(lastSafeIndex(in: gt, forTag: "<think>") == gt.endIndex)
        #expect(lastSafeIndex(in: bang, forTag: "<think>") == bang.endIndex)
    }

    @Test("Single-char tag — nothing is ever held back")
    func singleCharTag() {
        let buffer = "abc<"
        #expect(lastSafeIndex(in: buffer, forTag: "X") == buffer.endIndex)
    }
}
