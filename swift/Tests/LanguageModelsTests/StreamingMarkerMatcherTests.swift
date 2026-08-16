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
}
