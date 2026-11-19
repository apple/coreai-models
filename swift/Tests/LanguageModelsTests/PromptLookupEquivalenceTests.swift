// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Prompt lookup equivalence")
struct PromptLookupEquivalenceTests {
    @Test("The defaults are the shared speculative defaults")
    func defaults() {
        let lookup = PromptLookup()
        #expect(lookup.maxNgram == SpeculativeDefaults.lookupMaxNgram)
        #expect(lookup.minNgram == SpeculativeDefaults.lookupMinNgram)
        #expect(lookup.maxTokens == SpeculativeDefaults.lookupMaxTokens)
        let drafter = PromptLookupDrafter()
        #expect(drafter.maxContextLength == SpeculativeDefaults.maxContextLength)
    }

    @Test("Every context, however short, is searched without an out-of-range index")
    func shortContexts() {
        let lookup = PromptLookup(maxNgram: 1, minNgram: 1, maxTokens: 1)
        for context in [[], [5], [5, 5], [1, 2, 1], [4, 4, 4, 4]] as [[Int32]] {
            #expect(lookup.propose(context: context, count: 3).count <= 3)
        }
        #expect(lookup.propose(context: [4, 4, 4, 4], count: 3) == [4])
    }

    @Test("Bad parameters are refused at construction")
    func badParameters() async {
        await #expect(processExitsWith: .failure) { _ = PromptLookup(minNgram: 0) }
        await #expect(processExitsWith: .failure) { _ = PromptLookup(maxNgram: 2, minNgram: 3) }
        await #expect(processExitsWith: .failure) { _ = PromptLookup(maxTokens: 0) }
        await #expect(processExitsWith: .failure) { _ = PromptLookupDrafter(maxContextLength: 0) }
    }

    /// Element-by-element reference: longest suffix first, latest earlier occurrence first.
    private func reference(_ lookup: PromptLookup, _ context: [Int32], _ count: Int) -> [Int32] {
        guard count > 0 else { return [] }
        let end = context.count
        var n = Swift.min(lookup.maxNgram, end - 1)
        while n >= lookup.minNgram {
            var i = end - n - 1
            while i >= 0 {
                if (0..<n).allSatisfy({ context[i + $0] == context[end - n + $0] }) {
                    return Array(context[(i + n)..<Swift.min(i + n + count, end)])
                }
                i -= 1
            }
            n -= 1
        }
        return []
    }

    @Test("Proposals match the element-wise reference on random contexts")
    func matchesReference() {
        var rng = SplitMix(11)
        for _ in 0..<2000 {
            let minNgram = 1 + Int(rng.next() % 3)
            let lookup = PromptLookup(
                maxNgram: minNgram + Int(rng.next() % 3), minNgram: minNgram,
                maxTokens: 1 + Int(rng.next() % 8))
            let context = (0..<Int(rng.next() % 30)).map { _ in Int32(rng.next() % 3) }
            let count = Int(rng.next() % 9)
            #expect(lookup.propose(context: context, count: count) == reference(lookup, context, count))
        }
    }

    @Test("The latest earlier occurrence of the longest suffix is proposed")
    func latestOccurrence() {
        let lookup = PromptLookup(maxNgram: 2, minNgram: 2)
        #expect(lookup.propose(context: [1, 2, 9, 1, 2, 8, 1, 2], count: 2) == [8, 1])
        #expect(lookup.propose(context: [1, 2, 3], count: 2).isEmpty)
    }
}
