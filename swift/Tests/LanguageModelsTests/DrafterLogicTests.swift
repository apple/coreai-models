// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Drafter logic without a model")
struct DrafterLogicTests {
    private func features(rows: Int, start: Int) -> DrafterFeatures {
        // Row p holds the value p in both of its two channels.
        DrafterFeatures(
            values: (0..<rows).flatMap { [Float($0), Float($0)] }, positionCount: rows, hiddenDim: 2,
            startPosition: start)
    }

    @Test("The seed row is the last committed token's feature row")
    func seedRowSelectsLastCommitted() {
        // Rows are for positions 10...13; with 12 tokens committed the last is position 11 = row 1.
        #expect(features(rows: 4, start: 10).seedRow(confirmedLength: 12) == [1, 1])
        #expect(features(rows: 4, start: 10).seedRow(confirmedLength: 14) == [3, 3])
    }

    @Test("The seed row is clamped to the rows present")
    func seedRowIsClamped() {
        #expect(features(rows: 4, start: 10).seedRow(confirmedLength: 3) == [0, 0])
        #expect(features(rows: 4, start: 10).seedRow(confirmedLength: 99) == [3, 3])
    }

    @Test("No rows, no seed")
    func seedRowNeedsRows() {
        #expect(
            DrafterFeatures(values: [], positionCount: 0, hiddenDim: 2, startPosition: 0).seedRow(confirmedLength: 1)
                == nil)
        #expect(
            DrafterFeatures(values: [], positionCount: 3, hiddenDim: 0, startPosition: 0).seedRow(confirmedLength: 1)
                == nil)
    }

    @Test("The smallest window holding the context is chosen, else the largest")
    func windowSelection() {
        let windows = [512, 1024, 2048]
        #expect(StaticShapeDrafter.window(for: 1, in: windows) == 512)
        #expect(StaticShapeDrafter.window(for: 512, in: windows) == 512)
        #expect(StaticShapeDrafter.window(for: 513, in: windows) == 1024)
        #expect(StaticShapeDrafter.window(for: 2048, in: windows) == 2048)
        #expect(StaticShapeDrafter.window(for: 5000, in: windows) == 2048)
        #expect(StaticShapeDrafter.window(for: 7, in: [512]) == 512)
    }

    /// A fake ring cache: slot `s` holds the absolute position that was last written there.
    private func ringCache(ctx: Int, written: Int) -> [Int] {
        var cache = [Int](repeating: -1, count: ctx)
        for p in 0..<written { cache[p % ctx] = p }
        return cache
    }

    @Test("A ring window reads the most recent positions, wrapping once")
    func ringWindowIndexing() {
        for (ctx, kvLen, width) in [(8, 5, 6), (8, 8, 6), (8, 12, 6), (8, 13, 8), (576, 1000, 512), (576, 575, 512)] {
            let cache = ringCache(ctx: ctx, written: kvLen)
            let window = StaticShapeDrafter.KVWindow(kvLen: kvLen, width: width, ctx: ctx, ring: true)
            #expect(window.count == min(kvLen, width))
            for t in 0..<window.count {
                #expect(cache[window.slot(t)] == window.first + t, "ctx \(ctx) kvLen \(kvLen) t \(t)")
            }
        }
    }

    @Test("A flat window is the identity over the most recent positions")
    func flatWindowIndexing() {
        let window = StaticShapeDrafter.KVWindow(kvLen: 100, width: 64, ctx: 1000, ring: false)
        #expect(window.count == 64 && window.first == 36)
        #expect((0..<64).allSatisfy { window.slot($0) == 36 + $0 })
        let short = StaticShapeDrafter.KVWindow(kvLen: 10, width: 64, ctx: 1000, ring: false)
        #expect(short.count == 10 && short.first == 0 && short.slot(9) == 9)
    }
}
