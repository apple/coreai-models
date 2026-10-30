// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

/// The speculative verify paths pick a token from the raw logit row. They must pick the one plain
/// decoding picks: it caps the row, rounds to `LogitsScalarType`, and takes the first maximum.
@Suite("Logit row scan")
struct LogitRowScanTests {
    /// What plain decoding does with a row.
    private func plainGreedy(_ row: [LogitsScalarType], cap: Float?) -> Int32 {
        var logits = row
        if let cap { LogitSoftcapProcessor.apply(to: &logits, cap: cap) }
        return CompositeSampler.sample(from: &logits, config: .greedy)
    }

    private func scanArgmax(_ row: [LogitsScalarType], cap: Float?) -> Int32 {
        var widened = row.map { Float($0) }
        return widened.withUnsafeMutableBufferPointer {
            LogitRowScan.topK($0, k: 1, cap: cap).indices.first ?? 0
        }
    }

    @Test("Two logits that tie after the cap resolve to the lowest index, as in plain decoding")
    func capCollapsesNearTies() {
        var row = [LogitsScalarType](repeating: 0, count: 8)
        row[2] = 40.0
        row[6] = LogitsScalarType(40.0).nextUp
        #expect(plainGreedy(row, cap: 30) == 2)
        #expect(scanArgmax(row, cap: 30) == 2)
        // Without a cap the larger logit wins.
        #expect(scanArgmax(row, cap: nil) == 6)
    }

    @Test("The capped argmax equals plain decoding on random rows full of near-ties")
    func randomRowsMatchPlainDecoding() {
        var rng = SplitMix(5)
        var mismatches: [String] = []
        for cap: Float in [15, 30, 50] {
            for _ in 0..<1500 {
                let n = [3, 8, 64, 700][Int(rng.next() % 4)]
                let spread = [4.0, 40.0, 120.0][Int(rng.next() % 3)]
                let low = Double(rng.next() % 40) - 20
                let row = (0..<n).map { _ in
                    LogitsScalarType(Float(low + spread * Double(rng.next() % 4000) / 4000))
                }
                let expected = plainGreedy(row, cap: cap)
                let got = scanArgmax(row, cap: cap)
                if got != expected { mismatches.append("cap \(cap): scan \(got), plain \(expected)") }
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches, first: \(mismatches.first ?? "")")
    }

    @Test("Infinities and large magnitudes rank as plain decoding ranks them")
    func infinitiesMatchPlainDecoding() {
        var rng = SplitMix(8)
        let specials: [LogitsScalarType] = [
            .infinity, -.infinity, 65504, -65504, 0, 100, 29.99, 30, 30.03, 31,
        ]
        var mismatches = 0
        for _ in 0..<3000 {
            let n = 1 + Int(rng.next() % 12)
            let row = (0..<n).map { _ in specials[Int(rng.next() % UInt64(specials.count))] }
            for cap: Float? in [nil, 30] where plainGreedy(row, cap: cap) != scanArgmax(row, cap: cap) {
                mismatches += 1
            }
        }
        #expect(mismatches == 0)
    }

    @Test("The top k are ordered by capped value, then index, and match capping the whole row")
    func topKMatchesCappingTheWholeRow() {
        var rng = SplitMix(13)
        var mismatches: [String] = []
        for _ in 0..<2000 {
            let n = 1 + Int(rng.next() % 90)
            let k = 1 + Int(rng.next() % 10)
            let cap: Float? = rng.next() % 4 == 0 ? nil : [20, 30, 50][Int(rng.next() % 3)]
            let row = (0..<n).map { _ in LogitsScalarType(Float(rng.next() % 1200) / 20) }
            var scratch = row.map { Float($0) }
            let got = scratch.withUnsafeMutableBufferPointer { LogitRowScan.topK($0, k: k, cap: cap) }
            var capped = row
            if let cap { LogitSoftcapProcessor.apply(to: &capped, cap: cap) }
            let order = capped.indices.sorted { a, b in
                capped[a] != capped[b] ? capped[a] > capped[b] : a < b
            }.prefix(k)
            if got.indices != order.map({ Int32($0) }) || got.values != order.map({ Float(capped[$0]) }) {
                mismatches.append("n \(n) k \(k) cap \(String(describing: cap))")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches, first: \(mismatches.first ?? "")")
    }

    @Test("Fewer finite entries than k fills with the lowest unused indices, never a repeat")
    func topKWithFewFiniteEntries() {
        var scratch: [Float] = [-.infinity, 2, -.infinity, -.infinity, 1]
        let got = scratch.withUnsafeMutableBufferPointer { LogitRowScan.topK($0, k: 4, cap: nil) }
        #expect(got.indices == [1, 4, 0, 2])
        #expect(Set(got.indices).count == 4)
    }

    @Test("A NaN ranks as negative infinity")
    func nanRanksLowest() {
        var scratch: [Float] = [.nan, 1, 3, .nan, 3]
        let top = scratch.withUnsafeMutableBufferPointer { LogitRowScan.topK($0, k: 3, cap: nil) }
        #expect(top.indices == [2, 4, 1])
        var allNaN: [Float] = [.nan, .nan, .nan]
        #expect(allNaN.withUnsafeMutableBufferPointer { LogitRowScan.topK($0, k: 1, cap: 30).indices } == [0])
        // A NaN at index 0 does not pin the result to 0.
        let row: [Float] = [.nan, 0, 5, 1]
        #expect(row.withUnsafeBufferPointer { LogitRowScan.argmax($0) } == 2)
    }

    @Test("The engine's row scan applies the softcap the way plain decoding does")
    func engineRowScanMatchesPlainDecoding() {
        var rng = SplitMix(31)
        var scratch: [Float] = []
        var mismatches = 0
        for _ in 0..<400 {
            let vocab = [5, 300, 4096][Int(rng.next() % 3)]
            let row = (0..<vocab).map { _ in LogitsScalarType(Float(rng.next() % 1600) / 40) }
            for cap: Float? in [nil, 30] {
                let top = row.withUnsafeBufferPointer {
                    StaticShapeEngine.scanRow(
                        $0.baseAddress!, vocab: vocab, count: 1, softcap: cap, scratch: &scratch)
                }
                if top.indices.first != plainGreedy(row, cap: cap) { mismatches += 1 }
            }
        }
        #expect(mismatches == 0)
    }

    @Test("The first maximum wins a tie, whatever the row length")
    func firstMaximumWinsTies() {
        var rng = SplitMix(21)
        for n in [1, 2, 7, 8, 9, 31, 64, 1000, 4097, 262_144] {
            for _ in 0..<5 {
                let row = (0..<n).map { _ in Float(rng.next() % 3) }
                let expected = row.indices.first { row[$0] == row.max()! }!
                #expect(row.withUnsafeBufferPointer { LogitRowScan.argmax($0) } == expected, "n \(n)")
            }
        }
    }

    @Test("An empty row selects index 0")
    func emptyRow() {
        let row: [Float] = []
        #expect(row.withUnsafeBufferPointer { LogitRowScan.argmax($0) } == 0)
        var scratch: [Float] = []
        #expect(scratch.withUnsafeMutableBufferPointer { LogitRowScan.topK($0, k: 3, cap: 30).indices }.isEmpty)
    }
}
