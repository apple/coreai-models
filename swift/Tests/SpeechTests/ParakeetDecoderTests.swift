// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Argmax

@Suite("TDT argmax")
struct TDTArgmaxTests {
    @Test("Returns the index of the largest value")
    func returnsLargest() {
        #expect(ParakeetTDTDecoder.argmax([1, 5, 3], in: 0..<3) == 1)
    }

    @Test("Ties go to the lowest index")
    func tiesGoLow() {
        // Documented contract, and it differs from WhisperDecoder's `indices.max(by:)`, which
        // returns the *last* maximal element. Pinned so the two are not accidentally unified.
        #expect(ParakeetTDTDecoder.argmax([2, 2, 1], in: 0..<3) == 0)
    }

    @Test("An all-negative-infinity range returns zero")
    func allNegativeInfinityReturnsZero() {
        #expect(ParakeetTDTDecoder.argmax([-.infinity, -.infinity], in: 0..<2) == 0)
    }

    @Test("Indices are relative to the range lower bound")
    func indicesAreRelative() {
        // The duration argmax indexes `cfg.durations` with this result, so an absolute index here
        // would read the wrong duration or run off the end.
        #expect(ParakeetTDTDecoder.argmax([9, 9, 0, 7], in: 2..<4) == 1)
    }

    @Test("A single-element range returns zero")
    func singleElementRange() {
        #expect(ParakeetTDTDecoder.argmax([4, 8, 2], in: 1..<2) == 0)
    }

    /// Half of the invariant the reviewer questioned: `lastToken == blankTokenId` only means "the
    /// previous step emitted blank" if blank is a value the token argmax can actually produce.
    /// Blank sits at the top of the vocab range, so it must be reachable.
    @Test("Every vocab id including blank is reachable by the token argmax")
    func blankIsReachable() {
        let vocabSize = 1_030
        let blank = 1_024
        for id in [0, 1, 512, 1_023, blank, vocabSize - 1] {
            var logits = [Float](repeating: -1, count: vocabSize + 5)
            logits[id] = 10
            #expect(ParakeetTDTDecoder.argmax(logits, in: 0..<vocabSize) == id, "id \(id)")
        }
    }

    @Test("Duration indices stay inside the durations array")
    func durationIndicesAreInRange() {
        let vocabSize = 1_030
        let durations = [0, 1, 2, 3, 4]
        for j in durations.indices {
            var logits = [Float](repeating: -1, count: vocabSize + durations.count)
            logits[vocabSize + j] = 10
            let index = ParakeetTDTDecoder.argmax(
                logits, in: vocabSize..<(vocabSize + durations.count))
            #expect(index == j)
            #expect(durations.indices.contains(index))
        }
    }
}

// MARK: - Decode preconditions

@Suite("TDT decode preconditions")
struct TDTValidationTests {
    private static func config(
        vocabSize: Int = 1_025, blankTokenId: Int32 = 1_024, hidden: Int = 640,
        durations: [Int] = [0, 1, 2, 3, 4]
    ) -> ParakeetTDTConfig {
        ParakeetTDTConfig(
            vocabSize: vocabSize, blankTokenId: blankTokenId, decoderHiddenSize: hidden,
            numDecoderLayers: 2, maxSymbolsPerStep: 10, durations: durations,
            encoderNumMelBins: 128, encoderSubsamplingFactor: 8)
    }

    private static func validate(
        shape: [Int] = [1, 100, 640], logitsSize: Int = 1_030, config: ParakeetTDTConfig
    ) throws {
        try ParakeetTDTDecoder.validate(
            encoderOutputShape: shape, logitsSize: logitsSize, config: config)
    }

    @Test("A well-formed configuration validates")
    func wellFormedValidates() throws {
        try Self.validate(config: Self.config())
    }

    /// The other half of the reviewer's invariant, and the guard the blank bookkeeping rests on.
    /// A blank id outside the argmax range could never win, so `isBlank` would never fire and every
    /// frame's argmax would be emitted as a real token.
    @Test("A blank id outside the vocab range is rejected", arguments: [1_025, 1_030, -1])
    func blankOutsideVocabIsRejected(blank: Int) {
        let cfg = Self.config(blankTokenId: Int32(blank))
        #expect(throws: (any Error).self) { try Self.validate(config: cfg) }
        do {
            try Self.validate(config: cfg)
            Issue.record("expected a throw for blank id \(blank)")
        } catch {
            #expect(String(describing: error).contains("vocab range"))
        }
    }

    @Test("A blank id at the top of the vocab is accepted")
    func blankAtTopOfVocabAccepted() throws {
        // Boundary in the permitted direction: vocabSize - 1 is the largest legal blank id.
        try Self.validate(config: Self.config(vocabSize: 1_025, blankTokenId: 1_024))
    }

    @Test("A joint logits width mismatch is rejected", arguments: [1_029, 1_031])
    func logitsWidthMismatchRejected(width: Int) {
        #expect(throws: (any Error).self) {
            try Self.validate(logitsSize: width, config: Self.config())
        }
    }

    @Test(
        "A non-rank-three encoder output is rejected",
        arguments: [[1, 100], [1, 100, 640, 1], [640]])
    func nonRankThreeRejected(shape: [Int]) {
        #expect(throws: (any Error).self) {
            try Self.validate(shape: shape, config: Self.config())
        }
    }

    @Test("A batch size other than one is rejected")
    func nonUnitBatchRejected() {
        #expect(throws: (any Error).self) {
            try Self.validate(shape: [2, 100, 640], config: Self.config())
        }
    }

    @Test("A hidden size mismatch is rejected")
    func hiddenSizeMismatchRejected() {
        #expect(throws: (any Error).self) {
            try Self.validate(shape: [1, 100, 512], config: Self.config(hidden: 640))
        }
    }

    @Test("A zero-length encoder output passes validation")
    func zeroLengthPassesValidation() throws {
        // `decode` handles tEnc == 0 by returning early, so validation must not reject it.
        try Self.validate(shape: [1, 0, 640], config: Self.config())
    }
}

// MARK: - DecodeStats

@Suite("DecodeStats")
struct DecodeStatsTests {
    @Test("Aggregates match the step times")
    func aggregatesMatch() {
        let stats = DecodeStats(stepTimesMs: [10, 20, 30])
        #expect(stats.stepCount == 3)
        #expect(abs(stats.avgLatencyMs - 20) < 1e-9)
        #expect(stats.minLatencyMs == 10)
        #expect(stats.maxLatencyMs == 30)
        #expect(abs(stats.stepsPerSecond - 50) < 1e-9)
    }

    @Test("Empty stats report zeros rather than NaN")
    func emptyStatsAreZero() {
        let stats = DecodeStats(stepTimesMs: [])
        #expect(stats.stepCount == 0)
        #expect(stats.avgLatencyMs == 0)
        #expect(stats.minLatencyMs == 0)
        #expect(stats.maxLatencyMs == 0)
        #expect(stats.stepsPerSecond == 0)
        #expect(stats.avgLatencyMs.isFinite)
        #expect(stats.stepsPerSecond.isFinite)
    }

    @Test("A single step reports identical aggregates")
    func singleStep() {
        let stats = DecodeStats(stepTimesMs: [4])
        #expect(stats.avgLatencyMs == 4)
        #expect(stats.minLatencyMs == 4)
        #expect(stats.maxLatencyMs == 4)
        #expect(abs(stats.stepsPerSecond - 250) < 1e-9)
    }

    @Test("Zero-duration steps do not divide by zero")
    func zeroDurationSteps() {
        let stats = DecodeStats(stepTimesMs: [0, 0])
        #expect(stats.avgLatencyMs == 0)
        #expect(stats.stepsPerSecond == 0)
        #expect(stats.stepsPerSecond.isFinite)
    }

    @Test("Coverage starts at zero and compares field by field")
    func coverageDefaultsAndEquality() {
        let zero = DecodeStats.Coverage()
        #expect(zero == DecodeStats.Coverage())
        #expect(zero.blankSkipReuses == 0)
        #expect(zero.lstmStateAdvances == 0)
        #expect(zero.blankZeroDurationBreaks == 0)
        #expect(zero.positiveDurationBreaks == 0)
        #expect(zero.symbolCapExhaustions == 0)
        #expect(zero.multiTokenSteps == 0)
        #expect(zero.blankOnlySteps == 0)

        var bumped = zero
        bumped.blankSkipReuses = 1
        #expect(bumped != zero)
    }

    @Test("Stats default to empty coverage and no captured step")
    func statsDefaults() {
        let stats = DecodeStats(stepTimesMs: [1])
        #expect(stats.coverage == DecodeStats.Coverage())
        #expect(stats.firstStep == nil)
    }
}
