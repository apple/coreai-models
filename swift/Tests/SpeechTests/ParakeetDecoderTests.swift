// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Argmax

/// Exercised through an `NDArray`, which is what the decode loop scans (`buffers.logits`), so
/// these pin the same call the runtime makes — including the scalar-type dispatch and the
/// `[1, 1, N]` layout the joint emits.
@Suite("TDT argmax")
struct TDTArgmaxTests {
    /// A `[1, 1, values.count]` logits row, shaped like the joint's output.
    private func logitsRow(
        _ values: [Float], scalarType: NDArray.ScalarType = .float32
    ) -> NDArray {
        var array = NDArray(shape: [1, 1, values.count], scalarType: scalarType)
        fillFloatNDArray(&array, with: values)
        return array
    }

    @Test("Returns the index of the largest value")
    func returnsLargest() {
        #expect(argmaxFloat(logitsRow([1, 5, 3]), in: 0..<3) == 1)
    }

    @Test("Ties go to the lowest index")
    func tiesGoLow() {
        // Documented contract, and it differs from WhisperDecoder's `indices.max(by:)`, which
        // returns the *last* maximal element. Pinned so the two are not accidentally unified.
        #expect(argmaxFloat(logitsRow([2, 2, 1]), in: 0..<3) == 0)
    }

    @Test("An all-negative-infinity range returns zero")
    func allNegativeInfinityReturnsZero() {
        #expect(argmaxFloat(logitsRow([-.infinity, -.infinity]), in: 0..<2) == 0)
    }

    @Test("Indices are relative to the range lower bound")
    func indicesAreRelative() {
        // The duration argmax indexes `cfg.durations` with this result, so an absolute index here
        // would read the wrong duration or run off the end.
        #expect(argmaxFloat(logitsRow([9, 9, 0, 7]), in: 2..<4) == 1)
    }

    @Test("A single-element range returns zero")
    func singleElementRange() {
        #expect(argmaxFloat(logitsRow([4, 8, 2]), in: 1..<2) == 0)
    }

    /// The scan converts as it reads, so an f16 row — what a `--dtype float16` bundle emits,
    /// and the case the decode loop usually runs — must order identically to f32.
    @Test("An f16 row scans the same as f32")
    func float16RowMatches() {
        let values: [Float] = [1, 5, 3, 5, 2]
        #expect(argmaxFloat(logitsRow(values, scalarType: .float16), in: 0..<5) == 1)
        #expect(argmaxFloat(logitsRow(values, scalarType: .float16), in: 2..<5) == 1)
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
            #expect(argmaxFloat(logitsRow(logits), in: 0..<vocabSize) == id, "id \(id)")
        }
    }

    @Test("Duration indices stay inside the durations array")
    func durationIndicesAreInRange() {
        let vocabSize = 1_030
        let durations = [0, 1, 2, 3, 4]
        for j in durations.indices {
            var logits = [Float](repeating: -1, count: vocabSize + durations.count)
            logits[vocabSize + j] = 10
            let index = argmaxFloat(
                logitsRow(logits), in: vocabSize..<(vocabSize + durations.count))
            #expect(index == j)
            #expect(durations.indices.contains(index))
        }
    }
}

// MARK: - Partial reads

/// `floatElements` is how a streaming hop converts only the frames it decodes, leaving the
/// window's left and right context unconverted. Getting the range arithmetic wrong would hand
/// the joint a frame of the wrong audio, so the offsets are pinned here.
@Suite("Encoder frame slicing")
struct FloatElementsTests {
    /// A `[1, frames, hidden]` encoder output, the shape the streaming path slices.
    private func encoderOutput(
        frames: Int, hidden: Int, scalarType: NDArray.ScalarType = .float32
    ) -> NDArray {
        var array = NDArray(shape: [1, frames, hidden], scalarType: scalarType)
        fillFloatNDArray(&array, with: (0..<(frames * hidden)).map { Float($0) })
        return array
    }

    @Test("Converts exactly the requested range, in row-major order")
    func convertsRequestedRange() {
        let array = encoderOutput(frames: 4, hidden: 3)
        #expect(floatElements(array, in: 0..<3) == [0, 1, 2])
        // Frame 2 of a hidden-3 output: the slice the decode loop takes per step.
        #expect(floatElements(array, in: 6..<9) == [6, 7, 8])
        #expect(floatElements(array, in: 0..<12).count == 12)
    }

    @Test("An empty range converts to nothing")
    func emptyRange() {
        #expect(floatElements(encoderOutput(frames: 2, hidden: 3), in: 3..<3).isEmpty)
    }

    /// The usual case at runtime: a `--dtype float16` bundle's encoder output, converted up.
    @Test("An f16 output converts to the same values as f32")
    func float16Matches() {
        let expected: [Float] = [3, 4, 5]
        #expect(floatElements(encoderOutput(frames: 4, hidden: 3), in: 3..<6) == expected)
        #expect(
            floatElements(encoderOutput(frames: 4, hidden: 3, scalarType: .float16), in: 3..<6)
                == expected)
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
