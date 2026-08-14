// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Per-instance mean/std

/// Covers the review comment's third bullet: "the `perInstanceMeanStd` normalization path
/// iterating linearly regardless of `layout` (incorrect for `channelMajor`)".
///
/// No shipping preset uses `perInstanceMeanStd`, so this path is reachable only through a
/// hand-built `MelConfig` — which is exactly why an integration test cannot see it.
@Suite("Per-instance mean/std normalization")
struct PerInstanceNormalizationTests {
    // 4 bins x 5 allocated frames, of which 3 carry audio. channelMajor puts each bin's frames
    // contiguously, so the padding sits *between* bins rather than after all the data — the
    // arrangement a flat "first validFrames*nMelBins elements" loop gets wrong.
    private static let bins = 4
    private static let validFrames = 3
    private static let totalFrames = 5

    private static func index(_ t: Int, _ b: Int) -> Int { b * totalFrames + t }

    /// Valid entries hold 1...12 in bin-major order; padding holds 0.
    private static func makeChannelMajorInput() -> [Float] {
        var mel = [Float](repeating: 0, count: bins * totalFrames)
        for b in 0..<bins {
            for t in 0..<validFrames {
                mel[index(t, b)] = Float(b * validFrames + t + 1)
            }
        }
        return mel
    }

    @Test("Channel-major padding is left untouched")
    func channelMajorPaddingUntouched() {
        var mel = Self.makeChannelMajorInput()
        MelSpectrogram.normalize(
            &mel, normalization: .perInstanceMeanStd, validFrames: Self.validFrames,
            nMelBins: Self.bins, layout: .channelMajor)

        for b in 0..<Self.bins {
            for t in Self.validFrames..<Self.totalFrames {
                #expect(mel[Self.index(t, b)] == 0, "padding at bin \(b) frame \(t) was written")
            }
        }
    }

    @Test("Channel-major valid entries are z-scored over the whole valid set")
    func channelMajorValidEntriesAreZScored() {
        var mel = Self.makeChannelMajorInput()
        MelSpectrogram.normalize(
            &mel, normalization: .perInstanceMeanStd, validFrames: Self.validFrames,
            nMelBins: Self.bins, layout: .channelMajor)

        // Values 1...12: mean 6.5, population variance 143/12.
        let mean = 6.5
        let std = (143.0 / 12.0).squareRoot()
        for b in 0..<Self.bins {
            for t in 0..<Self.validFrames {
                let original = Double(b * Self.validFrames + t + 1)
                let expected = (original - mean) / max(std, 1e-5)
                let actual = Double(mel[Self.index(t, b)])
                #expect(abs(actual - expected) < 1e-5, "bin \(b) frame \(t)")
            }
        }
    }

    /// Control: with `timeMajor` the flat shortcut happens to be correct, so this proves the fix
    /// did not break the layout that already worked.
    @Test("Time-major valid entries are z-scored over the whole valid set")
    func timeMajorValidEntriesAreZScored() {
        var mel = [Float](repeating: 0, count: Self.bins * Self.totalFrames)
        for t in 0..<Self.validFrames {
            for b in 0..<Self.bins {
                mel[t * Self.bins + b] = Float(t * Self.bins + b + 1)
            }
        }
        MelSpectrogram.normalize(
            &mel, normalization: .perInstanceMeanStd, validFrames: Self.validFrames,
            nMelBins: Self.bins, layout: .timeMajor)

        let mean = 6.5
        let std = (143.0 / 12.0).squareRoot()
        for i in 0..<(Self.validFrames * Self.bins) {
            let expected = (Double(i + 1) - mean) / max(std, 1e-5)
            #expect(abs(Double(mel[i]) - expected) < 1e-5, "index \(i)")
        }
        for i in (Self.validFrames * Self.bins)..<mel.count {
            #expect(mel[i] == 0, "padding at \(i) was written")
        }
    }

    @Test("Zero variance does not divide by zero")
    func zeroVarianceIsSafe() {
        var mel = [Float](repeating: 7, count: 8)
        MelSpectrogram.normalize(
            &mel, normalization: .perInstanceMeanStd, validFrames: 2, nMelBins: 4,
            layout: .timeMajor)
        // std 0 is clamped to 1e-5, so every deviation of exactly 0 maps to exactly 0.
        #expect(mel.allSatisfy { $0 == 0 })
        #expect(mel.allSatisfy { $0.isFinite })
    }

    @Test("Zero valid frames is a no-op")
    func zeroValidFramesIsNoOp() {
        var mel: [Float] = [1, 2, 3, 4]
        MelSpectrogram.normalize(
            &mel, normalization: .perInstanceMeanStd, validFrames: 0, nMelBins: 4,
            layout: .timeMajor)
        #expect(mel == [1, 2, 3, 4])
    }
}

// MARK: - Per-bin mean/std

@Suite("Per-bin mean/std normalization")
struct PerBinNormalizationTests {
    @Test("Each bin is normalized independently")
    func binsAreIndependent() {
        // Bin 1 gets a wildly different range; bins 0 and 2 must be unaffected by it.
        func input() -> [Float] {
            var mel = [Float](repeating: 0, count: 3 * 4)
            for t in 0..<4 {
                mel[t * 3 + 0] = Float(t)
                mel[t * 3 + 1] = Float(t) * 10
                mel[t * 3 + 2] = Float(t) - 100
            }
            return mel
        }
        var base = input()
        MelSpectrogram.normalize(
            &base, normalization: .perBinMeanStd, validFrames: 4, nMelBins: 3, layout: .timeMajor)

        var shifted = input()
        for t in 0..<4 { shifted[t * 3 + 1] += 1_000 }
        MelSpectrogram.normalize(
            &shifted, normalization: .perBinMeanStd, validFrames: 4, nMelBins: 3,
            layout: .timeMajor)

        for t in 0..<4 {
            #expect(base[t * 3 + 0] == shifted[t * 3 + 0], "bin 0 changed at frame \(t)")
            #expect(base[t * 3 + 2] == shifted[t * 3 + 2], "bin 2 changed at frame \(t)")
        }
    }

    @Test("The standard deviation is Bessel-corrected")
    func standardDeviationIsBesselCorrected() {
        // One bin holding [0, 2]: mean 1, Bessel std sqrt(2/1) = 1.41421, population std 1.0.
        // The two conventions give 0.7071 vs 1.0, so this distinguishes them.
        var mel: [Float] = [0, 2]
        MelSpectrogram.normalize(
            &mel, normalization: .perBinMeanStd, validFrames: 2, nMelBins: 1, layout: .timeMajor)
        let expected = 1.0 / (2.0.squareRoot() + 1e-5)
        #expect(abs(Double(mel[1]) - expected) < 1e-5)
        #expect(abs(Double(mel[0]) + expected) < 1e-5)
    }

    @Test("Normalized bins have zero mean and unit deviation")
    func normalizedBinsAreStandardized() {
        let bins = 4
        let frames = 16
        var mel = deterministicNoise(count: bins * frames, seed: 21, amplitude: 8)
        MelSpectrogram.normalize(
            &mel, normalization: .perBinMeanStd, validFrames: frames, nMelBins: bins,
            layout: .timeMajor)
        for b in 0..<bins {
            let column = (0..<frames).map { mel[$0 * bins + b] }
            let (mean, std) = meanAndStd(column)
            #expect(abs(mean) < 1e-4, "bin \(b) mean \(mean)")
            #expect(abs(std - 1) < 1e-3, "bin \(b) std \(std)")
        }
    }

    @Test("Fewer than two valid frames is a no-op")
    func singleFrameIsNoOp() {
        // The Bessel denominator would be zero, so the implementation returns early and leaves
        // the raw log-mel in place.
        var mel: [Float] = [5, 6, 7]
        MelSpectrogram.normalize(
            &mel, normalization: .perBinMeanStd, validFrames: 1, nMelBins: 3, layout: .timeMajor)
        #expect(mel == [5, 6, 7])
    }

    @Test("Channel-major padding is left untouched")
    func channelMajorPaddingUntouched() {
        let bins = 3
        let total = 5
        let valid = 3
        var mel = [Float](repeating: 0, count: bins * total)
        for b in 0..<bins {
            for t in 0..<valid { mel[b * total + t] = Float(b * 10 + t) }
        }
        MelSpectrogram.normalize(
            &mel, normalization: .perBinMeanStd, validFrames: valid, nMelBins: bins,
            layout: .channelMajor)
        for b in 0..<bins {
            for t in valid..<total {
                #expect(mel[b * total + t] == 0, "padding at bin \(b) frame \(t) was written")
            }
        }
    }

    @Test("A constant bin normalizes to zero rather than NaN")
    func constantBinIsFinite() {
        // Epsilon is added inside the divide rather than clamping the std, so 0/(0+1e-5) == 0.
        var mel = [Float](repeating: -16.6, count: 4)
        MelSpectrogram.normalize(
            &mel, normalization: .perBinMeanStd, validFrames: 4, nMelBins: 1, layout: .timeMajor)
        #expect(mel.allSatisfy { $0.isFinite })
        #expect(mel.allSatisfy { abs($0) < 1e-3 })
    }
}

// MARK: - Whisper log-clip

@Suite("Whisper log-clip normalization")
struct WhisperNormalizationTests {
    @Test("Silence normalizes to a constant minus one and a half")
    func silenceNormalizesToConstant() {
        // log10(1e-10) = -10; the clip floor is max - 8 = -10, so every entry becomes
        // (-10 + 4)/4 = -1.5. Covers the review comment's "silence" edge case.
        let mel = MelSpectrogram.fromPCM(
            [Float](repeating: 0, count: 48_000), config: .whisper)
        #expect(mel.count == MelConfig.whisper.nMelBins * 3_000)
        #expect(mel.allSatisfy { abs($0 + 1.5) < 1e-6 })
    }

    @Test("Empty PCM behaves as silence")
    func emptyPCMBehavesAsSilence() {
        let mel = MelSpectrogram.fromPCM([], config: .whisper)
        #expect(mel.count == MelConfig.whisper.nMelBins * 3_000)
        #expect(mel.allSatisfy { $0.isFinite })
        #expect(mel.allSatisfy { abs($0 + 1.5) < 1e-6 })
    }

    @Test("Output spans exactly two units")
    func outputSpansTwoUnits() {
        // The maximum maps to (m+4)/4 and the floor to (m-8+4)/4, a difference of exactly 2.
        let mel = MelSpectrogram.fromPCM(deterministicNoise(count: 16_000, seed: 31), config: .whisper)
        let maxValue = mel.max() ?? 0
        let minValue = mel.min() ?? 0
        #expect(abs((maxValue - minValue) - 2.0) < 1e-5)
        #expect(mel.allSatisfy { $0 >= maxValue - 2.0 - 1e-5 })
    }

    @Test("Whisper normalizes its padded tail rather than masking it")
    func whisperNormalizesPaddedTail() {
        // Unlike perBinMeanStd, the whisper path computes every frame of the fixed window and
        // treats the zero-padded tail as log-silence, so nothing is left at exactly zero.
        let mel = MelSpectrogram.fromPCM(deterministicNoise(count: 16_000, seed: 33), config: .whisper)
        #expect(!mel.contains { $0 == 0 })
    }
}

// MARK: - Parakeet normalization edge cases

@Suite("Parakeet normalization edge cases")
struct ParakeetNormalizationEdgeTests {
    @Test("Sub-hop audio produces a single zero frame")
    func subHopAudioProducesOneZeroFrame() {
        // Fewer samples than one hop means zero valid frames, so nothing is computed and the
        // single allocated frame stays zero. Covers "very short audio".
        let mel = MelSpectrogram.fromPCM(deterministicNoise(count: 100, seed: 41), config: .parakeet)
        #expect(mel.count == MelConfig.parakeet.nMelBins)
        #expect(mel.allSatisfy { $0 == 0 })
    }

    @Test("Exactly one hop leaves the log-mel unnormalized")
    func oneHopLeavesLogMelUnnormalized() {
        // validFrames == 1 trips the Bessel early return, so frame 0 holds raw log values.
        // Covers "boundary-length clips".
        let mel = MelSpectrogram.fromPCM(deterministicNoise(count: 160, seed: 43), config: .parakeet)
        let bins = MelConfig.parakeet.nMelBins
        #expect(mel.count == bins * 2)
        #expect(mel.allSatisfy { $0.isFinite })
        // Raw log-mel sits far below zero on average (log of sub-unit power), unlike the
        // zero-mean output normalization would produce. Individual bins can exceed unit power,
        // so this is a statement about the mean, not about every entry.
        let (mean, _) = meanAndStd(Array(mel[0..<bins]))
        #expect(mean < -1, "frame 0 mean \(mean) looks normalized rather than raw")
        #expect(mel[bins..<(2 * bins)].allSatisfy { $0 == 0 })
    }

    @Test("Digital silence is finite and free of NaN")
    func silenceIsFinite() {
        // Every bin is constant, so the per-bin std is zero and the divide amplifies roundoff.
        // The reference implementation divides by `features_lengths - 1` here and produces NaN,
        // so only this side's behaviour can be asserted: finite, bounded, no NaN.
        let mel = MelSpectrogram.fromPCM(
            [Float](repeating: 0, count: 16_000), config: .parakeet)
        #expect(mel.allSatisfy { $0.isFinite })
        #expect(!mel.contains { $0.isNaN })
    }

    @Test(
        "Output stays finite across boundary lengths",
        arguments: [159, 160, 161, 319, 320, 321, 1_600, 1_601])
    func outputStaysFiniteAcrossLengths(length: Int) {
        let mel = MelSpectrogram.fromPCM(
            deterministicNoise(count: length, seed: 47), config: .parakeet)
        let expected =
            MelConfig.parakeet.nMelBins
            * MelSpectrogram.frameCount(forPCMLength: length, config: .parakeet)
        #expect(mel.count == expected)
        #expect(mel.allSatisfy { $0.isFinite })
    }
}
