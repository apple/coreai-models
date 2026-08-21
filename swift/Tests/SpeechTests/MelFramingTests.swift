// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Frame counting

/// Covers the review comment's first bullet: "off-by-one in frame counting for unusual audio
/// lengths". Every assertion here is integer arithmetic, so there is no tolerance to tune.
@Suite("Frame counting")
struct MelFrameCountTests {
    @Test("Dynamic frame count is one more than the valid count")
    func dynamicCountIsValidPlusOne() {
        // torch.stft(center: true) emits 1 + N/hop frames and HF masks the last one, so the
        // shapes must differ by exactly one for every length.
        for length in 0...800 {
            let total = MelSpectrogram.frameCount(forPCMLength: length, config: .parakeet)
            let valid = MelSpectrogram.validFrameCount(forPCMLength: length, config: .parakeet)
            #expect(total == valid + 1, "length \(length)")
        }
    }

    @Test(
        "Frame counts at and around hop boundaries",
        arguments: [0, 1, 159, 160, 161, 319, 320, 321, 16_000, 16_001])
    func frameCountsAtBoundaries(length: Int) {
        let config = MelConfig.parakeet
        #expect(
            MelSpectrogram.frameCount(forPCMLength: length, config: config)
                == 1 + length / config.hopLength)
        #expect(
            MelSpectrogram.validFrameCount(forPCMLength: length, config: config)
                == length / config.hopLength)
    }

    @Test("Output length always matches the advertised frame count")
    func outputLengthMatchesFrameCount() {
        // Sweep a whole hop's worth of lengths: an off-by-one in either direction shows up as a
        // mismatch between what `frameCount` promises and what `fromPCM` produces.
        let config = MelConfig.parakeet
        for length in stride(from: 0, through: 960, by: 37) {
            let mel = MelSpectrogram.fromPCM(deterministicNoise(count: length, seed: 7), config: config)
            let expected =
                config.nMelBins * MelSpectrogram.frameCount(forPCMLength: length, config: config)
            #expect(mel.count == expected, "length \(length)")
        }
    }

    @Test("Static configs clamp the valid count to the traced window")
    func staticConfigsClampValidCount() {
        let config = MelConfig.parakeet.withNFrames(40)
        #expect(MelSpectrogram.frameCount(forPCMLength: 3_200, config: config) == 40)
        #expect(MelSpectrogram.validFrameCount(forPCMLength: 3_200, config: config) == 20)
        #expect(MelSpectrogram.validFrameCount(forPCMLength: 6_400, config: config) == 40)
        // Audio longer than the window cannot report more valid frames than the window holds.
        #expect(MelSpectrogram.validFrameCount(forPCMLength: 128_000, config: config) == 40)
    }

    @Test("Empty PCM yields exactly one frame and does not crash")
    func emptyPCMYieldsOneFrame() {
        #expect(MelSpectrogram.frameCount(forPCMLength: 0, config: .parakeet) == 1)
        let mel = MelSpectrogram.fromPCM([], config: .parakeet)
        #expect(mel.count == MelConfig.parakeet.nMelBins)
        #expect(mel.allSatisfy { $0 == 0 })
    }

    @Test("The trailing dynamic frame is exactly zero in both layouts")
    func trailingFrameIsZero() {
        let pcm = deterministicNoise(count: 3_200, seed: 11)

        let timeMajor = MelSpectrogram.fromPCM(pcm, config: .parakeet)
        let bins = MelConfig.parakeet.nMelBins
        #expect(timeMajor.suffix(bins).allSatisfy { $0 == 0 })

        let channelConfig = MelConfig(
            sampleRate: 16_000, nFFT: 512, winLength: 400, hopLength: 160,
            nMelBins: bins, nFrames: nil, preemphasis: 0.97,
            normalization: .perBinMeanStd, layout: .channelMajor,
            padMode: .constant, melScale: .slaney)
        let channelMajor = MelSpectrogram.fromPCM(pcm, config: channelConfig)
        let total = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: channelConfig)
        for b in 0..<bins {
            #expect(channelMajor[b * total + (total - 1)] == 0, "bin \(b)")
        }
    }
}

// MARK: - Preprocessing primitives

@Suite("Preprocessing primitives")
struct MelPreprocessingTests {
    @Test("Preemphasis is a first-order difference")
    func preemphasisIsFirstOrderDifference() {
        let out = MelSpectrogram.applyPreemphasis([1, 2, 3, 4], alpha: 0.97)
        #expect(maxAbsDiff(out, [1, 2 - 0.97, 3 - 1.94, 4 - 2.91]) < 1e-6)
        // The first sample has no predecessor and passes through untouched.
        #expect(out[0] == 1)
    }

    @Test("Nil alpha and empty input pass through")
    func preemphasisPassthrough() {
        let x: [Float] = [3, 1, 4, 1, 5]
        #expect(MelSpectrogram.applyPreemphasis(x, alpha: nil) == x)
        #expect(MelSpectrogram.applyPreemphasis([], alpha: 0.97).isEmpty)
    }

    @Test("Constant padding surrounds the signal with zeros")
    func constantPadding() {
        #expect(
            MelSpectrogram.padAudio([1, 2, 3], pad: 2, mode: .constant) == [0, 0, 1, 2, 3, 0, 0])
    }

    @Test("Reflect padding mirrors without repeating the edge sample")
    func reflectPadding() {
        // Matches numpy/torch `pad_mode="reflect"`: the boundary sample is not duplicated.
        #expect(
            MelSpectrogram.padAudio([1, 2, 3, 4, 5], pad: 2, mode: .reflect)
                == [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }

    @Test("Padded length is the input plus twice the pad", arguments: [1, 2, 5])
    func paddedLength(pad: Int) {
        let x: [Float] = [1, 2, 3, 4, 5, 6]
        for mode in [MelConfig.PadMode.constant, .reflect] {
            #expect(MelSpectrogram.padAudio(x, pad: pad, mode: mode).count == x.count + 2 * pad)
        }
    }

    @Test("A static grid pads short audio and truncates long audio")
    func staticGridPadsAndTruncates() {
        let config = MelConfig.parakeet.withNFrames(40)  // 40 * 160 = 6400 samples
        let (short, shortValid) = MelSpectrogram.padToFrameGrid(
            deterministicNoise(count: 3_200, seed: 3), config: config)
        #expect(short.count == 6_400)
        #expect(shortValid == 20)

        let (long, longValid) = MelSpectrogram.padToFrameGrid(
            deterministicNoise(count: 12_800, seed: 3), config: config)
        #expect(long.count == 6_400)
        #expect(longValid == 40)
    }

    /// Regression: the dynamic branch used to round the waveform down to a whole number of hops,
    /// discarding up to `hopLength - 1` real samples that the final frames' windows still reach.
    @Test("The dynamic grid hands the waveform through whole")
    func dynamicGridDoesNotTruncate() {
        let pcm = deterministicNoise(count: 1_640, seed: 5)  // 1640 % 160 == 40
        let (audio, valid) = MelSpectrogram.padToFrameGrid(pcm, config: .parakeet)
        #expect(audio.count == 1_640)
        #expect(audio == pcm)
        #expect(valid == 10)
    }
}

// MARK: - STFT framing geometry

/// The alignment tests. These are the only ones that can detect a window read at the wrong
/// offset, and they work because an impulse is sharply localized in time: shifting a *steady*
/// signal by a fraction of a frame barely perturbs its magnitude spectrum, so a tone cannot
/// distinguish the two framings.
@Suite("STFT framing geometry")
struct MelFramingGeometryTests {
    /// Parakeet's geometry, but few enough mel bins to keep the test cheap. `winLength < nFFT`
    /// is essential — that is the only regime where the centring offset exists.
    private static let geometry = MelConfig(
        sampleRate: 16_000, nFFT: 512, winLength: 400, hopLength: 160,
        nMelBins: 16, nFrames: nil, preemphasis: nil,
        normalization: .perInstanceMeanStd, layout: .timeMajor,
        padMode: .constant, melScale: .slaney, windowPeriodicity: .symmetric)

    @Test("Audio outside a frame's window contributes nothing")
    func audioOutsideTheWindowIsExcluded() {
        // Correct framing windows padded[offset + frameOffset ..< offset + frameOffset + winLength].
        // With frameOffset 56 and one hop of audio that is source range [-200, 200), which contains
        // an impulse at index 150. Reading from `offset` instead windows [-256, 144) and misses it,
        // leaving every bin on the log-zero floor — and therefore all equal. So a *spread* across
        // bins is the signal: present when the impulse is inside the window, absent when it is not.
        let config = Self.geometry
        let pcm = makeImpulse(at: 150, count: 160)
        let mel = MelSpectrogram.fromPCM(pcm, config: config)

        let frame0 = Array(mel[0..<config.nMelBins])
        let spread = (frame0.max() ?? 0) - (frame0.min() ?? 0)
        #expect(spread > 1e-3, "frame 0 is flat, so the impulse fell outside the window")
    }

    @Test("Mel frames are centred on hop multiples")
    func framesAreCentredOnHopMultiples() {
        // The frame whose window peak sits on an impulse at padded index p is
        // round((p - 255.5)/160) when the window is read from `offset + frameOffset`, and
        // round((p - 199.5)/160) when it is read from `offset` — a 0.35-frame difference that only
        // changes the argmax where the fractional part crosses a half. Sample 856 (padded 1112) is
        // such a point: 5.35 rounds to 5, while the misaligned 5.70 rounds to 6. An impulse at a
        // hop multiple would peak at the same frame either way and prove nothing.
        let config = Self.geometry
        let pcm = makeImpulse(at: 856, count: 11 * config.hopLength)
        let mel = MelSpectrogram.fromPCM(pcm, config: config)

        let total = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: config)
        let valid = MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: config)
        #expect(total == 12)  // 1 + 1760/160
        #expect(valid == 11)

        // `perInstanceMeanStd` is a single affine map over the whole array, so it preserves the
        // ordering of per-frame energies and the argmax is unaffected by normalization.
        var energies: [Float] = []
        for t in 0..<valid {
            let frame = mel[(t * config.nMelBins)..<((t + 1) * config.nMelBins)]
            energies.append(frame.reduce(0, +))
        }
        let peak = energies.indices.max { energies[$0] < energies[$1] }
        #expect(peak == 5, "energy peaked at frame \(peak ?? -1), expected 5")
    }

    /// Regression: audio past the last whole hop used to be discarded. Dropping it leaves every
    /// frame identical, and `perBinMeanStd` then divides near-zero deviations by `std + 1e-5` —
    /// which amplifies float roundoff rather than producing zeros. So "contains a nonzero" is not
    /// a usable signal; the discriminator is *magnitude*. One deviant frame in ten yields a z-score
    /// around 2.8, whereas amplified roundoff stays far below 1.
    @Test("Tail samples beyond whole hops reach the last frame")
    func tailSamplesReachTheLastFrame() {
        let pcm = makeImpulse(at: 1_600, count: 1_640)  // 1640 = 10*160 + 40
        let mel = MelSpectrogram.fromPCM(pcm, config: .parakeet)
        let peak = mel.map(abs).max() ?? 0
        #expect(peak > 2.0, "peak magnitude \(peak) is too small — the tail sample was discarded")
    }

    @Test("Impulses anywhere in the final partial hop stay finite")
    func impulsesInFinalHopStayFinite() {
        for index in 1_600..<1_640 {
            let mel = MelSpectrogram.fromPCM(makeImpulse(at: index, count: 1_640), config: .parakeet)
            #expect(mel.allSatisfy { $0.isFinite }, "impulse at \(index) produced a non-finite value")
        }
    }
}

// MARK: - Encoder frame mapping

@Suite("Encoder frame mapping")
struct EncoderFrameMappingTests {
    @Test("Layout determines the encoder input shape")
    func layoutDeterminesShape() {
        #expect(
            SpeechRecognitionModel.encoderInputShape(nFrames: 3_000, config: .whisper)
                == [1, 128, 3_000])
        #expect(
            SpeechRecognitionModel.encoderInputShape(nFrames: 101, config: .parakeet)
                == [1, 101, 128])
    }

    @Test("Dynamic exports count every real-audio frame, excluding the trailing zero frame")
    func dynamicExcludesOnlyThePaddedFrame() {
        // 16 000 samples = 100 real mel frames; the dynamic path appends one zero frame to
        // match HF's shape, and 100 frames subsample to 13.
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 16_000, tEnc: 13, config: .parakeet, subsamplingFactor: 8) == 13)
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 0, tEnc: 5, config: .parakeet, subsamplingFactor: 8) == 0)
    }

    @Test("A static window's padded tail is excluded exactly")
    func staticExcludesPaddingExactly() {
        // Whisper: 1000 of 3000 mel frames are real, subsampling 2 -> (1000-1)/2+1 = 500.
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 1_000 * 160, tEnc: 1_500, config: .whisper, subsamplingFactor: 2) == 500)
    }

    /// The regression this pins: a proportional estimate rounded to nearest lands one frame
    /// low for ~25% of audio lengths against the shipped 21 s geometry, clipping a trailing
    /// token. Nine real mel frames subsample to 2 (9 -> 5 -> 3 -> 2), where the ratio
    /// 9/2101 x 263 rounds to 1.
    @Test("The boundary frame is derived, not rounded")
    func boundaryFrameIsExact() {
        let config = MelConfig.parakeet.withNFrames(2_101)
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 9 * 160, tEnc: 263, config: config, subsamplingFactor: 8) == 2)
        // 743 real frames (7.43 s) -> 743 -> 372 -> 186 -> 93.
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 743 * 160, tEnc: 263, config: config, subsamplingFactor: 8) == 93)
    }

    @Test("The valid count never exceeds the encoder's own length")
    func validCountIsClamped() {
        let config = MelConfig.parakeet.withNFrames(100)
        // Audio beyond the traced window cannot exceed the encoder's own output length.
        #expect(
            SpeechRecognitionModel.validEncoderFrames(
                pcmCount: 1_000_000, tEnc: 10, config: config, subsamplingFactor: 8) == 10)
    }

    /// Regression: the static-encoder path rebuilt `MelConfig` field by field and omitted
    /// `windowPeriodicity`, silently taking the initializer default.
    @Test("A static mel config differs from the preset only in nFrames")
    func staticConfigDiffersOnlyInFrameCount() {
        let derived = SpeechRecognitionBundle.melConfig(forEncoderTimeDim: 1_500)
        let preset = MelConfig.parakeet
        #expect(derived.nFrames == 1_500)
        #expect(derived.windowPeriodicity == preset.windowPeriodicity)
        #expect(derived.sampleRate == preset.sampleRate)
        #expect(derived.nFFT == preset.nFFT)
        #expect(derived.winLength == preset.winLength)
        #expect(derived.hopLength == preset.hopLength)
        #expect(derived.nMelBins == preset.nMelBins)
        #expect(derived.preemphasis == preset.preemphasis)
        #expect(derived.normalization == preset.normalization)
        #expect(derived.layout == preset.layout)
        #expect(derived.padMode == preset.padMode)
        #expect(derived.melScale == preset.melScale)
    }

    @Test("A dynamic encoder dimension leaves nFrames nil", arguments: [-1, 0])
    func dynamicEncoderDimLeavesFramesNil(dim: Int) {
        #expect(SpeechRecognitionBundle.melConfig(forEncoderTimeDim: dim).nFrames == nil)
    }
}
