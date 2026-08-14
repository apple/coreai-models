// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Golden values

/// Golden-value tests against the HuggingFace reference, for both `MelConfig` presets — the
/// remaining ask from the review comment about this file's lack of test coverage.
///
/// The expected numbers below were produced by running `ParakeetFeatureExtractor` and
/// `WhisperFeatureExtractor` over the signal `goldenTestSignal` builds, then pasted here. They are
/// **not** captured from this implementation's own output: values recorded from the code under test
/// would lock in whatever it happens to do, and would have passed happily while the front-end
/// carried a window misalignment, a truncated waveform, an HTK filterbank and a symmetric window.
///
/// To regenerate, build the same closed-form signal in Python and print
/// `fe(audio, sampling_rate=16000)["input_features"]` at the probe coordinates:
///
///     audio[n] = 0.50·sin(2π·220·n/16000) + 0.25·sin(2π·1500·n/16000) + 0.10·sin(2π·5000·n/16000)
///
/// accumulated in float64 and narrowed to float32, with n in 0..<1723. `math.sin` rather than
/// `numpy.sin`, so both languages call the same libm and see bit-identical input.
///
/// 1723 samples is deliberately not a whole number of hops (1723 % 160 == 123), so the framing tail
/// is inside the comparison.
@Suite("Mel golden values")
struct MelGoldenValueTests {
    private static let sampleCount = 1_723

    /// Tolerance is tighter than the ~2e-3 that a `Float`-domain DFT twiddle error produces, so
    /// that regression cannot slip past, while staying well above the ~1e-5 of genuine float
    /// disagreement between this implementation and the reference.
    private static let tolerance: Float = 5e-4

    // MARK: Parakeet — timeMajor [1, T, nMelBins], T = 11 for 1723 samples

    private static let parakeetProbes: [(frame: Int, bin: Int, value: Float)] = [
        (frame: 0, bin: 0, value: 2.6243355),
        (frame: 0, bin: 5, value: 2.8425014),
        (frame: 0, bin: 64, value: 2.6714416),
        (frame: 0, bin: 127, value: 2.4562137),
        (frame: 3, bin: 10, value: 0.3125439),
        (frame: 5, bin: 40, value: -0.4846673),
        (frame: 9, bin: 0, value: -0.8138530),
        (frame: 9, bin: 100, value: -0.4264047),
        (frame: 9, bin: 127, value: -0.4515118),
    ]

    @Test("Parakeet features match the HuggingFace reference")
    func parakeetMatchesReference() {
        let config = MelConfig.parakeet
        let mel = MelSpectrogram.fromPCM(goldenTestSignal(count: Self.sampleCount), config: config)
        let frames = MelSpectrogram.frameCount(forPCMLength: Self.sampleCount, config: config)
        #expect(frames == 11)
        #expect(mel.count == config.nMelBins * frames)

        for probe in Self.parakeetProbes {
            let actual = mel[probe.frame * config.nMelBins + probe.bin]
            #expect(
                abs(actual - probe.value) < Self.tolerance,
                "frame \(probe.frame) bin \(probe.bin): got \(actual), expected \(probe.value)")
        }
    }

    @Test("Parakeet frame statistics match the reference")
    func parakeetFrameStatisticsMatch() {
        // Cheap breadth over a whole frame, so a defect confined to bins the probes skip still
        // shows up.
        let config = MelConfig.parakeet
        let mel = MelSpectrogram.fromPCM(goldenTestSignal(count: Self.sampleCount), config: config)
        let frame = Array(mel[(5 * config.nMelBins)..<(6 * config.nMelBins)])
        let (mean, _) = meanAndStd(frame)
        #expect(abs(mean - -0.3842894) < 1e-3, "frame 5 mean \(mean)")
        #expect(abs((frame.min() ?? 0) - -0.7801333) < Self.tolerance)
        #expect(abs((frame.max() ?? 0) - 0.3192385) < Self.tolerance)
    }

    // MARK: Whisper — channelMajor [1, nMelBins, nFrames], nFrames = 3000

    /// Probes chosen away from the clip floor where possible: `whisperLogClip` saturates quiet bins
    /// to `(max - 8 + 4)/4`, and a floored value would agree even with a wrong filterbank.
    private static let whisperProbes: [(bin: Int, frame: Int, value: Float)] = [
        (bin: 0, frame: 0, value: 1.0817363),
        (bin: 0, frame: 5, value: 0.0705306),
        (bin: 10, frame: 3, value: 1.3274596),
        (bin: 64, frame: 2, value: 0.0001740),
        (bin: 127, frame: 0, value: 0.2965118),
        // One floored probe, to pin the clip behaviour itself.
        (bin: 40, frame: 5, value: -0.5615621),
    ]

    @Test("Whisper features match the HuggingFace reference")
    func whisperMatchesReference() {
        let config = MelConfig.whisper
        let mel = MelSpectrogram.fromPCM(goldenTestSignal(count: Self.sampleCount), config: config)
        let frames = 3_000
        #expect(mel.count == config.nMelBins * frames)

        for probe in Self.whisperProbes {
            let actual = mel[probe.bin * frames + probe.frame]
            #expect(
                abs(actual - probe.value) < Self.tolerance,
                "bin \(probe.bin) frame \(probe.frame): got \(actual), expected \(probe.value)")
        }
    }

    @Test("The Whisper padded tail sits on the clip floor")
    func whisperPaddedTailIsFloored() {
        // Whisper computes its entire fixed window, so frames past the audio hold log-silence
        // clipped to `max - 8`. The reference agrees on the exact value.
        let config = MelConfig.whisper
        let mel = MelSpectrogram.fromPCM(goldenTestSignal(count: Self.sampleCount), config: config)
        #expect(abs(mel[0 * 3_000 + 2_500] - -0.5615621) < Self.tolerance)
    }
}
