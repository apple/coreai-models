// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Audio loading

@Suite("Audio loading")
struct AudioLoadingTests {
    /// Regression: `loadAndResample` called `AVAudioFile.read(into:)` once, but that reads *up to*
    /// the buffer capacity and can stop at an internal packet boundary. A float32 WAV came back
    /// several hundred frames short while PCM_16 happened to arrive whole, so the defect was
    /// invisible on the usual test files. The length here spans more than one read chunk.
    @Test("A float32 WAV longer than one read chunk loads completely", arguments: [true, false])
    func longWAVLoadsCompletely(isFloat: Bool) throws {
        let frames = Int(MelSpectrogram.readChunkFrames) + 22_464  // 88_000
        let samples = (0..<frames).map { Float($0 % 1_000) / 1_000 }
        try withTempDirectory { dir in
            let url = dir.appending(path: "long.wav")
            try writeWAV(samples, to: url, sampleRate: 16_000, isFloat: isFloat)
            let loaded = try MelSpectrogram.loadAndResample(url, targetSampleRate: 16_000)
            #expect(loaded.count == frames)

            // Probe across the chunk boundary: a short read would truncate rather than corrupt,
            // but checking values guards against a mis-stitched loop too.
            let tolerance: Float = isFloat ? 1e-6 : 1e-4
            for index in [0, 65_535, 65_536, frames - 1] {
                #expect(abs(loaded[index] - samples[index]) < tolerance, "sample \(index)")
            }
        }
    }

    @Test("A short WAV loads exactly")
    func shortWAVLoadsExactly() throws {
        let samples = makeSine(frequency: 440, count: 1_000)
        try withTempDirectory { dir in
            let url = dir.appending(path: "short.wav")
            try writeWAV(samples, to: url)
            let loaded = try MelSpectrogram.loadAndResample(url, targetSampleRate: 16_000)
            #expect(loaded.count == 1_000)
            #expect(maxAbsDiff(loaded, samples) < 1e-6)
        }
    }

    @Test("fromFile agrees with fromPCM on the same audio")
    func fromFileAgreesWithFromPCM() throws {
        let samples = deterministicNoise(count: 4_000, seed: 61)
        try withTempDirectory { dir in
            let url = dir.appending(path: "clip.wav")
            try writeWAV(samples, to: url)
            let viaFile = try MelSpectrogram.fromFile(url, config: .parakeet)
            let viaPCM = MelSpectrogram.fromPCM(samples, config: .parakeet)
            #expect(maxAbsDiff(viaFile, viaPCM) == 0)
        }
    }

    @Test(
        "Resampling produces the rate-scaled sample count",
        arguments: [8_000.0, 22_050.0, 44_100.0, 48_000.0])
    func resamplingProducesScaledCount(sourceRate: Double) throws {
        // One second of audio at the source rate must yield about one second at 16 kHz. The bound
        // is deliberately loose — AVAudioConverter's output length is not contractually exact — but
        // it still catches the failure mode that matters, feeding the converter only its first
        // chunk, which would truncate the output dramatically.
        let frames = Int(sourceRate)
        let samples = makeSine(frequency: 440, sampleRate: sourceRate, count: frames)
        try withTempDirectory { dir in
            let url = dir.appending(path: "rate.wav")
            try writeWAV(samples, to: url, sampleRate: sourceRate)
            let loaded = try MelSpectrogram.loadAndResample(url, targetSampleRate: 16_000)
            let expected = Double(frames) * 16_000 / sourceRate
            #expect(
                abs(Double(loaded.count) - expected) < expected * 0.02,
                "\(sourceRate) Hz gave \(loaded.count), expected about \(Int(expected))")
        }
    }

    @Test("A stereo file is reduced to one channel")
    func stereoIsReducedToMono() throws {
        let samples = makeSine(frequency: 440, count: 4_000)
        try withTempDirectory { dir in
            let url = dir.appending(path: "stereo.wav")
            try writeWAV(samples, to: url, sampleRate: 16_000, isFloat: true, channels: 2)
            let loaded = try MelSpectrogram.loadAndResample(url, targetSampleRate: 16_000)
            // Both channels hold the same signal, so the count is what matters here.
            #expect(loaded.count == 4_000)
            #expect(loaded.allSatisfy { $0.isFinite })
        }
    }
}

// MARK: - Resampler quality

/// Quality assertions that need no reference implementation.
///
/// A unit test cannot compute soxr, and comparing against it numerically would be meaningless
/// anyway — the ~34 dB agreement measured between soxr and `AVAudioConverter` is a property of two
/// differing filter designs, not a correctness threshold. What *is* assertable in absolute terms is
/// the anti-aliasing behaviour, which is the dimension that actually separates a good resampler
/// from a bad one, and the thing that regresses if `sampleRateConverterQuality` is ever lowered.
@Suite("Resampler quality")
struct ResamplerQualityTests {
    private static let sourceRate = 22_050.0
    private static let targetRate = 16_000.0

    private func resample(_ samples: [Float], from rate: Double) throws -> [Float] {
        var result: [Float] = []
        try withTempDirectory { dir in
            let url = dir.appending(path: "tone.wav")
            try writeWAV(samples, to: url, sampleRate: rate)
            result = try MelSpectrogram.loadAndResample(url, targetSampleRate: Self.targetRate)
        }
        return result
    }

    @Test("Content above the target Nyquist is rejected rather than aliased")
    func aboveNyquistIsRejected() throws {
        // 9 kHz cannot be represented at 16 kHz (Nyquist 8 kHz). Without anti-aliasing it folds to
        // 16000 - 9000 = 7000 Hz, where it would be indistinguishable from real content.
        let toneAbove = makeSine(
            frequency: 9_000, sampleRate: Self.sourceRate, count: Int(Self.sourceRate),
            amplitude: 0.5)
        let aliasEnergy = toneEnergy(
            try resample(toneAbove, from: Self.sourceRate), frequency: 7_000,
            sampleRate: Self.targetRate)

        // Normalize against an in-band tone of the same amplitude so the figure is a rejection
        // ratio rather than an absolute level.
        let toneInBand = makeSine(
            frequency: 1_000, sampleRate: Self.sourceRate, count: Int(Self.sourceRate),
            amplitude: 0.5)
        let passEnergy = toneEnergy(
            try resample(toneInBand, from: Self.sourceRate), frequency: 1_000,
            sampleRate: Self.targetRate)

        // Measured 90.2 dB with sampleRateConverterQuality = .max. The floor sits 30 dB below
        // that: comfortably above the 20-40 dB a low-quality polyphase filter delivers, so
        // lowering the converter quality fails here, while OS-to-OS variation in the filter design
        // does not.
        let rejectionDB = 20 * log10(passEnergy / max(aliasEnergy, 1e-12))
        #expect(rejectionDB > 60, "measured alias rejection \(rejectionDB) dB")
    }

    @Test("An in-band tone survives resampling")
    func inBandTonePasses() throws {
        let tone = makeSine(
            frequency: 1_000, sampleRate: Self.sourceRate, count: Int(Self.sourceRate),
            amplitude: 0.5)
        let out = try resample(tone, from: Self.sourceRate)
        let atTone = toneEnergy(out, frequency: 1_000, sampleRate: Self.targetRate)
        let offTone = toneEnergy(out, frequency: 3_000, sampleRate: Self.targetRate)
        #expect(atTone > 0.1, "in-band tone amplitude \(atTone) is too low")
        // Measured 108.7 dB; floored at 70 to leave headroom for filter-design differences.
        let purityDB = 20 * log10(atTone / max(offTone, 1e-12))
        #expect(purityDB > 70, "measured spectral purity \(purityDB) dB")
    }
}
