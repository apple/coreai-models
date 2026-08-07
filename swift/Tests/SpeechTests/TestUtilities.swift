// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import Foundation

// Helpers shared across the SpeechTests suites. Target-local rather than in the shared
// `TestUtilities` target, matching LanguageModelsTests and GuidedGenerationTests, since
// nothing here is useful to another module.

// MARK: - Temporary files

/// Creates a unique temporary directory for the test and removes it after.
func withTempDirectory(_ body: (URL) throws -> Void) throws {
    let fm = FileManager.default
    let dir = fm.temporaryDirectory.appending(
        path: "speech-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: dir) }
    try body(dir)
}

/// Writes `samples` as a WAV that `AVAudioFile` can read back.
///
/// `isFloat` selects 32-bit float vs 16-bit integer samples. That distinction is the whole
/// point of some of these tests: `loadAndResample` once read float32 files short, because
/// `AVAudioFile.read(into:)` returns *up to* the buffer capacity and stopped early on that
/// encoding while PCM_16 happened to come back whole.
///
/// Written through `AVAudioFile` rather than a hand-assembled RIFF header so the reader under
/// test is guaranteed to accept the result.
func writeWAV(
    _ samples: [Float], to url: URL, sampleRate: Double = 16_000,
    isFloat: Bool = true, channels: AVAudioChannelCount = 1
) throws {
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: sampleRate,
        AVNumberOfChannelsKey: Int(channels),
        AVLinearPCMBitDepthKey: isFloat ? 32 : 16,
        AVLinearPCMIsFloatKey: isFloat,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
    ]
    let file = try AVAudioFile(
        forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: channels, interleaved: false),
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
    else {
        throw CocoaError(.fileWriteUnknown)
    }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    for channel in 0..<Int(channels) {
        let dst = buffer.floatChannelData![channel]
        for i in samples.indices { dst[i] = samples[i] }
    }
    try file.write(from: buffer)
}

// MARK: - Signal generation

/// `amplitude · sin(2π·frequency·n/sampleRate)`, accumulated in `Double` so the same signal
/// can be reproduced exactly by a reference implementation in another language.
func makeSine(
    frequency: Double, sampleRate: Double = 16_000, count: Int, amplitude: Double = 1.0
) -> [Float] {
    (0..<count).map { n in
        Float(amplitude * sin(2.0 * Double.pi * frequency * Double(n) / sampleRate))
    }
}

/// A single unit sample at `index`, zero elsewhere.
///
/// The sharp time localization is what makes this useful: it pins *which* frame a sample lands
/// in and with what window weight, which a steady tone cannot do — shifting a tone by a fraction
/// of a frame barely perturbs its magnitude spectrum.
func makeImpulse(at index: Int, count: Int, amplitude: Float = 1.0) -> [Float] {
    var samples = [Float](repeating: 0, count: count)
    samples[index] = amplitude
    return samples
}

/// The fixed multi-tone signal the golden-value tests use.
///
/// Deliberately closed-form and documented so a reference implementation can construct bit-identical
/// input without shipping an audio file. Three widely spaced partials keep energy in low, mid and
/// high mel bins, so a filterbank or window regression cannot hide in an unoccupied region.
func goldenTestSignal(count: Int, sampleRate: Double = 16_000) -> [Float] {
    (0..<count).map { n in
        let t = Double(n) / sampleRate
        return Float(
            0.50 * sin(2.0 * Double.pi * 220.0 * t)
                + 0.25 * sin(2.0 * Double.pi * 1_500.0 * t)
                + 0.10 * sin(2.0 * Double.pi * 5_000.0 * t))
    }
}

// MARK: - Reference spectra

/// Power spectrum of the first `nFFT` samples by direct O(n²) evaluation in `Double`.
///
/// Independent of the module's precomputed-table + BLAS path, which is the point: it can disagree
/// with it. Returns `nFFT/2 + 1` bins.
func naiveDFTPowerSpectrum(_ frame: [Float], nFFT: Int) -> [Double] {
    (0...(nFFT / 2)).map { k in
        var real = 0.0
        var imag = 0.0
        for n in 0..<nFFT {
            let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(nFFT)
            let x = n < frame.count ? Double(frame[n]) : 0.0
            real += x * cos(angle)
            imag -= x * sin(angle)
        }
        return real * real + imag * imag
    }
}

/// Normalized energy at a single frequency, via a one-bin DFT.
///
/// Cheaper than a full transform when only a couple of frequencies matter — the resampler tests
/// need the tone and its potential alias image, nothing else.
func toneEnergy(_ samples: [Float], frequency: Double, sampleRate: Double) -> Double {
    var real = 0.0
    var imag = 0.0
    for (n, x) in samples.enumerated() {
        let angle = 2.0 * Double.pi * frequency * Double(n) / sampleRate
        real += Double(x) * cos(angle)
        imag -= Double(x) * sin(angle)
    }
    return (real * real + imag * imag).squareRoot() / Double(samples.count)
}

// MARK: - Comparison

/// Uniform noise in `[-amplitude, amplitude]` from a seeded 64-bit LCG.
///
/// Deterministic on purpose: `Float.random` would make a failure unreproducible, and several of
/// these tests depend on the same input producing the same spectrum every run. Broadband content
/// also keeps every mel bin well above the log-zero floor, where Float and Double disagree by
/// O(1) on numerically meaningless power differences.
func deterministicNoise(count: Int, seed: UInt64, amplitude: Float = 0.5) -> [Float] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0..<count).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let unit = Float(state >> 11) / Float(1 << 53)  // [0, 1)
        return (unit * 2 - 1) * amplitude
    }
}

/// Largest absolute difference between two equal-length arrays; `.infinity` if the lengths differ
/// so a shape mismatch cannot masquerade as a small error.
func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return .infinity }
    return zip(a, b).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
}

/// Mean and sample standard deviation (Bessel-corrected, matching the module's `perBinMeanStd`).
func meanAndStd(_ values: [Float]) -> (mean: Double, std: Double) {
    guard values.count > 1 else { return (values.first.map(Double.init) ?? 0, 0) }
    let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
    let variance =
        values.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) }
        / Double(values.count - 1)
    return (mean, variance.squareRoot())
}
