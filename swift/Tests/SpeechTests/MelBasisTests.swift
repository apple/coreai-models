// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Hann window

@Suite("Hann window")
struct HannWindowTests {
    /// `torch.hann_window(N)` defaults to `periodic: true`, which divides by `N`. HF's Whisper
    /// extractor takes that default; Parakeet's passes `periodic=False`, dividing by `N - 1`.
    /// The distinction is one sample of taper, applied identically to every frame — small, and
    /// invisible unless asserted.
    @Test("Periodic window divides by N")
    func periodicDividesByN() {
        let w = MelSpectrogram.hannWindow(size: 8, periodicity: .periodic)
        let expected: [Float] = [0, 0.1464466, 0.5, 0.8535534, 1.0, 0.8535534, 0.5, 0.1464466]
        #expect(maxAbsDiff(w, expected) < 1e-6)
        // The defining property: the last sample is *not* zero, because N never appears as a
        // numerator. This single value is what separates the two conventions.
        #expect(w[7] > 0.1)
    }

    @Test("Symmetric window divides by N minus one")
    func symmetricDividesByNMinusOne() {
        let w = MelSpectrogram.hannWindow(size: 8, periodicity: .symmetric)
        #expect(w.count == 8)
        #expect(w[0] == 0)
        // Exactly zero at both ends: n = N-1 gives cos(2π) = 1.
        #expect(abs(w[7]) < 1e-7)
        // Peak straddles the half-sample centre, so the two middle taps are equal.
        #expect(abs(w[3] - w[4]) < 1e-7)
        #expect(w[3] > 0.9)
    }

    @Test("Presets select the periodicity their reference implementation uses")
    func presetsSelectPeriodicity() {
        // Policy pin: both of these were single-token defects. `MelConfig.whisper` shipped
        // `.symmetric` (inherited from a hardcoded window) and disagreed with HF by 0.0854
        // absolute until corrected.
        #expect(MelConfig.whisper.windowPeriodicity == .periodic)
        #expect(MelConfig.parakeet.windowPeriodicity == .symmetric)

        let whisperWindow = MelSpectrogram.Basis(config: .whisper).window
        let parakeetWindow = MelSpectrogram.Basis(config: .parakeet).window
        #expect(whisperWindow.count == MelConfig.whisper.winLength)
        #expect(parakeetWindow.count == MelConfig.parakeet.winLength)

        // The final tap is the whole difference between the conventions, and its size depends on
        // the window length: periodic gives 0.5·(1 − cos(2π/N)), which is 6.2e-5 at N = 400,
        // while symmetric gives exactly 0. Compare against the closed form rather than a fixed
        // threshold, which would either pass vacuously or fail on window length alone.
        let n = Double(MelConfig.whisper.winLength)
        let expectedLast = 0.5 * (1 - cos(2 * Double.pi / n))
        #expect(abs(Double(whisperWindow.last!) - expectedLast) < 1e-9)
        #expect(whisperWindow.last! > 0)
        #expect(parakeetWindow.last! == 0)
    }

    @Test("Window is bounded by zero and one", arguments: [MelConfig.whisper, MelConfig.parakeet])
    func windowIsBounded(config: MelConfig) {
        let w = MelSpectrogram.Basis(config: config).window
        #expect(w.allSatisfy { $0 >= 0 && $0 <= 1 })
    }
}

// MARK: - DFT basis

@Suite("DFT basis")
struct DFTBasisTests {
    /// The angles reach ~1600 rad at nFFT = 400. Forming them in `Float` leaves ~1e-4 of
    /// absolute error, which survives into the power spectrum; the implementation reduces
    /// `k*n` modulo nFFT and evaluates in `Double`.
    @Test("Basis matches a full-angle Double evaluation")
    func basisMatchesDoubleReference() {
        let nFFT = 400
        let (cosTable, sinTable) = MelSpectrogram.dftBasis(nFFT: nFFT)
        var worstCos = 0.0
        var worstSin = 0.0
        for k in 0...(nFFT / 2) {
            for n in 0..<nFFT {
                // Deliberately *not* reduced modulo nFFT: this is the naive formulation, so it is
                // an independent check rather than a restatement of the implementation.
                let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(nFFT)
                worstCos = max(worstCos, abs(Double(cosTable[k * nFFT + n]) - cos(angle)))
                worstSin = max(worstSin, abs(Double(sinTable[k * nFFT + n]) + sin(angle)))
            }
        }
        #expect(worstCos < 1e-6, "cos table drifted by \(worstCos)")
        #expect(worstSin < 1e-6, "sin table drifted by \(worstSin)")
    }

    /// Tolerance-free companion to the above. The basis is periodic in `k·n`, so every pair with
    /// the same residue must produce a bit-identical entry. Float-domain angle accumulation
    /// breaks this; modular reduction in `Double` cannot.
    @Test("Basis is exactly periodic in k times n modulo nFFT")
    func basisIsExactlyPeriodic() {
        let nFFT = 400
        let (cosTable, sinTable) = MelSpectrogram.dftBasis(nFFT: nFFT)
        var seenCos: [Int: Float] = [:]
        var seenSin: [Int: Float] = [:]
        var mismatches = 0
        for k in 0...(nFFT / 2) {
            for n in 0..<nFFT {
                let residue = (k * n) % nFFT
                let c = cosTable[k * nFFT + n]
                let s = sinTable[k * nFFT + n]
                if let first = seenCos[residue] {
                    if first != c || seenSin[residue] != s { mismatches += 1 }
                } else {
                    seenCos[residue] = c
                    seenSin[residue] = s
                }
            }
        }
        #expect(mismatches == 0, "\(mismatches) entries differed between equal residues")
    }

    @Test("Row zero is all ones and zeros")
    func rowZeroIsUnity() {
        let nFFT = 64
        let (cosTable, sinTable) = MelSpectrogram.dftBasis(nFFT: nFFT)
        #expect(cosTable[0..<nFFT].allSatisfy { $0 == 1.0 })
        #expect(sinTable[0..<nFFT].allSatisfy { abs($0) == 0.0 })
    }

    @Test("Basis shape is nFreqs by nFFT")
    func basisShape() {
        let nFFT = 60
        let (cosTable, sinTable) = MelSpectrogram.dftBasis(nFFT: nFFT)
        #expect(cosTable.count == (nFFT / 2 + 1) * nFFT)
        #expect(sinTable.count == (nFFT / 2 + 1) * nFFT)
    }
}

// MARK: - Mel filterbank

@Suite("Mel filterbank")
struct MelFilterbankTests {
    /// Policy pin. `MelConfig.whisper` shipped `.htk`, but HF's `WhisperFeatureExtractor` builds
    /// its filterbank with `mel_scale="slaney"` and reference OpenAI Whisper takes librosa's
    /// default, which is also Slaney. The two scales place every filter differently — the error
    /// was 1.65 absolute in normalized log-mel.
    @Test("Both presets use the Slaney mel scale")
    func presetsUseSlaney() {
        #expect(MelConfig.whisper.melScale == .slaney)
        #expect(MelConfig.parakeet.melScale == .slaney)
    }

    @Test("HTK and Slaney place filters differently")
    func htkAndSlaneyDiffer() {
        let slaney = MelSpectrogram.melFilterbank(config: MelConfig.whisper)
        let htkConfig = MelConfig(
            sampleRate: MelConfig.whisper.sampleRate, nFFT: MelConfig.whisper.nFFT,
            winLength: MelConfig.whisper.winLength, hopLength: MelConfig.whisper.hopLength,
            nMelBins: MelConfig.whisper.nMelBins, nFrames: MelConfig.whisper.nFrames,
            preemphasis: MelConfig.whisper.preemphasis,
            normalization: MelConfig.whisper.normalization, layout: MelConfig.whisper.layout,
            padMode: MelConfig.whisper.padMode, melScale: .htk,
            windowPeriodicity: MelConfig.whisper.windowPeriodicity)
        let htk = MelSpectrogram.melFilterbank(config: htkConfig)

        #expect(slaney.count == htk.count)
        // Substantially different, not a rounding difference: swapping the scale is exactly the
        // regression this guards.
        #expect(maxAbsDiff(slaney, htk) > 1e-3)
    }

    @Test("Every triangle is nonnegative and compactly supported")
    func trianglesAreWellFormed() {
        let config = MelConfig.parakeet
        let nFreqs = config.nFFT / 2 + 1
        let fb = MelSpectrogram.melFilterbank(config: config)
        #expect(fb.count == config.nMelBins * nFreqs)
        #expect(fb.allSatisfy { $0 >= 0 })

        for m in 0..<config.nMelBins {
            let row = Array(fb[(m * nFreqs)..<((m + 1) * nFreqs)])
            let nonzero = row.indices.filter { row[$0] > 0 }
            guard let first = nonzero.first, let last = nonzero.last else { continue }
            // A triangle's support is a single contiguous run; gaps would mean a construction bug.
            #expect(nonzero.count == last - first + 1, "bin \(m) support is not contiguous")
        }
    }

    /// Documents why the Nyquist power bin is unobservable end-to-end: `fMax` is `sampleRate/2`,
    /// so the last triangle's right edge lands exactly on it and every filter weights it zero.
    /// A test can pin the cause, but not the value that flows through it.
    @Test(
        "The Nyquist column is identically zero",
        arguments: [MelConfig.whisper, MelConfig.parakeet])
    func nyquistColumnIsZero(config: MelConfig) {
        let nFreqs = config.nFFT / 2 + 1
        let fb = MelSpectrogram.melFilterbank(config: config)
        for m in 0..<config.nMelBins {
            #expect(fb[m * nFreqs + (nFreqs - 1)] == 0)
        }
    }
}

// MARK: - Basis path selection

@Suite("Basis path selection")
struct MelBasisPathTests {
    @Test("A power-of-two nFFT selects the FFT path")
    func powerOfTwoSelectsFFT() {
        let basis = MelSpectrogram.Basis(config: .parakeet)
        #expect(basis.log2n == 9)  // 512 == 2^9
        #expect(basis.cosBasis.isEmpty)
        #expect(basis.sinBasis.isEmpty)
    }

    @Test("A non-power-of-two nFFT builds dense tables")
    func nonPowerOfTwoBuildsDenseTables() {
        let config = MelConfig.whisper  // nFFT 400 = 2^4 * 5^2
        let basis = MelSpectrogram.Basis(config: config)
        #expect(basis.log2n == nil)
        #expect(basis.cosBasis.count == (config.nFFT / 2 + 1) * config.nFFT)
        #expect(basis.sinBasis.count == (config.nFFT / 2 + 1) * config.nFFT)
    }

    @Test("Basis records the geometry it was built for")
    func basisRecordsGeometry() {
        // `fromPCM` compares these three against its config before trusting a handed-in basis.
        let basis = MelSpectrogram.Basis(config: .parakeet)
        #expect(basis.nFFT == MelConfig.parakeet.nFFT)
        #expect(basis.winLength == MelConfig.parakeet.winLength)
        #expect(basis.nMelBins == MelConfig.parakeet.nMelBins)
    }
}
