// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import Accelerate
import Foundation

// Whisper mel spectrogram: sr=16000, n_fft=400, hop=160, n_mels=128
// Slaney-normalised filterbank, reflect-padded audio, matches WhisperFeatureExtractor.
//
// vDSP DFT only supports f×2^n sizes (f ∈ {1,3,5,15}); 400=5²×2⁴ doesn't qualify.
// We precompute 201×400 DFT basis matrices and apply them with cblas_sgemv instead.

enum WhisperMel {
    static let sampleRate: Double = 16_000
    static let nFFT = 400  // analysis window (samples)
    static let hopLength = 160
    static let nMelBins = 128
    static let nFrames = 3_000
    static let nSamples = 480_000

    private static let nFreqs = nFFT / 2 + 1  // 201

    // MARK: - Public

    static func fromFile(_ url: URL) throws -> [Float] {
        return fromPCM(try loadAndResample(url))
    }

    // MARK: - Audio loading + resampling

    static func loadAndResample(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate, channels: 1, interleaved: false)!
        guard let conv = AVAudioConverter(from: file.processingFormat, to: fmt) else {
            throw NSError(
                domain: "WhisperMel", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Cannot convert \(file.processingFormat) → 16 kHz mono"
                ])
        }
        let cap = AVAudioFrameCount(
            ceil(Double(file.length) * sampleRate / file.processingFormat.sampleRate) + 1)
        let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap)!
        var fed = false
        var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            guard !fed else {
                status.pointee = .endOfStream
                return nil
            }
            fed = true
            let buf = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length))!
            try? file.read(into: buf)
            status.pointee = buf.frameLength > 0 ? .haveData : .endOfStream
            return buf
        }
        if let e = err { throw e }
        return Array(
            UnsafeBufferPointer(
                start: out.floatChannelData![0],
                count: Int(out.frameLength)))
    }

    // MARK: - Precomputed DFT basis (201 × 400)
    // cos_basis[k, n] =  cos(2π k n / 400)  →  Y[k].real = cos_basis @ x
    // sin_basis[k, n] = -sin(2π k n / 400)  →  Y[k].imag = sin_basis @ x

    static let cosBasis: [Float] = {
        var m = [Float](repeating: 0, count: (nFFT / 2 + 1) * nFFT)
        for k in 0...nFFT / 2 {
            for n in 0..<nFFT {
                m[k * nFFT + n] = cos(2 * Float.pi * Float(k) * Float(n) / Float(nFFT))
            }
        }
        return m
    }()

    static let sinBasis: [Float] = {
        var m = [Float](repeating: 0, count: (nFFT / 2 + 1) * nFFT)
        for k in 0...nFFT / 2 {
            for n in 0..<nFFT {
                m[k * nFFT + n] = -sin(2 * Float.pi * Float(k) * Float(n) / Float(nFFT))
            }
        }
        return m
    }()

    // MARK: - Mel filterbank (128 × 201, Slaney-normalised)

    static let melFilterbank: [Float] = makeMelFilterbank()

    // MARK: - Mel computation

    static func fromPCM(_ raw: [Float]) -> [Float] {
        // 1. Trim / zero-pad to nSamples
        var audio = raw
        if audio.count > nSamples {
            audio = Array(audio.prefix(nSamples))
        } else if audio.count < nSamples {
            audio += [Float](repeating: 0, count: nSamples - audio.count)
        }

        // 2. Reflect-pad by nFFT/2 (matches np.pad(..., mode='reflect'))
        let pad = nFFT / 2  // 200
        var padded = [Float](repeating: 0, count: nSamples + 2 * pad)
        for i in 0..<pad { padded[pad - 1 - i] = audio[i + 1] }
        for i in 0..<nSamples { padded[pad + i] = audio[i] }
        for i in 0..<pad { padded[pad + nSamples + i] = audio[nSamples - 2 - i] }

        // 3. Hann window
        var window = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            window[i] = Float(0.5 * (1 - cos(2 * Double.pi * Double(i) / Double(nFFT - 1))))
        }

        var frame = [Float](repeating: 0, count: nFFT)
        var yReal = [Float](repeating: 0, count: nFreqs)
        var yImag = [Float](repeating: 0, count: nFreqs)
        var powerSpec = [Float](repeating: 0, count: nFreqs)
        var melFrame = [Float](repeating: 0, count: nMelBins)
        var mel = [Float](repeating: 0, count: nMelBins * nFrames)

        for t in 0..<nFrames {
            let offset = t * hopLength

            // Apply Hann window
            vDSP_vmul(
                Array(padded[offset..<offset + nFFT]), 1,
                window, 1, &frame, 1, vDSP_Length(nFFT))

            // DFT via matrix multiply: Y[k] = cosBasis[k,:] @ frame - i × sinBasis[k,:] @ frame
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(nFreqs), Int32(nFFT), 1.0, cosBasis, Int32(nFFT),
                frame, 1, 0.0, &yReal, 1)
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(nFreqs), Int32(nFFT), 1.0, sinBasis, Int32(nFFT),
                frame, 1, 0.0, &yImag, 1)

            // Power spectrum |Y[k]|² = yReal² + yImag²
            vDSP_vmma(yReal, 1, yReal, 1, yImag, 1, yImag, 1, &powerSpec, 1, vDSP_Length(nFreqs))

            // Apply mel filterbank: (128×201) × (201) → (128)
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(nMelBins), Int32(nFreqs), 1.0, melFilterbank, Int32(nFreqs),
                powerSpec, 1, 0.0, &melFrame, 1)

            for i in 0..<nMelBins {
                mel[i * nFrames + t] = log10(max(melFrame[i], 1e-10))
            }
        }

        // Normalise: clamp to max−8, then (x+4)/4
        let maxVal = mel.max() ?? 0
        for i in 0..<mel.count { mel[i] = (max(mel[i], maxVal - 8) + 4) / 4 }
        return mel
    }

    // MARK: - Filterbank builder

    private static func makeMelFilterbank() -> [Float] {
        let fMax: Float = Float(sampleRate) / 2  // 8000 Hz

        func hzToMel(_ f: Float) -> Float { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Float) -> Float { 700 * (pow(10, m / 2595) - 1) }

        let melMin = hzToMel(0)
        let melMax = hzToMel(fMax)
        let nPts = nMelBins + 2
        let pts = (0..<nPts).map { i -> Float in
            melToHz(melMin + Float(i) / Float(nPts - 1) * (melMax - melMin))
        }
        // FFT bin frequencies for n_fft = 400
        let fftFreqs = (0..<nFreqs).map { Float($0) * Float(sampleRate) / Float(nFFT) }

        var fb = [Float](repeating: 0, count: nMelBins * nFreqs)
        for m in 0..<nMelBins {
            let fL = pts[m]
            let fC = pts[m + 1]
            let fR = pts[m + 2]
            let norm: Float = 2 / (fR - fL)
            for k in 0..<nFreqs {
                let f = fftFreqs[k]
                if f >= fL && f <= fC {
                    fb[m * nFreqs + k] = norm * (f - fL) / (fC - fL)
                } else if f > fC && f <= fR {
                    fb[m * nFreqs + k] = norm * (fR - f) / (fR - fC)
                }
            }
        }
        return fb
    }
}
