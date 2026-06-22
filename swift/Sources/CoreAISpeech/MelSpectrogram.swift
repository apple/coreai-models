// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import Accelerate
import CoreAIShared
import Foundation

// MARK: - MelConfig

/// Parameters for mel spectrogram computation.
public struct MelConfig: Sendable {
    public let sampleRate: Double
    public let nFFT: Int
    public let hopLength: Int
    public let nMelBins: Int
    public let nFrames: Int

    public var nSamples: Int { Int(sampleRate) * (nFrames * hopLength / Int(sampleRate / 100)) }

    /// Whisper / Parakeet shared parameters.
    public static let whisper = MelConfig(
        sampleRate: 16_000, nFFT: 400, hopLength: 160, nMelBins: 128, nFrames: 3_000)
}

// MARK: - MelSpectrogram

/// Computes a mel spectrogram from an audio file or raw PCM samples.
public enum MelSpectrogram {
    // MARK: Public API

    public static func fromFile(_ url: URL, config: MelConfig = .whisper) throws -> [Float] {
        return fromPCM(try loadAndResample(url, targetSampleRate: config.sampleRate), config: config)
    }

    public static func fromPCM(_ raw: [Float], config: MelConfig = .whisper) -> [Float] {
        let nSamples = config.nFrames * config.hopLength

        var audio = raw
        if audio.count > nSamples {
            audio = Array(audio.prefix(nSamples))
        } else if audio.count < nSamples {
            audio += [Float](repeating: 0, count: nSamples - audio.count)
        }

        let pad = config.nFFT / 2
        var padded = [Float](repeating: 0, count: nSamples + 2 * pad)
        for i in 0..<pad { padded[pad - 1 - i] = audio[i + 1] }
        for i in 0..<nSamples { padded[pad + i] = audio[i] }
        for i in 0..<pad { padded[pad + nSamples + i] = audio[nSamples - 2 - i] }

        let window = hannWindow(size: config.nFFT)
        let (cosBasis, sinBasis) = dftBasis(nFFT: config.nFFT)
        let filterbank = melFilterbank(config: config)
        let nFreqs = config.nFFT / 2 + 1

        var frame = [Float](repeating: 0, count: config.nFFT)
        var yReal = [Float](repeating: 0, count: nFreqs)
        var yImag = [Float](repeating: 0, count: nFreqs)
        var powerSpec = [Float](repeating: 0, count: nFreqs)
        var melFrame = [Float](repeating: 0, count: config.nMelBins)
        var mel = [Float](repeating: 0, count: config.nMelBins * config.nFrames)

        for t in 0..<config.nFrames {
            let offset = t * config.hopLength
            vDSP_vmul(
                Array(padded[offset..<offset + config.nFFT]), 1,
                window, 1, &frame, 1, vDSP_Length(config.nFFT))
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(nFreqs), Int32(config.nFFT), 1.0, cosBasis, Int32(config.nFFT),
                frame, 1, 0.0, &yReal, 1)
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(nFreqs), Int32(config.nFFT), 1.0, sinBasis, Int32(config.nFFT),
                frame, 1, 0.0, &yImag, 1)
            vDSP_vmma(yReal, 1, yReal, 1, yImag, 1, yImag, 1, &powerSpec, 1, vDSP_Length(nFreqs))
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(config.nMelBins), Int32(nFreqs), 1.0, filterbank, Int32(nFreqs),
                powerSpec, 1, 0.0, &melFrame, 1)
            for i in 0..<config.nMelBins {
                mel[i * config.nFrames + t] = log10(max(melFrame[i], 1e-10))
            }
        }

        let maxVal = mel.max() ?? 0
        for i in 0..<mel.count { mel[i] = (max(mel[i], maxVal - 8) + 4) / 4 }
        return mel
    }

    // MARK: Audio loading

    public static func loadAndResample(_ url: URL, targetSampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate, channels: 1, interleaved: false)!
        guard let conv = AVAudioConverter(from: file.processingFormat, to: fmt) else {
            throw SpeechError.invalidAudio(
                "Cannot resample \(file.processingFormat) to \(targetSampleRate) Hz mono")
        }
        let cap = AVAudioFrameCount(
            ceil(Double(file.length) * targetSampleRate / file.processingFormat.sampleRate) + 1)
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
        if let e = err { throw SpeechError.invalidAudio(e.localizedDescription) }
        return Array(
            UnsafeBufferPointer(
                start: out.floatChannelData![0],
                count: Int(out.frameLength)))
    }

    // MARK: Precomputed basis

    private static func hannWindow(size: Int) -> [Float] {
        (0..<size).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(size - 1)))) }
    }

    private static func dftBasis(nFFT: Int) -> ([Float], [Float]) {
        let nFreqs = nFFT / 2 + 1
        var cos = [Float](repeating: 0, count: nFreqs * nFFT)
        var sin = [Float](repeating: 0, count: nFreqs * nFFT)
        for k in 0..<nFreqs {
            for n in 0..<nFFT {
                let angle = 2 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                cos[k * nFFT + n] = Foundation.cos(angle)
                sin[k * nFFT + n] = -Foundation.sin(angle)
            }
        }
        return (cos, sin)
    }

    private static func melFilterbank(config: MelConfig) -> [Float] {
        let nFreqs = config.nFFT / 2 + 1
        let fMax = Float(config.sampleRate) / 2
        func h2m(_ f: Float) -> Float { 2595 * log10(1 + f / 700) }
        func m2h(_ m: Float) -> Float { 700 * (pow(10, m / 2595) - 1) }
        let pts = (0..<config.nMelBins + 2).map { i -> Float in
            m2h(h2m(0) + Float(i) / Float(config.nMelBins + 1) * (h2m(fMax) - h2m(0)))
        }
        let fftFreqs = (0..<nFreqs).map { Float($0) * Float(config.sampleRate) / Float(config.nFFT) }
        var fb = [Float](repeating: 0, count: config.nMelBins * nFreqs)
        for m in 0..<config.nMelBins {
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
