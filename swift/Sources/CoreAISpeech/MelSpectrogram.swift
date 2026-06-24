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
    public let winLength: Int
    public let hopLength: Int
    public let nMelBins: Int
    /// Fixed number of frames; `nil` lets the spectrogram length follow the audio length.
    public let nFrames: Int?
    /// Pre-emphasis coefficient applied as `y[t] = x[t] − α·x[t−1]`. `nil` disables it.
    public let preemphasis: Float?
    public let normalization: Normalization
    public let layout: Layout

    public enum Normalization: Sendable {
        /// Whisper: clip to `max−8`, shift+scale by `(x+4)/4`. log base 10.
        case whisperLogClip
        /// Per-instance scalar mean/std normalization on natural-log mel. NeMo/Parakeet convention.
        case perInstanceMeanStd
    }

    public enum Layout: Sendable {
        /// Whisper-style `[B, n_mels, n_frames]`.
        case channelMajor
        /// Parakeet-style `[B, n_frames, n_mels]`.
        case timeMajor
    }

    public init(
        sampleRate: Double, nFFT: Int, winLength: Int, hopLength: Int,
        nMelBins: Int, nFrames: Int?, preemphasis: Float?,
        normalization: Normalization, layout: Layout
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.winLength = winLength
        self.hopLength = hopLength
        self.nMelBins = nMelBins
        self.nFrames = nFrames
        self.preemphasis = preemphasis
        self.normalization = normalization
        self.layout = layout
    }

    /// Whisper v3-turbo parameters.
    public static let whisper = MelConfig(
        sampleRate: 16_000, nFFT: 400, winLength: 400, hopLength: 160,
        nMelBins: 128, nFrames: 3_000, preemphasis: nil,
        normalization: .whisperLogClip, layout: .channelMajor)

    /// Parakeet TDT v3 parameters (matches `ParakeetFeatureExtractor`).
    public static let parakeet = MelConfig(
        sampleRate: 16_000, nFFT: 512, winLength: 400, hopLength: 160,
        nMelBins: 128, nFrames: nil, preemphasis: 0.97,
        normalization: .perInstanceMeanStd, layout: .timeMajor)
}

// MARK: - MelSpectrogram

/// Computes a mel spectrogram from an audio file or raw PCM samples.
public enum MelSpectrogram {
    // MARK: Public API

    /// The number of mel frames the configured pipeline will emit for a given PCM length.
    public static func frameCount(forPCMLength count: Int, config: MelConfig) -> Int {
        if let n = config.nFrames { return n }
        // Variable: right-pad to a multiple of hopLength, one frame per hop.
        let rounded = ((count + config.hopLength - 1) / config.hopLength) * config.hopLength
        return rounded / config.hopLength
    }

    public static func fromFile(_ url: URL, config: MelConfig = .whisper) throws -> [Float] {
        return fromPCM(try loadAndResample(url, targetSampleRate: config.sampleRate), config: config)
    }

    public static func fromPCM(_ raw: [Float], config: MelConfig = .whisper) -> [Float] {
        let preemph = applyPreemphasis(raw, alpha: config.preemphasis)
        let (audio, nFrames) = padToFrameGrid(preemph, config: config)

        let pad = config.nFFT / 2
        let padded = reflectPad(audio, pad: pad)

        let window = hannWindow(size: config.winLength)
        let frameOffset = (config.nFFT - config.winLength) / 2
        let (cosBasis, sinBasis) = dftBasis(nFFT: config.nFFT)
        let filterbank = melFilterbank(config: config)
        let nFreqs = config.nFFT / 2 + 1

        var windowed = [Float](repeating: 0, count: config.winLength)
        var frame = [Float](repeating: 0, count: config.nFFT)
        var yReal = [Float](repeating: 0, count: nFreqs)
        var yImag = [Float](repeating: 0, count: nFreqs)
        var powerSpec = [Float](repeating: 0, count: nFreqs)
        var melFrame = [Float](repeating: 0, count: config.nMelBins)
        var mel = [Float](repeating: 0, count: config.nMelBins * nFrames)

        let logFloor: Float = 1e-10
        let useLog10 = (config.normalization == .whisperLogClip)

        for t in 0..<nFrames {
            let offset = t * config.hopLength
            vDSP_vmul(
                Array(padded[offset..<offset + config.winLength]), 1,
                window, 1, &windowed, 1, vDSP_Length(config.winLength))
            // Place the windowed slice into a zero-padded nFFT buffer.
            for i in 0..<config.nFFT { frame[i] = 0 }
            for i in 0..<config.winLength { frame[frameOffset + i] = windowed[i] }
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
                let v = max(melFrame[i], logFloor)
                let lv = useLog10 ? log10(v) : log(v)
                let idx = (config.layout == .channelMajor) ? (i * nFrames + t) : (t * config.nMelBins + i)
                mel[idx] = lv
            }
        }

        normalize(&mel, normalization: config.normalization)
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

    // MARK: Preprocessing

    private static func applyPreemphasis(_ raw: [Float], alpha: Float?) -> [Float] {
        guard let alpha, !raw.isEmpty else { return raw }
        var out = [Float](repeating: 0, count: raw.count)
        out[0] = raw[0]
        for i in 1..<raw.count { out[i] = raw[i] - alpha * raw[i - 1] }
        return out
    }

    private static func padToFrameGrid(_ raw: [Float], config: MelConfig) -> ([Float], Int) {
        if let target = config.nFrames {
            let n = target * config.hopLength
            var audio = raw
            if audio.count > n {
                audio = Array(audio.prefix(n))
            } else if audio.count < n {
                audio += [Float](repeating: 0, count: n - audio.count)
            }
            return (audio, target)
        }
        let nFrames = frameCount(forPCMLength: raw.count, config: config)
        let n = nFrames * config.hopLength
        var audio = raw
        if audio.count < n {
            audio += [Float](repeating: 0, count: n - audio.count)
        } else if audio.count > n {
            audio = Array(audio.prefix(n))
        }
        return (audio, nFrames)
    }

    private static func reflectPad(_ audio: [Float], pad: Int) -> [Float] {
        let n = audio.count
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { padded[pad - 1 - i] = audio[i + 1] }
        for i in 0..<n { padded[pad + i] = audio[i] }
        for i in 0..<pad { padded[pad + n + i] = audio[n - 2 - i] }
        return padded
    }

    private static func normalize(_ mel: inout [Float], normalization: MelConfig.Normalization) {
        switch normalization {
        case .whisperLogClip:
            let maxVal = mel.max() ?? 0
            for i in 0..<mel.count { mel[i] = (max(mel[i], maxVal - 8) + 4) / 4 }
        case .perInstanceMeanStd:
            if mel.isEmpty { return }
            var sum: Float = 0
            for v in mel { sum += v }
            let mean = sum / Float(mel.count)
            var sqSum: Float = 0
            for v in mel {
                let d = v - mean
                sqSum += d * d
            }
            let std = sqrt(sqSum / Float(mel.count))
            let denom = max(std, 1e-5)
            for i in 0..<mel.count { mel[i] = (mel[i] - mean) / denom }
        }
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
