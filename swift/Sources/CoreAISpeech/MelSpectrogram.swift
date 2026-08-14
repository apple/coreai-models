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
    public let padMode: PadMode

    public enum Normalization: Sendable {
        /// Whisper: clip to `max−8`, shift+scale by `(x+4)/4`. log base 10.
        case whisperLogClip
        /// Per-instance scalar mean/std normalization on natural-log mel.
        case perInstanceMeanStd
        /// Per-mel-bin mean/std over time with Bessel correction. NeMo `per_feature`
        /// convention; matches HF `ParakeetFeatureExtractor`.
        case perBinMeanStd
    }

    public enum Layout: Sendable {
        /// Whisper-style `[B, n_mels, n_frames]`.
        case channelMajor
        /// Parakeet-style `[B, n_frames, n_mels]`.
        case timeMajor
    }

    public enum PadMode: Sendable {
        /// Mirror the signal across the boundary (matches torch.stft `pad_mode="reflect"`).
        case reflect
        /// Zero-pad the boundary (matches torch.stft `pad_mode="constant"` — what HF
        /// `ParakeetFeatureExtractor` uses).
        case constant
    }

    public enum MelScale: Sendable {
        /// `2595·log10(1 + f/700)` — original HTK formula. Whisper's convention.
        case htk
        /// Slaney auditory toolbox: linear ≤ 1000Hz at `1 mel = 200/3 Hz`, log
        /// above with step `log(6.4)/27`. Default for `librosa.filters.mel` and
        /// what HF `ParakeetFeatureExtractor` uses.
        case slaney
    }

    public init(
        sampleRate: Double, nFFT: Int, winLength: Int, hopLength: Int,
        nMelBins: Int, nFrames: Int?, preemphasis: Float?,
        normalization: Normalization, layout: Layout,
        padMode: PadMode, melScale: MelScale,
        windowPeriodicity: WindowPeriodicity = .symmetric
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
        self.padMode = padMode
        self.melScale = melScale
        self.windowPeriodicity = windowPeriodicity
    }

    public let melScale: MelScale

    /// Which Hann window convention to build.
    ///
    /// `torch.hann_window(N)` defaults to `periodic: true`, dividing by `N`; passing
    /// `periodic: false` divides by `N - 1`. HF's Whisper extractor calls it bare and so
    /// gets the periodic form, while Parakeet's passes `periodic=False`. The two differ
    /// by one sample of taper — small, but applied identically to every frame.
    public enum WindowPeriodicity: Sendable {
        case periodic
        case symmetric
    }
    public let windowPeriodicity: WindowPeriodicity

    /// Whisper v3-turbo parameters.
    ///
    /// Slaney mel scale, not HTK: HF's `WhisperFeatureExtractor` builds its filterbank
    /// with `mel_scale="slaney"`, and reference OpenAI Whisper takes librosa's default
    /// (`htk=False`), which is also Slaney. The two scales place every filter
    /// differently, so HTK here shifts the whole spectrogram.
    public static let whisper = MelConfig(
        sampleRate: 16_000, nFFT: 400, winLength: 400, hopLength: 160,
        nMelBins: 128, nFrames: 3_000, preemphasis: nil,
        normalization: .whisperLogClip, layout: .channelMajor,
        padMode: .reflect, melScale: .slaney, windowPeriodicity: .periodic)

    /// Parakeet TDT v3 parameters. Matches HF `ParakeetFeatureExtractor`: Slaney mel scale,
    /// per-mel-bin Bessel-corrected normalization (NeMo `per_feature`), zero-pad boundary,
    /// and 16 kHz target sample rate. AVAudioConverter is used for resampling instead of
    /// `soxr_hq`; that remaining mismatch is small and tolerated by the model.
    public static let parakeet = MelConfig(
        sampleRate: 16_000, nFFT: 512, winLength: 400, hopLength: 160,
        nMelBins: 128, nFrames: nil, preemphasis: 0.97,
        normalization: .perBinMeanStd, layout: .timeMajor,
        padMode: .constant, melScale: .slaney, windowPeriodicity: .symmetric)

    /// A copy of this config with `nFrames` replaced.
    ///
    /// Exists so callers baking a traced frame count into a preset cannot silently drop a field:
    /// the previous open-coded rebuild listed eleven of twelve properties and omitted
    /// `windowPeriodicity`, which then fell back to the initializer default. That happened to match
    /// the preset, so static and dynamic exports agreed by luck rather than by construction.
    package func withNFrames(_ nFrames: Int?) -> MelConfig {
        MelConfig(
            sampleRate: sampleRate, nFFT: nFFT, winLength: winLength, hopLength: hopLength,
            nMelBins: nMelBins, nFrames: nFrames, preemphasis: preemphasis,
            normalization: normalization, layout: layout,
            padMode: padMode, melScale: melScale, windowPeriodicity: windowPeriodicity)
    }
}

// MARK: - MelSpectrogram

/// Computes a mel spectrogram from an audio file or raw PCM samples.
public enum MelSpectrogram {
    // MARK: Public API

    /// The number of mel frames the configured pipeline will emit for a given PCM length.
    public static func frameCount(forPCMLength count: Int, config: MelConfig) -> Int {
        if let n = config.nFrames { return n }
        // torch.stft with center=True emits `1 + N // hop` frames. HF's
        // ParakeetFeatureExtractor passes that whole tensor to the encoder
        // (with the trailing frame zeroed via attention_mask) — we match
        // its shape so the encoder sees the same number of time steps.
        return 1 + count / config.hopLength
    }

    /// Number of *valid* (non-padded) mel frames for a given PCM length — i.e. the
    /// frames that carry real audio, excluding a static window's zero padding. For
    /// a dynamic config every frame is real. Mirrors the count computed in
    /// `padToFrameGrid`.
    public static func validFrameCount(forPCMLength count: Int, config: MelConfig) -> Int {
        if let n = config.nFrames { return min(count / config.hopLength, n) }  // static: clamp to window
        return count / config.hopLength  // dynamic: all real
    }

    public static func fromFile(
        _ url: URL, config: MelConfig = .whisper, basis: Basis? = nil
    ) throws -> [Float] {
        return fromPCM(
            try loadAndResample(url, targetSampleRate: config.sampleRate), config: config, basis: basis)
    }

    /// Computes the log-mel spectrogram for `raw`.
    ///
    /// Pass a `Basis` built once for `config` to keep the window/DFT/filterbank
    /// construction off this path. Omitting it rebuilds the basis per call.
    public static func fromPCM(
        _ raw: [Float], config: MelConfig = .whisper, basis: Basis? = nil
    ) -> [Float] {
        let basis = basis ?? Basis(config: config)
        precondition(
            basis.nFFT == config.nFFT && basis.winLength == config.winLength
                && basis.nMelBins == config.nMelBins,
            "MelSpectrogram.Basis was built for a different MelConfig "
                + "(basis nFFT=\(basis.nFFT) winLength=\(basis.winLength) nMelBins=\(basis.nMelBins))")

        let preemph = applyPreemphasis(raw, alpha: config.preemphasis)
        let (audio, validFrames) = padToFrameGrid(preemph, config: config)
        // For static configs totalFrames == nFrames (the full traced shape, which
        // may be larger than validFrames when audio is shorter than the window).
        // For dynamic configs totalFrames == validFrames + 1 (trailing zero frame).
        let totalFrames = config.nFrames ?? (validFrames + 1)

        let pad = config.nFFT / 2
        let padded = padAudio(audio, pad: pad, mode: config.padMode)

        // Bind to locals so the frame loop below indexes arrays directly rather than
        // re-reading struct properties on every iteration.
        let window = basis.window
        let cosBasis = basis.cosBasis
        let sinBasis = basis.sinBasis
        let filterbank = basis.filterbank
        let frameOffset = (config.nFFT - config.winLength) / 2
        let nFreqs = config.nFFT / 2 + 1

        var windowed = [Float](repeating: 0, count: config.winLength)
        var frame = [Float](repeating: 0, count: config.nFFT)
        var yReal = [Float](repeating: 0, count: nFreqs)
        var yImag = [Float](repeating: 0, count: nFreqs)
        var powerSpec = [Float](repeating: 0, count: nFreqs)
        var melFrame = [Float](repeating: 0, count: config.nMelBins)
        var mel = [Float](repeating: 0, count: config.nMelBins * totalFrames)

        // Split-complex scratch for the FFT path. Raw buffers rather than Arrays so the
        // frame loop can hand vDSP a `DSPSplitComplex` without re-entering a
        // `withUnsafeMutableBufferPointer` closure on every frame.
        let half = max(config.nFFT / 2, 1)
        let fftReal = UnsafeMutablePointer<Float>.allocate(capacity: half)
        let fftImag = UnsafeMutablePointer<Float>.allocate(capacity: half)
        fftReal.initialize(repeating: 0, count: half)
        fftImag.initialize(repeating: 0, count: half)
        defer {
            fftReal.deallocate()
            fftImag.deallocate()
        }
        var split = DSPSplitComplex(realp: fftReal, imagp: fftImag)

        // The radix-2 setup for this frame size, when the size supports one. `Basis`
        // already validated that nFFT is a power of two, so a nil here is an allocation
        // failure for a table of nFFT constants — not a recoverable condition, and the
        // dense tables were deliberately not built for this config.
        let fftPlan: (setup: FFTSetup, log2n: vDSP_Length)?
        if let log2n = basis.log2n {
            guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
                preconditionFailure("vDSP_create_fftsetup failed for nFFT=\(config.nFFT)")
            }
            fftPlan = (setup, log2n)
        } else {
            fftPlan = nil
        }
        defer { vDSP_destroy_fftsetup(fftPlan?.setup) }

        let logFloor: Float = 1e-10
        // Whisper clamps the mel floor then takes log10 (OpenAI convention); Parakeet/NeMo
        // adds a small guard *inside* a natural log — HF ParakeetFeatureExtractor uses
        // LOG_ZERO_GUARD_VALUE = 2^-24. The two differ sharply for low-energy frames, so
        // honor each pipeline's exact convention rather than sharing one floor.
        let logGuard: Float = 0x1p-24
        let useLog10 = (config.normalization == .whisperLogClip)

        // Whisper computes and normalizes its entire fixed window (including the
        // zero-padded tail as log-silence) so results match reference OpenAI Whisper
        // for sub-window audio. Every other pipeline (Parakeet) fills only the valid
        // frames and leaves the masked tail at zero.
        let framesToCompute = (config.normalization == .whisperLogClip) ? totalFrames : validFrames

        for t in 0..<framesToCompute {
            let offset = t * config.hopLength
            // Multiply the window straight out of `padded` via a base-pointer offset —
            // avoids allocating an `Array` slice on every frame in this hot loop.
            //
            // `+ frameOffset` matters when winLength < nFFT (Parakeet: 400 in 512).
            // torch.stft pads the window to nFFT *centred* and multiplies the whole
            // nFFT-sample frame, so the taper lands on padded[offset+frameOffset...],
            // not padded[offset...]. Reading from `offset` here would window audio
            // `frameOffset` samples early while still writing it to the centred slot
            // below — a sub-hop misalignment that survives into every mel bin.
            padded.withUnsafeBufferPointer { p in
                vDSP_vmul(
                    p.baseAddress! + offset + frameOffset, 1,
                    window, 1, &windowed, 1, vDSP_Length(config.winLength))
            }
            // Place the windowed slice into a zero-padded nFFT buffer.
            for i in 0..<config.nFFT { frame[i] = 0 }
            for i in 0..<config.winLength { frame[frameOffset + i] = windowed[i] }
            if let fftPlan {
                // Real-to-complex FFT. vDSP takes the frame deinterleaved into the
                // split-complex halves (even samples → realp, odd → imagp).
                frame.withUnsafeBufferPointer { p in
                    p.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { c in
                        vDSP_ctoz(c, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(fftPlan.setup, &split, 1, fftPlan.log2n, FFTDirection(FFT_FORWARD))
                // zrip scales its output by 2, and packs the two purely-real bins into
                // element 0: realp[0] = 2·X[0], imagp[0] = 2·X[nFFT/2]. Undo both so
                // powerSpec holds |X[k]|², matching the dense path below.
                powerSpec[0] = 0.25 * fftReal[0] * fftReal[0]
                powerSpec[half] = 0.25 * fftImag[0] * fftImag[0]
                for k in 1..<half {
                    powerSpec[k] = 0.25 * (fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k])
                }
            } else {
                // No radix-2 FFT for this nFFT (e.g. Whisper's 400 = 2⁴·5²) — evaluate
                // the DFT as a dense matrix-vector product against the cos/sin tables.
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(nFreqs), Int32(config.nFFT), 1.0, cosBasis, Int32(config.nFFT),
                    frame, 1, 0.0, &yReal, 1)
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(nFreqs), Int32(config.nFFT), 1.0, sinBasis, Int32(config.nFFT),
                    frame, 1, 0.0, &yImag, 1)
                vDSP_vmma(yReal, 1, yReal, 1, yImag, 1, yImag, 1, &powerSpec, 1, vDSP_Length(nFreqs))
            }
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(config.nMelBins), Int32(nFreqs), 1.0, filterbank, Int32(nFreqs),
                powerSpec, 1, 0.0, &melFrame, 1)
            for i in 0..<config.nMelBins {
                let lv = useLog10 ? log10(max(melFrame[i], logFloor)) : log(melFrame[i] + logGuard)
                let idx = (config.layout == .channelMajor) ? (i * totalFrames + t) : (t * config.nMelBins + i)
                mel[idx] = lv
            }
        }

        // Normalize over the *valid* portion only — the trailing zero frame
        // (if any) stays zero, matching HF's masked-out-tail behavior.
        normalize(
            &mel, normalization: config.normalization,
            validFrames: validFrames, nMelBins: config.nMelBins, layout: config.layout)

        return mel
    }

    // MARK: Audio loading

    public static func loadAndResample(_ url: URL, targetSampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)

        // Fast path: file is already mono at the target rate. Read the float samples
        // directly instead of routing through AVAudioConverter. A 1:1 converter isn't
        // guaranteed bit-exact and diverges slightly from HF/librosa, which skip
        // resampling entirely when the rate already matches — keeping this path faithful.
        // (processingFormat is always standard deinterleaved float32, so floatChannelData
        // is valid.)
        let processing = file.processingFormat
        if processing.channelCount == 1 && processing.sampleRate == targetSampleRate {
            return try readAllSamples(file)
        }

        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate, channels: 1, interleaved: false)!
        guard let conv = AVAudioConverter(from: file.processingFormat, to: fmt) else {
            throw SpeechError.invalidAudio(
                "Cannot resample \(file.processingFormat) to \(targetSampleRate) Hz mono")
        }
        conv.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let cap = AVAudioFrameCount(
            ceil(Double(file.length) * targetSampleRate / file.processingFormat.sampleRate) + 1)
        let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap)!
        var readError: Error?
        var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            // Feed successive chunks: one read is not guaranteed to return the whole
            // file (see `readAllSamples`), and stopping after the first would hand the
            // converter a truncated signal.
            guard file.framePosition < file.length else {
                status.pointee = .endOfStream
                return nil
            }
            let remaining = min(Int64(readChunkFrames), file.length - file.framePosition)
            guard
                let buf = AVAudioPCMBuffer(
                    pcmFormat: file.processingFormat,
                    frameCapacity: AVAudioFrameCount(remaining))
            else {
                status.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: buf, frameCount: AVAudioFrameCount(remaining))
            } catch {
                // Surface the failure instead of silently ending the stream, which
                // would look like a short file rather than an error.
                readError = error
                status.pointee = .endOfStream
                return nil
            }
            guard buf.frameLength > 0 else {
                status.pointee = .endOfStream
                return nil
            }
            status.pointee = .haveData
            return buf
        }
        if let readError { throw readError }
        if let e = err { throw SpeechError.invalidAudio(e.localizedDescription) }
        return Array(
            UnsafeBufferPointer(
                start: out.floatChannelData![0],
                count: Int(out.frameLength)))
    }

    /// Frames per `AVAudioFile.read` call. Any value works; this just bounds the
    /// scratch buffer while keeping the number of reads small.
    package static let readChunkFrames: AVAudioFrameCount = 1 << 16

    /// Read an entire mono float file into `[Float]`.
    ///
    /// `AVAudioFile.read(into:)` reads *up to* the buffer's capacity and may stop early
    /// at an internal packet boundary — a float32 WAV comes back several hundred frames
    /// short of its length, silently truncating the audio and shifting every later mel
    /// frame. Reading in a loop and accumulating is format-agnostic. Note
    /// `read(into:frameCount:)` fills from the start of the buffer rather than
    /// appending, so the samples are copied out after each call.
    private static func readAllSamples(_ file: AVAudioFile) throws -> [Float] {
        guard
            let buf = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: readChunkFrames)
        else {
            throw SpeechError.invalidAudio("Cannot allocate read buffer for \(file.url.lastPathComponent)")
        }
        var samples = [Float]()
        samples.reserveCapacity(Int(file.length))
        // Bound by framePosition/length rather than reading until failure: a read that
        // runs past the end throws eofErr (OSStatus -39) instead of returning zero
        // frames, so "loop until empty" would surface as an error on every file.
        while file.framePosition < file.length {
            let remaining = min(Int64(readChunkFrames), file.length - file.framePosition)
            try file.read(into: buf, frameCount: AVAudioFrameCount(remaining))
            guard buf.frameLength > 0 else { break }
            guard let ch = buf.floatChannelData else {
                throw SpeechError.invalidAudio(
                    "Expected float samples reading \(file.url.lastPathComponent)")
            }
            samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(buf.frameLength)))
        }
        return samples
    }

    // MARK: Preprocessing

    package static func applyPreemphasis(_ raw: [Float], alpha: Float?) -> [Float] {
        guard let alpha, !raw.isEmpty else { return raw }
        var out = [Float](repeating: 0, count: raw.count)
        out[0] = raw[0]
        for i in 1..<raw.count { out[i] = raw[i] - alpha * raw[i - 1] }
        return out
    }

    package static func padToFrameGrid(_ raw: [Float], config: MelConfig) -> ([Float], Int) {
        if let target = config.nFrames {
            // Static model: pad or truncate to exactly target frames.
            // validFrames = min(actual, target) so normalization only covers
            // real audio, not the zero-padding.
            let n = target * config.hopLength
            let validFrames = validFrameCount(forPCMLength: raw.count, config: config)
            var audio = raw
            if audio.count > n {
                audio = Array(audio.prefix(n))
            } else if audio.count < n {
                audio += [Float](repeating: 0, count: n - audio.count)
            }
            return (audio, validFrames)
        }
        // Variable-length: HF emits `1 + N//hop` frames with the last zeroed
        // (torch.stft center=True semantics). We compute `validFrames = N//hop`
        // valid frames here; the caller appends a trailing zero frame to match
        // HF's tensor shape.
        //
        // The waveform is handed through whole. `validFrames` bounds how many frames
        // are computed and normalized, not how much audio the STFT may read: rounding
        // the input down to a whole number of hops would discard up to hopLength-1
        // real samples, which the final frames' windows still reach into.
        let validFrames = raw.count / config.hopLength
        return (raw, validFrames)
    }

    package static func padAudio(_ audio: [Float], pad: Int, mode: MelConfig.PadMode) -> [Float] {
        let n = audio.count
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<n { padded[pad + i] = audio[i] }
        switch mode {
        case .constant:
            break  // edges already zero-filled
        case .reflect:
            for i in 0..<pad { padded[pad - 1 - i] = audio[i + 1] }
            for i in 0..<pad { padded[pad + n + i] = audio[n - 2 - i] }
        }
        return padded
    }

    package static func normalize(
        _ mel: inout [Float],
        normalization: MelConfig.Normalization,
        validFrames: Int,
        nMelBins: Int,
        layout: MelConfig.Layout
    ) {
        let count = validFrames * nMelBins
        // Flat index of mel bin `b` at time step `t`. `totalFrames` may exceed
        // `validFrames` (a static window's zero-padded tail, or the dynamic path's single
        // trailing zero frame), and for channelMajor that padding sits *between* bins:
        // bin b occupies b*totalFrames..<(b+1)*totalFrames. So the valid entries are not
        // the first `count` elements of the array, and every statistic below has to walk
        // them by index rather than linearly.
        let totalFrames = mel.count / nMelBins
        func index(_ t: Int, _ b: Int) -> Int {
            (layout == .timeMajor) ? (t * nMelBins + b) : (b * totalFrames + t)
        }
        switch normalization {
        case .whisperLogClip:
            // Whisper fills its entire fixed window (see `framesToCompute`), so every
            // entry is real log-mel data — including the log-silence tail for sub-window
            // audio. Clip and scale over the whole array together, matching reference
            // OpenAI Whisper's global normalization.
            let maxVal = mel.max() ?? 0
            for i in 0..<mel.count { mel[i] = (max(mel[i], maxVal - 8) + 4) / 4 }
        case .perInstanceMeanStd:
            // One mean/std over every valid entry of every bin.
            if count == 0 { return }
            var sum: Float = 0
            for b in 0..<nMelBins {
                for t in 0..<validFrames { sum += mel[index(t, b)] }
            }
            let mean = sum / Float(count)
            var sqSum: Float = 0
            for b in 0..<nMelBins {
                for t in 0..<validFrames {
                    let d = mel[index(t, b)] - mean
                    sqSum += d * d
                }
            }
            let std = sqrt(sqSum / Float(count))
            let denom = max(std, 1e-5)
            for b in 0..<nMelBins {
                for t in 0..<validFrames {
                    let i = index(t, b)
                    mel[i] = (mel[i] - mean) / denom
                }
            }
        case .perBinMeanStd:
            // NeMo `per_feature`: normalize each mel bin independently over time.
            // Bessel-corrected std (divide by validFrames−1).
            if validFrames < 2 { return }
            for b in 0..<nMelBins {
                var sum: Float = 0
                for t in 0..<validFrames { sum += mel[index(t, b)] }
                let mean = sum / Float(validFrames)
                var sqSum: Float = 0
                for t in 0..<validFrames {
                    let d = mel[index(t, b)] - mean
                    sqSum += d * d
                }
                let std = sqrt(sqSum / Float(validFrames - 1))
                let denom = std + 1e-5  // HF adds EPSILON inside the divide (not a clamp)
                for t in 0..<validFrames {
                    let i = index(t, b)
                    mel[i] = (mel[i] - mean) / denom
                }
            }
        // Trailing entries beyond validFrames (if any) stay zero.
        }
    }

    // MARK: Precomputed basis

    /// The window, DFT basis, and mel filterbank — everything in the mel pipeline
    /// that depends only on `MelConfig`.
    ///
    /// Split out so a caller transcribing repeatedly can build it once and hand it to
    /// `fromPCM`, rather than reconstructing identical tables on every call.
    public struct Basis: Sendable {
        package let window: [Float]
        package let filterbank: [Float]

        /// `log2(nFFT)` when `nFFT` is a power of two — the sizes `kFFTRadix2` handles. In
        /// that case the STFT runs as an FFT and `cosBasis`/`sinBasis` stay empty;
        /// otherwise this is nil and the dense tables below are built and used.
        ///
        /// Only the exponent is stored, not a vDSP setup: the setup is an opaque pointer
        /// needing manual destruction, which would make `Basis` a reference type carrying
        /// lifetime and concurrency annotations. `fromPCM` creates one per call instead —
        /// it prepares a table of `nFFT` constants, far below one frame's work.
        package let log2n: vDSP_Length?
        package let cosBasis: [Float]
        package let sinBasis: [Float]

        // Retained so `fromPCM` can check the basis it was handed matches its config.
        package let nFFT: Int
        package let winLength: Int
        package let nMelBins: Int

        public init(config: MelConfig) {
            self.window = MelSpectrogram.hannWindow(
                size: config.winLength, periodicity: config.windowPeriodicity)
            self.filterbank = MelSpectrogram.melFilterbank(config: config)
            if config.nFFT > 1, config.nFFT & (config.nFFT - 1) == 0 {
                self.log2n = vDSP_Length(config.nFFT.trailingZeroBitCount)
                self.cosBasis = []
                self.sinBasis = []
            } else {
                self.log2n = nil
                let (cos, sin) = MelSpectrogram.dftBasis(nFFT: config.nFFT)
                self.cosBasis = cos
                self.sinBasis = sin
            }
            self.nFFT = config.nFFT
            self.winLength = config.winLength
            self.nMelBins = config.nMelBins
        }
    }

    package static func hannWindow(
        size: Int, periodicity: MelConfig.WindowPeriodicity
    ) -> [Float] {
        let denom = Double(periodicity == .periodic ? size : size - 1)
        return (0..<size).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / denom))) }
    }

    package static func dftBasis(nFFT: Int) -> ([Float], [Float]) {
        let nFreqs = nFFT / 2 + 1
        var cos = [Float](repeating: 0, count: nFreqs * nFFT)
        var sin = [Float](repeating: 0, count: nFreqs * nFFT)
        for k in 0..<nFreqs {
            for n in 0..<nFFT {
                // Reduce k·n modulo nFFT before scaling, and evaluate in Double: the
                // basis is periodic in k·n, but forming 2π·k·n/nFFT directly pushes the
                // angle into the thousands of radians where Float's absolute error
                // (~1e-4 rad) shows up in the resulting power spectrum.
                let angle = 2 * Double.pi * Double((k * n) % nFFT) / Double(nFFT)
                cos[k * nFFT + n] = Float(Foundation.cos(angle))
                sin[k * nFFT + n] = Float(-Foundation.sin(angle))
            }
        }
        return (cos, sin)
    }

    package static func melFilterbank(config: MelConfig) -> [Float] {
        let nFreqs = config.nFFT / 2 + 1
        let fMax = Double(config.sampleRate) / 2

        // HTK mel — original 1980 toolkit. Whisper uses this.
        func htkH2M(_ f: Double) -> Double { 2595 * Foundation.log10(1 + f / 700) }
        func htkM2H(_ m: Double) -> Double { 700 * (Foundation.pow(10, m / 2595) - 1) }
        // Slaney auditory toolbox — librosa default; HF Parakeet uses this.
        let fSp: Double = 200.0 / 3
        let minLogHz: Double = 1000
        let minLogMel: Double = minLogHz / fSp  // = 15
        let logstep: Double = Foundation.log(6.4) / 27
        func slaneyH2M(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + Foundation.log(f / minLogHz) / logstep : f / fSp
        }
        func slaneyM2H(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * Foundation.exp(logstep * (m - minLogMel)) : fSp * m
        }

        let h2m: (Double) -> Double
        let m2h: (Double) -> Double
        switch config.melScale {
        case .htk:
            h2m = htkH2M
            m2h = htkM2H
        case .slaney:
            h2m = slaneyH2M
            m2h = slaneyM2H
        }

        let pts = (0..<config.nMelBins + 2).map { i -> Double in
            m2h(h2m(0) + Double(i) / Double(config.nMelBins + 1) * (h2m(fMax) - h2m(0)))
        }
        let fftFreqs = (0..<nFreqs).map { Double($0) * Double(config.sampleRate) / Double(config.nFFT) }
        var fb = [Float](repeating: 0, count: config.nMelBins * nFreqs)
        for m in 0..<config.nMelBins {
            let fL = pts[m]
            let fC = pts[m + 1]
            let fR = pts[m + 2]
            let norm: Double = 2 / (fR - fL)
            for k in 0..<nFreqs {
                let f = fftFreqs[k]
                if f >= fL && f <= fC {
                    fb[m * nFreqs + k] = Float(norm * (f - fL) / (fC - fL))
                } else if f > fC && f <= fR {
                    fb[m * nFreqs + k] = Float(norm * (fR - f) / (fR - fC))
                }
            }
        }
        return fb
    }
}
