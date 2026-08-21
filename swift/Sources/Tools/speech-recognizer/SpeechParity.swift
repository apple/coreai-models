// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import ArgumentParser
import CoreAI
import CoreAIShared
import CoreAISpeech
import Foundation

// MARK: - Parity test
//
// Compares the *Swift runtime's* own intermediates against PyTorch reference traces,
// mirroring the `image-segmenter --parity-test` / `diffusion-runner --parity-test`
// harnesses. This exercises the code that ships, as distinct from checking an exported
// graph in isolation: if a graph reproduces its reference but the runtime does not, the
// fault is in Swift.
//
// The trace directory is expected to contain:
//
//   audio.wav                      input both sides consume, at its original rate
//   meta.json                      {"architecture": "parakeet" | "whisper"}
//   ref_input_features.npy         feature-extractor output, row-major
//   ref_encoder_hidden_states.npy  encoder output      (parakeet; optional)
//   ref_tokens.npy                 emitted token ids, int32 (parakeet; optional)
//   ref_transcript.txt             expected text       (optional)
//
// Every `ref_*` file is optional: whichever are present become rows, so a trace set can
// cover only the front-end. Arrays must be C-contiguous — see `NpyArray.load`.

struct SpeechParity {
    let directory: URL
    let modelPath: String?
    let psnrFloor: Double
    let melPsnrFloor: Double
    /// PSNR floor for the mel row when the runtime had to resample the input.
    let resampledMelPsnrFloor: Double
    /// PSNR floor for the encoder row in that case.
    ///
    /// Lower than the mel's: the encoder compounds the front-end difference through the
    /// conformer stack, so it sits several dB below by construction. The mel floor is
    /// backed by a measured distribution across resamplers; the encoder's own spread has
    /// not been characterised, so this is set to leave real margin over the observed
    /// values (30.4 dB static, 32.1 dB dynamic) rather than to sit just under them.
    let resampledEncoderPsnrFloor: Double
    /// Cosine floor for both rows in that case.
    let resampledCosineFloor: Double
    let cosineFloor: Double

    func run() async throws {
        let meta = try Metadata(directory: directory)
        print("=== Speech runtime parity (\(meta.architecture)) ===")
        print("  traces: \(directory.path)")
        let audioURL = directory.appending(path: "audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("--parity-test dir is missing audio.wav")
        }
        let format = try Self.audioFormat(audioURL)
        print("  audio:  \(Int(format.rate)) Hz, \(format.channels) channel(s)")
        if !meta.comparable {
            print("  tensor rows skipped — \(meta.uncomparableReason ?? "reference not comparable")")
        }

        var rows: [ParityRow]
        let transcript: String?
        switch meta.architecture {
        case "parakeet":
            (rows, transcript) = try await parakeetRows(
                audioURL: audioURL, compareTensors: meta.comparable,
                compareTokens: meta.comparableTokens)
        case "whisper":
            (rows, transcript) = (
                try whisperRows(audioURL: audioURL, compareTensors: meta.comparable), nil
            )
        default:
            throw ValidationError("unsupported architecture '\(meta.architecture)' in meta.json")
        }
        if let reference = try referenceTranscript() {
            rows.append(transcriptRow(actual: transcript, reference: reference))
        }

        print("")
        for row in rows {
            let label = row.name.padding(toLength: 22, withPad: " ", startingAt: 0)
            print("  \(label) \(row.status)  \(row.ok ? "✓" : "✗")")
        }
        if rows.contains(where: { !$0.ok }) {
            print("\nParity FAILED.")
            throw ExitCode.failure
        }
        print("\nAll compared stages within tolerance.")
    }

    // MARK: Architectures

    /// Full pipeline: mel, encoder hidden states, emitted tokens, transcript.
    private func parakeetRows(
        audioURL: URL, compareTensors: Bool, compareTokens: Bool
    ) async throws -> ([ParityRow], String) {
        guard let modelPath else {
            throw ValidationError("--model is required for parakeet parity traces")
        }
        let model = try await SpeechRecognitionModel(
            resourcesAt: URL(fileURLWithPath: modelPath))
        let targetRate = await model.sampleRate
        let resampled = try Self.requiresConversion(audioURL, targetRate: targetRate)
        let pcm = try MelSpectrogram.loadAndResample(audioURL, targetSampleRate: targetRate)
        let capture = try await model.transcribeCapturingStages(pcm: pcm)

        var rows: [ParityRow] = []
        defer { printCoverage(capture.stats.coverage) }
        guard compareTensors else {
            if compareTokens, let ref = try? reference("ref_tokens.npy") {
                rows.append(tokenRow(actual: capture.tokens, ref: try ref.asInt32()))
            } else {
                print("  token row skipped — reference tokens derive from those features")
            }
            return (rows, capture.text)
        }
        rows.append(
            melRow(
                name: "input_features", actual: capture.mel, shape: capture.melShape,
                resampled: resampled))

        // Compare only the encoder frames the runtime decodes. A static export's tail is
        // derived from zero padding and is discarded before decoding, so including it
        // would score a region no output depends on.
        //
        // `validEncoderFrames` derives the count from the subsampling arithmetic rather than
        // estimating it, so it must match the reference exactly. This previously tolerated a
        // one-frame difference to absorb a proportional estimate.
        if let ref = try? reference("ref_encoder_hidden_states.npy") {
            let hidden = capture.encoderShape.last ?? 1
            let valid = capture.validEncoderFrames
            let refFrames = ref.shape.count == 3 ? ref.shape[1] : 0
            if refFrames != valid {
                rows.append(
                    ParityRow(
                        name: "encoder_hidden_states",
                        status: "valid-frame mismatch — runtime treats \(valid) of "
                            + "\(capture.encoderShape[1]) frames as real audio, ref has \(refFrames)",
                        ok: false))
            } else {
                let refValues = try ref.asFloat()
                let common = min(valid, refFrames)
                var row = metricRow(
                    name: "encoder_hidden_states",
                    actual: Array(capture.encoderHiddenStates.prefix(common * hidden)),
                    ref: Array(refValues.prefix(common * hidden)),
                    psnrFloor: resampled ? resampledEncoderPsnrFloor : psnrFloor,
                    cosineFloor: resampled ? resampledCosineFloor : cosineFloor)
                if resampled {
                    row.status +=
                        "  [resampled floors: PSNR \(String(format: "%g", resampledEncoderPsnrFloor))"
                        + ", cosine \(String(format: "%g", resampledCosineFloor))]"
                }
                if valid < capture.encoderShape[1] {
                    row.status += "  (\(capture.encoderShape[1] - valid) padding frames excluded)"
                }
                if valid != refFrames {
                    row.status += "  (valid estimate \(valid) vs ref \(refFrames); compared \(common))"
                }
                rows.append(row)
            }
        }

        if let ref = try? reference("ref_tokens.npy") {
            rows.append(tokenRow(actual: capture.tokens, ref: try ref.asInt32()))
        }
        rows.append(contentsOf: try firstStepRows(capture.stats.firstStep, resampled: resampled))
        return (rows, capture.text)
    }

    /// Whisper: mel only.
    ///
    /// The repo exports Whisper as one monolithic asset, so there is no encoder
    /// component to compare. The mel is the point anyway — it is the only coverage of
    /// `MelConfig.whisper`: nFFT=400 takes the non-radix-2 dense DFT path, with
    /// `whisperLogClip` normalization and `channelMajor` layout, none of which the
    /// Parakeet traces reach.
    private func whisperRows(audioURL: URL, compareTensors: Bool) throws -> [ParityRow] {
        guard compareTensors else { return [] }
        let config = MelConfig.whisper
        let resampled = try Self.requiresConversion(audioURL, targetRate: config.sampleRate)
        let pcm = try MelSpectrogram.loadAndResample(audioURL, targetSampleRate: config.sampleRate)
        let mel = MelSpectrogram.fromPCM(pcm, config: config)
        let frames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: config)
        return [
            melRow(
                name: "input_features", actual: mel,
                shape: [1, config.nMelBins, frames], resampled: resampled)
        ]
    }

    // MARK: Rows

    /// Tensor rows for the first `decoder_step` and `joint` call of the decode.
    ///
    /// Without these, both graphs are only checked through the emitted tokens — and
    /// argmax is discrete, so it absorbs drift that would show up here.
    private func firstStepRows(
        _ step: DecodeStats.FirstStep?, resampled: Bool
    ) throws -> [ParityRow] {
        guard let step else { return [] }
        let psnr = resampled ? resampledEncoderPsnrFloor : psnrFloor
        let cosine = resampled ? resampledCosineFloor : cosineFloor
        var rows: [ParityRow] = []
        for (file, name, actual) in [
            ("ref_decoder_output.npy", "decoder_output", step.decoderOutput),
            ("ref_new_hidden_state.npy", "new_hidden_state", step.newHiddenState),
            ("ref_new_cell_state.npy", "new_cell_state", step.newCellState),
            ("ref_joint_logits.npy", "joint_logits", step.jointLogits),
        ] {
            guard let ref = try? reference(file) else { continue }
            var row = metricRow(
                name: name, actual: actual, ref: try ref.asFloat(),
                psnrFloor: psnr, cosineFloor: cosine)
            if resampled {
                row.status +=
                    "  [resampled floors: PSNR \(String(format: "%g", psnr))"
                    + ", cosine \(String(format: "%g", cosine))]"
            }
            rows.append(row)
        }
        return rows
    }

    /// Report which branches of the TDT loop this input reached.
    ///
    /// Printed rather than gated by default: which branches a given audio clip exercises
    /// is a property of the clip, not a defect. It is here so coverage is a measured fact
    /// instead of an assumption. Nothing asserts a particular branch is reached yet —
    /// two of them are not reached by any input tried, so a gate would have to be
    /// per-trace-set rather than global.
    private func printCoverage(_ c: DecodeStats.Coverage) {
        print("  decode-loop coverage:")
        for (label, count) in Self.branchCounts(c) {
            print("    \(label.padding(toLength: 26, withPad: " ", startingAt: 0)) \(count)")
        }
    }

    static func branchCounts(_ c: DecodeStats.Coverage) -> [(String, Int)] {
        [
            ("blankSkipReuses", c.blankSkipReuses),
            ("lstmStateAdvances", c.lstmStateAdvances),
            ("blankZeroDurationBreaks", c.blankZeroDurationBreaks),
            ("positiveDurationBreaks", c.positiveDurationBreaks),
            ("symbolCapExhaustions", c.symbolCapExhaustions),
            ("multiTokenSteps", c.multiTokenSteps),
            ("blankOnlySteps", c.blankOnlySteps),
        ]
    }

    /// Mel row, held to a tighter floor than the graph stages.
    ///
    /// The front-end is deterministic CPU arithmetic against a reference running the
    /// same algorithm, so it agrees to ~124 dB. Scoring it against the ~30 dB that
    /// suits an f16 graph would pass errors large enough to shift every mel bin.
    ///
    /// Resampled traces get `resampledMelPsnrFloor` instead: the reference resamples with
    /// soxr_hq and the runtime with AVAudioConverter, and those filter designs genuinely
    /// differ in the transition band below 8 kHz — the error is confined to the top ~16
    /// mel bins. `resampledMelPsnrFloor`'s default separates AVAudioConverter and other
    /// correct resamplers from degraded ones; it is not fitted to one observation.
    private func melRow(
        name: String, actual: [Float], shape: [Int], resampled: Bool
    ) -> ParityRow {
        guard let ref = try? reference("ref_input_features.npy") else {
            return ParityRow(name: name, status: "ref_input_features.npy missing", ok: false)
        }
        guard let refValues = try? ref.asFloat() else {
            return ParityRow(name: name, status: "could not read ref_input_features.npy", ok: false)
        }

        var compared = actual
        var note = ""
        if refValues.count != actual.count {
            // A static export pads the mel out to its traced frame count. Compare the
            // leading real-audio frames; anything else is a genuine disagreement.
            guard
                let prefix = Self.melPrefix(
                    runtime: actual, runtimeShape: shape, refShape: ref.shape)
            else {
                return ParityRow(
                    name: name,
                    status: "element-count mismatch — runtime \(shape) (\(actual.count)) "
                        + "vs ref \(ref.shape) (\(refValues.count))",
                    ok: false)
            }
            compared = prefix.values
            note = "  (\(prefix.droppedFrames) padded frames excluded)"
        }

        var row = metricRow(
            name: name, actual: compared, ref: refValues,
            psnrFloor: resampled ? resampledMelPsnrFloor : melPsnrFloor,
            cosineFloor: resampled ? resampledCosineFloor : cosineFloor)
        row.status += note
        if resampled {
            row.status +=
                "  [resampled floors: PSNR \(String(format: "%g", resampledMelPsnrFloor))"
                + ", cosine \(String(format: "%g", resampledCosineFloor))]"
        }
        return row
    }

    /// Leading `refFrames` time steps of a runtime mel padded out to a longer traced
    /// length, in whichever layout the two shapes imply.
    ///
    /// `[1, T, nMels]` (timeMajor) is frame-contiguous, so the prefix is a plain slice.
    /// `[1, nMels, T]` (channelMajor) interleaves the padding between bins, so each
    /// bin's leading frames have to be gathered.
    private static func melPrefix(
        runtime: [Float], runtimeShape: [Int], refShape: [Int]
    ) -> (values: [Float], droppedFrames: Int)? {
        guard runtimeShape.count == 3, refShape.count == 3, runtimeShape[0] == refShape[0]
        else { return nil }
        if runtimeShape[2] == refShape[2], runtimeShape[1] > refShape[1] {
            let refFrames = refShape[1]
            return (
                Array(runtime.prefix(refFrames * refShape[2])),
                runtimeShape[1] - refFrames
            )
        }
        if runtimeShape[1] == refShape[1], runtimeShape[2] > refShape[2] {
            let bins = refShape[1]
            let runtimeFrames = runtimeShape[2]
            let refFrames = refShape[2]
            var out = [Float]()
            out.reserveCapacity(bins * refFrames)
            for b in 0..<bins {
                let start = b * runtimeFrames
                out.append(contentsOf: runtime[start..<(start + refFrames)])
            }
            return (out, runtimeFrames - refFrames)
        }
        return nil
    }

    private func tokenRow(actual: [Int32], ref: [Int32]) -> ParityRow {
        if actual == ref {
            return ParityRow(name: "tokens", status: "\(actual.count)/\(actual.count) tokens match", ok: true)
        }
        let n = min(actual.count, ref.count)
        if let first = (0..<n).first(where: { actual[$0] != ref[$0] }) {
            return ParityRow(
                name: "tokens",
                status: "first diff at \(first): runtime=\(actual[first]) vs ref=\(ref[first]) "
                    + "(\(actual.count) vs \(ref.count) tokens)",
                ok: false)
        }
        return ParityRow(
            name: "tokens",
            status: "length mismatch: runtime=\(actual.count) vs ref=\(ref.count) (prefix equal)",
            ok: false)
    }

    private func transcriptRow(actual: String?, reference: String) -> ParityRow {
        // An empty reference means the traces contain no speech (synthetic audio), so
        // there is nothing to assert; say so rather than passing a vacuous compare.
        if reference.isEmpty {
            return ParityRow(name: "transcript", status: "skipped — reference is empty", ok: true)
        }
        guard let actual else {
            return ParityRow(
                name: "transcript", status: "skipped — runtime produced no transcript", ok: true)
        }
        if actual == reference {
            return ParityRow(
                name: "transcript", status: "exact match (\(actual.count) chars)", ok: true)
        }
        return ParityRow(
            name: "transcript",
            status: "MISMATCH\n      runtime: \(actual)\n      ref    : \(reference)",
            ok: false)
    }

    // MARK: Audio format

    /// The source format, read from the file itself.
    private static func audioFormat(_ url: URL) throws -> (rate: Double, channels: UInt32) {
        let format = try AVAudioFile(forReading: url).fileFormat
        return (format.sampleRate, format.channelCount)
    }

    /// Whether `loadAndResample` will have to convert rather than read straight through.
    ///
    /// Determined from the file, not from `meta.json`: the thresholds applied downstream
    /// depend on this, so a generator that mislabelled its own traces would otherwise
    /// silently select the loose floors. Mirrors the fast-path condition in
    /// `MelSpectrogram.loadAndResample`, which requires mono at the target rate.
    private static func requiresConversion(_ url: URL, targetRate: Double) throws -> Bool {
        let format = try audioFormat(url)
        return format.rate != targetRate || format.channels != 1
    }

    // MARK: Loading

    private func reference(_ name: String) throws -> NpyArray {
        try NpyArray.load(directory.appending(path: name))
    }

    private func referenceTranscript() throws -> String? {
        let url = directory.appending(path: "ref_transcript.txt")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private struct Metadata {
        let architecture: String
        /// False when the trace's reference tensors are degenerate — see
        /// `uncomparableReason`. Token, transcript and coverage still apply.
        let comparable: Bool
        /// False when the reference token sequence is meaningless too — see the
        /// generator's notes. Distinct from `comparable`: degenerate features do not
        /// always imply a degenerate decode.
        let comparableTokens: Bool
        let uncomparableReason: String?

        init(directory: URL) throws {
            let url = directory.appending(path: "meta.json")
            guard let data = try? Data(contentsOf: url),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let architecture = json["architecture"] as? String
            else {
                throw ValidationError(
                    """
                    --parity-test dir needs a meta.json containing an "architecture" \
                    field set to "parakeet" or "whisper", alongside audio.wav and at \
                    least ref_input_features.npy.
                    """)
            }
            self.architecture = architecture
            self.comparable = (json["comparable"] as? Bool) ?? true
            self.comparableTokens = (json["comparable_tokens"] as? Bool) ?? true
            self.uncomparableReason = json["uncomparable_reason"] as? String
        }
    }
}

// MARK: - Metrics

private struct ParityRow {
    let name: String
    var status: String
    let ok: Bool
}

/// PSNR + cosine row.
///
/// `cosineFloor` of 0 effectively disables the cosine gate — used for resampled traces,
/// where the measured cosine (0.96 at the encoder) is far below what an exact comparison
/// yields and no distribution has been characterised for it. Zero still catches sign
/// inversion, which would be catastrophic rather than a tolerance question.
private func metricRow(
    name: String, actual: [Float], ref: [Float],
    psnrFloor: Double, cosineFloor: Double
) -> ParityRow {
    // Counts stay gated even when the metrics are not: a length disagreement is exact
    // regardless of which resampler produced the samples, and it is the signature of
    // dropped frames in the converter.
    if actual.count != ref.count {
        return ParityRow(
            name: name,
            status: "element-count mismatch — runtime \(actual.count) vs ref \(ref.count)",
            ok: false)
    }
    let p = psnr(actual, ref)
    let c = cosineSimilarity(actual, ref)
    let maxAbs = zip(actual, ref).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
    let ok = (p.isInfinite || p >= psnrFloor) && c >= cosineFloor
    let psnrStr = p.isInfinite ? "INF" : String(format: "%.2f", p)
    return ParityRow(
        name: name,
        status: "PSNR=\(psnrStr) dB  cosine=\(String(format: "%.6f", c))  "
            + "maxAbs=\(String(format: "%.3g", maxAbs))",
        ok: ok)
}

private func psnr(_ a: [Float], _ b: [Float]) -> Double {
    var mse = 0.0
    var peak = 0.0
    for i in a.indices {
        let d = Double(a[i]) - Double(b[i])
        mse += d * d
        peak = max(peak, abs(Double(b[i])))
    }
    mse /= Double(a.count)
    if mse == 0 { return .infinity }
    if peak == 0 { peak = 1 }
    return 10 * log10(peak * peak / mse)
}

private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in a.indices {
        dot += Double(a[i]) * Double(b[i])
        na += Double(a[i]) * Double(a[i])
        nb += Double(b[i]) * Double(b[i])
    }
    guard na > 0 && nb > 0 else { return na == nb ? 1 : 0 }
    return dot / (na.squareRoot() * nb.squareRoot())
}

// MARK: - .npy reader

/// Minimal `.npy` reader for the dtypes these traces use.
///
/// The image-segmenter and diffusion-runner tools each carry their own copy of this;
/// a third is the consistent choice until one is promoted into a shared module.
private struct NpyArray {
    enum DType { case float16, float32, int32, int64 }
    let shape: [Int]
    let dtype: DType
    let data: Data

    static func load(_ url: URL) throws -> NpyArray {
        let raw = try Data(contentsOf: url)
        guard raw.count > 10, raw[0] == 0x93, raw[1] == 0x4E else {
            throw ValidationError("not a .npy file: \(url.path)")
        }
        let headerLen: Int
        let headerStart: Int
        if raw[6] == 1 {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
            headerStart = 10
        } else {
            headerLen =
                Int(raw[8]) | (Int(raw[9]) << 8) | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let header = String(data: raw[headerStart..<dataStart], encoding: .ascii) ?? ""

        // Reject column-major files rather than reading them as C-order. numpy writes
        // `fortran_order: True` for any transposed view (a `permute` in the trace
        // script is enough), and interpreting that as row-major yields plausible-looking
        // garbage — near-zero cosine on data that is actually correct. Generators should
        // pass the array through `np.ascontiguousarray` first.
        if header.contains("'fortran_order': True") {
            throw ValidationError(
                "\(url.lastPathComponent) is Fortran-ordered; re-save it C-contiguous "
                    + "(np.ascontiguousarray) — this reader only handles row-major data")
        }

        let dtype: DType
        if header.contains("f2") {
            dtype = .float16
        } else if header.contains("f4") {
            dtype = .float32
        } else if header.contains("i4") {
            dtype = .int32
        } else if header.contains("i8") {
            dtype = .int64
        } else {
            throw ValidationError("unsupported .npy dtype in \(url.lastPathComponent)")
        }
        return NpyArray(
            shape: Self.parseShape(from: header),
            dtype: dtype,
            data: raw.subdata(in: dataStart..<raw.count))
    }

    private static func parseShape(from header: String) -> [Int] {
        guard let start = header.range(of: "("),
            let end = header.range(of: ")", range: start.upperBound..<header.endIndex)
        else { return [] }
        return header[start.upperBound..<end.lowerBound]
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    var count: Int { shape.reduce(1, *) }

    func asFloat() throws -> [Float] {
        let n = count
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { ptr in
            switch dtype {
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                let src = ptr.bindMemory(to: Float16.self)
                for i in 0..<n { out[i] = Float(src[i]) }
                #else
                fatalError("Float16 is not supported on this platform")
                #endif
            case .float32:
                let src = ptr.bindMemory(to: Float.self)
                for i in 0..<n { out[i] = src[i] }
            case .int32:
                let src = ptr.bindMemory(to: Int32.self)
                for i in 0..<n { out[i] = Float(src[i]) }
            case .int64:
                let src = ptr.bindMemory(to: Int64.self)
                for i in 0..<n { out[i] = Float(src[i]) }
            }
        }
        return out
    }

    func asInt32() throws -> [Int32] {
        let n = count
        var out = [Int32](repeating: 0, count: n)
        data.withUnsafeBytes { ptr in
            switch dtype {
            case .int32:
                let src = ptr.bindMemory(to: Int32.self)
                for i in 0..<n { out[i] = src[i] }
            case .int64:
                let src = ptr.bindMemory(to: Int64.self)
                for i in 0..<n { out[i] = Int32(truncatingIfNeeded: src[i]) }
            case .float32:
                let src = ptr.bindMemory(to: Float.self)
                for i in 0..<n { out[i] = Int32(src[i]) }
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                let src = ptr.bindMemory(to: Float16.self)
                for i in 0..<n { out[i] = Int32(src[i]) }
                #else
                fatalError("Float16 is not supported on this platform")
                #endif
            }
        }
        return out
    }
}
