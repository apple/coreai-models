// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - SpeechRecognitionModel

/// On-device speech recognition model.
///
/// Loads a CoreAISpeech bundle and transcribes audio. Supports both Whisper-style
/// encoder-decoder bundles and Parakeet TDT bundles; the architecture is
/// auto-detected from the bundle's metadata.json (or the legacy
/// encoder/decoder filename convention for Whisper).
public actor SpeechRecognitionModel {
    private let bundle: SpeechRecognitionBundle
    private let decoder: any SpeechDecoder
    private let melConfig: MelConfig
    private let resources: DecoderResources

    /// Encoder function and its input descriptors, resolved once at init. These are
    /// shape-independent, so caching them keeps `loadFunction`/descriptor lookups off
    /// the per-transcription path; only `resolvingDynamicDimensions` is per-call.
    private let encoderFunction: InferenceFunction
    private let melInputDescriptor: NDArrayDescriptor
    /// Non-nil only for bundles exported with the optional `attention_mask` input.
    private let maskInputDescriptor: NDArrayDescriptor?

    /// Mel window/DFT/filterbank, built once for `melConfig`. Derived purely from the
    /// config, so rebuilding it on every transcription is wasted work.
    private let melBasis: MelSpectrogram.Basis

    public init(resourcesAt url: URL) async throws {
        self.bundle = try await SpeechRecognitionBundle(at: url)
        let encoder: AIModel
        switch bundle.kind {
        case .whisper(let assets):
            self.decoder = WhisperDecoder()
            self.melConfig = assets.melConfig
            self.resources = .whisper(decoder: assets.decoder, generationConfig: assets.generationConfig)
            encoder = assets.encoder
        case .parakeetTDT(let assets):
            self.decoder = try ParakeetTDTDecoder(decoderStep: assets.decoderStep, joint: assets.joint)
            self.melConfig = assets.melConfig
            self.resources = .parakeetTDT(
                decoderStep: assets.decoderStep, joint: assets.joint, config: assets.config)
            encoder = assets.encoder
        }
        self.melBasis = MelSpectrogram.Basis(config: self.melConfig)

        guard let fn = try encoder.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in encoder")
        }
        guard let encDesc = encoder.functionDescriptor(for: "main") else {
            throw SpeechError.missingModel("No 'main' descriptor in encoder")
        }
        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input descriptor") }
        self.encoderFunction = fn
        self.melInputDescriptor = melNDDesc
        if encDesc.inputNames.contains("attention_mask"),
            case .ndArray(let maskDesc) = encDesc.inputDescriptor(of: "attention_mask")
        {
            self.maskInputDescriptor = maskDesc
        } else {
            self.maskInputDescriptor = nil
        }

        try await warmUp()
    }

    /// Human-readable architecture label (for logging).
    public var architecture: String {
        switch bundle.kind {
        case .whisper: return "Whisper"
        case .parakeetTDT: return "Parakeet TDT"
        }
    }

    public var sampleRate: Double { melConfig.sampleRate }

    /// The mel features this model would hand its encoder for `pcm`, with the shape
    /// the encoder expects them in.
    ///
    /// The transcription path computes these internally; this exposes them so a
    /// verification harness can compare the Swift front-end against a reference
    /// implementation without re-deriving the bundle's `MelConfig`.
    public func melFeatures(pcm: [Float]) -> (values: [Float], shape: [Int]) {
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig, basis: melBasis)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        return (mel, encoderInputShape(nFrames: nFrames))
    }

    /// Warm the encoder and decoder for a specific PCM sample count so MPSGraph
    /// compiles and caches graphs for that input shape before real audio arrives.
    ///
    /// The encoder runs at the full `sampleCount`, since its graph is compiled per input
    /// shape and that is the compilation worth paying for up front. The decoder graphs are
    /// shape-static, so the frame loop is capped at one frame rather than decoding the
    /// whole silence buffer — one pass compiles the same graphs.
    public func prewarm(sampleCount: Int) async throws {
        let (encOut, encShape) = try await runEncoder(
            pcm: [Float](repeating: 0, count: sampleCount))
        _ = try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: 1,
            resources: resources)
    }

    // MARK: - Transcription

    /// Transcribe an audio file, returning the full text and decode stats.
    public func transcribe(audioURL: URL) async throws -> (String, DecodeStats) {
        let (tokens, stats) = try await decodeAudio(from: audioURL)
        return try (detokenize(tokens), stats)
    }

    /// Transcribe raw 16 kHz mono PCM samples, returning the full text and decode stats.
    public func transcribe(pcm: [Float]) async throws -> (String, DecodeStats) {
        let (tokens, stats) = try await decodeAudio(pcm: pcm)
        return try (detokenize(tokens), stats)
    }

    // MARK: - Parity Testing

    /// Intermediates from one transcription, for stage-by-stage parity checking.
    ///
    /// `transcribe` returns only text, which is a weak signal: a front-end error large
    /// enough to shift every mel bin can still detokenize to the correct sentence. These
    /// are the values a reference implementation can actually be compared against.
    public struct StageCapture: Sendable {
        public let mel: [Float]
        public let melShape: [Int]
        public let encoderHiddenStates: [Float]
        public let encoderShape: [Int]
        /// Emitted tokens, before detokenization — blanks already filtered for TDT.
        public let tokens: [Int32]
        public let text: String
        /// Leading encoder frames treated as real audio (== `encoderShape[1]` when dynamic).
        public let validEncoderFrames: Int
        /// Decode-loop branch counts and first-step tensors for this transcription.
        public let stats: DecodeStats
    }

    /// Transcribe `pcm`, returning each stage's output alongside the text.
    ///
    /// Recomputes the mel separately from `runEncoder`'s internal copy so the encoder
    /// path stays untouched; that costs one extra front-end pass, which is irrelevant
    /// on a verification path.
    public func transcribeCapturingStages(pcm: [Float]) async throws -> StageCapture {
        let (melValues, melShape) = melFeatures(pcm: pcm)
        let (encOut, encShape) = try await runEncoder(pcm: pcm)
        let validEnc = validEncoderFrames(pcmCount: pcm.count, tEnc: encShape[1])
        // Copy the encoder output out *before* decoding. `decode` issues ~2 graph runs
        // per emitted token, and reading `encOut` afterwards would report whatever those
        // left in the buffer if the runtime recycles output storage.
        let encoderValues = flattenAsFloat(encOut)
        let (tokens, stats) = try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEnc,
            resources: resources)
        return StageCapture(
            mel: melValues, melShape: melShape,
            encoderHiddenStates: encoderValues, encoderShape: encShape,
            tokens: tokens, text: try detokenize(tokens), validEncoderFrames: validEnc,
            stats: stats)
    }

    // MARK: - Internals

    private func warmUp() async throws {
        switch bundle.kind {
        case .whisper:
            let nSamples = (melConfig.nFrames ?? 3_000) * melConfig.hopLength
            _ = try await runEncoder(pcm: [Float](repeating: 0, count: nSamples))
        case .parakeetTDT:
            // Static exports have nFrames set; warm up at exactly that size.
            // Dynamic exports skip init warmup — callers use prewarm(sampleCount:)
            // once the actual audio length is known, avoiding a wasted compilation.
            guard let nFrames = melConfig.nFrames else { return }
            _ = try await runEncoder(pcm: [Float](repeating: 0, count: nFrames * melConfig.hopLength))
        }
    }

    /// Run the encoder over PCM and return the encoder hidden states + concrete shape.
    private func runEncoder(pcm: [Float]) async throws -> (NDArray, [Int]) {
        let start = ContinuousClock.now
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig, basis: melBasis)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        let inputShape = encoderInputShape(nFrames: nFrames)
        var melArray = NDArray(descriptor: melInputDescriptor.resolvingDynamicDimensions(inputShape))
        fillFloatNDArray(&melArray, with: mel)

        // Attention mask (B, T_audio): true for real-audio frames, false for the
        // static window's zero-padding tail (and the dynamic path's trailing zero
        // frame). Lets the encoder exclude padding from self-attention and the conv
        // modules, matching HF. Guarded so bundles exported without the input still run.
        var inputs: [String: NDArray] = ["input_features": melArray]
        if let maskDesc = maskInputDescriptor {
            let validFrames = min(
                MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: melConfig), nFrames)
            var maskArray = NDArray(descriptor: maskDesc.resolvingDynamicDimensions([1, nFrames]))
            fillNDArray(&maskArray, as: Bool.self, count: nFrames) { $0 < validFrames }
            inputs["attention_mask"] = maskArray
        }
        let preprocessDuration = ContinuousClock.now - start
        CLILogger.log("The preprocessing took \(preprocessDuration.inMilliseconds) ms", level: 1)
        let startEncode = ContinuousClock.now
        var outputs = try await encoderFunction.run(inputs: inputs)
        let encodeDuration = ContinuousClock.now - startEncode
        CLILogger.log("The encoding took \(encodeDuration.inMilliseconds) ms", level: 1)
        guard let encOut = outputs.remove("encoder_hidden_states")?.ndArray else {
            throw SpeechError.missingModel("Encoder did not produce 'encoder_hidden_states'")
        }
        return (encOut, encOut.shape)
    }

    private func encoderInputShape(nFrames: Int) -> [Int] {
        Self.encoderInputShape(nFrames: nFrames, config: melConfig)
    }

    /// The encoder's `input_features` shape for `nFrames` mel frames, in the config's layout.
    ///
    /// `static` and `MelConfig`-parameterized so it can be unit-tested: the actor's only
    /// initializer loads a bundle and warms the graphs, so an instance method here would be
    /// unreachable without model assets on disk.
    package static func encoderInputShape(nFrames: Int, config: MelConfig) -> [Int] {
        switch config.layout {
        case .channelMajor: return [1, config.nMelBins, nFrames]
        case .timeMajor: return [1, nFrames, config.nMelBins]
        }
    }

    private func decodeAudio(from url: URL) async throws -> ([Int32], DecodeStats) {
        let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: melConfig.sampleRate)
        return try await decodeAudio(pcm: pcm)
    }

    private func decodeAudio(pcm: [Float]) async throws -> ([Int32], DecodeStats) {
        let (encOut, encShape) = try await runEncoder(pcm: pcm)
        return try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEncoderFrames(pcmCount: pcm.count, tEnc: encShape[1]),
            resources: resources)
    }

    /// How many leading encoder frames carry real audio.
    ///
    /// Excludes a static window's zero-padded tail: decode only the encoder frames
    /// that carry real audio. Estimated proportionally from the mel valid/total ratio
    /// and the encoder's actual output length (robust to conv edge effects). Rounds to
    /// nearest so the boundary frame is kept only when it's majority real audio — this
    /// drops the mostly-padding tail frame (a spurious trailing period) while keeping
    /// the final token. Dynamic exports have no padding, so this returns `tEnc`.
    ///
    /// `static` and `MelConfig`-parameterized so the rounding heuristic can be unit-tested;
    /// the actor cannot be constructed without model assets.
    package static func validEncoderFrames(pcmCount: Int, tEnc: Int, config: MelConfig) -> Int {
        guard let total = config.nFrames else { return tEnc }
        let validMel = MelSpectrogram.validFrameCount(forPCMLength: pcmCount, config: config)
        return min(tEnc, max(1, Int((Double(validMel) / Double(total) * Double(tEnc)).rounded())))
    }

    private func validEncoderFrames(pcmCount: Int, tEnc: Int) -> Int {
        Self.validEncoderFrames(pcmCount: pcmCount, tEnc: tEnc, config: melConfig)
    }

    private func detokenize(_ tokens: [Int32]) throws -> String {
        guard let tokenizer = bundle.tokenizer else { throw SpeechError.missingTokenizer }
        let ids: [Int]
        switch bundle.kind {
        case .whisper(let assets):
            ids = tokens.filter { $0 < assets.generationConfig.eotToken }.map { Int($0) }
        case .parakeetTDT:
            // Decoder already filters blanks; pass everything through.
            ids = tokens.map { Int($0) }
        }
        return tokenizer.decode(tokens: ids).trimmingCharacters(in: .whitespaces)
    }
}
