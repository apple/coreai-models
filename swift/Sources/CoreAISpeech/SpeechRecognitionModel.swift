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
    package let bundle: SpeechRecognitionBundle
    package let decoder: any SpeechDecoder
    package let melConfig: MelConfig
    private let resources: DecoderResources
    /// Non-nil for Parakeet bundles. Streaming geometry and the exact
    /// `ceil(validMel / subsampling)` frame rule are both derived from it.
    package let tdtConfig: ParakeetTDTConfig?
    /// Streaming session state, non-nil between `startStream` and `finishStream`.
    package var streaming: StreamingSessionState?
    /// Window geometry recorded by a `--streaming` export. Authoritative when present.
    package let bundleStreamingConfig: StreamingConfig?

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
            self.tdtConfig = nil
            self.bundleStreamingConfig = nil
            self.melConfig = assets.melConfig
            self.resources = .whisper(decoder: assets.decoder, generationConfig: assets.generationConfig)
            encoder = assets.encoder
        case .parakeetTDT(let assets):
            self.decoder = try ParakeetTDTDecoder(decoderStep: assets.decoderStep, joint: assets.joint)
            self.tdtConfig = assets.config
            self.bundleStreamingConfig = assets.streamingConfig
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
        let (encOut, encShape, _) = try await runEncoder(
            pcm: [Float](repeating: 0, count: sampleCount))
        _ = try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: 1,
            resources: resources)
    }

    // MARK: - Transcription

    /// Transcribe an audio file, returning the full text and decode stats.
    ///
    /// - Parameter resetAfterSilenceFrames: See `ParakeetTDTDecoder.decode`. 0 keeps this path
    ///   reference-exact.
    public func transcribe(
        audioURL: URL, resetAfterSilenceFrames: Int = 0
    ) async throws -> (String, DecodeStats) {
        let (tokens, stats) = try await decodeAudio(
            from: audioURL, resetAfterSilenceFrames: resetAfterSilenceFrames)
        return try (detokenize(tokens), stats)
    }

    /// Transcribe raw 16 kHz mono PCM samples, returning the full text and decode stats.
    ///
    /// - Parameter resetAfterSilenceFrames: See `transcribe(audioURL:resetAfterSilenceFrames:)`.
    public func transcribe(
        pcm: [Float], resetAfterSilenceFrames: Int = 0
    ) async throws -> (String, DecodeStats) {
        let (tokens, stats) = try await decodeAudio(
            pcm: pcm, resetAfterSilenceFrames: resetAfterSilenceFrames)
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
        let (encOut, encShape, validEnc) = try await runEncoder(pcm: pcm)
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
    ///
    /// Also returns the exact number of leading encoder frames that carry real audio.
    ///
    /// `fromPCM` pads up to the traced window itself and `padToFrameGrid` derives the
    /// valid-frame count from `pcm.count`, so what a caller passes sets how far the mask and
    /// the per-bin normalization statistics reach. Offline passes real samples only. The
    /// streaming path deliberately zero-fills its ramp-up windows first, so that a frame's
    /// representation does not shift as the window fills — see `runHopIfReady`.
    package func runEncoder(pcm: [Float]) async throws -> (NDArray, [Int], Int) {
        let start = ContinuousClock.now
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig, basis: melBasis)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        let inputShape = encoderInputShape(nFrames: nFrames)
        var melArray = NDArray(descriptor: melInputDescriptor.resolvingDynamicDimensions(inputShape))
        fillFloatNDArray(&melArray, with: mel)

        let validMelFrames = min(
            MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: melConfig), nFrames)

        // Attention mask (B, T_audio): true for real-audio frames, false for the
        // static window's zero-padding tail (and the dynamic path's trailing zero
        // frame). Lets the encoder exclude padding from self-attention and the conv
        // modules, matching HF. Guarded so bundles exported without the input still run.
        var inputs: [String: NDArray] = ["input_features": melArray]
        if let maskDesc = maskInputDescriptor {
            var maskArray = NDArray(descriptor: maskDesc.resolvingDynamicDimensions([1, nFrames]))
            fillNDArray(&maskArray, as: Bool.self, count: nFrames) { $0 < validMelFrames }
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
        // How many leading frames real audio backs. Both the offline decode cap and the
        // streaming path's frame slicing come from this one value.
        let validEnc = Self.validEncoderFrames(
            pcmCount: pcm.count, tEnc: encOut.shape[1], config: melConfig,
            subsamplingFactor: tdtConfig?.encoderSubsamplingFactor ?? 1)
        return (encOut, encOut.shape, validEnc)
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

    private func decodeAudio(
        from url: URL, resetAfterSilenceFrames: Int = 0
    ) async throws -> ([Int32], DecodeStats) {
        let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: melConfig.sampleRate)
        return try await decodeAudio(pcm: pcm, resetAfterSilenceFrames: resetAfterSilenceFrames)
    }

    private func decodeAudio(
        pcm: [Float], resetAfterSilenceFrames: Int = 0
    ) async throws -> ([Int32], DecodeStats) {
        let (encOut, encShape, validEnc) = try await runEncoder(pcm: pcm)
        // Not on `SpeechDecoder`, so reached by downcast the way `startStream` does. Thrown
        // rather than ignored elsewhere: silently dropping a flag the caller passed is how a
        // missed effect reads as a clean run.
        if resetAfterSilenceFrames > 0 {
            guard let parakeet = decoder as? ParakeetTDTDecoder else {
                throw SpeechError.incompatibleResources(
                    "resetAfterSilenceFrames applies to a transducer's predictor state; "
                        + "\(architecture) has none")
            }
            return try await parakeet.decode(
                encoderOutput: encOut,
                encoderOutputShape: encShape,
                validEncoderFrames: validEnc,
                resources: resources,
                resetAfterSilenceFrames: resetAfterSilenceFrames)
        }
        return try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEnc,
            resources: resources)
    }

    /// How many leading encoder frames carry real audio.
    ///
    /// Exact, not proportional. The FastConformer front end is a stack of stride-2 convs,
    /// each mapping `T` to `(T - 1) / 2 + 1`, so applying that per stage to the *valid* mel
    /// count yields the frame count backed by real audio — no ratio, no rounding. A
    /// proportional estimate lands one frame low for ~25% of audio lengths against the
    /// shipped geometry, which clips a trailing token.
    ///
    /// Sub-hop audio yields 0 here; `ParakeetTDTDecoder.decode` floors its own cap at one
    /// frame, so a tiny clip still decodes something.
    ///
    /// `static` and parameterized so it can be unit-tested — the actor cannot be constructed
    /// without model assets. `runEncoder` is the only caller, so offline and streaming share
    /// one definition rather than each deriving their own.
    package static func validEncoderFrames(
        pcmCount: Int, tEnc: Int, config: MelConfig, subsamplingFactor: Int
    ) -> Int {
        let frameCount = MelSpectrogram.frameCount(forPCMLength: pcmCount, config: config)
        let validMel = min(
            MelSpectrogram.validFrameCount(forPCMLength: pcmCount, config: config), frameCount)
        return min(
            tEnc, encoderFrameCount(melFrames: validMel, subsamplingFactor: subsamplingFactor))
    }

    package func detokenize(_ tokens: [Int32]) throws -> String {
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
