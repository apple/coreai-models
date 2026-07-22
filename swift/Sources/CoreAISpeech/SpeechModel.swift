// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - SpeechModel

/// On-device speech recognition model.
///
/// Loads a CoreAISpeech bundle and transcribes audio. Supports both Whisper-style
/// encoder-decoder bundles and Parakeet TDT bundles; the architecture is
/// auto-detected from the bundle's metadata.json (or the legacy
/// encoder/decoder filename convention for Whisper).
public actor SpeechModel {
    private let bundle: SpeechBundle
    private let decoder: any SpeechDecoder
    private let melConfig: MelConfig
    private let resources: DecoderResources

    public init(resourcesAt url: URL) async throws {
        self.bundle = try await SpeechBundle(at: url)
        switch bundle.kind {
        case .whisper(let assets):
            self.decoder = WhisperDecoder()
            self.melConfig = assets.melConfig
            self.resources = .whisper(decoder: assets.decoder, generationConfig: assets.generationConfig)
        case .parakeetTDT(let assets):
            self.decoder = ParakeetTDTDecoder()
            self.melConfig = assets.melConfig
            self.resources = .parakeetTDT(
                decoderStep: assets.decoderStep, joint: assets.joint, config: assets.config)
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

    /// Warm the encoder and decoder for a specific PCM sample count so MPSGraph
    /// compiles and caches graphs for that input shape before real audio arrives.
    public func prewarm(sampleCount: Int) async throws {
        _ = try await decodeAudio(pcm: [Float](repeating: 0, count: sampleCount))
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

    // MARK: - Internals

    private var encoder: AIModel {
        switch bundle.kind {
        case .whisper(let a): return a.encoder
        case .parakeetTDT(let a): return a.encoder
        }
    }

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
        guard let fn = try encoder.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in encoder")
        }
        let encDesc = encoder.functionDescriptor(for: "main")!
        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input descriptor") }
        let start = SuspendingClock().now
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        let inputShape = encoderInputShape(nFrames: nFrames)
        var melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions(inputShape))
        fillNDArray(&melArray, as: Float.self, with: mel)

        // Attention mask (B, T_audio): true for real-audio frames, false for the
        // static window's zero-padding tail (and the dynamic path's trailing zero
        // frame). Lets the encoder exclude padding from self-attention and the conv
        // modules, matching HF. Guarded so bundles exported without the input still run.
        var inputs: [String: NDArray] = ["input_features": melArray]
        if encDesc.inputNames.contains("attention_mask"),
            case .ndArray(let maskDesc) = encDesc.inputDescriptor(of: "attention_mask")
        {
            let validFrames = min(
                MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: melConfig), nFrames)
            var maskArray = NDArray(descriptor: maskDesc.resolvingDynamicDimensions([1, nFrames]))
            fillNDArray(&maskArray, as: Bool.self, count: nFrames) { $0 < validFrames }
            inputs["attention_mask"] = maskArray
        }
        let preprocessDuration = SuspendingClock().now - start
        CLILogger.log("The preprocessing took \(preprocessDuration)", level: 2)
        let startEncode = SuspendingClock().now
        var outputs = try await fn.run(inputs: inputs)
        let encodeDuration = SuspendingClock().now - startEncode
        CLILogger.log("The encoding took \(encodeDuration)", level: 2)
        guard let encOut = outputs.remove("encoder_hidden_states")?.ndArray else {
            throw SpeechError.missingModel("Encoder did not produce 'encoder_hidden_states'")
        }
        return (encOut, encOut.shape)
    }

    private func encoderInputShape(nFrames: Int) -> [Int] {
        switch melConfig.layout {
        case .channelMajor: return [1, melConfig.nMelBins, nFrames]
        case .timeMajor: return [1, nFrames, melConfig.nMelBins]
        }
    }

    private func decodeAudio(from url: URL) async throws -> ([Int32], DecodeStats) {
        let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: melConfig.sampleRate)
        return try await decodeAudio(pcm: pcm)
    }

    private func decodeAudio(pcm: [Float]) async throws -> ([Int32], DecodeStats) {
        let (encOut, encShape) = try await runEncoder(pcm: pcm)
        let tEnc = encShape[1]
        // Exclude a static window's zero-padded tail: decode only the encoder frames
        // that carry real audio. Estimated proportionally from the mel valid/total
        // ratio and the encoder's actual output length (robust to conv edge effects).
        // Round to nearest so the boundary frame is kept only when it's majority real
        // audio — this drops the mostly-padding tail frame (a spurious trailing period)
        // while keeping the final token. Dynamic exports have no padding (validEnc == tEnc).
        let validEnc: Int
        if let total = melConfig.nFrames {
            let validMel = MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: melConfig)
            validEnc = min(tEnc, max(1, Int((Double(validMel) / Double(total) * Double(tEnc)).rounded())))
        } else {
            validEnc = tEnc
        }
        return try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEnc,
            resources: resources)
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
