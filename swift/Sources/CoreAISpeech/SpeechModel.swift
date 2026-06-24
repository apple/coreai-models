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

    // MARK: - Transcription

    /// Transcribe an audio file, returning the full text.
    public func transcribe(audioURL: URL) async throws -> String {
        let tokens = try await decodeAudio(from: audioURL)
        return try detokenize(tokens)
    }

    /// Transcribe raw 16 kHz mono PCM samples.
    public func transcribe(pcm: [Float]) async throws -> String {
        let tokens = try await decodeAudio(pcm: pcm)
        return try detokenize(tokens)
    }

    // MARK: - Internals

    private var encoder: AIModel {
        switch bundle.kind {
        case .whisper(let a): return a.encoder
        case .parakeetTDT(let a): return a.encoder
        }
    }

    private func warmUp() async throws {
        let nSamples: Int
        switch bundle.kind {
        case .whisper:
            nSamples = (melConfig.nFrames ?? 3_000) * melConfig.hopLength
        case .parakeetTDT:
            // 5 s of silence — matches the static export's traced shape and is
            // representative for dynamic exports too.
            nSamples = Int(melConfig.sampleRate) * 5
        }
        _ = try await runEncoder(pcm: [Float](repeating: 0, count: nSamples))
    }

    /// Run the encoder over PCM and return the encoder hidden states + concrete shape.
    private func runEncoder(pcm: [Float]) async throws -> (NDArray, [Int]) {
        guard let fn = try encoder.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in encoder")
        }
        let encDesc = encoder.functionDescriptor(for: "main")!
        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input descriptor") }

        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        let inputShape = encoderInputShape(nFrames: nFrames)
        var melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions(inputShape))
        fillNDArray(&melArray, as: Float.self, with: mel)

        var outputs = try await fn.run(inputs: ["input_features": melArray])
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

    private func decodeAudio(from url: URL) async throws -> [Int32] {
        let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: melConfig.sampleRate)
        return try await decodeAudio(pcm: pcm)
    }

    private func decodeAudio(pcm: [Float]) async throws -> [Int32] {
        let (encOut, encShape) = try await runEncoder(pcm: pcm)
        return try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
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
