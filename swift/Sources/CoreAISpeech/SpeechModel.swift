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
/// Loads a CoreAISpeech bundle (encoder.aimodel + decoder.aimodel) and transcribes
/// audio files. The decoder architecture is pluggable via ``SpeechDecoder``
public actor SpeechModel {
    private let bundle: SpeechBundle
    private let decoder: any SpeechDecoder
    private let melConfig: MelConfig

    // Encoder function and descriptor, cached after first load
    private var encFn: InferenceFunction?
    private var encOutShape: [Int]?

    /// Load a model from a bundle directory.
    ///
    /// - Parameters:
    ///   - url: Directory containing encoder.aimodel and decoder.aimodel.
    ///   - decoder: Decode strategy. Defaults to ``WhisperDecoder``.
    ///   - melConfig: Mel spectrogram parameters. Defaults to ``MelConfig/whisper``.
    public init(
        resourcesAt url: URL,
        decoder: any SpeechDecoder = WhisperDecoder(),
        melConfig: MelConfig = .whisper
    ) async throws {
        self.bundle = try await SpeechBundle(at: url)
        self.decoder = decoder
        self.melConfig = melConfig
        try await warmUp()
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

    private func warmUp() async throws {
        // Run the encoder once with silence to trigger JIT compilation
        guard let fn = try bundle.encoder.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in encoder")
        }
        encFn = fn
        let encDesc = bundle.encoder.functionDescriptor(for: "main")!
        guard case .ndArray(let encOutNDDesc) = encDesc.outputDescriptor(of: "encoder_hidden_states")
        else { throw SpeechError.missingModel("Unexpected encoder output descriptor") }
        encOutShape = encOutNDDesc.shape

        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input descriptor") }

        var silence = NDArray(
            descriptor: melNDDesc.resolvingDynamicDimensions([1, melConfig.nMelBins, melConfig.nFrames]))
        fillNDArray(&silence, as: Float.self, count: melConfig.nMelBins * melConfig.nFrames) { _ in 0.0 }
        var encOut = NDArray(descriptor: encOutNDDesc.resolvingDynamicDimensions(encOutNDDesc.shape))
        var out = InferenceFunction.MutableViews()
        out.insert(&encOut, for: "encoder_hidden_states")
        _ = try await fn.run(
            inputs: ["input_features": silence],
            states: InferenceFunction.MutableViews(), outputViews: consume out)
    }

    private func runEncoder(_ melArray: inout NDArray) async throws -> NDArray {
        guard let fn = encFn, let shape = encOutShape else {
            throw SpeechError.missingModel("Encoder not initialised")
        }
        let encDesc = bundle.encoder.functionDescriptor(for: "main")!
        guard case .ndArray(let encOutNDDesc) = encDesc.outputDescriptor(of: "encoder_hidden_states")
        else { throw SpeechError.missingModel("Unexpected encoder output") }
        var encOut = NDArray(descriptor: encOutNDDesc.resolvingDynamicDimensions(shape))
        var out = InferenceFunction.MutableViews()
        out.insert(&encOut, for: "encoder_hidden_states")
        _ = try await fn.run(
            inputs: ["input_features": melArray],
            states: InferenceFunction.MutableViews(), outputViews: consume out)
        return encOut
    }

    private func decodeAudio(from url: URL) async throws -> [Int32] {
        let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: melConfig.sampleRate)
        return try await decodeAudio(pcm: pcm)
    }

    private func decodeAudio(pcm: [Float]) async throws -> [Int32] {
        let encDesc = bundle.encoder.functionDescriptor(for: "main")!
        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input") }

        let floats = MelSpectrogram.fromPCM(pcm, config: melConfig)
        var melArray = NDArray(
            descriptor: melNDDesc.resolvingDynamicDimensions(
                [1, melConfig.nMelBins, melConfig.nFrames]))
        fillNDArray(&melArray, as: Float.self, with: floats)

        let encoderOutput = try await runEncoder(&melArray)
        let shape = encOutShape ?? [1, 1500, 1280]
        return try await decoder.decode(
            encoderOutput: encoderOutput,
            encoderOutputShape: shape,
            decoderModel: bundle.decoder,
            config: bundle.generationConfig)
    }

    private func detokenize(_ tokens: [Int32]) throws -> String {
        guard let tokenizer = bundle.tokenizer else { throw SpeechError.missingTokenizer }
        let ids = tokens.filter { $0 < bundle.generationConfig.eotToken }.map { Int($0) }
        return tokenizer.decode(tokens: ids).trimmingCharacters(in: .whitespaces)
    }
}
