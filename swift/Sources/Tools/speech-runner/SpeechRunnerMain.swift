// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAIShared
import CoreAISpeech
import Foundation
import Tokenizers

// MARK: - Entry point

@main
struct SpeechRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech-runner",
        abstract: "Transcribe audio using a CoreAI speech model bundle"
    )

    @Argument(help: "Bundle dir (encoder.aimodel + decoder.aimodel) or single .aimodel (legacy)")
    var modelPath: String

    @Argument(help: "Audio file (wav, flac, m4a, …). Omit for latency benchmarking with silence.")
    var audioPath: String?

    func run() async throws {
        let bundleURL = URL(fileURLWithPath: modelPath)
        if FileManager.default.fileExists(atPath: bundleURL.appending(path: "encoder.aimodel").path) {
            try await runBundle(bundleURL: bundleURL, audioPath: audioPath)
        } else {
            try await runLegacy(modelPath: modelPath, audioPath: audioPath)
        }
    }
}

// MARK: - Split bundle via CoreAISpeech

func runBundle(bundleURL: URL, audioPath: String?) async throws {
    print("Format: split (encoder + decoder, KV cache)")
    let model = try await SpeechModel(resourcesAt: bundleURL)

    if let path = audioPath {
        let url = URL(fileURLWithPath: path)
        print("Transcribing \(url.lastPathComponent)…")
        let t0 = ContinuousClock.now
        let text = try await model.transcribe(audioURL: url)
        let ms = (ContinuousClock.now - t0).inMilliseconds
        print(String(format: "  %.1f ms total", ms))
        print("\n── Transcription ──────────────────────────────────────────────────────")
        print("  \(text)")
    } else {
        print("No audio — silence benchmark")
        let pcm = [Float](repeating: 0, count: 480_000)
        let t0 = ContinuousClock.now
        _ = try await model.transcribe(pcm: pcm)
        print(String(format: "  %.1f ms (silence)", (ContinuousClock.now - t0).inMilliseconds))
    }
}

// MARK: - Legacy monolithic model

func runLegacy(modelPath: String, audioPath: String?) async throws {
    print("Format: legacy (monolithic, no KV cache)")

    let model = try await AIModel(contentsOf: URL(fileURLWithPath: modelPath))
    guard let fn = try model.loadFunction(named: "main")
    else { throw RuntimeError("No 'main' function in model") }
    let desc = model.functionDescriptor(for: "main")!

    guard case .ndArray(let melNDDesc) = desc.inputDescriptor(of: "input_features"),
        case .ndArray(let idsNDDesc) = desc.inputDescriptor(of: "decoder_input_ids"),
        case .ndArray(let logitsDesc) = desc.outputDescriptor(of: "logits")
    else { throw RuntimeError("Unexpected model descriptors") }

    let vocabSize = logitsDesc.shape.last!
    let isStaticIds = !idsNDDesc.shape.contains(where: { $0 < 0 })
    if isStaticIds {
        print("  ⚠️  decoder_input_ids has static shape — no past context per step")
    }

    var melArray: NDArray
    if let path = audioPath {
        let pcm = try MelSpectrogram.loadAndResample(
            URL(fileURLWithPath: path), targetSampleRate: 16_000)
        let floats = MelSpectrogram.fromPCM(pcm)
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillNDArray(&melArray, as: Float.self, with: floats)
    } else {
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillNDArray(&melArray, as: Float.self, count: 128 * 3000) { _ in 0.0 }
    }

    // Warmup
    do {
        var ids = NDArray(descriptor: idsNDDesc.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&ids, as: Int32.self, with: [50258])
        var lw = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, 1, vocabSize]))
        var out = InferenceFunction.MutableViews()
        out.insert(&lw, for: "logits")
        _ = try await fn.run(
            inputs: ["input_features": melArray, "decoder_input_ids": ids],
            states: InferenceFunction.MutableViews(), outputViews: consume out)
    }

    let config = GenerationConfig.whisper
    var tokens: [Int32] = config.forcedPrefix
    var stepTimesMs: [Double] = []

    print("\n── Decode ─────────────────────────────────────────────────────────────")

    while stepTimesMs.count < config.maxDecodeSteps {
        let inputTokens: [Int32] = isStaticIds ? [tokens.last!] : tokens
        let seqLen = inputTokens.count
        var ids = NDArray(descriptor: idsNDDesc.resolvingDynamicDimensions([1, seqLen]))
        fillNDArray(&ids, as: Int32.self, with: inputTokens)
        var la = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, seqLen, vocabSize]))
        var out = InferenceFunction.MutableViews()
        out.insert(&la, for: "logits")
        let t0 = ContinuousClock.now
        _ = try await fn.run(
            inputs: ["input_features": melArray, "decoder_input_ids": ids],
            states: InferenceFunction.MutableViews(), outputViews: consume out)
        stepTimesMs.append((ContinuousClock.now - t0).inMilliseconds)
        let logits = flattenAsFloat(la)
        let base = (seqLen - 1) * vocabSize
        let next = Int32(
            (0..<vocabSize).max(by: { logits[base + $0] < logits[base + $1] })!)
        tokens.append(next)
        if next == config.eotToken { break }
    }

    let avgMs = stepTimesMs.reduce(0, +) / Double(stepTimesMs.count)
    print(
        String(
            format: "  steps: %d  latency: %.1f ms/tok  speed: %.1f tok/s",
            stepTimesMs.count, avgMs, 1000 / avgMs))

    print("\n── Transcription ──────────────────────────────────────────────────────")
    let cacheBase = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: ".cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots")
    if let snap = try? FileManager.default.contentsOfDirectory(atPath: cacheBase.path).first,
        let tok = try? await AutoTokenizer.from(modelFolder: cacheBase.appending(path: snap))
    {
        let ids = tokens.filter { $0 < config.eotToken }.map { Int($0) }
        print("  \(tok.decode(tokens: ids).trimmingCharacters(in: .whitespaces))")
    } else {
        print("  token ids: \(tokens)")
    }
}

// MARK: - Helpers

struct RuntimeError: Error, CustomStringConvertible {
    let description: String
    init(_ msg: String) { description = msg }
}

extension Duration {
    var inMilliseconds: Double {
        Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
    }
}
