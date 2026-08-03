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
struct SpeechRecognizer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech-recognizer",
        abstract: "Transcribe audio using a CoreAI speech model bundle"
    )

    @Option(help: "Bundle dir (metadata.json + .aimodel assets) or single .aimodel (legacy)")
    var model: String

    @Option(help: "Audio file (wav, flac, m4a, …). Omit for latency benchmarking with silence.")
    var audioPath: String?

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this model before loading (forces re-specialization)"
    )
    var clearCoreAICache: Bool = false

    @Flag(name: .long, help: "Run a full transcription pass (encode + decode) on silence before timing.")
    var warmup = false

    @Flag(name: .long, help: "Print verbose debug output")
    var verbose = false

    func run() async throws {
        let bundleURL = URL(fileURLWithPath: model)
        if clearCoreAICache {
            try clearCache(bundleURL: bundleURL)
        }
        if FileManager.default.fileExists(atPath: bundleURL.appending(path: "metadata.json").path) {
            try await runBundle(bundleURL: bundleURL, audioPath: audioPath, warmup: warmup, verbose: verbose)
        } else {
            try await runLegacy(model: model, audioPath: audioPath, warmup: warmup)
        }
    }

    /// Clear the specialization cache for this model's asset component(s).
    ///
    /// Delegates to `PreparedModel.clearCache`, which scans the bundle directory for every
    /// `.aimodel`/`.aimodelc` component (or treats `bundleURL` as a single asset), so it stays
    /// correct regardless of component filenames.
    private func clearCache(bundleURL: URL) throws {
        let cleared = try PreparedModel.clearCache(at: bundleURL)
        print("🗑️  Cleared specialization cache for \(bundleURL.lastPathComponent) (\(cleared.count) component(s))")
    }
}

// MARK: - Split bundle via CoreAISpeech

func runBundle(bundleURL: URL, audioPath: String?, warmup: Bool, verbose: Bool) async throws {
    print("Format: split (encoder + decoder, KV cache)")

    // Detect an existing cached specialization before loading so we can annotate the load time
    // below. Only inspects the cache; never specializes. `SpeechBundle` loads each asset via
    // `AIModel(contentsOf:)`, which uses `.default` options — match that, and require both.
    let encoderURL = bundleURL.appending(path: "encoder.aimodel")
    let decoderURL = bundleURL.appending(path: "decoder.aimodel")
    let cacheHit =
        PreparedModel.isCached(at: encoderURL, options: .default)
        && PreparedModel.isCached(at: decoderURL, options: .default)

    print("⏳ Preparing AI asset...", terminator: "")
    fflush(stdout)
    let loadStart = ContinuousClock.now
    let model = try await SpeechRecognitionModel(resourcesAt: bundleURL)
    let loadElapsed = ContinuousClock.now - loadStart
    print(" done in \(String(format: "%.3f", loadElapsed.inSeconds))s\(cacheHit ? " (cache hit)" : "")")
    print("Format: bundle (\(await model.architecture))")

    if verbose {
        CLILogger.level = 1
    }

    // Resolve the PCM buffer once — either the decoded audio file or a fixed
    // silence buffer for latency benchmarking — then share the warmup / transcribe
    // / stats path for both.
    let pcm: [Float]
    let audioURL: URL?
    if let path = audioPath {
        audioURL = URL(fileURLWithPath: path)
        pcm = try MelSpectrogram.loadAndResample(audioURL!, targetSampleRate: await model.sampleRate)
    } else {
        audioURL = nil
        print("No audio — silence benchmark")
        pcm = [Float](repeating: 0, count: 480_000)
    }

    if warmup {
        print("Warming up…")
        try await model.prewarm(sampleCount: pcm.count)
    }

    if let audioURL { print("Transcribing \(audioURL.lastPathComponent)…") }
    let t0 = ContinuousClock.now
    let (text, stats) = try await model.transcribe(pcm: pcm)
    let totalTranscribeTime = (ContinuousClock.now - t0)
    let totalMs = totalTranscribeTime.inMilliseconds
    print("\n── Decode ─────────────────────────────────────────────────────────────")
    print(
        String(
            format:
                "  steps: %d  latency: %.1f ms/step  speed: %.1f steps/s  min: %.1f ms  max: %.1f ms  [%.1f ms total]",
            stats.stepCount, stats.avgLatencyMs, stats.stepsPerSecond,
            stats.minLatencyMs, stats.maxLatencyMs, totalMs))
    // Silence benchmark has no meaningful transcript to print.
    if audioURL != nil {
        print("\n── Transcription ──────────────────────────────────────────────────────")
        print("  \(text)")
    }
}

// MARK: - Legacy monolithic model

func runLegacy(model: String, audioPath: String?, warmup: Bool) async throws {
    print("Format: legacy (monolithic, no KV cache)")

    let modelURL = URL(fileURLWithPath: model)
    // Detect an existing cached specialization before loading. Only inspects the cache; never
    // specializes. The legacy path loads via `AIModel(contentsOf:)`, which uses `.default` options.
    let cacheHit = PreparedModel.isCached(at: modelURL, options: .default)

    print("⏳ Preparing AI asset...", terminator: "")
    fflush(stdout)
    let loadStart = ContinuousClock.now
    let model = try await AIModel(contentsOf: modelURL)
    let loadElapsed = ContinuousClock.now - loadStart
    print(" done in \(String(format: "%.3f", loadElapsed.inSeconds))s\(cacheHit ? " (cache hit)" : "")")
    guard let fn = try model.loadFunction(named: "main")
    else { throw RuntimeError("No 'main' function in model") }
    guard let desc = model.functionDescriptor(for: "main")
    else { throw RuntimeError("No 'main' descriptor in model") }

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

    if warmup {
        print("Warming up…")
        var warmupTokens: [Int32] = config.forcedPrefix
        while warmupTokens.count - config.forcedPrefix.count < config.maxDecodeSteps {
            let inputTokens: [Int32] = isStaticIds ? [warmupTokens.last!] : warmupTokens
            let seqLen = inputTokens.count
            var ids = NDArray(descriptor: idsNDDesc.resolvingDynamicDimensions([1, seqLen]))
            fillNDArray(&ids, as: Int32.self, with: inputTokens)
            var la = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, seqLen, vocabSize]))
            var out = InferenceFunction.MutableViews()
            out.insert(&la, for: "logits")
            _ = try await fn.run(
                inputs: ["input_features": melArray, "decoder_input_ids": ids],
                states: InferenceFunction.MutableViews(), outputViews: consume out)
            let logits = flattenAsFloat(la)
            let base = (seqLen - 1) * vocabSize
            let next = Int32(
                (0..<vocabSize).max(by: { logits[base + $0] < logits[base + $1] })!)
            warmupTokens.append(next)
            if next == config.eotToken { break }
        }
    }

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
        let thisStepTime = ContinuousClock.now - t0
        stepTimesMs.append(thisStepTime.inMilliseconds)
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
            format: "  steps: %d  latency: %.1f ms/step  speed: %.1f steps/s",
            stepTimesMs.count, avgMs, 1000 / avgMs))

    print("\n── Transcription ──────────────────────────────────────────────────────")
    #if os(macOS)
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
    #else
    // iOS has no user HF cache to load the Whisper tokenizer from.
    print("  token ids: \(tokens)")
    #endif
}

// MARK: - Helpers

struct RuntimeError: Error, CustomStringConvertible {
    let description: String
    init(_ msg: String) { description = msg }
}
