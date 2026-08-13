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
    var model: String?

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

    @Option(
        help:
            "Write the computed mel features to this path as raw little-endian f32, plus a <path>.json sidecar with the shape. For front-end parity checks."
    )
    var dumpMel: String?

    @Option(
        name: .customLong("parity-test"),
        help:
            "Path to a directory of PyTorch reference traces (audio.wav, meta.json, ref_*.npy). Compares the runtime's own mel / encoder / tokens / transcript against them and exits non-zero on failure."
    )
    var parityTest: String?

    @Option(help: "Minimum PSNR (dB) per tensor in --parity-test mode.")
    var psnrFloor: Double = 29.5

    @Option(
        help:
            "Minimum PSNR (dB) for the mel front-end in --parity-test mode. Tighter than --psnr-floor: the front-end is deterministic CPU arithmetic, so it agrees with the reference to ~100 dB."
    )
    var melPsnrFloor: Double = 80.0

    @Option(
        help:
            "Minimum PSNR (dB) for the mel front-end in --parity-test mode when the input had to be resampled. The runtime resamples with AVAudioConverter and the reference with soxr_hq; their anti-aliasing filters differ in the transition band, so exact parity is not available. 30 dB separates AVAudioConverter and other correct resamplers (scipy resample_poly, soxr_vhq) from degraded ones (soxr_lq and below)."
    )
    var resampledMelPsnrFloor: Double = 30.0

    @Option(
        help:
            "Minimum PSNR (dB) for the encoder in --parity-test mode when the input had to be resampled. Lower than the mel floor: the encoder compounds the front-end difference through the conformer stack."
    )
    var resampledEncoderPsnrFloor: Double = 25.0

    @Option(
        help:
            "Minimum cosine similarity per tensor in --parity-test mode when the input had to be resampled."
    )
    var resampledCosineFloor: Double = 0.95

    @Option(help: "Minimum cosine similarity per tensor in --parity-test mode. Not applied to resampled traces.")
    var cosineFloor: Double = 0.999

    func validate() throws {
        if parityTest == nil && model == nil {
            throw ValidationError("--model is required (unless --parity-test is set).")
        }
    }

    func run() async throws {
        if let parityTest {
            try await SpeechParityTest(
                directory: URL(fileURLWithPath: parityTest),
                modelPath: model,
                psnrFloor: psnrFloor,
                melPsnrFloor: melPsnrFloor,
                resampledMelPsnrFloor: resampledMelPsnrFloor,
                resampledEncoderPsnrFloor: resampledEncoderPsnrFloor,
                resampledCosineFloor: resampledCosineFloor,
                cosineFloor: cosineFloor
            ).run()
            return
        }
        guard let model else {
            throw ValidationError("--model is required.")
        }
        let bundleURL = URL(fileURLWithPath: model)
        if clearCoreAICache {
            try clearCache(bundleURL: bundleURL)
        }
        // A single asset is identified by its extension; a bundle by carrying
        // metadata.json. Both contain a metadata.json, so the extension has to be
        // checked first or `.aimodel` assets get misrouted to the bundle path.
        let assetExtensions: Set<String> = ["aimodel", "aimodelc"]
        let isBundle =
            !assetExtensions.contains(bundleURL.pathExtension)
            && FileManager.default.fileExists(atPath: bundleURL.appending(path: "metadata.json").path)
        if isBundle {
            try await runBundle(
                bundleURL: bundleURL, audioPath: audioPath, warmup: warmup, verbose: verbose,
                dumpMel: dumpMel)
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

func runBundle(
    bundleURL: URL, audioPath: String?, warmup: Bool, verbose: Bool, dumpMel: String? = nil
) async throws {
    // The architecture is printed after loading, once the bundle has reported it.
    // Detect an existing cached specialization before loading so we can annotate the load time
    // below. Only inspects the cache; never specializes. Components are discovered by scanning
    // rather than by hardcoded filenames — a Parakeet bundle prefixes each asset with the
    // variant name — via the same helper `--clear-coreai-cache` uses. `SpeechRecognitionBundle`
    // loads each asset with `AIModel(contentsOf:)`, which uses `.default` options: match that,
    // and require every component.
    let assetURLs = (try? PreparedModel.modelAssetURLs(at: bundleURL)) ?? []
    let cacheHit =
        !assetURLs.isEmpty
        && assetURLs.allSatisfy { PreparedModel.isCached(at: $0, options: .default) }

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

    if let dumpMel {
        let (values, shape) = await model.melFeatures(pcm: pcm)
        let url = URL(fileURLWithPath: dumpMel)
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
        try JSONSerialization
            .data(withJSONObject: ["shape": shape], options: [.prettyPrinted])
            .write(to: URL(fileURLWithPath: dumpMel + ".json"))
        print("Wrote mel \(shape) (\(values.count) floats) to \(dumpMel)")
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
    // TODO: Pinned to CPU because the monolithic f16/f32 Whisper export decodes incorrectly on the
    // default compute path
    let options = SpecializationOptions(preferredComputeUnitKind: .cpu)
    // Detect an existing cached specialization before loading. Only inspects the cache; never
    // specializes. Probed with the same options the load uses: cache entries are keyed by them,
    // so probing `.default` against a non-default load always reports a miss.
    let cacheHit = PreparedModel.isCached(at: modelURL, options: options)

    print("⏳ Preparing AI asset...", terminator: "")
    fflush(stdout)
    let loadStart = ContinuousClock.now
    let model = try await AIModel(contentsOf: modelURL, options: options)
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
        fillFloatNDArray(&melArray, with: floats)
    } else {
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillFloatNDArray(&melArray, with: [Float](repeating: 0, count: 128 * 3000))
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
    if let snapshot = huggingFaceCacheSnapshot(forModelName: "openai/whisper-large-v3-turbo"),
        let tok = try? await AutoTokenizer.from(modelFolder: snapshot)
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
