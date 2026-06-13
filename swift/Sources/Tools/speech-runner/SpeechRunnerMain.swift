// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// Whisper forced prefix: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
private let forcedPrefix: [Int32] = [50258, 50259, 50360, 50364]
private let eotToken: Int32 = 50257
private let maxTargetPositions = 448
private let maxDecodeSteps = 50
private let melElements = 128 * 3000

// MARK: - Entry point

// Usage: speech-runner <model-path> [audio-or-mel]
//
// model-path  A bundle dir with encoder.aimodel + decoder.aimodel (--mode coreai),
//             or a single .aimodel file (--mode legacy).
//
// audio-or-mel  An audio file (wav, flac, m4a, …) or a precomputed mel .bin
//               from tools/compute_mel.py. Omit for silence benchmarking.
@main
struct Main {
    static func main() async {
        guard CommandLine.arguments.count > 1 else {
            print("Usage: speech-runner <model-path> [audio-or-mel]")
            exit(1)
        }
        let modelPath = CommandLine.arguments[1]
        let audioPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : nil
        do {
            let encURL = URL(fileURLWithPath: "\(modelPath)/encoder.aimodel")
            if FileManager.default.fileExists(atPath: encURL.path) {
                try await runSplit(bundleDir: modelPath, audioPath: audioPath)
            } else {
                try await runLegacy(modelPath: modelPath, audioPath: audioPath)
            }
        } catch {
            print("Fatal: \(error)")
            exit(1)
        }
    }
}

// MARK: - Mel loading

private let audioExtensions: Set<String> = ["wav", "flac", "m4a", "mp3", "aiff", "aif", "caf"]

private func loadMelArray(from path: String, descriptor: NDArrayDescriptor) throws -> NDArray {
    let url = URL(fileURLWithPath: path)
    let floats: [Float]
    if audioExtensions.contains(url.pathExtension.lowercased()) {
        print("Computing mel from audio file…")
        floats = try WhisperMel.fromFile(url)
    } else {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        guard count == melElements else {
            fatalError("mel bin has \(count) floats, expected \(melElements) (128×3000)")
        }
        floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, 128, 3000]))
    fillNDArray(&array, as: Float.self, with: floats)
    return array
}

// MARK: - Results

private func printResults(tokens: [Int32], stepTimesMs: [Double]) async {
    let avgMs = stepTimesMs.reduce(0, +) / Double(stepTimesMs.count)
    print(String(format: "  steps:    %d", stepTimesMs.count))
    print(String(format: "  latency:  %.1f ms/tok", avgMs))
    print(String(format: "  speed:    %.1f tok/s", 1000.0 / avgMs))
    if let lo = stepTimesMs.min(), let hi = stepTimesMs.max() {
        print(String(format: "  min/max:  %.1f / %.1f ms", lo, hi))
    }
    print("\n── Transcription ──────────────────────────────────────────────────────")
    if let tokenizer = try? await AutoTokenizer.from(pretrained: "openai/whisper-large-v3-turbo") {
        let ids = tokens.filter { $0 < 50257 }.map { Int($0) }
        print("  \(tokenizer.decode(tokens: ids))")
    } else {
        print("  (tokenizer unavailable — token ids: \(tokens))")
    }
}

// MARK: - Split runner (encoder + decoder with KV cache)

func runSplit(bundleDir: String, audioPath: String?) async throws {
    print("Format: split (encoder + decoder, KV cache)")

    let encModel = try await AIModel(contentsOf: URL(fileURLWithPath: "\(bundleDir)/encoder.aimodel"))
    let decModel = try await AIModel(contentsOf: URL(fileURLWithPath: "\(bundleDir)/decoder.aimodel"))

    guard let encFn = try encModel.loadFunction(named: "main"),
          let decFn = try decModel.loadFunction(named: "main")
    else { fatalError("No 'main' function") }

    let encDesc = encModel.functionDescriptor(for: "main")!
    let decDesc = decModel.functionDescriptor(for: "main")!

    guard case .ndArray(let melNDDesc)    = encDesc.inputDescriptor(of: "input_features"),
          case .ndArray(let encOutNDDesc) = encDesc.outputDescriptor(of: "encoder_hidden_states")
    else { fatalError("Unexpected encoder descriptors") }

    let encOutShape = encOutNDDesc.shape

    var melArray: NDArray
    if let path = audioPath {
        melArray = try loadMelArray(from: path, descriptor: melNDDesc)
    } else {
        print("No audio — using silence for benchmarking")
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillNDArray(&melArray, as: Float.self, count: melElements) { _ in 0.0 }
    }
    var encOutArray = NDArray(descriptor: encOutNDDesc.resolvingDynamicDimensions(encOutShape))

    // Warmup
    do {
        var out = InferenceFunction.MutableViews()
        out.insert(&encOutArray, for: "encoder_hidden_states")
        _ = try await encFn.run(inputs: ["input_features": melArray],
                                states: InferenceFunction.MutableViews(), outputViews: consume out)
    }
    print("\n── Encoder ────────────────────────────────────────────────────────────")
    let encT0 = Date()
    do {
        var out = InferenceFunction.MutableViews()
        out.insert(&encOutArray, for: "encoder_hidden_states")
        _ = try await encFn.run(inputs: ["input_features": melArray],
                                states: InferenceFunction.MutableViews(), outputViews: consume out)
    }
    print(String(format: "  latency: %.1f ms", Date().timeIntervalSince(encT0) * 1000))

    guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
          case .ndArray(let posIdsNDDesc)   = decDesc.inputDescriptor(of: "position_ids"),
          case .ndArray(let encHSNDDesc)    = decDesc.inputDescriptor(of: "encoder_hidden_states"),
          case .ndArray(let keyCacheNDDesc) = decDesc.stateDescriptor(of: "keyCache"),
          case .ndArray(let valCacheNDDesc) = decDesc.stateDescriptor(of: "valueCache"),
          case .ndArray(let logitsNDDesc)   = decDesc.outputDescriptor(of: "logits")
    else { fatalError("Unexpected decoder descriptors") }

    let vocabSize = logitsNDDesc.shape.last!
    let kcShape = keyCacheNDDesc.shape.map { $0 < 0 ? maxTargetPositions : $0 }
    let vcShape = valCacheNDDesc.shape.map { $0 < 0 ? maxTargetPositions : $0 }
    var keyCache   = NDArray(descriptor: keyCacheNDDesc.resolvingDynamicDimensions(kcShape))
    var valueCache = NDArray(descriptor: valCacheNDDesc.resolvingDynamicDimensions(vcShape))

    let encFlat = readNDArray(encOutArray, as: Float.self, count: encOutShape.reduce(1, *))
    var encHSArray = NDArray(descriptor: encHSNDDesc.resolvingDynamicDimensions(encOutShape))
    fillNDArray(&encHSArray, as: Float.self, with: encFlat)
    var logitsArray = NDArray(descriptor: logitsNDDesc.resolvingDynamicDimensions([1, 1, vocabSize]))

    print("\n── Decoder ────────────────────────────────────────────────────────────")

    var tokens: [Int32] = forcedPrefix
    var pos = 0
    for tok in forcedPrefix {
        var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        var posIds = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, pos + 1]))
        fillNDArray(&ids, as: Int32.self, with: [tok])
        fillNDArray(&posIds, as: Int32.self, count: pos + 1) { Int32($0) }
        var st = InferenceFunction.MutableViews()
        st.insert(&keyCache, for: "keyCache"); st.insert(&valueCache, for: "valueCache")
        var out = InferenceFunction.MutableViews(); out.insert(&logitsArray, for: "logits")
        _ = try await decFn.run(
            inputs: ["input_ids": ids, "position_ids": posIds, "encoder_hidden_states": encHSArray],
            states: consume st, outputViews: consume out)
        pos += 1
    }

    var stepTimesMs: [Double] = []
    while stepTimesMs.count < maxDecodeSteps {
        var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        var posIds = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, pos + 1]))
        fillNDArray(&ids, as: Int32.self, with: [tokens.last!])
        fillNDArray(&posIds, as: Int32.self, count: pos + 1) { Int32($0) }
        var st = InferenceFunction.MutableViews()
        st.insert(&keyCache, for: "keyCache"); st.insert(&valueCache, for: "valueCache")
        var out = InferenceFunction.MutableViews(); out.insert(&logitsArray, for: "logits")
        let t0 = Date()
        _ = try await decFn.run(
            inputs: ["input_ids": ids, "position_ids": posIds, "encoder_hidden_states": encHSArray],
            states: consume st, outputViews: consume out)
        stepTimesMs.append(Date().timeIntervalSince(t0) * 1000)
        let logits = flattenAsFloat(logitsArray)
        let next = Int32(logits.indices.max(by: { logits[$0] < logits[$1] })!)
        tokens.append(next); pos += 1
        if next == eotToken { break }
    }

    await printResults(tokens: tokens, stepTimesMs: stepTimesMs)
}

// MARK: - Legacy runner (monolithic model, no KV cache)

func runLegacy(modelPath: String, audioPath: String?) async throws {
    print("Format: legacy (monolithic, no KV cache)")

    let model = try await AIModel(contentsOf: URL(fileURLWithPath: modelPath))
    guard let fn = try model.loadFunction(named: "main") else { fatalError("No 'main' function") }
    let desc = model.functionDescriptor(for: "main")!

    guard case .ndArray(let melNDDesc)  = desc.inputDescriptor(of: "input_features"),
          case .ndArray(let idsNDDesc)  = desc.inputDescriptor(of: "decoder_input_ids"),
          case .ndArray(let logitsDesc) = desc.outputDescriptor(of: "logits")
    else { fatalError("Unexpected model descriptors") }

    let vocabSize = logitsDesc.shape.last!
    let isStaticIds = !idsNDDesc.shape.contains(where: { $0 < 0 })
    if isStaticIds {
        print("decoder_input_ids exported with static shape — no past context per step")
        print("Re-export with --mode legacy to get dynamic shapes")
    }

    var melArray: NDArray
    if let path = audioPath {
        melArray = try loadMelArray(from: path, descriptor: melNDDesc)
    } else {
        print("No audio — using silence for benchmarking")
        melArray = NDArray(descriptor: melNDDesc.resolvingDynamicDimensions([1, 128, 3000]))
        fillNDArray(&melArray, as: Float.self, count: melElements) { _ in 0.0 }
    }

    print("\n── Decode ─────────────────────────────────────────────────────────────")

    var tokens: [Int32] = forcedPrefix
    var stepTimesMs: [Double] = []

    while stepTimesMs.count < maxDecodeSteps {
        let inputTokens: [Int32] = isStaticIds ? [tokens.last!] : tokens
        let seqLen = inputTokens.count
        var ids = NDArray(descriptor: idsNDDesc.resolvingDynamicDimensions([1, seqLen]))
        fillNDArray(&ids, as: Int32.self, with: inputTokens)
        var logitsArray = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, seqLen, vocabSize]))
        var out = InferenceFunction.MutableViews(); out.insert(&logitsArray, for: "logits")
        let t0 = Date()
        _ = try await fn.run(
            inputs: ["input_features": melArray, "decoder_input_ids": ids],
            states: InferenceFunction.MutableViews(), outputViews: consume out)
        stepTimesMs.append(Date().timeIntervalSince(t0) * 1000)
        let logits = flattenAsFloat(logitsArray)
        let lastStart = (seqLen - 1) * vocabSize
        let lastLogits = Array(logits[lastStart ..< lastStart + vocabSize])
        let next = Int32(lastLogits.indices.max(by: { lastLogits[$0] < lastLogits[$1] })!)
        tokens.append(next)
        if next == eotToken { break }
    }

    await printResults(tokens: tokens, stepTimesMs: stepTimesMs)
}
