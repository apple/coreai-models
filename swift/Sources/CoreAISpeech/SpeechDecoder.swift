// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - SpeechDecoder protocol

/// Model-specific decode logic.
public protocol SpeechDecoder: Sendable {
    func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        decoderModel: AIModel,
        config: GenerationConfig
    ) async throws -> [Int32]
}

// MARK: - WhisperDecoder

/// Greedy decoder for Whisper (encoder-decoder, cross-attention, KV cache).
public struct WhisperDecoder: SpeechDecoder {
    public init() {}

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        decoderModel: AIModel,
        config: GenerationConfig
    ) async throws -> [Int32] {
        guard let decFn = try decoderModel.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in decoder")
        }
        let decDesc = decoderModel.functionDescriptor(for: "main")!

        guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let posIdsNDDesc) = decDesc.inputDescriptor(of: "position_ids"),
            case .ndArray(let encHSNDDesc) = decDesc.inputDescriptor(of: "encoder_hidden_states"),
            case .ndArray(let keyCacheNDDesc) = decDesc.stateDescriptor(of: "keyCache"),
            case .ndArray(let valCacheNDDesc) = decDesc.stateDescriptor(of: "valueCache"),
            case .ndArray(let logitsNDDesc) = decDesc.outputDescriptor(of: "logits")
        else { throw SpeechError.missingModel("Unexpected decoder descriptors") }

        let vocabSize = logitsNDDesc.shape.last!
        let maxTargetPos = 448
        let kcShape = keyCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }
        let vcShape = valCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }
        var keyCache = NDArray(descriptor: keyCacheNDDesc.resolvingDynamicDimensions(kcShape))
        var valueCache = NDArray(descriptor: valCacheNDDesc.resolvingDynamicDimensions(vcShape))

        var encHSArray = NDArray(descriptor: encHSNDDesc.resolvingDynamicDimensions(encoderOutputShape))
        let encFlat = readNDArray(encoderOutput, as: Float.self, count: encoderOutputShape.reduce(1, *))
        fillNDArray(&encHSArray, as: Float.self, with: encFlat)

        var logitsArray = NDArray(descriptor: logitsNDDesc.resolvingDynamicDimensions([1, 1, vocabSize]))

        func step(_ tok: Int32, pos: Int) async throws {
            var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
            var posIds = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, pos + 1]))
            fillNDArray(&ids, as: Int32.self, with: [tok])
            fillNDArray(&posIds, as: Int32.self, count: pos + 1) { Int32($0) }
            var st = InferenceFunction.MutableViews()
            st.insert(&keyCache, for: "keyCache")
            st.insert(&valueCache, for: "valueCache")
            var out = InferenceFunction.MutableViews()
            out.insert(&logitsArray, for: "logits")
            _ = try await decFn.run(
                inputs: [
                    "input_ids": ids, "position_ids": posIds,
                    "encoder_hidden_states": encHSArray,
                ],
                states: consume st, outputViews: consume out)
        }

        // Prime KV cache with forced prefix
        var tokens: [Int32] = config.forcedPrefix
        for (i, tok) in config.forcedPrefix.enumerated() {
            try await step(tok, pos: i)
        }

        // Greedy decode
        var pos = config.forcedPrefix.count
        while tokens.count - config.forcedPrefix.count < config.maxDecodeSteps {
            try await step(tokens.last!, pos: pos)
            let logits = flattenAsFloat(logitsArray)
            let next = Int32(logits.indices.max(by: { logits[$0] < logits[$1] })!)
            tokens.append(next)
            pos += 1
            if next == config.eotToken { break }
        }

        return tokens
    }
}
