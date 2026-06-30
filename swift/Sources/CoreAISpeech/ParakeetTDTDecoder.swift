// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Greedy decoder for Parakeet TDT.
///
/// Drives the autoregressive (token, duration) loop in Swift, calling the exported
/// `decoder_step` (single LSTM step) and `joint` graphs per emission. The LSTM
/// hidden/cell state is owned here, seeded with zeros and only advanced when a
/// non-blank token is emitted — matching the HF `ParakeetTDTDecoderCache.update(...,
/// mask=~blank_mask)` semantics.
public struct ParakeetTDTDecoder: SpeechDecoder {
    public init() {}

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        resources: DecoderResources
    ) async throws -> [Int32] {
        guard case .parakeetTDT(let decoderStep, let joint, let cfg) = resources else {
            throw SpeechError.incompatibleResources("ParakeetTDTDecoder requires .parakeetTDT resources")
        }

        guard encoderOutputShape.count == 3 else {
            throw SpeechError.missingModel(
                "Parakeet encoder must output rank-3 [B, T, H], got \(encoderOutputShape)")
        }
        let tEnc = encoderOutputShape[1]
        let hidden = cfg.decoderHiddenSize
        guard encoderOutputShape[0] == 1 && encoderOutputShape[2] == hidden else {
            throw SpeechError.missingModel(
                "Encoder output shape \(encoderOutputShape) doesn't match config (B=1, H=\(hidden))")
        }
        if tEnc == 0 { return [] }

        // Resolve descriptors for the two callable graphs.
        guard let stepFn = try decoderStep.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in decoder_step")
        }
        guard let jointFn = try joint.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in joint")
        }
        let stepDesc = decoderStep.functionDescriptor(for: "main")!
        let jointDesc = joint.functionDescriptor(for: "main")!

        guard case .ndArray(let inputIdsDesc) = stepDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let hiddenInDesc) = stepDesc.inputDescriptor(of: "hidden_state"),
            case .ndArray(let cellInDesc) = stepDesc.inputDescriptor(of: "cell_state"),
            case .ndArray(let decoderOutDesc) = stepDesc.outputDescriptor(of: "decoder_output"),
            case .ndArray(let newHiddenDesc) = stepDesc.outputDescriptor(of: "new_hidden_state"),
            case .ndArray(let newCellDesc) = stepDesc.outputDescriptor(of: "new_cell_state")
        else { throw SpeechError.missingModel("Unexpected decoder_step descriptors") }

        guard case .ndArray(let jointDecDesc) = jointDesc.inputDescriptor(of: "decoder_hidden_states"),
            case .ndArray(let jointEncDesc) = jointDesc.inputDescriptor(of: "encoder_hidden_states"),
            case .ndArray(let logitsDesc) = jointDesc.outputDescriptor(of: "logits")
        else { throw SpeechError.missingModel("Unexpected joint descriptors") }

        let lstmShape = [cfg.numDecoderLayers, 1, hidden]
        let lstmCount = cfg.numDecoderLayers * hidden
        let logitsSize = logitsDesc.shape.last!  // vocab + |durations|

        // Pull the full encoder output once; slice frame-by-frame in pure Swift.
        let encFlat = readNDArray(encoderOutput, as: Float.self, count: tEnc * hidden)

        // Persistent buffers (reused across all steps).
        var inputIds = NDArray(descriptor: inputIdsDesc.resolvingDynamicDimensions([1, 1]))
        var hIn = NDArray(descriptor: hiddenInDesc.resolvingDynamicDimensions(lstmShape))
        var cIn = NDArray(descriptor: cellInDesc.resolvingDynamicDimensions(lstmShape))
        var decOut = NDArray(descriptor: decoderOutDesc.resolvingDynamicDimensions([1, 1, hidden]))
        var hOut = NDArray(descriptor: newHiddenDesc.resolvingDynamicDimensions(lstmShape))
        var cOut = NDArray(descriptor: newCellDesc.resolvingDynamicDimensions(lstmShape))

        var jointDecIn = NDArray(descriptor: jointDecDesc.resolvingDynamicDimensions([1, 1, hidden]))
        var jointEncIn = NDArray(descriptor: jointEncDesc.resolvingDynamicDimensions([1, 1, hidden]))
        var logits = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, 1, logitsSize]))

        // Swift-side LSTM state — zero-seeded, only advanced on non-blank emissions.
        var hState = [Float](repeating: 0, count: lstmCount)
        var cState = [Float](repeating: 0, count: lstmCount)

        var lastToken: Int32 = cfg.blankTokenId
        var emitted: [Int32] = []
        var frame = 0
        var firstStep = true
        var cachedDecBuf: [Float]? = nil  // last decoder output; reused on blank-input frames
        let emitCap = tEnc * cfg.maxSymbolsPerStep
        let vocabSize = cfg.vocabSize

        while frame < tEnc && emitted.count < emitCap {
            var advance = 0
            for _ in 0..<cfg.maxSymbolsPerStep {
                let wasInputBlank = (lastToken == cfg.blankTokenId)

                // HF blank-skip: when the cache is warm and the input is blank, reuse the
                // last cached decoder output without re-running the LSTM step.
                let decBuf: [Float]
                if !firstStep && wasInputBlank, let cached = cachedDecBuf {
                    decBuf = cached
                } else {
                    fillNDArray(&inputIds, as: Int32.self, with: [lastToken])
                    fillNDArray(&hIn, as: Float.self, with: hState)
                    fillNDArray(&cIn, as: Float.self, with: cState)

                    var stepOut = InferenceFunction.MutableViews()
                    stepOut.insert(&decOut, for: "decoder_output")
                    stepOut.insert(&hOut, for: "new_hidden_state")
                    stepOut.insert(&cOut, for: "new_cell_state")
                    _ = try await stepFn.run(
                        inputs: [
                            "input_ids": inputIds,
                            "hidden_state": hIn,
                            "cell_state": cIn,
                        ],
                        states: InferenceFunction.MutableViews(),
                        outputViews: consume stepOut)

                    decBuf = flattenAsFloat(decOut)
                    cachedDecBuf = decBuf

                    // LSTM-state update rule: always on the very first call; afterwards only
                    // when the input was non-blank (matches HF cache.update mask=~blank_mask).
                    if firstStep || !wasInputBlank {
                        hState = flattenAsFloat(hOut)
                        cState = flattenAsFloat(cOut)
                    }
                    firstStep = false
                }

                // Joint(decoder_output, encoder[:, frame:frame+1, :]).
                fillNDArray(&jointDecIn, as: Float.self, with: decBuf)
                let encOffset = frame * hidden
                let encSlice = Array(encFlat[encOffset..<encOffset + hidden])
                fillNDArray(&jointEncIn, as: Float.self, with: encSlice)

                var jointOutViews = InferenceFunction.MutableViews()
                jointOutViews.insert(&logits, for: "logits")
                _ = try await jointFn.run(
                    inputs: [
                        "decoder_hidden_states": jointDecIn,
                        "encoder_hidden_states": jointEncIn,
                    ],
                    states: InferenceFunction.MutableViews(),
                    outputViews: consume jointOutViews)

                // Argmax over [vocab] and [durations] sub-ranges.
                let logitsFlat = flattenAsFloat(logits)
                var bestTok: Int32 = 0
                var bestTokVal: Float = -.infinity
                for i in 0..<vocabSize where logitsFlat[i] > bestTokVal {
                    bestTokVal = logitsFlat[i]
                    bestTok = Int32(i)
                }
                var bestDurIdx = 0
                var bestDurVal: Float = -.infinity
                for i in 0..<cfg.durations.count where logitsFlat[vocabSize + i] > bestDurVal {
                    bestDurVal = logitsFlat[vocabSize + i]
                    bestDurIdx = i
                }
                let dur = cfg.durations[bestDurIdx]

                if bestTok != cfg.blankTokenId {
                    emitted.append(bestTok)
                    lastToken = bestTok
                }
                // (If blank, lastToken stays as it was — typically still blank or the
                // most recently emitted real token.)

                if dur > 0 {
                    advance = dur
                    break
                }
            }
            // No duration > 0 was selected within max_symbols_per_step — force one frame
            // forward to guarantee outer-loop progress.
            if advance == 0 { advance = 1 }
            frame += advance
        }

        return emitted
    }
}
