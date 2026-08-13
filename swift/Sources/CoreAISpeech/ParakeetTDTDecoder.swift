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
/// hidden/cell state is owned here, seeded with zeros and only advanced when the
/// step's input symbol is non-blank, matching the HF
/// `ParakeetTDTDecoderCache.update(..., mask=~blank_mask)` semantics.
public struct ParakeetTDTDecoder: SpeechDecoder {
    /// The `decoder_step` graph plus the descriptors its buffers are built from.
    private struct StepGraph {
        let fn: InferenceFunction
        let inputIds: NDArrayDescriptor
        let hiddenIn: NDArrayDescriptor
        let cellIn: NDArrayDescriptor
        let decoderOut: NDArrayDescriptor
        let newHidden: NDArrayDescriptor
        let newCell: NDArrayDescriptor
    }

    /// The `joint` graph plus the descriptors its buffers are built from.
    ///
    /// No `decoderIn`: the step graph's `decoder_output` buffer is handed to the joint
    /// as `decoder_hidden_states` directly, so there's no second buffer to allocate.
    private struct JointGraph {
        let fn: InferenceFunction
        let encoderIn: NDArrayDescriptor
        let logits: NDArrayDescriptor
    }

    /// Every NDArray a decode call needs, allocated once up front and rewritten in
    /// place on each step rather than reallocated per emission.
    private struct Buffers {
        var inputIds: NDArray
        /// LSTM state, double-buffered: a step reads `hIn`/`cIn` and writes `hOut`/`cOut`,
        /// and the call site swaps each pair to adopt the new state.
        var hIn: NDArray
        var cIn: NDArray
        var hOut: NDArray
        var cOut: NDArray
        /// Doubles as the joint's `decoder_hidden_states` input: same `[1, 1, hidden]` shape,
        /// same precision, so the step's output needs no copy to become the joint's input.
        var decOut: NDArray
        var jointEncIn: NDArray
        var logits: NDArray

        init(
            step: StepGraph, joint: JointGraph,
            lstmShape: [Int], hidden: Int, logitsSize: Int
        ) {
            inputIds = NDArray(descriptor: step.inputIds.resolvingDynamicDimensions([1, 1]))
            hIn = NDArray(descriptor: step.hiddenIn.resolvingDynamicDimensions(lstmShape))
            cIn = NDArray(descriptor: step.cellIn.resolvingDynamicDimensions(lstmShape))
            hOut = NDArray(descriptor: step.newHidden.resolvingDynamicDimensions(lstmShape))
            cOut = NDArray(descriptor: step.newCell.resolvingDynamicDimensions(lstmShape))
            decOut = NDArray(descriptor: step.decoderOut.resolvingDynamicDimensions([1, 1, hidden]))
            jointEncIn = NDArray(
                descriptor: joint.encoderIn.resolvingDynamicDimensions([1, 1, hidden]))
            logits = NDArray(descriptor: joint.logits.resolvingDynamicDimensions([1, 1, logitsSize]))

            // The first step reads `hIn`/`cIn` before anything has written them, so seed the
            // zero state here rather than relying on fresh-allocation contents.
            let zeros = [Float](repeating: 0, count: lstmShape.reduce(1, *))
            fillFloatNDArray(&hIn, with: zeros)
            fillFloatNDArray(&cIn, with: zeros)
        }
    }

    /// Loaded once at init and reused across every `decode` call — the graphs and
    /// their descriptors are shape-independent, so only `resolvingDynamicDimensions`
    /// belongs on the per-transcription path.
    private let stepGraph: StepGraph
    private let jointGraph: JointGraph

    public init(decoderStep: AIModel, joint: AIModel) throws {
        guard let stepFn = try decoderStep.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in decoder_step")
        }
        guard let jointFn = try joint.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in joint")
        }
        guard let stepDesc = decoderStep.functionDescriptor(for: "main") else {
            throw SpeechError.missingModel("No 'main' descriptor in decoder_step")
        }
        guard let jointDesc = joint.functionDescriptor(for: "main") else {
            throw SpeechError.missingModel("No 'main' descriptor in joint")
        }

        guard case .ndArray(let inputIdsDesc) = stepDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let hiddenInDesc) = stepDesc.inputDescriptor(of: "hidden_state"),
            case .ndArray(let cellInDesc) = stepDesc.inputDescriptor(of: "cell_state"),
            case .ndArray(let decoderOutDesc) = stepDesc.outputDescriptor(of: "decoder_output"),
            case .ndArray(let newHiddenDesc) = stepDesc.outputDescriptor(of: "new_hidden_state"),
            case .ndArray(let newCellDesc) = stepDesc.outputDescriptor(of: "new_cell_state")
        else { throw SpeechError.missingModel("Unexpected decoder_step descriptors") }

        guard case .ndArray(_) = jointDesc.inputDescriptor(of: "decoder_hidden_states"),
            case .ndArray(let jointEncDesc) = jointDesc.inputDescriptor(of: "encoder_hidden_states"),
            case .ndArray(let logitsDesc) = jointDesc.outputDescriptor(of: "logits")
        else { throw SpeechError.missingModel("Unexpected joint descriptors") }

        self.stepGraph = StepGraph(
            fn: stepFn, inputIds: inputIdsDesc, hiddenIn: hiddenInDesc, cellIn: cellInDesc,
            decoderOut: decoderOutDesc, newHidden: newHiddenDesc, newCell: newCellDesc)
        self.jointGraph = JointGraph(fn: jointFn, encoderIn: jointEncDesc, logits: logitsDesc)
    }

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        // Only the config is read from `resources`; the two graphs come from the same
        // models this decoder was initialized with, already loaded in `init`.
        guard case .parakeetTDT(_, _, let cfg) = resources else {
            throw SpeechError.incompatibleResources("ParakeetTDTDecoder requires .parakeetTDT resources")
        }
        let hidden = cfg.decoderHiddenSize
        let vocabSize = cfg.vocabSize
        let logitsSize = jointGraph.logits.shape.last!
        try Self.validate(
            encoderOutputShape: encoderOutputShape, logitsSize: logitsSize, config: cfg)

        let tEnc = encoderOutputShape[1]
        if tEnc == 0 { return (tokens: [], stats: DecodeStats(stepTimesMs: [])) }

        // For a static (padded) encoder the tail frames are zero-padding; decoding
        // them yields spurious tokens (trailing periods). Cap the loop at the number
        // of frames that carry real audio. Dynamic exports pass tEnc, so cap == tEnc.
        let cap = max(1, min(tEnc, validEncoderFrames))
        let lstmShape = [cfg.numDecoderLayers, 1, hidden]

        // Pull the full encoder output once; slice frame-by-frame in pure Swift.
        // flattenAsFloat inspects the array's own scalar type, so this reads an
        // f16 encoder output correctly (a raw `as: Float.self` read would not).
        let encFlat = flattenAsFloat(encoderOutput)
        var buffers = Buffers(
            step: stepGraph, joint: jointGraph,
            lstmShape: lstmShape, hidden: hidden, logitsSize: logitsSize)

        // Previous iteration's symbol, blanks included — fed back as the next `input_ids`.
        // Distinct from `emitted`, which keeps only the non-blank symbols.
        var previousSymbol: Int32 = cfg.blankTokenId
        var emitted: [Int32] = []
        var frame = 0
        var firstStep = true
        let emitCap = cap * cfg.maxSymbolsPerStep

        var stepTimesMs: [Double] = []
        var coverage = DecodeStats.Coverage()
        var capturedStep: (decoderOutput: [Float], newHidden: [Float], newCell: [Float])?
        var capturedLogits: [Float]?

        while frame < cap && emitted.count < emitCap {
            let t0 = ContinuousClock.now
            var advance = 0
            let emittedAtStepStart = emitted.count
            for _ in 0..<cfg.maxSymbolsPerStep {
                // Per-iteration, not per-step: one step runs this loop up to
                // `maxSymbolsPerStep` times, so the input here may be a symbol the same step
                // just emitted. `firstStep` covers SOS.
                let inputIsBlank = (previousSymbol == cfg.blankTokenId)

                // Blank-skip (optimization): a blank input reproduces the last decoder
                // output, since the state was held too — and `buffers.decOut` still holds
                // it, because this branch doesn't run the step graph that would overwrite it.
                if !firstStep, inputIsBlank {
                    coverage.blankSkipReuses += 1
                } else {
                    try await runDecoderStep(token: previousSymbol, buffers: &buffers)

                    // Parity capture only: gating on nil keeps the flattens to once per decode
                    // instead of once per step, and reads `hOut`/`cOut` before the swap below.
                    if capturedStep == nil {
                        capturedStep = (
                            decoderOutput: flattenAsFloat(buffers.decOut),
                            newHidden: flattenAsFloat(buffers.hOut),
                            newCell: flattenAsFloat(buffers.cOut)
                        )
                    }

                    // Load-bearing: a blank carries no label, so the state must not absorb
                    // it. Mirrors HF `cache.update(..., mask=~blank_mask)`. Blanks normally
                    // never get here at all — they take the reuse branch above — but the
                    // guard still has to hold if that branch is ever changed. Not adopting
                    // means not swapping: `hIn`/`cIn` keep the state the step ran from, and
                    // the next step overwrites `hOut`/`cOut`.
                    if firstStep || !inputIsBlank {
                        swap(&buffers.hIn, &buffers.hOut)
                        swap(&buffers.cIn, &buffers.cOut)
                        coverage.lstmStateAdvances += 1
                    }
                    firstStep = false
                }

                // Joint(decoder_output, encoder[:, frame:frame+1, :]).
                let encOffset = frame * hidden
                let logitsFlat = try await runJoint(
                    encoderFrame: encFlat[encOffset..<encOffset + hidden],
                    buffers: &buffers)
                if capturedLogits == nil { capturedLogits = logitsFlat }

                let symbol = Int32(Self.argmax(logitsFlat, in: 0..<vocabSize))
                let dur = cfg.durations[Self.argmax(logitsFlat, in: vocabSize..<logitsSize)]
                // Output-side counterpart to `inputIsBlank`; becomes it next iteration.
                let isBlank = (symbol == cfg.blankTokenId)

                if !isBlank {
                    emitted.append(symbol)
                }
                // Carries blanks too, matching HF. Holding the last non-blank instead would
                // re-feed it and advance the state as if the label repeated.
                previousSymbol = symbol

                // A blank that picks duration 0 still moves one frame forward, matching HF's
                // `torch.where(blank_mask & (durations == 0), 1, durations)`. Without this the
                // inner loop re-runs the joint on the same frame with the same (cached)
                // decoder output until maxSymbolsPerStep is exhausted.
                if isBlank && dur == 0 {
                    coverage.blankZeroDurationBreaks += 1
                    advance = 1
                    break
                }

                if dur > 0 {
                    coverage.positiveDurationBreaks += 1
                    advance = dur
                    break
                }
            }
            // No duration > 0 was selected within max_symbols_per_step — force one frame
            // forward to guarantee outer-loop progress.
            if advance == 0 {
                coverage.symbolCapExhaustions += 1
                advance = 1
            }
            switch emitted.count - emittedAtStepStart {
            case 0: coverage.blankOnlySteps += 1
            case 1: break
            default: coverage.multiTokenSteps += 1
            }
            stepTimesMs.append((ContinuousClock.now - t0).inMilliseconds)
            frame += advance
        }

        var capture: DecodeStats.FirstStep?
        if let step = capturedStep, let logits = capturedLogits {
            capture = DecodeStats.FirstStep(
                decoderOutput: step.decoderOutput, newHiddenState: step.newHidden,
                newCellState: step.newCell, jointLogits: logits)
        }
        return (
            tokens: emitted,
            stats: DecodeStats(
                stepTimesMs: stepTimesMs, coverage: coverage, firstStep: capture)
        )
    }

    /// Preconditions the decode loop's arithmetic depends on.
    ///
    /// Extracted from `decode` so it can be tested: the initializer needs two loaded `AIModel`s,
    /// so guards left inline would be unreachable without model assets on disk.
    package static func validate(
        encoderOutputShape: [Int], logitsSize: Int, config: ParakeetTDTConfig
    ) throws {
        guard encoderOutputShape.count == 3 else {
            throw SpeechError.missingModel(
                "Parakeet encoder must output rank-3 [B, T, H], got \(encoderOutputShape)")
        }
        guard encoderOutputShape[0] == 1 && encoderOutputShape[2] == config.decoderHiddenSize else {
            throw SpeechError.missingModel(
                "Encoder output shape \(encoderOutputShape) doesn't match config "
                    + "(B=1, H=\(config.decoderHiddenSize))")
        }
        // The token argmax scans `0..<vocabSize`, so a blank id outside that range could
        // never win: `isBlank` would never fire, every frame's argmax would be emitted as
        // a real token, and the blank-skip path would be dead code. The loop's blank
        // bookkeeping rests on blank being a producible, unshared id — check it here
        // rather than silently degrading to a garbage transcript.
        guard config.blankTokenId >= 0, Int(config.blankTokenId) < config.vocabSize else {
            throw SpeechError.missingModel(
                "blank_token_id \(config.blankTokenId) is outside the vocab range "
                    + "0..<\(config.vocabSize)")
        }
        // The joint emits `[vocab | durations]` along its last axis; both argmaxes in the loop
        // index that layout directly, so a mismatch would read duration logits out of the
        // vocab range (or off the end of the row).
        guard logitsSize == config.vocabSize + config.durations.count else {
            throw SpeechError.missingModel(
                "Joint logits width \(logitsSize) doesn't match vocab \(config.vocabSize) + "
                    + "\(config.durations.count) durations")
        }
    }

    // MARK: - Graph invocations

    /// One LSTM step: feed `token` at the state in `buffers.hIn`/`cIn`, leaving the decoder
    /// output in `buffers.decOut` and the resulting state in `buffers.hOut`/`cOut`.
    ///
    /// Writing the new state to the output buffers rather than adopting it keeps the decision
    /// of *whether* to adopt — the HF `mask=~blank_mask` rule — at the call site with the
    /// blank bookkeeping it depends on. The call site adopts by swapping the buffer pairs.
    private func runDecoderStep(token: Int32, buffers: inout Buffers) async throws {
        fillNDArray(&buffers.inputIds, as: Int32.self, with: [token])

        var stepOut = InferenceFunction.MutableViews()
        stepOut.insert(&buffers.decOut, for: "decoder_output")
        stepOut.insert(&buffers.hOut, for: "new_hidden_state")
        stepOut.insert(&buffers.cOut, for: "new_cell_state")
        _ = try await stepGraph.fn.run(
            inputs: [
                "input_ids": buffers.inputIds,
                "hidden_state": buffers.hIn,
                "cell_state": buffers.cIn,
            ],
            states: InferenceFunction.MutableViews(),
            outputViews: consume stepOut)
    }

    /// `joint(decoder_output, encoder_frame)` → the flattened `[vocab | durations]` row.
    ///
    /// The decoder side comes straight from `buffers.decOut`, whether the last step wrote it
    /// or the blank-skip branch left it in place.
    private func runJoint(
        encoderFrame: ArraySlice<Float>, buffers: inout Buffers
    ) async throws -> [Float] {
        fillFloatNDArray(&buffers.jointEncIn, with: encoderFrame)

        var jointOut = InferenceFunction.MutableViews()
        jointOut.insert(&buffers.logits, for: "logits")
        _ = try await jointGraph.fn.run(
            inputs: [
                "decoder_hidden_states": buffers.decOut,
                "encoder_hidden_states": buffers.jointEncIn,
            ],
            states: InferenceFunction.MutableViews(),
            outputViews: consume jointOut)

        return flattenAsFloat(buffers.logits)
    }

    /// Index of the largest value in `values[range]`, relative to `range.lowerBound`.
    /// Ties go to the lowest index, and an all-`-infinity` range yields 0 — matching the
    /// hand-rolled scans this replaces.
    ///
    /// `static` because it reads no instance state, which also lets it be unit-tested: the
    /// initializer needs two loaded `AIModel`s, so an instance method here would be reachable
    /// only with model assets on disk.
    package static func argmax(_ values: [Float], in range: Range<Int>) -> Int {
        var best = range.lowerBound
        var bestVal: Float = -.infinity
        for i in range where values[i] > bestVal {
            bestVal = values[i]
            best = i
        }
        return best - range.lowerBound
    }
}
