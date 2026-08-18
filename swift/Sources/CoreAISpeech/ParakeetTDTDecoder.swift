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

    /// Transducer state that outlives a single call, so a streaming session can decode
    /// one chunk of encoder frames at a time and keep going where it left off.
    ///
    /// The fields mirror NeMo's `BatchedLabelLoopingState`
    /// (`nemo/collections/asr/parts/submodules/transducer_decoding/label_looping_base.py:41-49`)
    public final class Stream: @unchecked Sendable {
        private let decoder: ParakeetTDTDecoder
        private let cfg: ParakeetTDTConfig
        private let logitsSize: Int
        private let lstmShape: [Int]

        /// LSTM state carried across chunks (NeMo `predictor_states`).
        private var hiddenState: [Float]
        private var cellState: [Float]
        /// Last decoder output (NeMo `predictor_outputs`). Load-bearing across a chunk
        /// boundary: the blank-skip branch reuses it without re-running the step graph, so
        /// dropping it would make the first step of every chunk read a stale value.
        private var decoderOutput: [Float]?

        /// Previous iteration's symbol, blanks included — fed back as the next `input_ids`.
        private var previousSymbol: Int32
        private var firstStep: Bool

        /// Frames a TDT duration overshot the last chunk by, to be skipped at the start of
        /// the next one.
        public private(set) var timeJump: Int = 0

        /// Consecutive encoder frames consumed without emitting anything, duration-weighted.
        ///
        /// Read by the streaming endpointer: a blank carrying duration 4 skips 320 ms in one
        /// step, so a step count would under-measure silence by up to 4x, and a per-hop count
        /// can only say "this whole chunk was quiet".
        public private(set) var silentFrames: Int = 0

        init(decoder: ParakeetTDTDecoder, config: ParakeetTDTConfig) {
            self.decoder = decoder
            self.cfg = config
            self.logitsSize = decoder.jointGraph.logits.shape.last!
            self.lstmShape = [config.numDecoderLayers, 1, config.decoderHiddenSize]
            let stateCount = lstmShape.reduce(1, *)
            self.hiddenState = [Float](repeating: 0, count: stateCount)
            self.cellState = [Float](repeating: 0, count: stateCount)
            self.decoderOutput = nil
            self.previousSymbol = config.blankTokenId
            self.firstStep = true
        }

        /// Start a new segment: zero the LSTM and re-seed the blank as the previous label.
        public func resetSegment() {
            let stateCount = lstmShape.reduce(1, *)
            hiddenState = [Float](repeating: 0, count: stateCount)
            cellState = [Float](repeating: 0, count: stateCount)
            decoderOutput = nil
            previousSymbol = cfg.blankTokenId
            firstStep = true
            silentFrames = 0
        }

        /// Decode the global encoder frames in `frames`, where local index 0 of
        /// `encoderOutput` is global frame `windowStartFrame`.
        ///
        /// The loop body is the offline one verbatim; only the frame pointer's coordinate
        /// system (global rather than window-local) and the `timeJump` carry are new.
        /// `collectStats` off skips the first-step tensor capture and the per-step timings,
        /// which only the offline parity harness reads.
        public func decodeFrames(
            encoderOutput: NDArray,
            encoderOutputShape: [Int],
            frames: Range<Int>,
            windowStartFrame: Int,
            collectStats: Bool = true,
            resetAfterSilenceFrames: Int = 0
        ) async throws -> (tokens: [Int32], stats: DecodeStats) {
            try ParakeetTDTDecoder.validate(
                encoderOutputShape: encoderOutputShape, logitsSize: logitsSize, config: cfg)
            try checkFrames(
                frames, windowStartFrame: windowStartFrame,
                windowEncoderFrames: encoderOutputShape[1])

            // Convert only the frames this call reads. The window also carries the left and
            // right context the loop never indexes — at the default geometry that is 12 frames
            // of 151 — and `floatElements` inspects the array's own scalar type, so an f16
            // encoder output reads correctly (a raw `as: Float.self` read would not).
            let hidden = cfg.decoderHiddenSize
            let lower = (frames.lowerBound - windowStartFrame) * hidden
            let upper = (frames.upperBound - windowStartFrame) * hidden
            return try await decodeFrames(
                encoderFlat: floatElements(encoderOutput, in: lower..<upper),
                encoderOutputShape: [1, frames.count, hidden],
                frames: frames,
                windowStartFrame: frames.lowerBound,
                collectStats: collectStats,
                resetAfterSilenceFrames: resetAfterSilenceFrames)
        }

        /// As above, for a caller that already holds the encoder output as floats — e.g.
        /// `--deferred-decode`, which concatenates per-chunk encoder outputs and decodes once.
        public func decodeFrames(
            encoderFlat: [Float],
            encoderOutputShape: [Int],
            frames: Range<Int>,
            windowStartFrame: Int,
            collectStats: Bool = true,
            resetAfterSilenceFrames: Int = 0
        ) async throws -> (tokens: [Int32], stats: DecodeStats) {
            try ParakeetTDTDecoder.validate(
                encoderOutputShape: encoderOutputShape, logitsSize: logitsSize, config: cfg)
            try checkFrames(
                frames, windowStartFrame: windowStartFrame,
                windowEncoderFrames: encoderOutputShape[1])
            if frames.isEmpty { return (tokens: [], stats: DecodeStats(stepTimesMs: [])) }

            let hidden = cfg.decoderHiddenSize
            let vocabSize = cfg.vocabSize

            var buffers = Buffers(
                step: decoder.stepGraph, joint: decoder.jointGraph,
                lstmShape: lstmShape, hidden: hidden, logitsSize: logitsSize)
            // Restore the state this stream left off with. `Buffers.init` already seeded
            // zeros, so a fresh stream's first chunk is unaffected by these writes.
            fillFloatNDArray(&buffers.hIn, with: hiddenState)
            fillFloatNDArray(&buffers.cIn, with: cellState)
            if let previousDecoderOutput = decoderOutput {
                fillFloatNDArray(&buffers.decOut, with: previousDecoderOutput)
            }

            var emitted: [Int32] = []
            // Resume where the last chunk's duration jump landed, then clear the debt.
            var frame = frames.lowerBound + timeJump
            timeJump = 0
            // Per-hop, not per-utterance: bounds this chunk's work only.
            let emitCap = frames.count * cfg.maxSymbolsPerStep

            var stepTimesMs: [Double] = []
            var coverage = DecodeStats.Coverage()
            var capturedStep: (decoderOutput: [Float], newHidden: [Float], newCell: [Float])?
            var capturedLogits: [Float]?

            while frame < frames.upperBound && emitted.count < emitCap {
                let t0 = ContinuousClock.now
                var advance = 0
                let emittedAtStepStart = emitted.count
                for _ in 0..<cfg.maxSymbolsPerStep {
                    // Per-iteration, not per-step: one step runs this loop up to
                    // `maxSymbolsPerStep` times, so the input here may be a symbol the same step
                    // just emitted. `firstStep` covers the initial blank, which is the
                    // start-of-sequence condition rather than a real previous label.
                    let inputIsBlank = (previousSymbol == cfg.blankTokenId)

                    // Blank-skip (optimization): a blank input reproduces the last decoder
                    // output, since the state was held too — and `buffers.decOut` still holds
                    // it, because this branch doesn't run the step graph that would overwrite it.
                    if !firstStep, inputIsBlank {
                        coverage.blankSkipReuses += 1
                    } else {
                        try await decoder.runDecoderStep(token: previousSymbol, buffers: &buffers)

                        // Parity capture only: gating on nil keeps the flattens to once per call
                        // instead of once per step, and reads `hOut`/`cOut` before the swap below.
                        if collectStats, capturedStep == nil {
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

                    // Joint(decoder_output, encoder[:, frame, :]) — `frame` is global, so it
                    // has to be rebased onto this window's local indexing.
                    let encOffset = (frame - windowStartFrame) * hidden
                    try await decoder.runJoint(
                        encoderFrame: encoderFlat[encOffset..<encOffset + hidden],
                        buffers: &buffers)
                    // Scanned in place: materializing the row would allocate and convert the
                    // whole vocab per emitted symbol, only to read two argmaxes off it.
                    if collectStats, capturedLogits == nil {
                        capturedLogits = flattenAsFloat(buffers.logits)
                    }

                    let symbol = Int32(argmaxFloat(buffers.logits, in: 0..<vocabSize))
                    let dur = cfg.durations[argmaxFloat(buffers.logits, in: vocabSize..<logitsSize)]
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
                let emittedThisStep = emitted.count - emittedAtStepStart
                switch emittedThisStep {
                case 0: coverage.blankOnlySteps += 1
                case 1: break
                default: coverage.multiTokenSteps += 1
                }
                // Endpointing counts *frames of audio*, not steps: a blank with duration 4
                // skips 320 ms in one step, so counting steps would under-measure silence 4×.
                silentFrames = emittedThisStep == 0 ? silentFrames + advance : 0

                // Past a long gap, drop the label history — see
                // `EndpointingConfig.resetAfterSilenceFrames`. In the loop rather than at the
                // caller's chunk boundary because that is the only place streaming,
                // `--deferred-decode` and offline all share, so the modes still agree.
                if resetAfterSilenceFrames > 0, silentFrames >= resetAfterSilenceFrames {
                    // `resetSegment` writes the stream's properties, so re-seed the buffers the
                    // loop actually reads. `firstStep` then forces `decOut` to be recomputed
                    // rather than reused via the blank-skip branch.
                    resetSegment()
                    fillFloatNDArray(&buffers.hIn, with: hiddenState)
                    fillFloatNDArray(&buffers.cIn, with: cellState)
                    coverage.predictorResets += 1
                }
                if collectStats {
                    stepTimesMs.append((ContinuousClock.now - t0).inMilliseconds)
                }
                frame += advance
            }

            // Carry the overshoot rather than clamping it. See `timeJump`.
            timeJump = max(0, frame - frames.upperBound)

            // Hand the state to the next chunk. `hIn`/`cIn` are read after the swaps, so
            // they hold the state the *next* step should run from.
            hiddenState = flattenAsFloat(buffers.hIn)
            cellState = flattenAsFloat(buffers.cIn)
            decoderOutput = flattenAsFloat(buffers.decOut)

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

        /// Preconditions the frame arithmetic depends on, shared by both overloads.
        private func checkFrames(
            _ frames: Range<Int>, windowStartFrame: Int, windowEncoderFrames: Int
        ) throws {
            guard frames.lowerBound >= windowStartFrame,
                frames.upperBound - windowStartFrame <= windowEncoderFrames
            else {
                throw SpeechError.invalidStreamingConfig(
                    "frames \(frames) fall outside the window starting at \(windowStartFrame) "
                        + "with \(windowEncoderFrames) encoder frames")
            }
        }
    }

    /// A fresh streaming state for this decoder's graphs.
    public func makeStream(config: ParakeetTDTConfig) -> Stream {
        Stream(decoder: self, config: config)
    }

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        try await decode(
            encoderOutput: encoderOutput,
            encoderOutputShape: encoderOutputShape,
            validEncoderFrames: validEncoderFrames,
            resources: resources,
            resetAfterSilenceFrames: 0)
    }

    /// As the protocol's `decode`, with the transducer's long-silence predictor reset.
    ///
    /// Not on `SpeechDecoder` because Whisper has no predictor to re-seed. Callers reach it the
    /// way `startStream` does, by downcasting to this type.
    ///
    /// - Parameter resetAfterSilenceFrames: Silent encoder frames after which the predictor is
    ///   reset, or 0 to never. 0 is the right default offline: it keeps this path identical to
    ///   HF's `generate()`, which is what `--parity-test` rests on. See
    ///   `EndpointingConfig.resetAfterSilenceFrames`.
    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources,
        resetAfterSilenceFrames: Int
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        // Only the config is read from `resources`; the two graphs come from the same
        // models this decoder was initialized with, already loaded in `init`.
        guard case .parakeetTDT(_, _, let cfg) = resources else {
            throw SpeechError.incompatibleResources("ParakeetTDTDecoder requires .parakeetTDT resources")
        }
        try Self.validate(
            encoderOutputShape: encoderOutputShape,
            logitsSize: jointGraph.logits.shape.last!, config: cfg)

        let tEnc = encoderOutputShape[1]
        if tEnc == 0 { return (tokens: [], stats: DecodeStats(stepTimesMs: [])) }

        // For a static (padded) encoder the tail frames are zero-padding; decoding
        // them yields spurious tokens (trailing periods). Cap the loop at the number
        // of frames that carry real audio. Dynamic exports pass tEnc, so cap == tEnc.
        let cap = max(1, min(tEnc, validEncoderFrames))

        // The offline path is one chunk covering the whole utterance, over state that
        // nothing else will touch — so at the default reset of 0 it is byte-for-byte the
        // previous behaviour.
        return try await makeStream(config: cfg).decodeFrames(
            encoderOutput: encoderOutput,
            encoderOutputShape: encoderOutputShape,
            frames: 0..<cap,
            windowStartFrame: 0,
            resetAfterSilenceFrames: resetAfterSilenceFrames)
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

    /// `joint(decoder_output, encoder_frame)`, leaving the `[vocab | durations]` row in
    /// `buffers.logits` for the caller to scan in place.
    ///
    /// The decoder side comes straight from `buffers.decOut`, whether the last step wrote it
    /// or the blank-skip branch left it in place.
    private func runJoint(
        encoderFrame: ArraySlice<Float>, buffers: inout Buffers
    ) async throws {
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
    }
}
