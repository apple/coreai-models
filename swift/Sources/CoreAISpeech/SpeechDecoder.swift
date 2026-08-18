// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - DecoderResources

/// Architecture-specific assets handed to a `SpeechDecoder` per call.
public enum DecoderResources: Sendable {
    case whisper(decoder: AIModel, generationConfig: GenerationConfig)
    case parakeetTDT(decoderStep: AIModel, joint: AIModel, config: ParakeetTDTConfig)
}

// MARK: - DecodeStats

/// Per-step timing collected during the autoregressive decode loop, plus enough
/// diagnostics to tell which branches of that loop an input actually reached.
public struct DecodeStats: Sendable {
    public let stepTimesMs: [Double]

    /// How often each branch of the TDT loop fired.
    ///
    /// The token sequence is a discrete, all-or-nothing check: argmax absorbs numeric
    /// drift, so "tokens matched" says little about whether the blank-skip and duration
    /// bookkeeping were exercised at all. These counts let a verification harness assert
    /// that a given input reaches a given branch instead of assuming it does. Zero for
    /// architectures without a transducer loop.
    public struct Coverage: Sendable, Equatable {
        /// Inner-loop iterations that reused the cached decoder output instead of
        /// re-running the LSTM. This is how a blank input avoids advancing the state.
        public var blankSkipReuses = 0
        /// Iterations that ran the LSTM and adopted its new hidden/cell state.
        public var lstmStateAdvances = 0
        /// `isBlank && dur == 0` — a blank choosing duration 0, forced forward one frame.
        public var blankZeroDurationBreaks = 0
        /// `dur > 0` — the ordinary exit from the inner symbol loop.
        public var positiveDurationBreaks = 0
        /// `maxSymbolsPerStep` exhausted without any duration > 0 being chosen.
        public var symbolCapExhaustions = 0
        /// Outer steps that emitted more than one non-blank token.
        public var multiTokenSteps = 0
        /// Outer steps that emitted no token at all.
        public var blankOnlySteps = 0
        /// Times the predictor was zeroed after a long silence, which only happens when a
        /// caller opts in with `resetAfterSilenceFrames`. 0 on the offline path.
        public var predictorResets = 0

        public init() {}
    }

    public let coverage: Coverage

    /// The first `decoder_step` and `joint` outputs of the decode.
    ///
    /// Retained so a harness can compare those graphs tensor-to-tensor as the runtime
    /// actually drives them, rather than only through the token sequence. Costs one
    /// retained copy of values already computed on that step.
    public struct FirstStep: Sendable {
        public let decoderOutput: [Float]
        public let newHiddenState: [Float]
        public let newCellState: [Float]
        public let jointLogits: [Float]
    }

    public let firstStep: FirstStep?

    public init(
        stepTimesMs: [Double], coverage: Coverage = Coverage(), firstStep: FirstStep? = nil
    ) {
        self.stepTimesMs = stepTimesMs
        self.coverage = coverage
        self.firstStep = firstStep
    }

    public var stepCount: Int { stepTimesMs.count }
    public var avgLatencyMs: Double {
        guard !stepTimesMs.isEmpty else { return 0 }
        return stepTimesMs.reduce(0, +) / Double(stepTimesMs.count)
    }
    public var minLatencyMs: Double { stepTimesMs.min() ?? 0 }
    public var maxLatencyMs: Double { stepTimesMs.max() ?? 0 }
    public var stepsPerSecond: Double { avgLatencyMs > 0 ? 1000 / avgLatencyMs : 0 }
}

// MARK: - SpeechDecoder protocol

/// Model-specific decode logic.
public protocol SpeechDecoder: Sendable {
    func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats)
}

// MARK: - Helpers

/// Greedy decoder for Whisper (encoder-decoder, cross-attention, KV cache).
public struct WhisperDecoder: SpeechDecoder {
    public init() {}

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        // `validEncoderFrames` is ignored: Whisper is encoder-decoder and stops on
        // `eotToken`, not on encoder-frame exhaustion, so padded encoder frames don't
        // produce trailing tokens the way the Parakeet TDT frame loop does.
        guard case .whisper(let decoderModel, let config) = resources else {
            throw SpeechError.incompatibleResources("WhisperDecoder requires .whisper resources")
        }
        guard let decFn = try decoderModel.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in decoder")
        }
        let decDesc = decoderModel.functionDescriptor(for: "main")!

        guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let posIdsNDDesc) = decDesc.inputDescriptor(of: "position_ids"),
            case .ndArray(_) = decDesc.inputDescriptor(of: "encoder_hidden_states"),
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
                    "encoder_hidden_states": encoderOutput,
                ],
                states: consume st, outputViews: consume out)
        }

        // Prime KV cache with forced prefix
        var tokens: [Int32] = config.forcedPrefix
        for (i, tok) in config.forcedPrefix.enumerated() {
            try await step(tok, pos: i)
        }

        // Greedy decode
        var stepTimesMs: [Double] = []
        var pos = config.forcedPrefix.count
        while tokens.count - config.forcedPrefix.count < config.maxDecodeSteps {
            let t0 = ContinuousClock.now
            try await step(tokens.last!, pos: pos)
            stepTimesMs.append((ContinuousClock.now - t0).inMilliseconds)
            let logits = flattenAsFloat(logitsArray)
            let next = Int32(logits.indices.max(by: { logits[$0] < logits[$1] })!)
            tokens.append(next)
            pos += 1
            if next == config.eotToken { break }
        }

        return (tokens: tokens, stats: DecodeStats(stepTimesMs: stepTimesMs))
    }
}
