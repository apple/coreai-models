// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// How a decode loop turns logits into the next token. Fixed per engine at init, so each engine
/// keeps its own sampling behavior.
struct SamplingPolicy: Sendable {
    /// Final-logit soft cap, applied on the CPU before sampling and to returned logits, for
    /// exports that leave `c · tanh(logits / c)` out of the graph. `nil` for none.
    let logitSoftcap: Double?
    /// Whether the sampling configuration goes through `SamplingConfiguration.normalized()`.
    let normalizesSampling: Bool
}

/// Decode-loop bookkeeping shared by the sequential and static-shape engines' iterators.
///
/// Owns the growing token sequence, the step count, stop and cancel handling, and next-token
/// selection. An iterator drives it from `next()` and supplies its engine's forward pass.
struct DecodeLoop {
    /// Prompt tokens followed by every token generated so far.
    private(set) var inputTokens: [Int32]
    let generationStartOffset: Int
    let maxTokens: Int
    let forcedContinuation: [Int32]?
    let returnsLogits: Bool
    let stopReasonStore: StopReasonStore
    var step: Int = 0
    private(set) var finished: Bool = false

    private let samplingConfiguration: SamplingConfiguration
    private let logitSoftcap: Double?
    private let generationToken: GenerationToken
    private let release: () -> Void

    /// - Parameter release: Clears the engine's active generation token. Called once, when
    ///   generation finishes.
    init(
        input: [Int32],
        samplingConfiguration: SamplingConfiguration,
        samplingPolicy: SamplingPolicy,
        inferenceOptions: InferenceOptions,
        maxContextLength: Int,
        stopReasonStore: StopReasonStore,
        generationToken: GenerationToken,
        release: @escaping () -> Void
    ) {
        self.inputTokens = input
        self.generationStartOffset = input.count
        self.maxTokens = Self.clampMaxTokens(
            requested: inferenceOptions.maxTokens,
            forcedCount: inferenceOptions.forcedContinuation?.count,
            inputCount: input.count,
            maxContextLength: maxContextLength
        )
        self.forcedContinuation = inferenceOptions.forcedContinuation
        self.returnsLogits = inferenceOptions.returnsLogits
        self.samplingConfiguration =
            samplingPolicy.normalizesSampling ? samplingConfiguration.normalized() : samplingConfiguration
        self.logitSoftcap = samplingPolicy.logitSoftcap
        self.stopReasonStore = stopReasonStore
        self.generationToken = generationToken
        self.release = release
    }

    /// Whether another step may run. Records the stop reason and finishes when generation is
    /// finished, cancelled, or out of budget.
    mutating func shouldContinue() -> Bool {
        if finished { return false }

        if generationToken.isCancelled {
            stopReasonStore.set(.cancelled)
            finish()
            return false
        }

        guard step < maxTokens else {
            // Natural exhaustion. Don't clobber a reason a decoder set (e.g. `.eos`).
            stopReasonStore.setIfUnset(.maxTokens)
            finish()
            return false
        }
        return true
    }

    /// Run one decode step: `forward` processes the not-yet-seen tokens of `inputTokens` and
    /// returns the last position's raw logits; the next token is then selected and appended.
    /// Returned logits carry the policy's soft cap.
    ///
    /// Returns `nil` when generation was cancelled during the forward pass. Errors from
    /// `forward` finish the loop with `.cancelled` or `.error` and are rethrown.
    mutating func decodeStep(
        forward: ([Int32]) async throws -> [LogitsScalarType]
    ) async throws -> InferenceOutput? {
        do {
            try Task.checkCancellation()

            let logitBuffer = try await forward(inputTokens)

            // Check cancellation after inference step
            if generationToken.isCancelled {
                stopReasonStore.set(.cancelled)
                finish()
                return nil
            }

            let nextToken = Self.nextToken(
                fromLogits: logitBuffer,
                forced: forcedContinuation,
                step: step,
                sampling: samplingConfiguration,
                tokenHistory: inputTokens[generationStartOffset...],
                logitSoftcap: logitSoftcap
            )

            inputTokens.append(nextToken)
            step += 1

            var returnedLogits = returnsLogits ? logitBuffer : nil
            if let logitSoftcap, returnedLogits != nil {
                LogitSoftcapProcessor.apply(to: &returnedLogits!, cap: Float(logitSoftcap))
            }
            return InferenceOutput(tokenId: nextToken, logits: returnedLogits)
        } catch is CancellationError {
            stopReasonStore.set(.cancelled)
            finish()
            throw CancellationError()
        } catch {
            stopReasonStore.set(.error)
            finish()
            throw error
        }
    }

    /// The generation-length cap for an iterator.
    ///
    /// A forced continuation replays exactly its own tokens; otherwise the requested budget
    /// (`nil` meaning "unbounded") is clamped to the context left after the prompt.
    static func clampMaxTokens(
        requested: Int?,
        forcedCount: Int?,
        inputCount: Int,
        maxContextLength: Int
    ) -> Int {
        if let forcedCount {
            return forcedCount
        }
        return min(requested ?? Int.max, max(0, maxContextLength - inputCount))
    }

    /// Select the next token: the forced-continuation token when replaying, otherwise the
    /// sampler's choice, with `logitSoftcap` applied first.
    ///
    /// `logits` is taken by value; the sampler mutates a copy-on-write copy so the caller's
    /// buffer (which it may also return to the consumer) is left untouched.
    static func nextToken(
        fromLogits logits: [LogitsScalarType],
        forced: [Int32]?,
        step: Int,
        sampling: SamplingConfiguration,
        tokenHistory: ArraySlice<Int32>,
        logitSoftcap: Double? = nil
    ) -> Int32 {
        if let forced {
            return forced[step]
        }
        var logits = logits
        return sampling.fallbackSampler(
            from: &logits, tokenHistory: tokenHistory, step: step, logitSoftcap: logitSoftcap)
    }

    /// Mark generation finished and release the engine's active token. Idempotent.
    mutating func finish() {
        guard !finished else {
            return
        }
        finished = true
        release()
    }
}
