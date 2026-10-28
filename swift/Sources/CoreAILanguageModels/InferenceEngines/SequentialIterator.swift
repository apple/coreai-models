// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Token-count clamp, next-token selection, and prefill-strategy dispatch shared by the
/// sequential engines' iterators (see `SequentialDecodeLoop`).
enum SequentialIterator {
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
    /// sampler's choice.
    ///
    /// `logits` is taken by value; the sampler mutates a copy-on-write copy so the caller's
    /// buffer (which it may also return to the consumer) is left untouched.
    static func nextToken(
        fromLogits logits: [LogitsScalarType],
        forced: [Int32]?,
        step: Int,
        sampling: SamplingConfiguration,
        tokenHistory: ArraySlice<Int32>
    ) -> Int32 {
        if let forced {
            return forced[step]
        }
        return BatchedSampler.sample(
            rows: [logits],
            configurations: [sampling],
            histories: [tokenHistory],
            steps: [step]
        )[0]
    }

    /// Run a prompt through the forward pass with the given prefill strategy, returning the
    /// logits of its last position.
    ///
    /// `chunked` runs a chunked prefill and returns last-position logits. `batch` runs one
    /// forward pass over a slice.
    static func lastPositionLogits(
        for tokens: ArraySlice<Int32>,
        strategy: PrefillStrategy,
        vocabSize: Int,
        chunked: (ArraySlice<Int32>, Int) async throws -> [LogitsScalarType],
        batch: (ArraySlice<Int32>) async throws -> [LogitsScalarType]
    ) async throws -> [LogitsScalarType] {
        switch strategy {
        case .chunked(let chunkSize):
            return try await chunked(tokens, chunkSize)
        case .wholeBatch:
            let allLogits = try await batch(tokens)
            return lastTokenLogits(from: allLogits, vocabSize: vocabSize)
        case .oneAtATime:
            var lastLogits: [LogitsScalarType] = []
            for j in tokens.indices {
                lastLogits = try await batch(tokens[j...j])
            }
            return lastLogits
        }
    }
}

/// Decode-loop bookkeeping shared by the sequential engines' iterators.
///
/// Owns the growing token sequence, the step count, stop and cancel handling, and next-token
/// selection. An iterator drives it from `next()` and supplies its engine's forward pass.
struct SequentialDecodeLoop {
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
    private let generationToken: GenerationToken
    private let release: () -> Void

    /// - Parameter release: Clears the engine's active generation token. Called once, when
    ///   generation finishes.
    init(
        input: [Int32],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions,
        maxContextLength: Int,
        stopReasonStore: StopReasonStore,
        generationToken: GenerationToken,
        release: @escaping () -> Void
    ) {
        self.inputTokens = input
        self.generationStartOffset = input.count
        self.maxTokens = SequentialIterator.clampMaxTokens(
            requested: inferenceOptions.maxTokens,
            forcedCount: inferenceOptions.forcedContinuation?.count,
            inputCount: input.count,
            maxContextLength: maxContextLength
        )
        self.forcedContinuation = inferenceOptions.forcedContinuation
        self.returnsLogits = inferenceOptions.returnsLogits
        self.samplingConfiguration = samplingConfiguration.normalized()
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
    /// returns the last position's logits; the next token is then selected and appended.
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

            let nextToken = SequentialIterator.nextToken(
                fromLogits: logitBuffer,
                forced: forcedContinuation,
                step: step,
                sampling: samplingConfiguration,
                tokenHistory: inputTokens[generationStartOffset...]
            )

            inputTokens.append(nextToken)
            step += 1

            return InferenceOutput(
                tokenId: nextToken,
                logits: returnsLogits ? logitBuffer : nil
            )
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

    /// Mark generation finished and release the engine's active token. Idempotent.
    mutating func finish() {
        guard !finished else {
            return
        }
        finished = true
        release()
    }
}
