// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Prefill Strategy

/// Determines the optimal prefill strategy based on prompt size.
enum PrefillStrategy {
    case chunked(chunkSize: Int)
    case wholeBatch
    case oneAtATime
}

// MARK: - Core AI Sequential Clean Engine

/// Clean Core AI inference engine built from scratch using only public APIs.
///
/// ## Model Contract
///
/// Expects a `.aimodel` with:
/// - **2 inputs**: `input_ids` (Int32), `position_ids` (Int32)
/// - **1 output**: `logits` (LogitsScalarType)
/// - **2–4 states**: KV cache pair + optional persistent states (hybrid models), updated in-place
///
/// KV cache NDArrays start small (256 tokens) and grow dynamically with 2× expansion.
/// Passed as `states` on every forward pass; the model graph updates them in-place.
public final class CoreAISequentialEngine: InferenceEngine, IdempotentEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }
    public var vocabSize: Int { config.vocabSize }
    public var hasRecurrentState: Bool { session.hasNonTruncatableStates }
    public let config: ModelConfig

    // Core AI function handle
    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    // Optional prefill graph. Prefill chunks run here when the asset has it. It produces
    // no logits, so the last prompt token still goes through `function`.
    private let prefillFunction: InferenceFunction?

    // I/O names from descriptor
    private let logitsName: String

    // Input handling — handler owns allocation and fill logic
    private var inputHandler: TokenInputHandler

    /// Retained to build per-session state handlers in `makeSessionState()`.
    private let options: EngineOptions

    // Per-request mutable state. The `generate()` shim owns one internal session and
    // reuses it across calls, preserving implicit prefix-cache reuse and reset(to:)
    // semantics. Idempotent callers pass their own session instead.
    private let session: GenerationSessionState

    // Logits descriptor and buffer
    private let logitsDescriptor: NDArrayDescriptor
    private var logitsArray: NDArray
    private var cachedLogitsBatchSize: Int

    // Ring buffer mode: handled by TokenInputHandler.useCompactPositionIds

    // Track processed tokens for incremental inference (delegated to the shim session).
    public var processedTokenCount: Int { session.processedTokenCount }

    public private(set) var lastPrefixHitCount: Int = 0

    // Track in-flight generation via token (replaces simple bool lock)
    private let tokenBox = GenerationTokenBox()

    public var isBusy: Bool { tokenBox.isBusy }

    /// Clear the engine's active token if it matches the given token.
    /// Called by the iterator when generation finishes or is cancelled.
    func clearTokenIfActive(_ token: GenerationToken) {
        tokenBox.clearIfActive(token)
    }

    // MARK: - Init

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        self.config = config

        let modelLoadSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAICleanModelLoading",
            details: "Loading \(config.name) from prepared asset"
        )

        let model = preparedModel.model

        // Get function descriptor
        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(config.function)' in model")
        }
        self.functionDescriptor = descriptor

        // Validate model architecture: 2 inputs, 1+ output, at least KV cache pair.
        // Hybrid models may declare additional persistent fixed-shape states.
        guard descriptor.inputNames.count == 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "Expected 2 inputs, got \(descriptor.inputNames.count): \(descriptor.inputNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count): \(descriptor.outputNames)")
        }
        guard descriptor.stateNames.count >= 2 && descriptor.stateNames.count <= 4 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected 2–4 states (KV cache + optional persistent states), got \(descriptor.stateNames.count): "
                    + "states=\(descriptor.stateNames), outputs=\(descriptor.outputNames)")
        }

        // Create state handlers from descriptor
        let stateHandlers = try StateHandlerFactory.createSyncHandlers(
            descriptor: descriptor,
            maxContextLength: config.maxContextLength,
            options: options
        )
        self.options = options
        self.session = GenerationSessionState(
            kvCache: stateHandlers.kvCache,
            additionalStates: stateHandlers.additionalStates,
            hasNonTruncatableStates: stateHandlers.hasNonTruncatableStates
        )

        let layout = try InputLayout.analyze(
            model: model, functionName: config.function, config: config,
            useCompactPositionIds: stateHandlers.isAllSlidingCache)

        self.logitsName = layout.logitsName
        let inputIdsName = layout.inputIdsName
        let positionIdsName = layout.positionIdsName

        // Create input handler from descriptors
        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(inputIdsName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(positionIdsName)'")
        }

        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }
        self.logitsDescriptor = logitsDesc

        self.inputHandler = TokenInputHandler(
            inputIdsName: inputIdsName,
            positionIdsName: positionIdsName,
            inputIdsDescriptor: inputIdsDesc,
            positionIdsDescriptor: posIdsDesc,
            useCompactPositionIds: layout.positionPolicy == .compact
        )

        CLILogger.log(
            "KV cache: capacity=\(stateHandlers.kvCache.currentCapacity), states=\(stateHandlers.kvCache.stateNames)"
        )
        if let additional = stateHandlers.additionalStates {
            CLILogger.log(
                "Additional persistent states: \(additional.stateNames.joined(separator: ", "))")
        }

        // Allocate initial logits (1 token — will be reallocated per batch)
        let initLogitsDesc = logitsDesc.resolvingDynamicDimensions([1, 1, config.vocabSize])
        self.logitsArray = NDArray(descriptor: initLogitsDesc)
        self.cachedLogitsBatchSize = 1

        // Load inference function
        self.prefillFunction = try loadPrefillGraph(
            from: model, matching: descriptor, mainName: config.function)
        if self.prefillFunction != nil {
            CLILogger.log("Found '\(prefillGraphFunctionName)' graph — prefill skips the LM head")
        }

        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot load function '\(config.function)'")
        }
        self.function = fn

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAICleanModelLoading",
            signpostID: modelLoadSignpost
        )

        CLILogger.log(
            "CoreAI clean engine initialized — inputs: \(descriptor.inputNames), outputs: \(descriptor.outputNames), states: \(descriptor.stateNames)"
        )
    }

    /// Convenience initializer with direct model URL.
    public convenience init(
        config: ModelConfig,
        modelURL: URL,
        options: EngineOptions = EngineOptions()
    ) async throws {
        CLILogger.log("Loading CoreAI model asset from: \(modelURL.lastPathComponent)")
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(config: config, preparedModel: preparedModel, options: options)
    }

    // MARK: - Prefill Strategy

    private func selectPrefillStrategy(newTokenCount: Int) -> PrefillStrategy {
        // With a prefill graph, chunking is cheaper at any size: every chunk but the last
        // token skips the LM head, so there is no threshold to clear.
        if shouldChunkPrefill(
            tokenCount: newTokenCount,
            hasPrefillGraph: prefillFunction != nil,
            chunkThreshold: config.chunkThreshold)
        {
            return .chunked(chunkSize: config.prefillChunkSize)
        }
        return .wholeBatch
    }

    // MARK: - Token Batch Processing

    /// Process a batch of tokens in a single forward pass.
    private func processTokenBatch(
        _ tokens: ArraySlice<Int32>, sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        let batchSize = tokens.count
        guard batchSize > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot process empty token batch")
        }

        _ = try sessionState.kvCache.ensureCapacity(
            forContextLength: sessionState.processedTokenCount + batchSize)

        let batchSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Batch",
            details: "\(batchSize) tokens at position \(sessionState.processedTokenCount)"
        )

        let context = InputContext.dynamic(
            tokens: tokens, processedTokenCount: sessionState.processedTokenCount)
        let inputs = try await inputHandler.prepare(context)

        // Reuse pre-allocated logits when the batch size is unchanged.
        if cachedLogitsBatchSize != batchSize {
            let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions([1, batchSize, config.vocabSize])
            logitsArray = NDArray(descriptor: resolvedLogitsDesc)
            cachedLogitsBatchSize = batchSize
        }

        // Bind states, build output views, and execute
        try await runWithStates(
            function: function,
            inputs: inputs,
            primary: sessionState.kvCache,
            secondary: sessionState.additionalStates,
            outputArray: &logitsArray,
            outputName: logitsName
        )

        // Read logits from NDArray
        let totalLogits = batchSize * config.vocabSize
        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: totalLogits)

        sessionState.processedTokenCount += batchSize

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIClean Batch",
            signpostID: batchSignpost
        )

        return logitBuffer
    }

    /// Run one prefill chunk on the prefill graph: KV cache writes only, no logits.
    private func encodePrefillChunk(
        _ tokens: ArraySlice<Int32>, using prefillFn: InferenceFunction,
        sessionState: GenerationSessionState
    ) async throws {
        let batchSize = tokens.count
        _ = try sessionState.kvCache.ensureCapacity(
            forContextLength: sessionState.processedTokenCount + batchSize)

        let context = InputContext.dynamic(
            tokens: tokens, processedTokenCount: sessionState.processedTokenCount)
        let inputs = try await inputHandler.prepare(context)

        try await runWithStatesNoOutputs(
            function: prefillFn,
            inputs: inputs,
            primary: sessionState.kvCache,
            secondary: sessionState.additionalStates)

        sessionState.processedTokenCount += batchSize
    }

    // MARK: - Chunked Prefill

    private func processChunkedPrompt(
        tokens: ArraySlice<Int32>,
        chunkSize: Int,
        sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        // The prefill graph produces no logits, so hold the final token back for
        // `function`: it is the one whose logits seed sampling. Without one, nothing is
        // held back and the trailing chunk carries the logits.
        let heldBack = prefillHeldBackTokens(hasPrefillGraph: prefillFunction != nil)

        let chunkSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Chunked Prefill",
            details: "\(tokens.count) tokens, chunkSize \(chunkSize)"
        )
        defer {
            InstrumentsProfiler.endCustomInterval(
                name: "CoreAIClean Chunked Prefill",
                signpostID: chunkSignpost
            )
        }

        return try await runChunkedPrefill(
            tokens: tokens,
            chunkSize: chunkSize,
            heldBack: heldBack,
            vocabSize: config.vocabSize
        ) { chunk, isHeldBack in
            // Held-back tail (and every chunk when there is no prefill graph) runs through
            // `main` for logits; earlier chunks fill the KV cache via the prefill graph.
            if !isHeldBack, let prefillFn = self.prefillFunction {
                try await self.encodePrefillChunk(chunk, using: prefillFn, sessionState: sessionState)
                return []
            }
            return try await self.processTokenBatch(chunk, sessionState: sessionState)
        }
    }

    /// Process tokens in chunks, returning ALL position logits (not just last token).
    /// Used for batched PPL evaluation where every position's logits are needed.
    func processChunkedPromptAllLogits(
        tokens: ArraySlice<Int32>,
        chunkSize: Int,
        sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        var allLogits: [LogitsScalarType] = []
        var remainingTokens = tokens

        while !remainingTokens.isEmpty {
            let currentChunkSize = min(chunkSize, remainingTokens.count)
            let chunkEnd = remainingTokens.startIndex + currentChunkSize
            let chunk = remainingTokens[remainingTokens.startIndex..<chunkEnd]
            let chunkLogits = try await processTokenBatch(chunk, sessionState: sessionState)
            allLogits.append(contentsOf: chunkLogits)
            remainingTokens = remainingTokens[chunkEnd...]
        }

        return allLogits
    }

    // MARK: - Generate (primary API)

    /// Shim over the idempotent generate path: drives generation on the engine's single
    /// internal session, preserving today's implicit prefix-cache reuse and reset(to:)
    /// semantics for existing callers.
    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        try await generate(
            with: input,
            sessionState: session,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions
        )
    }

    // MARK: - Lifecycle

    /// Wait for any in-flight generate() Task to finish.
    private func drain() {
        var attempts = 0
        while tokenBox.isBusy {
            attempts += 1
            if attempts > 5000 {
                fatalError("Sequential engine drain() timeout — generation Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    public func cancel() async throws {
        tokenBox.cancelActive()
    }

    public func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= session.processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(session.processedTokenCount)]")
        if tokenIndex != 0 && session.hasNonTruncatableStates {
            throw InferenceRuntimeError.invalidState(
                "Partial reset is not supported for hybrid models with recurrent state. "
                    + "Use reset(to: 0) and replay the prefix.")
        }
        tokenBox.cancelActive()
        internalReset(to: tokenIndex, sessionState: session)
    }

    /// Internal reset without cancelling the active generation token.
    /// Used by `reset(to:)` and the prefix-cache routing in `generate(with:sessionState:)`.
    func internalReset(to tokenIndex: Int, sessionState: GenerationSessionState) {
        let resetSpan = InstrumentsProfiler.beginReset(engine: "CoreAIClean")
        if tokenIndex == 0 {
            sessionState.processedTokenCount = 0
            sessionState.history.clear()
            sessionState.kvCache.reset()
            sessionState.additionalStates?.reset()
        } else {
            sessionState.processedTokenCount = tokenIndex
            sessionState.history.truncate(to: tokenIndex)
        }
        resetSpan.end()
    }

    public func cleanup() {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAIClean")
        CLILogger.log("CoreAI clean engine cleanup complete")
        cleanupSpan.end()
    }

    // MARK: - Helpers
}

// MARK: - Idempotent Engine Conformance

extension CoreAISequentialEngine {
    /// Mint a fresh, independent per-request state (fresh KV cache + empty history).
    package func makeSessionState() throws -> GenerationSessionState {
        let stateHandlers = try StateHandlerFactory.createSyncHandlers(
            descriptor: functionDescriptor,
            maxContextLength: config.maxContextLength,
            options: options)
        return GenerationSessionState(
            kvCache: stateHandlers.kvCache,
            additionalStates: stateHandlers.additionalStates,
            hasNonTruncatableStates: stateHandlers.hasNonTruncatableStates)
    }

    /// Drive generation over the caller-supplied state instead of engine-owned state.
    package func generate(
        with input: [TokenId],
        sessionState: GenerationSessionState,
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        // Cancel any prior generation so its Iterator stops on next poll.
        tokenBox.cancelActive()

        // Implicit prefix caching: resolve input against history.
        // For hybrid models with recurrent states, we must full-reset on any
        // rewind because recurrent state summarizes the whole prefix and cannot
        // be truncated by moving a KV cursor. Future: checkpoint/restore.
        // Default to 0 so a fresh session (no history) always reports a clean
        // value instead of leaking the previous session's hit count.
        lastPrefixHitCount = 0
        if sessionState.history.count > 0 {
            let (commonPrefix, _) = sessionState.history.resolve(input: input)
            let plan = Self.prefixResetPlan(
                commonPrefix: commonPrefix,
                historyCount: sessionState.history.count,
                processedTokenCount: sessionState.processedTokenCount,
                inputCount: input.count,
                hasNonTruncatableStates: sessionState.hasNonTruncatableStates
            )
            if let resetTo = plan.resetTo {
                internalReset(to: resetTo, sessionState: sessionState)
            }
            lastPrefixHitCount = plan.prefixHitCount
        }

        let token = GenerationToken()
        tokenBox.install(token)
        return GenerationSequence(
            engine: self,
            input: input,
            sessionState: sessionState,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    /// Pure decision for the implicit prefix-cache routing in `generate(with:sessionState:)`.
    ///
    /// Given the common-prefix length resolved against the session history and the
    /// current cursor state, decides how far to rewind the KV cache and what to
    /// report as the prefix hit count. `resetTo == nil` means no reset is needed.
    ///
    /// Extracted so the four-way routing can be unit-tested without a compiled model;
    /// the logic is equivalent to the inline decision it replaced.
    static func prefixResetPlan(
        commonPrefix: Int,
        historyCount: Int,
        processedTokenCount: Int,
        inputCount: Int,
        hasNonTruncatableStates: Bool
    ) -> (resetTo: Int?, prefixHitCount: Int) {
        if hasNonTruncatableStates {
            // Hybrid model: recurrent state can't be partially rewound.
            // Full reset and replay the entire prompt.
            if commonPrefix < historyCount || processedTokenCount >= inputCount {
                return (0, 0)
            }
            return (nil, 0)
        } else if commonPrefix < inputCount && commonPrefix < historyCount {
            // Divergence: input differs from history. Full reset needed.
            return (0, commonPrefix)
        } else if processedTokenCount >= inputCount {
            // Pure extension: all input tokens match history. Rewind for seeding.
            return (Swift.max(0, commonPrefix - 1), commonPrefix)
        } else {
            return (nil, commonPrefix)
        }
    }
}

extension CoreAISequentialEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let engine: CoreAISequentialEngine
        let input: [CoreAISequentialEngine.TokenId]
        let sessionState: GenerationSessionState
        let samplingConfiguration: SamplingConfiguration
        let inferenceOptions: InferenceOptions
        let generationToken: GenerationToken

        /// Shared with the iterator so the caller can read why generation ended.
        let stopReasonStore = StopReasonStore()

        public var stopReason: StopReason? { stopReasonStore.stopReason }

        public func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                engine: engine,
                input: input,
                sessionState: sessionState,
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore,
                generationToken: generationToken
            )
        }
    }
}

extension CoreAISequentialEngine.GenerationSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: CoreAISequentialEngine
        private let sessionState: GenerationSessionState
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [CoreAISequentialEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore
        private let generationToken: GenerationToken

        private var inputTokens: [CoreAISequentialEngine.TokenId]
        private let generationStartOffset: Int
        private var step: Int = 0
        private var finished: Bool = false
        // Pre-computed logits for batched forcedContinuation evaluation.
        // When non-nil, next() yields from this buffer instead of running inference.
        private var batchedLogitsBuffer: [[LogitsScalarType]]?

        init(
            engine: CoreAISequentialEngine,
            input: [CoreAISequentialEngine.TokenId],
            sessionState: GenerationSessionState,
            samplingConfiguration: SamplingConfiguration,
            inferenceOptions: InferenceOptions,
            stopReasonStore: StopReasonStore,
            generationToken: GenerationToken
        ) {
            self.engine = engine
            self.sessionState = sessionState
            self.samplingConfiguration = samplingConfiguration.normalized()
            self.returnsLogits = inferenceOptions.includeLogits
            self.forcedContinuation = inferenceOptions.forcedContinuation
            self.stopReasonStore = stopReasonStore
            self.generationToken = generationToken
            self.inputTokens = input
            self.generationStartOffset = input.count
            self.maxTokens = SequentialIterator.clampMaxTokens(
                requested: inferenceOptions.maxTokens,
                forcedCount: inferenceOptions.forcedContinuation?.count,
                inputCount: input.count,
                maxContextLength: engine.config.maxContextLength
            )
        }

        deinit {
            engine.clearTokenIfActive(generationToken)
        }

        public func next() async throws -> InferenceOutput? {
            if finished { return nil }

            if generationToken.isCancelled {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                return nil
            }

            guard step < maxTokens else {
                // Natural exhaustion. Don't clobber a reason a decoder set (e.g. `.eos`).
                stopReasonStore.setIfUnset(.maxTokens)
                finishAndRelease()
                return nil
            }

            // Fast path: batched forcedContinuation with logits.
            // All tokens were processed in one prefill; yield pre-computed logits.
            if let buffer = batchedLogitsBuffer {
                let logits = buffer[step]
                let token = forcedContinuation![step]
                step += 1
                if step >= maxTokens {
                    stopReasonStore.setIfUnset(.maxTokens)
                    finishAndRelease()
                }
                return InferenceOutput(tokenId: token, logits: logits)
            }

            // First call with forcedContinuation + logits: batch-process all tokens at once.
            if let forced = forcedContinuation, returnsLogits, step == 0 {
                let allTokens = inputTokens + forced.map { $0 }
                let vocabSize = engine.config.vocabSize

                let allLogits: [LogitsScalarType]
                let strategy = engine.selectPrefillStrategy(newTokenCount: allTokens.count)
                switch strategy {
                case .chunked(let chunkSize):
                    allLogits = try await engine.processChunkedPromptAllLogits(
                        tokens: allTokens[...], chunkSize: chunkSize, sessionState: sessionState)
                case .wholeBatch:
                    allLogits = try await engine.processTokenBatch(
                        allTokens[...], sessionState: sessionState)
                case .oneAtATime:
                    var collected: [LogitsScalarType] = []
                    for j in allTokens.indices {
                        collected.append(
                            contentsOf: try await engine.processTokenBatch(
                                allTokens[j...j], sessionState: sessionState))
                    }
                    allLogits = collected
                }

                // Split into per-position logit vectors.
                // Skip the prompt positions (inputTokens.count - 1 positions);
                // we want logits that predict each forced token.
                let promptLen = inputTokens.count
                var buffer: [[LogitsScalarType]] = []
                for i in 0..<forced.count {
                    let offset = (promptLen - 1 + i) * vocabSize
                    let endOffset = offset + vocabSize
                    guard endOffset <= allLogits.count else {
                        throw InferenceRuntimeError.invalidState(
                            "Batched logits underflow at position \(i): need \(endOffset), got \(allLogits.count)")
                    }
                    buffer.append(Array(allLogits[offset..<endOffset]))
                }
                batchedLogitsBuffer = buffer

                // Update engine state
                sessionState.history.append(contentsOf: allTokens[...])

                // Yield first result
                let logits = buffer[step]
                let token = forced[step]
                step += 1
                return InferenceOutput(tokenId: token, logits: logits)
            }

            do {
                try Task.checkCancellation()

                guard sessionState.processedTokenCount < inputTokens.count else {
                    throw InferenceRuntimeError.invalidState("No new tokens to process")
                }

                let oldProcessedCount = sessionState.processedTokenCount
                let newTokens = inputTokens[sessionState.processedTokenCount...]
                let strategy = engine.selectPrefillStrategy(newTokenCount: newTokens.count)

                let logitBuffer: [LogitsScalarType]
                switch strategy {
                case .chunked(let chunkSize):
                    logitBuffer = try await engine.processChunkedPrompt(
                        tokens: newTokens, chunkSize: chunkSize, sessionState: sessionState)
                case .wholeBatch:
                    let allLogits = try await engine.processTokenBatch(
                        newTokens, sessionState: sessionState)
                    logitBuffer = lastTokenLogits(from: allLogits, vocabSize: engine.config.vocabSize)
                case .oneAtATime:
                    var lastLogits: [LogitsScalarType] = []
                    for j in newTokens.indices {
                        lastLogits = try await engine.processTokenBatch(
                            newTokens[j...j], sessionState: sessionState)
                    }
                    logitBuffer = lastLogits
                }

                // Update history with newly processed tokens
                let processedSlice = inputTokens[oldProcessedCount..<sessionState.processedTokenCount]
                sessionState.history.append(contentsOf: processedSlice)

                // Check cancellation after inference step
                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
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
                finishAndRelease()
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                finishAndRelease()
                throw error
            }
        }

        private func finishAndRelease() {
            guard !finished else {
                return
            }
            finished = true
            engine.clearTokenIfActive(generationToken)
        }
    }
}
